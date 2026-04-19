"""QLoRA-дообучение Qwen2.5-1.5B на Lenta и профайлинг через torch.profiler."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .data import format_training_text


@dataclass(frozen=True)
class TrainConfig:
    # Colab T4 (sm75) -> fp16; bf16 на Turing недоступен.
    model_name: str = "Qwen/Qwen2.5-1.5B"
    max_seq_length: int = 512
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    num_train_epochs: float = 1.0
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_strategy: str = "no"
    seed: int = 42
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    profile_active_steps: int = 4
    profile_warmup_steps: int = 1
    profile_wait_steps: int = 1


def load_base_model(model_name: str, seed: int) -> tuple[torch.nn.Module, AutoTokenizer]:
    torch.manual_seed(seed)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    model.config.use_cache = False
    return model, tokenizer


def wrap_with_lora(model: torch.nn.Module, cfg: TrainConfig) -> torch.nn.Module:
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.lora_targets),
        bias="none",
        task_type="CAUSAL_LM",
    )
    return get_peft_model(model, lora)


def build_dataset(df: pd.DataFrame, tokenizer: AutoTokenizer, max_length: int) -> Dataset:
    texts = [format_training_text(t, x) for t, x in zip(df["title"], df["text"])]
    ds = Dataset.from_dict({"text": texts})

    def _tok(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(_tok, batched=True, remove_columns=["text"])


class ProfilerCallback(TrainerCallback):
    """Активирует torch.profiler на первые N шагов обучения, потом отключается."""

    def __init__(self, profiler: torch.profiler.profile, stop_after: int):
        self.profiler = profiler
        self.stop_after = stop_after
        self._active = False

    def on_train_begin(self, args, state, control, **kwargs):
        self.profiler.__enter__()
        self._active = True

    def on_step_end(self, args, state, control, **kwargs):
        if self._active:
            self.profiler.step()
            if state.global_step >= self.stop_after:
                self.profiler.__exit__(None, None, None)
                self._active = False

    def on_train_end(self, args, state, control, **kwargs):
        if self._active:
            self.profiler.__exit__(None, None, None)
            self._active = False


def build_profiler(trace_dir: Path, cfg: TrainConfig) -> torch.profiler.profile:
    trace_dir.mkdir(parents=True, exist_ok=True)
    return torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=cfg.profile_wait_steps,
            warmup=cfg.profile_warmup_steps,
            active=cfg.profile_active_steps,
            repeat=1,
        ),
        record_shapes=True,
        profile_memory=True,
        with_stack=False,
        on_trace_ready=torch.profiler.tensorboard_trace_handler(str(trace_dir)),
    )


def save_profiler_summary(prof: torch.profiler.profile, out_path: Path, top_k: int = 15) -> None:
    cuda_tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=top_k)
    cpu_tbl = prof.key_averages().table(sort_by="cpu_time_total", row_limit=top_k)
    mem_tbl = prof.key_averages().table(sort_by="self_cuda_memory_usage", row_limit=top_k)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        "# torch.profiler summary\n\n"
        f"## Top {top_k} по cuda_time_total\n\n```\n{cuda_tbl}\n```\n\n"
        f"## Top {top_k} по cpu_time_total\n\n```\n{cpu_tbl}\n```\n\n"
        f"## Top {top_k} по self_cuda_memory_usage\n\n```\n{mem_tbl}\n```\n",
        encoding="utf-8",
    )


def train_qlora(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: TrainConfig,
    output_dir: Path,
    profiler_trace_dir: Path | None = None,
) -> dict:
    model, tokenizer = load_base_model(cfg.model_name, cfg.seed)
    model = wrap_with_lora(model, cfg)
    train_ds = build_dataset(train_df, tokenizer, cfg.max_seq_length)
    eval_ds = build_dataset(eval_df, tokenizer, cfg.max_seq_length)

    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_strategy=cfg.save_strategy,
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    callbacks = []
    prof = None
    if profiler_trace_dir is not None:
        prof = build_profiler(profiler_trace_dir, cfg)
        stop_after = cfg.profile_wait_steps + cfg.profile_warmup_steps + cfg.profile_active_steps
        callbacks.append(ProfilerCallback(prof, stop_after))

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
        callbacks=callbacks,
    )
    train_result = trainer.train()
    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "train_loss": train_result.metrics.get("train_loss"),
        "train_runtime_sec": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "num_train_epochs": cfg.num_train_epochs,
        "effective_batch_size": cfg.per_device_batch_size * cfg.gradient_accumulation_steps,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if prof is not None:
        save_profiler_summary(prof, output_dir / "profiler_summary.md")
    return metrics
