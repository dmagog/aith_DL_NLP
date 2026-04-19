"""QLoRA-дообучение Qwen2.5-1.5B на Lenta и профайлинг через torch.profiler.

Профайлер и основной train — отдельные стадии: это даёт частичную
устойчивость к обрывам Colab-сессии (profiler_summary.md сохраняется ещё
до длинного train) и стандартный HF-чекпойнтинг для ручного resume.
"""
from __future__ import annotations

import gc
import json
from dataclasses import dataclass
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
    model_name: str = "Qwen/Qwen2.5-1.5B"
    # Colab T4 (Turing, sm75) поддерживает только fp16.
    # Короткие новостные сниппеты укладываются в 384 токена, это ускоряет шаг.
    max_seq_length: int = 384
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    # Половины эпохи достаточно для заметного эффекта QLoRA; вторая половина
    # удлиняет прогон, но не меняет выводы — и повышает риск отвала T4-сессии.
    num_train_epochs: float = 0.5
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "cosine"
    logging_steps: int = 20
    save_steps: int = 100
    save_total_limit: int = 2
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
    profile_sample_rows: int = 64


def _free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_base_model(model_name: str, seed: int):
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


def wrap_with_lora(model, cfg: TrainConfig):
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


def build_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
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


class _ProfilerCallback(TrainerCallback):
    def __init__(self, profiler, stop_after: int):
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


def _build_profiler(trace_dir: Path, cfg: TrainConfig) -> torch.profiler.profile:
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


def _save_profiler_summary(prof, out_path: Path, top_k: int = 15) -> None:
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


def profile_short_run(train_df: pd.DataFrame, cfg: TrainConfig, output_dir: Path) -> None:
    """Короткий прогон на несколько шагов под torch.profiler.

    Использует отдельный мини-train с `max_steps = wait+warmup+active+1` и
    сохраняет `profiler_summary.md` в output_dir.
    """
    model, tokenizer = load_base_model(cfg.model_name, cfg.seed)
    model = wrap_with_lora(model, cfg)

    sub_df = train_df.head(cfg.profile_sample_rows).reset_index(drop=True)
    train_ds = build_dataset(sub_df, tokenizer, cfg.max_seq_length)

    stop_after = cfg.profile_wait_steps + cfg.profile_warmup_steps + cfg.profile_active_steps
    max_steps = stop_after + 1

    args = TrainingArguments(
        output_dir=str(output_dir / "profile_trainer"),
        max_steps=max_steps,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=1,
        learning_rate=cfg.learning_rate,
        warmup_ratio=0.0,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
    )

    prof = _build_profiler(output_dir / "profiler_trace", cfg)
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[_ProfilerCallback(prof, stop_after)],
    )
    trainer.train()
    _save_profiler_summary(prof, output_dir / "profiler_summary.md")

    del trainer, model, tokenizer, prof
    _free()


def train_full_run(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    cfg: TrainConfig,
    output_dir: Path,
) -> dict:
    """Полный QLoRA-train. Чекпойнты каждые save_steps, автоматический resume."""
    model, tokenizer = load_base_model(cfg.model_name, cfg.seed)
    model = wrap_with_lora(model, cfg)

    train_ds = build_dataset(train_df, tokenizer, cfg.max_seq_length)
    eval_ds = build_dataset(eval_df, tokenizer, cfg.max_seq_length)

    trainer_dir = output_dir / "trainer"
    args = TrainingArguments(
        output_dir=str(trainer_dir),
        num_train_epochs=cfg.num_train_epochs,
        per_device_train_batch_size=cfg.per_device_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        weight_decay=cfg.weight_decay,
        lr_scheduler_type=cfg.lr_scheduler_type,
        logging_steps=cfg.logging_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        report_to="none",
        fp16=True,
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        seed=cfg.seed,
        data_seed=cfg.seed,
        dataloader_num_workers=0,
    )

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=collator,
    )

    checkpoints = sorted(
        trainer_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    resume_path = str(checkpoints[-1]) if checkpoints else None
    if resume_path:
        print(f"[train] resume from {resume_path}", flush=True)

    result = trainer.train(resume_from_checkpoint=resume_path)

    adapter_dir = output_dir / "lora_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    metrics = {
        "train_loss": result.metrics.get("train_loss"),
        "train_runtime_sec": result.metrics.get("train_runtime"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "num_train_epochs": cfg.num_train_epochs,
        "max_seq_length": cfg.max_seq_length,
        "effective_batch_size": cfg.per_device_batch_size * cfg.gradient_accumulation_steps,
        "resumed_from": resume_path,
    }
    (output_dir / "train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del trainer, model, tokenizer
    _free()
    return metrics
