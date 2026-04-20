"""QLoRA-DPO обучение аблитерированной модели на наших парах.

Схема:
- База = аблитерированная модель (из `artifacts_hw6/abliterated_model/`),
  которая уже сохранена как обычный HF-чекпойнт.
- Грузим её в 4-bit NF4 + double-quant, compute dtype fp16.
- LoRA r=16, α=32 по всем линейным (как в HW5).
- ref_model=None: TRL сам отключает адаптер при подсчёте reference-
  logprob'ов, это стандартная оптимизация для PEFT-DPO.
- seed=42 всюду.

Параметры выбраны так, чтобы на T4 за ~20-40 мин уложился один проход
по ~50 harmful-prompt'ам с effective batch 4 и beta=0.1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


@dataclass(frozen=True)
class DpoTrainConfig:
    abliterated_model_dir: str = "artifacts_hw6/abliterated_model"
    max_prompt_length: int = 384
    max_length: int = 768
    per_device_batch: int = 2
    gradient_accumulation_steps: int = 2   # eff. batch 4 — DPO чувствителен
    num_train_epochs: float = 1.0
    learning_rate: float = 5e-5            # LoRA-DPO: 5e-5 — частый разумный выбор
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.1
    beta: float = 0.1                      # DPO регуляризация; дефолт TRL
    logging_steps: int = 5
    save_steps: int = 50
    save_total_limit: int = 2
    seed: int = 42

    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_targets: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )


def _load_base_4bit(model_dir: str, seed: int):
    torch.manual_seed(seed)
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_dir,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    mdl.config.use_cache = False
    return mdl, tok


def _wrap_lora(model, cfg: DpoTrainConfig):
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


def _build_ds(pairs: list[dict]) -> Dataset:
    """TRL.DPOTrainer ждёт колонки `prompt`, `chosen`, `rejected`."""
    return Dataset.from_list(pairs)


def _new_dpo_trainer(model, tokenizer, train_ds, val_ds, cfg: DpoTrainConfig, out_dir: Path):
    """Пробуем DPOConfig (TRL≥0.11), если его нет — fallback на TrainingArguments."""
    from trl import DPOTrainer

    try:
        from trl import DPOConfig
        args = DPOConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=cfg.per_device_batch,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
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
            beta=cfg.beta,
            max_prompt_length=cfg.max_prompt_length,
            max_length=cfg.max_length,
        )
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            processing_class=tokenizer,
        )
    except (ImportError, TypeError):
        # Fallback для старого TRL с TrainingArguments.
        from transformers import TrainingArguments
        args = TrainingArguments(
            output_dir=str(out_dir),
            per_device_train_batch_size=cfg.per_device_batch,
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            num_train_epochs=cfg.num_train_epochs,
            learning_rate=cfg.learning_rate,
            lr_scheduler_type=cfg.lr_scheduler_type,
            warmup_ratio=cfg.warmup_ratio,
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
        return DPOTrainer(
            model=model,
            ref_model=None,
            args=args,
            beta=cfg.beta,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,
            max_prompt_length=cfg.max_prompt_length,
            max_length=cfg.max_length,
        )


def train_dpo(
    train_pairs: list[dict],
    val_pairs: list[dict],
    cfg: DpoTrainConfig,
    out_dir: Path,
) -> dict:
    """Тренируем DPO-LoRA поверх аблитерированной модели."""
    model, tokenizer = _load_base_4bit(cfg.abliterated_model_dir, cfg.seed)
    model = _wrap_lora(model, cfg)

    train_ds = _build_ds(train_pairs)
    val_ds = _build_ds(val_pairs)

    trainer_dir = out_dir / "trainer"
    trainer = _new_dpo_trainer(model, tokenizer, train_ds, val_ds, cfg, trainer_dir)

    # Auto-resume с последнего чекпойнта, как в HW5.
    checkpoints = sorted(
        trainer_dir.glob("checkpoint-*"),
        key=lambda p: int(p.name.split("-")[1]),
    )
    resume_path = str(checkpoints[-1]) if checkpoints else None
    if resume_path:
        print(f"[dpo_train] resume from {resume_path}", flush=True)

    result = trainer.train(resume_from_checkpoint=resume_path)

    adapter_dir = out_dir / "dpo_adapter"
    model.save_pretrained(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))

    # Сохраняем log_history в отдельный лёгкий артефакт (trainer/ gitignored).
    log_payload = {
        "global_step": trainer.state.global_step,
        "epoch": trainer.state.epoch,
        "log_history": trainer.state.log_history,
    }
    (out_dir / "dpo_log_history.json").write_text(
        json.dumps(log_payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    metrics = {
        "train_loss": result.metrics.get("train_loss"),
        "train_runtime_sec": result.metrics.get("train_runtime"),
        "train_samples_per_second": result.metrics.get("train_samples_per_second"),
        "num_train_epochs": cfg.num_train_epochs,
        "effective_batch_size": cfg.per_device_batch * cfg.gradient_accumulation_steps,
        "beta": cfg.beta,
        "learning_rate": cfg.learning_rate,
        "lr_scheduler_type": cfg.lr_scheduler_type,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "resumed_from": resume_path,
    }
    (out_dir / "dpo_train_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    del trainer, model, tokenizer
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return metrics
