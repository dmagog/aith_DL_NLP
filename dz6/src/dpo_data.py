"""Сбор (prompt, chosen, rejected) пар для DPO.

Дизайн:
- `prompt`   — goal из `masterkristall/harmful_behaviors_ru`, уже обёрнут
  под user-турн + assistant-generation-prefix через chat_template.
  Это ровно то, что модель "видит" при генерации, и совместимо с
  TRL.DPOTrainer (раздел "Formatting the data").
- `rejected` — `target` из того же датасета (harmful continuation).
  Фаст, без генерации; условие ДЗ это позволяет.
- `chosen`   — генерация ОРИГИНАЛЬНОЙ instruct-модели (не аблитерированной).
  По условию "для генерации harmless вариантов можно использовать любую
  LLM (кроме аблитерированных версий)". Instruct-модель на harmful prompts
  отказывается, значит `chosen` — естественный refusal/safe-answer.

Итог: DPO учит аблитерированную модель снова предпочитать refusal над
harmful continuation — восстанавливает safety-поведение, которое
убрала аблитерация.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .evaluate import GenConfig, apply_chat


@dataclass(frozen=True)
class DpoDataConfig:
    # Модель, чьи ответы идут в `chosen`. По умолчанию — оригинальная
    # instruct-версия той же архитектуры, что и наша target-модель.
    chosen_source_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens: int = 256
    # Для chosen важен естественный, стабильный refusal — greedy хватит.
    do_sample: bool = False
    seed: int = 42


def _load_instruct_4bit(model_name: str):
    """Грузим reference-модель в 4-bit — это дешевле fp16 и достаточно для
    генерации `chosen`-сэмплов."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def _generate_chosen(model, tokenizer, user_prompt: str, cfg: DpoDataConfig) -> str:
    """Генерим `chosen` — ответ reference-модели на harmful-запрос."""
    text = apply_chat(tokenizer, user_prompt)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=cfg.max_new_tokens,
        do_sample=cfg.do_sample,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )
    cont_ids = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(cont_ids, skip_special_tokens=True).strip()


def _format_prompt_for_dpo(tokenizer, goal: str) -> str:
    """Обёртка-prompt для DPO: chat_template + generation prefix."""
    return apply_chat(tokenizer, goal)


def build_pairs(
    rows: list[dict],        # items {goal, target}
    model,
    tokenizer,
    cfg: DpoDataConfig,
    desc: str = "dpo pairs",
) -> list[dict]:
    """Возвращает список {prompt, chosen, rejected} в формате DPOTrainer."""
    torch.manual_seed(cfg.seed)
    pairs: list[dict] = []
    for row in tqdm(rows, desc=desc):
        goal = row["goal"]
        rejected = row["target"].strip()
        chosen = _generate_chosen(model, tokenizer, goal, cfg)
        pairs.append({
            "prompt": _format_prompt_for_dpo(tokenizer, goal),
            "chosen": chosen,
            "rejected": rejected,
        })
    return pairs


def save_pairs(pairs: list[dict], out_path: Path) -> None:
    """Сохраняем в JSON (массив), а не JSONL — проще читать в отчёте."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_all(
    dpo_train_rows: list[dict],
    dpo_val_rows: list[dict],
    out_dir: Path,
    cfg: DpoDataConfig = DpoDataConfig(),
) -> dict:
    """End-to-end: грузим reference-модель, строим пары для train и val,
    сохраняем. Возвращает короткую сводку."""
    model, tokenizer = _load_instruct_4bit(cfg.chosen_source_model)
    try:
        train_pairs = build_pairs(
            dpo_train_rows, model, tokenizer, cfg, desc="dpo train pairs"
        )
        val_pairs = build_pairs(
            dpo_val_rows, model, tokenizer, cfg, desc="dpo val pairs"
        )
    finally:
        del model, tokenizer
        import gc as _gc
        _gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_pairs(train_pairs, out_dir / "dpo_train_pairs.json")
    save_pairs(val_pairs, out_dir / "dpo_val_pairs.json")

    info = {
        "chosen_source_model": cfg.chosen_source_model,
        "n_train_pairs": len(train_pairs),
        "n_val_pairs": len(val_pairs),
        "max_new_tokens": cfg.max_new_tokens,
        "do_sample": cfg.do_sample,
        "seed": cfg.seed,
    }
    (out_dir / "dpo_data_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return info
