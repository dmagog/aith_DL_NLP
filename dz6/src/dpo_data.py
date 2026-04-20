"""Сбор (prompt, chosen, rejected) пар для DPO.

Датасет `masterkristall/harmful_behaviors_ru` содержит только harmful
prompt'ы (колонка `text`), без заранее заготовленного harmful-ответа.
Значит `rejected` надо сгенерировать — и разумнее всего использовать
для этого саму аблитерированную модель (условие ДЗ запрещает только
генерацию harmless-ответов аблитерированной моделью).

Дизайн пар:
- `prompt`   — goal (harmful RU-prompt), обёрнут user-турном и
  generation-prefix'ом через chat_template. Это ровно то, что модель
  видит при inference, и совместимо с TRL.DPOTrainer.
- `chosen`   — генерация ОРИГИНАЛЬНОЙ instruct-модели (greedy). На
  harmful-prompt'ах она стабильно отказывается — это и есть safety-
  ответ, к которому мы тянем аблитерированную.
- `rejected` — генерация АБЛИТЕРИРОВАННОЙ модели (greedy). Она перестала
  отказываться, значит выдаёт harmful continuation — именно его мы
  штрафуем через DPO.

Порядок загрузки — последовательный: сначала оригинал (4-bit) для
chosen, высвобождаем память, затем аблит. модель (4-bit) для rejected.
Иначе на T4 (16 ГБ) две 1.5B-модели одновременно могут не влезть вместе
с KV-кэшем.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
from tqdm import tqdm

from .evaluate import apply_chat


@dataclass(frozen=True)
class DpoDataConfig:
    # Для chosen — любая референс-модель; по умолчанию оригинальный
    # instruct той же архитектуры.
    chosen_source_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # rejected-модель — путь к аблитерированной на диске. Обязателен,
    # заполняется из run_full.py (см. stage_dpo_data).
    rejected_source_dir: str = ""
    max_new_tokens: int = 256
    do_sample: bool = False   # greedy — нужен детерминированный ответ
    seed: int = 42


def _load_instruct_4bit(model_name_or_path: str):
    """4-bit NF4 + double-quant, compute fp16. Работает и для HF-хаб
    репы, и для локального abliterated_model/."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=bnb,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    mdl.eval()
    return mdl, tok


@torch.no_grad()
def _generate(model, tokenizer, user_prompt: str, cfg: DpoDataConfig) -> str:
    """Genegenerate один ответ на harmful-prompt. Возвращаем только
    continuation — prompt-часть срезается."""
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


def _free() -> None:
    import gc as _gc
    _gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _gen_for_all(
    prompts: list[str],
    model_name_or_path: str,
    cfg: DpoDataConfig,
    desc: str,
) -> tuple[list[str], list[str]]:
    """Грузим модель ОДИН раз, генерим continuation для каждого prompt,
    параллельно считаем форматированный prompt (одинаковый chat_template
    для всех моделей — Qwen-Instruct и аблит. версия используют тот же
    токенайзер, но mind-the-contract: prompt фиксируем по первой модели
    и переиспользуем). Здесь возвращаем и prompts_formatted, и responses."""
    torch.manual_seed(cfg.seed)
    mdl, tok = _load_instruct_4bit(model_name_or_path)
    try:
        prompts_fmt: list[str] = []
        responses: list[str] = []
        for goal in tqdm(prompts, desc=desc):
            prompts_fmt.append(apply_chat(tok, goal))
            responses.append(_generate(mdl, tok, goal, cfg))
    finally:
        del mdl, tok
        _free()
    return prompts_fmt, responses


def build_pairs_two_pass(
    goals: list[str],
    cfg: DpoDataConfig,
    desc_prefix: str = "dpo",
) -> list[dict]:
    """Двухпроходная сборка: сначала оригинал → chosen, потом аблит → rejected."""
    if not cfg.rejected_source_dir:
        raise ValueError("DpoDataConfig.rejected_source_dir must be set")

    prompts_fmt_a, chosen = _gen_for_all(
        goals, cfg.chosen_source_model, cfg, f"{desc_prefix} chosen (original)"
    )
    prompts_fmt_b, rejected = _gen_for_all(
        goals, cfg.rejected_source_dir, cfg, f"{desc_prefix} rejected (abliterated)"
    )

    # Обе модели имеют один и тот же chat_template (tokenizer переписан
    # из аблит. чекпойнта). На всякий случай используем версию от
    # первой модели — она канонична.
    assert len(prompts_fmt_a) == len(prompts_fmt_b) == len(chosen) == len(rejected)
    pairs: list[dict] = []
    for prompt_fmt, c, r in zip(prompts_fmt_a, chosen, rejected):
        pairs.append({"prompt": prompt_fmt, "chosen": c, "rejected": r})
    return pairs


def save_pairs(pairs: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(pairs, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_all(
    dpo_train_goals: list[str],
    dpo_val_goals: list[str],
    out_dir: Path,
    cfg: DpoDataConfig,
) -> dict:
    """End-to-end: строим train- и val-пары, сохраняем. На T4 каждый
    проход — это загрузка одной 4-bit 1.5B-модели + ~N generations."""
    # Склеиваем train+val в один прогон каждой модели, чтобы не грузить
    # веса дважды. Потом режем обратно по длинам.
    all_goals = list(dpo_train_goals) + list(dpo_val_goals)
    n_train = len(dpo_train_goals)

    all_pairs = build_pairs_two_pass(all_goals, cfg, desc_prefix="dpo")
    train_pairs = all_pairs[:n_train]
    val_pairs = all_pairs[n_train:]

    save_pairs(train_pairs, out_dir / "dpo_train_pairs.json")
    save_pairs(val_pairs, out_dir / "dpo_val_pairs.json")

    info = {
        "chosen_source_model": cfg.chosen_source_model,
        "rejected_source_dir": cfg.rejected_source_dir,
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
