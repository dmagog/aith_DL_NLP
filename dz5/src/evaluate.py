"""Оценка модели: perplexity на валидации + lm-evaluation-harness."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import torch

from .data import format_training_text

DEFAULT_HARNESS_TASKS: tuple[str, ...] = ("xnli_ru", "xwinograd_ru")


@torch.no_grad()
def compute_perplexity(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    max_length: int = 512,
    max_samples: int | None = 200,
) -> float:
    model.eval()
    device = next(model.parameters()).device
    rows = eval_df if max_samples is None else eval_df.head(max_samples)

    total_loss = 0.0
    total_tokens = 0
    for title, text in zip(rows["title"], rows["text"]):
        enc = tokenizer(
            format_training_text(title, text),
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        labels = enc["input_ids"].clone()
        out = model(**enc, labels=labels)
        n_tokens = int(enc["attention_mask"].sum().item())
        total_loss += float(out.loss.item()) * n_tokens
        total_tokens += n_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    return math.exp(mean_loss)


def run_harness(
    model,
    tokenizer,
    tasks: tuple[str, ...] = DEFAULT_HARNESS_TASKS,
    num_fewshot: int = 0,
    limit: int | None = None,
    batch_size: int = 4,
) -> dict:
    from lm_eval import simple_evaluate
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
    results = simple_evaluate(
        model=lm,
        tasks=list(tasks),
        num_fewshot=num_fewshot,
        limit=limit,
    )
    return results


def harness_to_summary(results: dict) -> dict:
    out: dict[str, dict] = {}
    for task, metrics in (results or {}).get("results", {}).items():
        out[task] = {
            k: v for k, v in metrics.items()
            if isinstance(v, (int, float)) and not k.endswith("_stderr")
        }
    return out


def save_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
