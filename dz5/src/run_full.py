"""Полный прогон HW5: подготовка данных, baseline, QLoRA, post-train, артефакты.

Запуск (в Colab или на любой CUDA-машине):
    python -m src.run_full \
        --corpus-path /path/to/lenta-ru-news.csv.bz2 \
        --out artifacts_hw5

Все долгие шаги делаются один раз; ноутбук `hw5.ipynb` потом читает артефакты
без повторной тренировки.
"""
from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import pandas as pd
import torch

from .basket import GenerationConfig, run_basket, save_basket_md
from .data import SampleConfig, prepare_lenta_sample, save_sample
from .evaluate import (
    DEFAULT_HARNESS_TASKS,
    compute_perplexity,
    harness_to_summary,
    run_harness,
    save_json,
)
from .train import TrainConfig, load_base_model, train_qlora


def _free(obj) -> None:
    del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_adapter(base_model_name: str, adapter_dir: Path, seed: int):
    from peft import PeftModel
    model, tokenizer = load_base_model(base_model_name, seed)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    return model, tokenizer


def _evaluate_stage(
    model,
    tokenizer,
    eval_df: pd.DataFrame,
    out_dir: Path,
    stage: str,
    harness_tasks: tuple[str, ...],
    harness_limit: int | None,
    perplexity_samples: int,
    gen_cfg: GenerationConfig,
) -> dict:
    basket_items = run_basket(model, tokenizer, gen_cfg=gen_cfg)
    save_basket_md(basket_items, out_dir / f"basket_{stage}.md", header=f"Basket: {stage}")

    ppl = compute_perplexity(model, tokenizer, eval_df, max_samples=perplexity_samples)

    harness_raw = run_harness(model, tokenizer, tasks=harness_tasks, limit=harness_limit)
    save_json(harness_raw, out_dir / f"harness_{stage}.json")
    harness_summary = harness_to_summary(harness_raw)

    metrics = {
        "stage": stage,
        "perplexity": ppl,
        "harness": harness_summary,
        "harness_tasks": list(harness_tasks),
        "harness_limit": harness_limit,
        "perplexity_samples": perplexity_samples,
    }
    save_json(metrics, out_dir / f"metrics_{stage}.json")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="HW5 end-to-end runner (Colab/GPU)")
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--eval-size", type=int, default=500)
    parser.add_argument("--text-char-limit", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--harness-limit", type=int, default=200,
                        help="Limit per harness task (None = full); по умолчанию 200 для скорости.")
    parser.add_argument("--perplexity-samples", type=int, default=200)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-post", action="store_true")
    args = parser.parse_args()

    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    sample_cfg = SampleConfig(
        sample_size=args.sample_size,
        eval_size=args.eval_size,
        seed=args.seed,
        text_char_limit=args.text_char_limit,
    )
    sample_dir = out / "sample"
    train_csv = sample_dir / "train.csv"
    eval_csv = sample_dir / "eval.csv"
    if not train_csv.exists() or not eval_csv.exists():
        train_df, eval_df = prepare_lenta_sample(args.corpus_path, sample_cfg)
        save_sample(train_df, eval_df, sample_dir, sample_cfg)
    else:
        train_df = pd.read_csv(train_csv)
        eval_df = pd.read_csv(eval_csv)

    train_cfg = TrainConfig(num_train_epochs=args.num_train_epochs, seed=args.seed)
    gen_cfg = GenerationConfig(seed=args.seed)
    harness_tasks = DEFAULT_HARNESS_TASKS

    if not args.skip_baseline:
        model, tokenizer = load_base_model(train_cfg.model_name, train_cfg.seed)
        _evaluate_stage(
            model, tokenizer, eval_df, out, stage="before",
            harness_tasks=harness_tasks, harness_limit=args.harness_limit,
            perplexity_samples=args.perplexity_samples, gen_cfg=gen_cfg,
        )
        _free(model)

    adapter_dir = out / "lora_adapter"
    if not args.skip_train:
        train_qlora(
            train_df=train_df,
            eval_df=eval_df,
            cfg=train_cfg,
            output_dir=out,
            profiler_trace_dir=out / "profiler_trace",
        )

    if not args.skip_post:
        model, tokenizer = _load_adapter(train_cfg.model_name, adapter_dir, train_cfg.seed)
        _evaluate_stage(
            model, tokenizer, eval_df, out, stage="after",
            harness_tasks=harness_tasks, harness_limit=args.harness_limit,
            perplexity_samples=args.perplexity_samples, gen_cfg=gen_cfg,
        )
        _free(model)

    summary = {
        "sample": json.loads((sample_dir / "dataset_info.json").read_text(encoding="utf-8")),
        "before": json.loads((out / "metrics_before.json").read_text(encoding="utf-8"))
        if (out / "metrics_before.json").exists() else None,
        "after": json.loads((out / "metrics_after.json").read_text(encoding="utf-8"))
        if (out / "metrics_after.json").exists() else None,
        "train": json.loads((out / "train_metrics.json").read_text(encoding="utf-8"))
        if (out / "train_metrics.json").exists() else None,
    }
    save_json(summary, out / "run_summary.json")


if __name__ == "__main__":
    main()
