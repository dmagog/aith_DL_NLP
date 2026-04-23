"""End-to-end пайплайн HW6.

Стадии (у каждой — `--skip-*` флаг и idempotent-проверка артефактов):
  1. prepare        — data.prepare_all → сплиты в artifacts_hw6/sample/
  2. eval_pretrained — генерация + refusal на оригинальной Qwen-Instruct
  3. abliterate     — вычисление refusal direction + weight orthogonalization,
                       сохранение модифицированной модели в fp16 на диск
  4. eval_abliterated — refusal на аблитерированной
  5. push           — загрузка аблитерированной модели в HF Hub (dmagog/...)
  6. dpo_data       — генерация `chosen`-ответов reference-моделью →
                       пары для DPO
  7. dpo_train      — QLoRA-DPO поверх аблитерированной
  8. eval_dpo       — refusal после DPO (LoRA-адаптер + аблит. база)
  9. summary        — итоговый run_summary.json

Дизайн вдохновлён run_full из HW5: всё идемпотентно, каждый стадий либо
читает уже существующий артефакт, либо пересчитывает с нуля. Это важно на
free-tier Colab: сессия может оборваться, перезапуск подхватит прогресс.
"""
from __future__ import annotations

import argparse
import gc
import json
from dataclasses import asdict
from pathlib import Path

import torch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _free_cuda() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _load_instruct_4bit(model_name_or_path: str):
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


def _load_instruct_fp16(model_name_or_path: str):
    """fp16 без bnb — нужен для weight orthogonalization."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return mdl, tok


def _load_lora_on_abliterated(abliterated_dir: str, adapter_dir: str):
    """Собираем eval-модель для стадии eval_dpo."""
    from peft import PeftModel
    mdl, tok = _load_instruct_4bit(abliterated_dir)
    mdl = PeftModel.from_pretrained(mdl, adapter_dir)
    mdl.eval()
    return mdl, tok


# ---------------------------------------------------------------------------
# Стадии
# ---------------------------------------------------------------------------

def stage_prepare(out_dir: Path, seed: int) -> dict:
    from .data import SplitConfig, prepare_all

    sample_dir = out_dir / "sample"
    splits_info_path = sample_dir / "splits_info.json"
    if splits_info_path.exists():
        print("[prepare] cached splits found, skipping", flush=True)
        return _read_json(splits_info_path)

    cfg = SplitConfig(seed=seed)
    prepare_all(sample_dir, cfg)
    return _read_json(splits_info_path)


def _load_splits(out_dir: Path) -> dict:
    sample_dir = out_dir / "sample"
    return {
        "abliteration": {
            "harmful": _read_json(sample_dir / "abliteration_harmful.json"),
            "harmless": _read_json(sample_dir / "abliteration_harmless.json"),
        },
        "eval": {
            "harmful": _read_json(sample_dir / "eval_harmful.json"),
            "harmless": _read_json(sample_dir / "eval_harmless.json"),
        },
        "dpo_train": _read_json(sample_dir / "dpo_train.json"),
        "dpo_val": _read_json(sample_dir / "dpo_val.json"),
    }


def stage_eval(
    out_dir: Path,
    tag: str,
    model_loader,
    eval_prompts: dict,
) -> dict:
    """Generic eval-стадия: загрузчик модели, harmful+harmless prompts, tag."""
    from .evaluate import GenConfig, run_eval, save_eval

    metrics_path = out_dir / f"metrics_{tag}.json"
    if metrics_path.exists():
        print(f"[eval:{tag}] cached, skipping", flush=True)
        return _read_json(metrics_path)

    mdl, tok = model_loader()
    try:
        gen = GenConfig()
        # Все сплиты в HW6 — list[str]: harmful-датасет содержит только
        # prompt'ы (колонка `text`), без target'а.
        harmful_prompts = list(eval_prompts["harmful"])
        harmless_prompts = list(eval_prompts["harmless"])
        print(f"[eval:{tag}] harmful ({len(harmful_prompts)})", flush=True)
        rows_h = run_eval(mdl, tok, harmful_prompts, gen, desc=f"{tag} harmful")
        print(f"[eval:{tag}] harmless ({len(harmless_prompts)})", flush=True)
        rows_n = run_eval(mdl, tok, harmless_prompts, gen, desc=f"{tag} harmless")
        summary = save_eval(rows_h, rows_n, out_dir, tag)
    finally:
        del mdl, tok
        _free_cuda()
    return summary


def stage_abliterate(
    base_model_name: str,
    out_dir: Path,
    splits: dict,
    seed: int,
) -> dict:
    from .abliterate import AbliterateConfig, abliterate

    abliterated_dir = out_dir / "abliterated_model"
    info_path = out_dir / "abliteration_info.json"
    if info_path.exists() and (abliterated_dir / "config.json").exists():
        print("[abliterate] cached, skipping", flush=True)
        return _read_json(info_path)

    torch.manual_seed(seed)
    mdl, tok = _load_instruct_fp16(base_model_name)
    try:
        cfg = AbliterateConfig()
        harmful_prompts = list(splits["abliteration"]["harmful"])
        harmless_prompts = list(splits["abliteration"]["harmless"])
        info = abliterate(mdl, tok, harmful_prompts, harmless_prompts, cfg, out_dir)

        # Сохраняем модифицированную модель как обычный HF-чекпойнт (fp16).
        abliterated_dir.mkdir(parents=True, exist_ok=True)
        mdl.save_pretrained(str(abliterated_dir), safe_serialization=True)
        tok.save_pretrained(str(abliterated_dir))
    finally:
        del mdl, tok
        _free_cuda()
    return info


def stage_push(out_dir: Path, repo_id: str) -> dict:
    from huggingface_hub import HfApi, create_repo

    marker = out_dir / "hf_push_info.json"
    if marker.exists():
        print("[push] cached, skipping", flush=True)
        return _read_json(marker)

    abliterated_dir = out_dir / "abliterated_model"
    if not abliterated_dir.exists():
        raise RuntimeError("abliterated_model/ not found — run abliterate first")

    create_repo(repo_id=repo_id, exist_ok=True, private=False)
    api = HfApi()
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(abliterated_dir),
        commit_message="Upload abliterated Qwen2.5-1.5B-Instruct (HW6)",
    )
    info = {"hf_repo_id": repo_id}
    marker.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


def stage_dpo_data(
    base_model_name: str,
    out_dir: Path,
    splits: dict,
    seed: int,
) -> dict:
    from .dpo_data import DpoDataConfig, build_all

    info_path = out_dir / "dpo_data_info.json"
    if info_path.exists():
        print("[dpo_data] cached, skipping", flush=True)
        return _read_json(info_path)

    cfg = DpoDataConfig(
        chosen_source_model=base_model_name,
        rejected_source_dir=str(out_dir / "abliterated_model"),
        seed=seed,
    )
    return build_all(splits["dpo_train"], splits["dpo_val"], out_dir, cfg)


def stage_dpo_train(out_dir: Path, seed: int) -> dict:
    from .dpo_train import DpoTrainConfig, train_dpo

    metrics_path = out_dir / "dpo_train_metrics.json"
    if metrics_path.exists():
        print("[dpo_train] cached, skipping", flush=True)
        return _read_json(metrics_path)

    train_pairs = _read_json(out_dir / "dpo_train_pairs.json")
    val_pairs = _read_json(out_dir / "dpo_val_pairs.json")

    cfg = DpoTrainConfig(
        abliterated_model_dir=str(out_dir / "abliterated_model"),
        seed=seed,
    )
    return train_dpo(train_pairs, val_pairs, cfg, out_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _log_env_diagnostics() -> None:
    """Первые строчки `main()`: печатаем encoding/locale.
    Если `PYTHONUTF8=1` не долетел — это сразу видно в логе."""
    import locale as _locale
    import sys as _sys
    print(
        "[env] python=%s stdout=%s stderr=%s fs=%s locale=%s utf8_mode=%s"
        % (
            _sys.version.split()[0],
            _sys.stdout.encoding,
            _sys.stderr.encoding,
            _sys.getfilesystemencoding(),
            _locale.getpreferredencoding(False),
            _sys.flags.utf8_mode,
        ),
        flush=True,
    )


def main() -> None:
    _log_env_diagnostics()
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("artifacts_hw6"))
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--hf-repo-id", type=str,
                   default="dmagog/Qwen2.5-1.5B-Instruct-ru-abliterated")
    p.add_argument("--seed", type=int, default=42)
    for name in (
        "prepare", "eval-pretrained", "abliterate", "eval-abliterated",
        "push", "dpo-data", "dpo-train", "eval-dpo",
    ):
        p.add_argument(f"--skip-{name}", action="store_true")
    args = p.parse_args()

    out_dir: Path = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1. Prepare ---
    if not args.skip_prepare:
        stage_prepare(out_dir, args.seed)

    splits = _load_splits(out_dir)

    # --- 2. Eval pretrained ---
    if not args.skip_eval_pretrained:
        stage_eval(
            out_dir,
            "pretrained",
            lambda: _load_instruct_4bit(args.model),
            splits["eval"],
        )

    # --- 3. Abliterate ---
    if not args.skip_abliterate:
        stage_abliterate(args.model, out_dir, splits, args.seed)

    # --- 4. Eval abliterated ---
    if not args.skip_eval_abliterated:
        stage_eval(
            out_dir,
            "abliterated",
            lambda: _load_instruct_4bit(str(out_dir / "abliterated_model")),
            splits["eval"],
        )

    # --- 5. Push ---
    if not args.skip_push:
        stage_push(out_dir, args.hf_repo_id)

    # --- 6. DPO data ---
    if not args.skip_dpo_data:
        stage_dpo_data(args.model, out_dir, splits, args.seed)

    # --- 7. DPO train ---
    if not args.skip_dpo_train:
        stage_dpo_train(out_dir, args.seed)

    # --- 8. Eval DPO ---
    if not args.skip_eval_dpo:
        stage_eval(
            out_dir,
            "dpo",
            lambda: _load_lora_on_abliterated(
                str(out_dir / "abliterated_model"),
                str(out_dir / "dpo_adapter"),
            ),
            splits["eval"],
        )

    # --- 9. Summary ---
    run_summary = {
        "model": args.model,
        "hf_repo_id": args.hf_repo_id,
        "seed": args.seed,
    }
    for tag in ("pretrained", "abliterated", "dpo"):
        m_path = out_dir / f"metrics_{tag}.json"
        if m_path.exists():
            run_summary[f"metrics_{tag}"] = _read_json(m_path)
    (out_dir / "run_summary.json").write_text(
        json.dumps(run_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("[run_full] done", flush=True)


if __name__ == "__main__":
    main()
