from __future__ import annotations

import argparse
from pathlib import Path

from datasets import DatasetDict

from hw3.data import (
    build_json_dataset_from_rows,
    detect_token_label_schema,
    load_synthetic_dataset,
    load_token_classification_dataset,
    merge_gold_and_synthetic,
    normalize_label_ids,
    remap_labels_to_target_schema,
    save_dataset_preview,
)
from hw3.pseudo_label import generate_pseudo_labeled_rows
from hw3.reporting import collect_run_summaries, format_markdown_table
from hw3.training import train_and_evaluate_ner, train_mlm
from hw3.utils import dump_json, ensure_dir, get_device, set_seed


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-name", default="gusevski/factrueval2016")
    parser.add_argument("--dataset-config-name", default=None)
    parser.add_argument("--cache-dir", default="data/cache")


def namespace_to_jsonable_dict(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {}
    for key, value in vars(args).items():
        if callable(value):
            continue
        payload[key] = value
    return payload


def add_split_limit_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)


def maybe_limit_splits(dataset_dict: DatasetDict, args: argparse.Namespace) -> DatasetDict:
    limits = {
        "train": args.max_train_samples,
        "validation": args.max_validation_samples,
        "test": args.max_test_samples,
    }
    limited = dataset_dict
    for split_name, limit in limits.items():
        if limit is None or split_name not in limited:
            continue
        limited[split_name] = limited[split_name].select(range(min(limit, len(limited[split_name]))))
    return limited


def load_gold_dataset(args: argparse.Namespace) -> tuple[DatasetDict, object]:
    dataset_dict = load_token_classification_dataset(
        dataset_name=args.dataset_name,
        dataset_config_name=args.dataset_config_name,
        cache_dir=args.cache_dir,
    )
    dataset_dict = maybe_limit_splits(dataset_dict, args)
    schema = detect_token_label_schema(dataset_dict)
    dataset_dict = normalize_label_ids(dataset_dict, schema)
    schema = detect_token_label_schema(dataset_dict)
    return dataset_dict, schema


def command_inspect_dataset(args: argparse.Namespace) -> None:
    dataset_dict, schema = load_gold_dataset(args)
    ensure_dir(args.output_dir)
    preview_path = Path(args.output_dir) / "dataset_preview.txt"
    save_dataset_preview(dataset_dict, preview_path, schema)
    dump_json(
        {
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "splits": {split_name: len(split) for split_name, split in dataset_dict.items()},
            "token_column": schema.token_column,
            "label_column": schema.label_column,
            "label_names": schema.label_names,
        },
        Path(args.output_dir) / "dataset_info.json",
    )


def command_train_ner(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset_dict, schema = load_gold_dataset(args)

    synthetic_dataset = None
    if args.synthetic_path:
        synthetic_dataset = load_synthetic_dataset(args.synthetic_path)
        synthetic_schema = detect_token_label_schema(synthetic_dataset)
        synthetic_dataset = normalize_label_ids(synthetic_dataset, synthetic_schema)
        synthetic_schema = detect_token_label_schema(synthetic_dataset)
        synthetic_dataset = remap_labels_to_target_schema(
            synthetic_dataset,
            source_schema=synthetic_schema,
            target_label_names=schema.label_names,
        )
        synthetic_dataset = synthetic_dataset.rename_column("tokens", schema.token_column) if "tokens" != schema.token_column else synthetic_dataset
        synthetic_dataset = synthetic_dataset.rename_column("ner_tags", schema.label_column) if "ner_tags" != schema.label_column else synthetic_dataset

    merged_dataset = merge_gold_and_synthetic(dataset_dict, synthetic_dataset, schema)
    metrics = train_and_evaluate_ner(
        dataset_dict=merged_dataset,
        schema=schema,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    dump_json(
        {
            "command": "train-ner",
            **namespace_to_jsonable_dict(args),
            "device": get_device(),
            "metrics": metrics,
        },
        Path(args.output_dir) / "run_config.json",
    )


def command_train_mlm(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    dataset_dict, schema = load_gold_dataset(args)
    metrics = train_mlm(
        dataset_dict=dataset_dict,
        schema=schema,
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        masking_strategy=args.masking_strategy,
        max_length=args.max_length,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        weight_decay=args.weight_decay,
        mlm_probability=args.mlm_probability,
        seed=args.seed,
    )
    dump_json(
        {
            "command": "train-mlm",
            **namespace_to_jsonable_dict(args),
            "device": get_device(),
            "metrics": metrics,
        },
        Path(args.output_dir) / "run_config.json",
    )


def command_pseudo_label(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    supported_labels = {label.strip().upper() for label in args.supported_labels.split(",")}
    rows = generate_pseudo_labeled_rows(
        teacher_model=args.teacher_model,
        source_dataset_name=args.source_dataset_name,
        source_split=args.source_split,
        text_column=args.text_column,
        supported_labels=supported_labels,
        dataset_config_name=args.source_dataset_config_name,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        device=0 if get_device() == "cuda" else -1,
        max_text_chars=args.max_text_chars,
    )
    ensure_dir(Path(args.output_path).parent)
    dataset = build_json_dataset_from_rows(rows)
    dataset["train"].to_json(args.output_path, force_ascii=False)
    dump_json(
        {
            "command": "pseudo-label",
            **namespace_to_jsonable_dict(args),
            "saved_rows": len(dataset["train"]),
            "supported_labels": sorted(supported_labels),
        },
        Path(args.output_path).with_suffix(".meta.json"),
    )


def command_compare_runs(args: argparse.Namespace) -> None:
    rows = collect_run_summaries(args.root_dir)
    if args.sort_by:
        present_rows = [row for row in rows if row.get(args.sort_by) is not None]
        missing_rows = [row for row in rows if row.get(args.sort_by) is None]
        present_rows = sorted(
            present_rows,
            key=lambda row: row.get(args.sort_by),
            reverse=not args.ascending,
        )
        rows = present_rows + missing_rows
    output_text = format_markdown_table(rows)
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    output_path.write_text(output_text, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW3 NER pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset")
    add_common_dataset_args(inspect_parser)
    add_split_limit_args(inspect_parser)
    inspect_parser.add_argument("--output-dir", default="artifacts_hw3/dataset_inspect")
    inspect_parser.set_defaults(func=command_inspect_dataset)

    train_ner_parser = subparsers.add_parser("train-ner")
    add_common_dataset_args(train_ner_parser)
    add_split_limit_args(train_ner_parser)
    train_ner_parser.add_argument("--model-name-or-path", default="DeepPavlov/rubert-base-cased")
    train_ner_parser.add_argument("--synthetic-path", default=None)
    train_ner_parser.add_argument("--output-dir", required=True)
    train_ner_parser.add_argument("--max-length", type=int, default=256)
    train_ner_parser.add_argument("--num-train-epochs", type=float, default=5.0)
    train_ner_parser.add_argument("--learning-rate", type=float, default=3e-5)
    train_ner_parser.add_argument("--per-device-train-batch-size", type=int, default=8)
    train_ner_parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    train_ner_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_ner_parser.add_argument("--seed", type=int, default=42)
    train_ner_parser.set_defaults(func=command_train_ner)

    train_mlm_parser = subparsers.add_parser("train-mlm")
    add_common_dataset_args(train_mlm_parser)
    add_split_limit_args(train_mlm_parser)
    train_mlm_parser.add_argument("--model-name-or-path", default="DeepPavlov/rubert-base-cased")
    train_mlm_parser.add_argument("--output-dir", required=True)
    train_mlm_parser.add_argument(
        "--masking-strategy",
        choices=("random", "whole_word", "entity"),
        default="whole_word",
    )
    train_mlm_parser.add_argument("--max-length", type=int, default=256)
    train_mlm_parser.add_argument("--num-train-epochs", type=float, default=3.0)
    train_mlm_parser.add_argument("--learning-rate", type=float, default=5e-5)
    train_mlm_parser.add_argument("--per-device-train-batch-size", type=int, default=16)
    train_mlm_parser.add_argument("--per-device-eval-batch-size", type=int, default=16)
    train_mlm_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_mlm_parser.add_argument("--mlm-probability", type=float, default=0.15)
    train_mlm_parser.add_argument("--seed", type=int, default=42)
    train_mlm_parser.set_defaults(func=command_train_mlm)

    pseudo_parser = subparsers.add_parser("pseudo-label")
    pseudo_parser.add_argument("--teacher-model", required=True)
    pseudo_parser.add_argument("--source-dataset-name", required=True)
    pseudo_parser.add_argument("--source-dataset-config-name", default=None)
    pseudo_parser.add_argument("--source-split", default="train")
    pseudo_parser.add_argument("--text-column", default="text")
    pseudo_parser.add_argument("--supported-labels", default="PER,ORG,LOC")
    pseudo_parser.add_argument("--max-samples", type=int, default=10000)
    pseudo_parser.add_argument("--batch-size", type=int, default=8)
    pseudo_parser.add_argument("--max-text-chars", type=int, default=None)
    pseudo_parser.add_argument("--output-path", required=True)
    pseudo_parser.add_argument("--seed", type=int, default=42)
    pseudo_parser.set_defaults(func=command_pseudo_label)

    compare_parser = subparsers.add_parser("compare-runs")
    compare_parser.add_argument("--root-dir", default="artifacts_hw3")
    compare_parser.add_argument("--output-path", default="artifacts_hw3/run_summary.md")
    compare_parser.add_argument("--sort-by", default="test_after_f1")
    compare_parser.add_argument("--ascending", action="store_true")
    compare_parser.set_defaults(func=command_compare_runs)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
