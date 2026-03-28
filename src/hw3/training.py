from __future__ import annotations

from pathlib import Path
import inspect
from typing import Any

from datasets import Dataset, DatasetDict
from transformers import (
    AutoModelForMaskedLM,
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    Trainer,
    TrainingArguments,
)

from hw3.data import (
    TokenLabelSchema,
    prepare_tokenized_mlm_dataset,
    prepare_tokenized_ner_datasets,
)
from hw3.metrics import build_compute_metrics
from hw3.mlm import build_mlm_collator
from hw3.utils import dump_json, ensure_dir, get_device, resolve_split_name


def build_tokenizer(model_name_or_path: str) -> Any:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("A fast tokenizer is required for word alignment and masking.")
    return tokenizer


def build_token_classifier(
    model_name_or_path: str,
    schema: TokenLabelSchema,
) -> Any:
    label2id = {label: index for index, label in enumerate(schema.label_names)}
    id2label = {index: label for label, index in label2id.items()}
    return AutoModelForTokenClassification.from_pretrained(
        model_name_or_path,
        num_labels=len(schema.label_names),
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )


def build_trainer(
    model: Any,
    tokenizer: Any,
    train_dataset: Dataset,
    eval_dataset: Dataset,
    output_dir: str,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    seed: int,
    label_names: list[str],
) -> Trainer:
    args = build_training_arguments(
        output_dir=output_dir,
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        seed=seed,
    )
    return Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(label_names),
    )


def build_training_arguments(
    output_dir: str,
    learning_rate: float,
    num_train_epochs: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    load_best_model_at_end: bool,
    metric_for_best_model: str,
    greater_is_better: bool,
    seed: int,
    remove_unused_columns: bool = True,
) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__).parameters
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "learning_rate": learning_rate,
        "num_train_epochs": num_train_epochs,
        "per_device_train_batch_size": per_device_train_batch_size,
        "per_device_eval_batch_size": per_device_eval_batch_size,
        "weight_decay": weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "epoch",
        "load_best_model_at_end": load_best_model_at_end,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": greater_is_better,
        "report_to": [],
        "seed": seed,
        "save_total_limit": 2,
        "dataloader_pin_memory": get_device() == "cuda",
        "remove_unused_columns": remove_unused_columns,
    }
    if "evaluation_strategy" in signature:
        kwargs["evaluation_strategy"] = "epoch"
    else:
        kwargs["eval_strategy"] = "epoch"
    return TrainingArguments(**kwargs)


def train_and_evaluate_ner(
    dataset_dict: DatasetDict,
    schema: TokenLabelSchema,
    model_name_or_path: str,
    output_dir: str,
    max_length: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    seed: int,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    tokenizer = build_tokenizer(model_name_or_path)
    tokenized = prepare_tokenized_ner_datasets(
        dataset_dict=dataset_dict,
        tokenizer=tokenizer,
        schema=schema,
        max_length=max_length,
    )

    train_split = resolve_split_name(tokenized, ("train",))
    eval_split = resolve_split_name(tokenized, ("validation", "dev", "test"))
    test_split = resolve_split_name(tokenized, ("test", "validation", "dev"))

    model = build_token_classifier(model_name_or_path, schema)
    trainer = build_trainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=tokenized[train_split],
        eval_dataset=tokenized[eval_split],
        output_dir=str(output_path),
        num_train_epochs=num_train_epochs,
        learning_rate=learning_rate,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        seed=seed,
        label_names=schema.label_names,
    )

    before_metrics = trainer.evaluate(tokenized[test_split], metric_key_prefix="test_before")
    trainer.train()
    after_metrics = trainer.evaluate(tokenized[test_split], metric_key_prefix="test_after")
    trainer.save_model(str(output_path / "checkpoint-best"))
    tokenizer.save_pretrained(str(output_path / "checkpoint-best"))

    metrics = {
        "before_finetuning": before_metrics,
        "after_finetuning": after_metrics,
        "train_split": train_split,
        "eval_split": eval_split,
        "test_split": test_split,
    }
    dump_json(metrics, output_path / "metrics.json")
    return metrics


def train_mlm(
    dataset_dict: DatasetDict,
    schema: TokenLabelSchema,
    model_name_or_path: str,
    output_dir: str,
    masking_strategy: str,
    max_length: int,
    num_train_epochs: float,
    learning_rate: float,
    per_device_train_batch_size: int,
    per_device_eval_batch_size: int,
    weight_decay: float,
    mlm_probability: float,
    seed: int,
) -> dict[str, Any]:
    output_path = ensure_dir(output_dir)
    tokenizer = build_tokenizer(model_name_or_path)
    train_split = resolve_split_name(dataset_dict, ("train",))
    eval_split = resolve_split_name(dataset_dict, ("validation", "dev", "test"))

    tokenized_train = prepare_tokenized_mlm_dataset(
        dataset=dataset_dict[train_split],
        tokenizer=tokenizer,
        token_column=schema.token_column,
        label_column=schema.label_column if masking_strategy == "entity" else None,
        max_length=max_length,
    )
    tokenized_eval = prepare_tokenized_mlm_dataset(
        dataset=dataset_dict[eval_split],
        tokenizer=tokenizer,
        token_column=schema.token_column,
        label_column=schema.label_column if masking_strategy == "entity" else None,
        max_length=max_length,
    )

    model = AutoModelForMaskedLM.from_pretrained(model_name_or_path)
    collator = build_mlm_collator(masking_strategy, tokenizer, mlm_probability)

    args = build_training_arguments(
        output_dir=str(output_path),
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        seed=seed,
        remove_unused_columns=False,
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=collator,
    )

    trainer.train()
    metrics = trainer.evaluate(metric_key_prefix="eval")
    trainer.save_model(str(output_path / "checkpoint-best"))
    tokenizer.save_pretrained(str(output_path / "checkpoint-best"))
    dump_json(metrics, Path(output_path) / "metrics.json")
    return metrics
