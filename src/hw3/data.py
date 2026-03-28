from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from datasets import ClassLabel, Dataset, DatasetDict, Features, Sequence, Value, concatenate_datasets, load_dataset
from transformers import PreTrainedTokenizerBase


@dataclass(slots=True)
class TokenLabelSchema:
    token_column: str
    label_column: str
    label_names: list[str]


def load_token_classification_dataset(
    dataset_name: str,
    dataset_config_name: str | None = None,
    cache_dir: str | None = None,
) -> DatasetDict:
    dataset_dict = load_dataset(dataset_name, dataset_config_name, cache_dir=cache_dir)
    return maybe_flatten_nested_data_column(dataset_dict)


def maybe_flatten_nested_data_column(dataset_dict: DatasetDict) -> DatasetDict:
    sample_split_name = next(iter(dataset_dict.keys()))
    sample_columns = dataset_dict[sample_split_name].column_names
    if sample_columns != ["data"]:
        return dataset_dict

    flattened_splits: dict[str, Dataset] = {}
    for split_name, split in dataset_dict.items():
        rows: list[dict[str, Any]] = []
        for nested_rows in split["data"]:
            rows.extend(nested_rows)
        flattened_splits[split_name] = Dataset.from_list(rows)
    return DatasetDict(flattened_splits)


def detect_token_label_schema(dataset_dict: DatasetDict) -> TokenLabelSchema:
    sample_split_name = next(iter(dataset_dict.keys()))
    features: Features = dataset_dict[sample_split_name].features

    token_candidates = ("tokens", "token", "words")
    label_candidates = ("ner_tags", "labels", "tags")

    token_column = next((name for name in token_candidates if name in features), None)
    label_column = next((name for name in label_candidates if name in features), None)

    if token_column is None or label_column is None:
        available = ", ".join(features.keys())
        raise ValueError(
            "Could not detect token and label columns automatically. "
            f"Available columns: {available}"
        )

    label_feature = features[label_column]
    if isinstance(label_feature, Sequence) and isinstance(label_feature.feature, ClassLabel):
        label_names = list(label_feature.feature.names)
    elif isinstance(label_feature, ClassLabel):
        label_names = list(label_feature.names)
    elif label_column == "ner_tags" and "ner_tags_str" in features:
        id_to_name: dict[int, str] = {}
        for split in dataset_dict.values():
            for label_ids, label_names_row in zip(split["ner_tags"], split["ner_tags_str"]):
                for label_id, label_name in zip(label_ids, label_names_row):
                    id_to_name[int(label_id)] = str(label_name)
        label_names = [id_to_name[index] for index in sorted(id_to_name)]
    else:
        label_values: set[str] = set()
        for split in dataset_dict.values():
            for row in split[label_column]:
                for item in row:
                    label_values.add(str(item))
        label_names = sorted(label_values)

    return TokenLabelSchema(
        token_column=token_column,
        label_column=label_column,
        label_names=label_names,
    )


def normalize_label_ids(dataset_dict: DatasetDict, schema: TokenLabelSchema) -> DatasetDict:
    sample_split_name = next(iter(dataset_dict.keys()))
    label_feature = dataset_dict[sample_split_name].features[schema.label_column]
    if isinstance(label_feature, Sequence) and isinstance(label_feature.feature, ClassLabel):
        return dataset_dict
    if all(label.startswith(("B-", "I-", "O")) for label in schema.label_names):
        if isinstance(label_feature, Sequence) and isinstance(label_feature.feature, Value):
            if label_feature.feature.dtype.startswith("int"):
                return dataset_dict.cast_column(
                    schema.label_column,
                    Sequence(ClassLabel(names=schema.label_names)),
                )
        label_to_id = {label: index for index, label in enumerate(schema.label_names)}

        def convert_row(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
            return {
                schema.label_column: [
                    [label_to_id[str(label)] for label in labels]
                    for labels in batch[schema.label_column]
                ]
            }

        encoded = dataset_dict.map(convert_row, batched=True)
        encoded = encoded.cast_column(
            schema.label_column,
            Sequence(ClassLabel(names=schema.label_names)),
        )
        return encoded
    raise ValueError("Unsupported label format. Expected BIO labels or ClassLabel ids.")


def remap_labels_to_target_schema(
    dataset_dict: DatasetDict,
    source_schema: TokenLabelSchema,
    target_label_names: list[str],
) -> DatasetDict:
    target_label_to_id = {label: index for index, label in enumerate(target_label_names)}

    def convert_row(batch: dict[str, list[Any]]) -> dict[str, list[list[int]]]:
        converted_labels: list[list[int]] = []
        for label_row in batch[source_schema.label_column]:
            remapped_row: list[int] = []
            for label_value in label_row:
                if isinstance(label_value, str):
                    label_name = label_value
                else:
                    label_name = source_schema.label_names[int(label_value)]
                if label_name not in target_label_to_id:
                    raise ValueError(f"Label {label_name!r} is not present in target schema.")
                remapped_row.append(target_label_to_id[label_name])
            converted_labels.append(remapped_row)
        return {source_schema.label_column: converted_labels}

    remapped = dataset_dict.map(convert_row, batched=True)
    remapped = remapped.cast_column(
        source_schema.label_column,
        Sequence(ClassLabel(names=target_label_names)),
    )
    return remapped


def load_synthetic_dataset(path: str) -> DatasetDict:
    synthetic = load_dataset("json", data_files={"train": path})["train"]
    return DatasetDict({"train": synthetic})


def merge_gold_and_synthetic(
    gold_dataset: DatasetDict,
    synthetic_dataset: DatasetDict | None,
    schema: TokenLabelSchema,
) -> DatasetDict:
    if synthetic_dataset is None:
        return gold_dataset

    train_split_name = "train"
    if train_split_name not in gold_dataset:
        raise ValueError("Gold dataset must contain a train split.")
    if train_split_name not in synthetic_dataset:
        raise ValueError("Synthetic dataset must contain a train split.")

    gold_train = gold_dataset[train_split_name]
    synthetic_train = synthetic_dataset[train_split_name]
    required_columns = {schema.token_column, schema.label_column}
    missing_columns = required_columns - set(synthetic_train.column_names)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise ValueError(f"Synthetic dataset is missing required columns: {missing}")

    if synthetic_train.features[schema.label_column] != gold_train.features[schema.label_column]:
        synthetic_train = synthetic_train.cast_column(
            schema.label_column,
            gold_train.features[schema.label_column],
        )

    merged = DatasetDict(gold_dataset)
    merged["train"] = concatenate_datasets([gold_train, synthetic_train])
    return merged


def tokenize_and_align_labels(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    token_column: str,
    label_column: str,
    max_length: int,
) -> dict[str, Any]:
    tokenized = tokenizer(
        batch[token_column],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    aligned_labels: list[list[int]] = []
    for batch_index, word_labels in enumerate(batch[label_column]):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        previous_word_id: int | None = None
        token_labels: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                token_labels.append(-100)
            elif word_id != previous_word_id:
                token_labels.append(int(word_labels[word_id]))
            else:
                token_labels.append(-100)
            previous_word_id = word_id
        aligned_labels.append(token_labels)

    tokenized["labels"] = aligned_labels
    return tokenized


def tokenize_for_mlm(
    batch: dict[str, list[Any]],
    tokenizer: PreTrainedTokenizerBase,
    token_column: str,
    label_column: str | None,
    max_length: int,
) -> dict[str, Any]:
    tokenized = tokenizer(
        batch[token_column],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )

    all_word_ids: list[list[int | None]] = []
    entity_word_masks: list[list[int]] = []
    for batch_index in range(len(batch[token_column])):
        word_ids = tokenized.word_ids(batch_index=batch_index)
        all_word_ids.append(word_ids)

        if label_column is None:
            entity_word_masks.append([0] * len(word_ids))
            continue

        labels = batch[label_column][batch_index]
        example_entity_mask: list[int] = []
        for word_id in word_ids:
            if word_id is None:
                example_entity_mask.append(0)
            else:
                label_value = labels[word_id]
                example_entity_mask.append(0 if int(label_value) == 0 else 1)
        entity_word_masks.append(example_entity_mask)

    tokenized["word_ids"] = all_word_ids
    tokenized["entity_token_mask"] = entity_word_masks
    return tokenized


def prepare_tokenized_ner_datasets(
    dataset_dict: DatasetDict,
    tokenizer: PreTrainedTokenizerBase,
    schema: TokenLabelSchema,
    max_length: int,
) -> DatasetDict:
    keep_columns = [
        column
        for column in dataset_dict[next(iter(dataset_dict.keys()))].column_names
        if column not in {schema.token_column, schema.label_column}
    ]
    return dataset_dict.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "token_column": schema.token_column,
            "label_column": schema.label_column,
            "max_length": max_length,
        },
        remove_columns=keep_columns + [schema.token_column, schema.label_column],
        desc="Tokenizing and aligning NER labels",
    )


def prepare_tokenized_mlm_dataset(
    dataset: Dataset,
    tokenizer: PreTrainedTokenizerBase,
    token_column: str,
    label_column: str | None,
    max_length: int,
) -> Dataset:
    keep_columns = [
        column
        for column in dataset.column_names
        if column not in {token_column, label_column}
    ]
    return dataset.map(
        tokenize_for_mlm,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "token_column": token_column,
            "label_column": label_column,
            "max_length": max_length,
        },
        remove_columns=keep_columns + [token_column] + ([label_column] if label_column else []),
        desc="Tokenizing MLM corpus",
    )


def save_dataset_preview(dataset_dict: DatasetDict, output_path: str | Path, schema: TokenLabelSchema) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_name = next(iter(dataset_dict.keys()))
    sample = dataset_dict[split_name][0]
    preview = {
        "split": split_name,
        "token_column": schema.token_column,
        "label_column": schema.label_column,
        "label_names": schema.label_names,
        "example_tokens": sample[schema.token_column][:30],
        "example_labels": sample[schema.label_column][:30],
    }
    output_path.write_text(str(preview), encoding="utf-8")


def build_json_dataset_from_rows(rows: list[dict[str, Any]]) -> DatasetDict:
    features = Features(
        {
            "tokens": Sequence(Value("string")),
            "ner_tags": Sequence(Value("string")),
        }
    )
    dataset = Dataset.from_list(rows, features=features)
    return DatasetDict({"train": dataset})
