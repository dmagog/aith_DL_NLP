from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoModelForTokenClassification, AutoTokenizer, pipeline


TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", flags=re.UNICODE)
ENTITY_PREFIX_PATTERN = re.compile(r"^[BIES]-")


def simple_tokenize(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def build_teacher_pipeline(teacher_model: str, device: int) -> Any:
    config = AutoConfig.from_pretrained(teacher_model)
    if getattr(config, "label2id", None):
        config.id2label = {index: label for label, index in config.label2id.items()}

    model = AutoModelForTokenClassification.from_pretrained(teacher_model, config=config)
    tokenizer = AutoTokenizer.from_pretrained(teacher_model, use_fast=True)
    return pipeline(
        task="token-classification",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="simple",
        device=device,
    )


def normalize_entity_label(label: str) -> str:
    return ENTITY_PREFIX_PATTERN.sub("", label.strip().upper())


def span_to_bio_tags(
    token_spans: list[tuple[str, int, int]],
    entity_spans: list[dict[str, Any]],
    supported_labels: set[str],
) -> tuple[list[str], list[str]]:
    tokens = [token for token, _, _ in token_spans]
    tags = ["O"] * len(token_spans)

    for entity in entity_spans:
        raw_label = entity.get("entity_group") or entity.get("entity") or ""
        label = normalize_entity_label(str(raw_label))
        if label not in supported_labels:
            continue

        entity_start = int(entity["start"])
        entity_end = int(entity["end"])
        matched_token_indices = [
            index
            for index, (_, token_start, token_end) in enumerate(token_spans)
            if token_start < entity_end and token_end > entity_start
        ]
        if not matched_token_indices:
            continue
        tags[matched_token_indices[0]] = f"B-{label}"
        for token_index in matched_token_indices[1:]:
            tags[token_index] = f"I-{label}"

    return tokens, tags


def generate_pseudo_labeled_rows(
    teacher_model: str,
    source_dataset_name: str,
    source_split: str,
    text_column: str,
    supported_labels: set[str],
    dataset_config_name: str | None = None,
    max_samples: int | None = None,
    batch_size: int = 8,
    device: int = -1,
    max_text_chars: int | None = None,
) -> list[dict[str, list[str]]]:
    source_dataset = load_dataset(source_dataset_name, dataset_config_name, split=source_split)
    if max_samples is not None:
        source_dataset = source_dataset.select(range(min(max_samples, len(source_dataset))))

    ner_pipe = build_teacher_pipeline(teacher_model, device)

    rows: list[dict[str, list[str]]] = []
    texts: list[str] = []
    for value in source_dataset[text_column]:
        text = str(value).strip()
        if not text:
            continue
        if max_text_chars is not None:
            text = text[:max_text_chars]
        if text:
            texts.append(text)
    for batch_start in tqdm(range(0, len(texts), batch_size), desc="Pseudo-labeling"):
        batch_texts = texts[batch_start : batch_start + batch_size]
        batch_predictions = ner_pipe(batch_texts)
        for text, prediction in zip(batch_texts, batch_predictions):
            token_spans = simple_tokenize(text)
            if not token_spans:
                continue
            tokens, ner_tags = span_to_bio_tags(token_spans, prediction, supported_labels)
            if not any(tag != "O" for tag in ner_tags):
                continue
            rows.append({"tokens": tokens, "ner_tags": ner_tags})
    return rows
