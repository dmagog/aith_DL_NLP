from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from transformers import DataCollatorForLanguageModeling


@dataclass
class RandomMLMCollator:
    tokenizer: Any
    mlm_probability: float = 0.15

    def __post_init__(self) -> None:
        self.base_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm_probability=self.mlm_probability,
        )

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        cleaned_examples = []
        for example in examples:
            cleaned_example = dict(example)
            cleaned_example.pop("word_ids", None)
            cleaned_example.pop("entity_token_mask", None)
            cleaned_examples.append(cleaned_example)
        return self.base_collator(cleaned_examples)


def _group_word_positions(word_ids: list[int | None]) -> dict[int, list[int]]:
    positions: dict[int, list[int]] = {}
    for token_position, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        positions.setdefault(int(word_id), []).append(token_position)
    return positions


def _sample_word_mask(
    word_ids: list[int | None],
    mlm_probability: float,
    eligible_word_ids: set[int] | None = None,
) -> list[int]:
    grouped_positions = _group_word_positions(word_ids)
    if not grouped_positions:
        return [0] * len(word_ids)

    candidate_word_ids = sorted(
        word_id
        for word_id in grouped_positions
        if eligible_word_ids is None or word_id in eligible_word_ids
    )
    if not candidate_word_ids:
        candidate_word_ids = sorted(grouped_positions)

    num_to_mask = max(1, int(round(len(candidate_word_ids) * mlm_probability)))
    chosen_word_ids = set(
        np.random.choice(
            candidate_word_ids,
            size=min(num_to_mask, len(candidate_word_ids)),
            replace=False,
        ).tolist()
    )

    token_mask = [0] * len(word_ids)
    for word_id in chosen_word_ids:
        for token_position in grouped_positions[word_id]:
            token_mask[token_position] = 1
    return token_mask


@dataclass
class WholeWordMLMCollator:
    tokenizer: Any
    mlm_probability: float = 0.15

    def _torch_mask_tokens(
        self,
        inputs: torch.Tensor,
        mask_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.clone()
        probability_matrix = mask_labels.bool()

        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
            for val in labels.tolist()
        ]
        probability_matrix &= ~torch.tensor(special_tokens_mask, dtype=torch.bool)
        labels[~probability_matrix] = -100

        masked_indices = probability_matrix
        replace_prob = torch.full(labels.shape, 0.8)
        indices_replaced = torch.bernoulli(replace_prob).bool() & masked_indices
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

        random_prob = torch.full(labels.shape, 0.5)
        indices_random = torch.bernoulli(random_prob).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        inputs[indices_random] = random_words[indices_random]
        return inputs, labels

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        cleaned_examples = []
        mask_labels: list[list[int]] = []
        for example in examples:
            word_ids = example.get("word_ids")
            cleaned_example = dict(example)
            cleaned_example.pop("word_ids", None)
            cleaned_example.pop("entity_token_mask", None)
            cleaned_examples.append(cleaned_example)
            mask_labels.append(_sample_word_mask(word_ids, self.mlm_probability))

        batch = self.tokenizer.pad(cleaned_examples, return_tensors="pt")
        padded_mask_labels = torch.zeros_like(batch["input_ids"])
        for row_index, row_mask in enumerate(mask_labels):
            padded_mask_labels[row_index, : len(row_mask)] = torch.tensor(row_mask, dtype=torch.long)
        input_ids, labels = self._torch_mask_tokens(batch["input_ids"], padded_mask_labels)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch


@dataclass
class DataCollatorForEntityMask(WholeWordMLMCollator):
    tokenizer: Any
    mlm_probability: float = 0.15

    def __call__(self, examples: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        cleaned_examples = []
        mask_labels: list[list[int]] = []
        for example in examples:
            word_ids = example.get("word_ids")
            entity_token_mask = example.get("entity_token_mask")
            cleaned_example = dict(example)
            cleaned_example.pop("word_ids", None)
            cleaned_example.pop("entity_token_mask", None)
            cleaned_examples.append(cleaned_example)

            eligible_word_ids = {
                int(word_id)
                for token_position, word_id in enumerate(word_ids)
                if word_id is not None and entity_token_mask[token_position]
            }
            mask_labels.append(
                _sample_word_mask(
                    word_ids=word_ids,
                    mlm_probability=self.mlm_probability,
                    eligible_word_ids=eligible_word_ids,
                )
            )

        batch = self.tokenizer.pad(cleaned_examples, return_tensors="pt")
        padded_mask_labels = torch.zeros_like(batch["input_ids"])
        for row_index, row_mask in enumerate(mask_labels):
            padded_mask_labels[row_index, : len(row_mask)] = torch.tensor(row_mask, dtype=torch.long)
        input_ids, labels = self._torch_mask_tokens(batch["input_ids"], padded_mask_labels)
        batch["input_ids"] = input_ids
        batch["labels"] = labels
        return batch


def build_mlm_collator(masking_strategy: str, tokenizer: Any, mlm_probability: float) -> Any:
    if masking_strategy == "random":
        return RandomMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
    if masking_strategy == "whole_word":
        return WholeWordMLMCollator(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
    if masking_strategy == "entity":
        return DataCollatorForEntityMask(
            tokenizer=tokenizer,
            mlm_probability=mlm_probability,
        )
    raise ValueError(f"Unsupported masking strategy: {masking_strategy}")
