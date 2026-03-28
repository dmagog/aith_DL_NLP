from __future__ import annotations

from typing import Any

import numpy as np
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score


def build_compute_metrics(label_names: list[str]):
    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        logits, labels = eval_prediction
        predictions = np.argmax(logits, axis=-1)

        true_predictions: list[list[str]] = []
        true_labels: list[list[str]] = []
        for prediction_row, label_row in zip(predictions, labels):
            filtered_predictions: list[str] = []
            filtered_labels: list[str] = []
            for predicted_label_id, label_id in zip(prediction_row, label_row):
                if label_id == -100:
                    continue
                filtered_predictions.append(label_names[int(predicted_label_id)])
                filtered_labels.append(label_names[int(label_id)])
            true_predictions.append(filtered_predictions)
            true_labels.append(filtered_labels)

        return {
            "precision": float(precision_score(true_labels, true_predictions)),
            "recall": float(recall_score(true_labels, true_predictions)),
            "f1": float(f1_score(true_labels, true_predictions)),
            "accuracy": float(accuracy_score(true_labels, true_predictions)),
        }

    return compute_metrics
