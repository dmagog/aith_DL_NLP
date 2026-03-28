from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def collect_run_summaries(root_dir: str | Path) -> list[dict[str, Any]]:
    root_path = Path(root_dir)
    rows: list[dict[str, Any]] = []
    if not root_path.exists():
        return rows

    for metrics_path in sorted(root_path.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        run_name = run_dir.name
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))

        run_config_path = run_dir / "run_config.json"
        run_config = {}
        if run_config_path.exists():
            run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

        row: dict[str, Any] = {
            "run_name": run_name,
            "command": run_config.get("command"),
            "model_name_or_path": run_config.get("model_name_or_path"),
        }

        before_metrics = metrics.get("before_finetuning", {})
        after_metrics = metrics.get("after_finetuning", {})
        if before_metrics or after_metrics:
            row.update(
                {
                    "test_before_f1": before_metrics.get("test_before_f1"),
                    "test_after_f1": after_metrics.get("test_after_f1"),
                    "delta_f1": None
                    if before_metrics.get("test_before_f1") is None or after_metrics.get("test_after_f1") is None
                    else after_metrics["test_after_f1"] - before_metrics["test_before_f1"],
                    "test_after_precision": after_metrics.get("test_after_precision"),
                    "test_after_recall": after_metrics.get("test_after_recall"),
                }
            )
        else:
            row.update(metrics)

        rows.append(row)
    return rows


def format_markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "No runs found."

    preferred_headers = [
        "run_name",
        "command",
        "model_name_or_path",
        "test_before_f1",
        "test_after_f1",
        "delta_f1",
        "test_after_precision",
        "test_after_recall",
        "eval_loss",
    ]
    discovered_headers: list[str] = []
    for row in rows:
        for key in row.keys():
            if key not in discovered_headers:
                discovered_headers.append(key)

    headers = [header for header in preferred_headers if header in discovered_headers]
    headers.extend(header for header in discovered_headers if header not in headers)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)
