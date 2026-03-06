#!/usr/bin/env python3
"""End-to-end pipeline for ITMO NLP HW1 (Lenta topic classification)."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import requests
from corus import load_lenta, load_lenta2
from joblib import dump
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

DATASET_URLS = {
    "v1.0": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz",
    "v1.1": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2",
}

NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")
MULTISPACE_RE = re.compile(r"\s+")
BASE_MODEL_NAMES = ("count_logreg", "tfidf_logreg")


@dataclass(frozen=True)
class SplitData:
    x_train: pd.Series
    x_val: pd.Series
    x_test: pd.Series
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_columns(df: pd.DataFrame, required_columns: List[str], context: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def sanitize_sample_df(sampled_df: pd.DataFrame) -> pd.DataFrame:
    out = sampled_df.copy()
    out = out.dropna(subset=["title", "text", "topic"]).reset_index(drop=True)
    out["title"] = out["title"].astype(str)
    out["text"] = out["text"].astype(str)
    out["topic"] = out["topic"].astype(str).str.strip()
    out = out[out["topic"].str.len() > 0].reset_index(drop=True)
    return out


def sanitize_processed_df(processed_df: pd.DataFrame) -> pd.DataFrame:
    out = processed_df.copy()
    out = out.dropna(subset=["processed_text", "topic"]).reset_index(drop=True)
    out["processed_text"] = out["processed_text"].astype(str).str.strip()
    out["topic"] = out["topic"].astype(str).str.strip()
    out = out[(out["processed_text"].str.len() > 0) & (out["topic"].str.len() > 0)].reset_index(drop=True)
    return out


def maybe_warn_sample_size(expected: int, actual: int, label: str) -> None:
    if actual != expected:
        print(f"Warning: {label} has {actual} rows, expected {expected}. Continuing with available rows.")


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    response = requests.get(url, stream=True, timeout=30)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))

    with destination.open("wb") as fout, tqdm(
        total=total,
        unit="B",
        unit_scale=True,
        desc=f"Downloading {destination.name}",
    ) as progress:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if not chunk:
                continue
            fout.write(chunk)
            progress.update(len(chunk))


def ensure_dataset(data_dir: Path, version: str) -> Path:
    if version not in DATASET_URLS:
        available = ", ".join(sorted(DATASET_URLS.keys()))
        raise ValueError(f"Unknown version '{version}'. Available: {available}")

    url = DATASET_URLS[version]
    filename = url.rsplit("/", maxsplit=1)[-1]
    dataset_path = data_dir / filename

    if dataset_path.exists() and dataset_path.stat().st_size > 0:
        return dataset_path

    print(f"Dataset not found, downloading {version} to {dataset_path} ...")
    ensure_dir(data_dir)
    download_file(url, dataset_path)
    return dataset_path


def iterate_records(dataset_path: Path):
    suffix = dataset_path.suffix.lower()
    if suffix == ".gz":
        return load_lenta(dataset_path)
    if suffix == ".bz2":
        return load_lenta2(dataset_path)
    raise ValueError(f"Unsupported dataset format: {dataset_path}")


def is_valid_record(record) -> bool:
    return bool(record.title and record.text and record.topic)


def count_topics(dataset_path: Path) -> Counter:
    counts: Counter = Counter()
    for record in tqdm(iterate_records(dataset_path), desc="Counting topics"):
        if not is_valid_record(record):
            continue
        counts[record.topic] += 1
    return counts


def allocate_targets(topic_counts: Counter, sample_size: int) -> Dict[str, int]:
    total = sum(topic_counts.values())
    if sample_size > total:
        raise ValueError(f"sample_size={sample_size} exceeds valid records={total}")

    exact = {topic: sample_size * count / total for topic, count in topic_counts.items()}
    targets = {topic: int(np.floor(value)) for topic, value in exact.items()}
    remainder = sample_size - sum(targets.values())

    for topic, _ in sorted(
        exact.items(), key=lambda item: item[1] - np.floor(item[1]), reverse=True
    )[:remainder]:
        targets[topic] += 1

    return targets


def stratified_reservoir_sample(
    dataset_path: Path,
    targets: Dict[str, int],
    seed: int,
) -> pd.DataFrame:
    rng = random.Random(seed)
    seen_by_topic: Counter = Counter()
    reservoirs: Dict[str, List[Dict[str, str]]] = {topic: [] for topic in targets}

    for record in tqdm(iterate_records(dataset_path), desc="Sampling 100k"):
        if not is_valid_record(record):
            continue

        topic = record.topic
        if topic not in targets:
            continue

        target_size = targets[topic]
        if target_size <= 0:
            continue

        seen_by_topic[topic] += 1
        row = {
            "title": record.title,
            "text": record.text,
            "topic": topic,
        }

        topic_bucket = reservoirs[topic]
        if len(topic_bucket) < target_size:
            topic_bucket.append(row)
            continue

        replace_index = rng.randrange(seen_by_topic[topic])
        if replace_index < target_size:
            topic_bucket[replace_index] = row

    sampled_rows: List[Dict[str, str]] = []
    for topic, rows in reservoirs.items():
        expected = targets[topic]
        if len(rows) < expected:
            raise RuntimeError(
                f"Topic '{topic}' has only {len(rows)} sampled rows, expected {expected}."
            )
        sampled_rows.extend(rows)

    sampled_df = pd.DataFrame(sampled_rows)
    sampled_df = sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    return sampled_df


def preprocess_text(title: str, text: str) -> str:
    merged = f"{title} {text}".lower().strip()
    merged = NON_ALNUM_RE.sub(" ", merged)
    merged = MULTISPACE_RE.sub(" ", merged)
    return merged.strip()


def apply_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["processed_text"] = [
        preprocess_text(title, text)
        for title, text in tqdm(
            zip(out["title"].fillna(""), out["text"].fillna("")),
            total=len(out),
            desc="Preprocessing",
        )
    ]
    out = out[out["processed_text"].str.len() > 0].reset_index(drop=True)
    return out


def split_data(texts: pd.Series, labels: np.ndarray, seed: int) -> SplitData:
    x_train, x_temp, y_train, y_temp = train_test_split(
        texts,
        labels,
        test_size=0.4,
        random_state=seed,
        stratify=labels,
    )

    x_val, x_test, y_val, y_test = train_test_split(
        x_temp,
        y_temp,
        test_size=0.5,
        random_state=seed,
        stratify=y_temp,
    )

    return SplitData(
        x_train=x_train,
        x_val=x_val,
        x_test=x_test,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def evaluate_model(model, x_train, y_train, x_eval, y_eval) -> Tuple[Dict[str, float], float]:
    start = perf_counter()
    model.fit(x_train, y_train)
    train_seconds = perf_counter() - start

    predictions = model.predict(x_eval)
    metrics = compute_metrics(y_eval, predictions)
    metrics["train_seconds"] = train_seconds
    return metrics, train_seconds


def build_base_pipelines(seed: int) -> Dict[str, Pipeline]:
    base_lr = LogisticRegression(
        solver="saga",
        penalty="l2",
        C=1.0,
        max_iter=800,
        random_state=seed,
    )

    return {
        "count_logreg": Pipeline(
            steps=[
                (
                    "vectorizer",
                    CountVectorizer(
                        min_df=3,
                        max_df=0.95,
                        ngram_range=(1, 2),
                        max_features=120_000,
                    ),
                ),
                ("classifier", clone(base_lr)),
            ]
        ),
        "tfidf_logreg": Pipeline(
            steps=[
                (
                    "vectorizer",
                    TfidfVectorizer(
                        min_df=3,
                        max_df=0.95,
                        ngram_range=(1, 2),
                        max_features=120_000,
                    ),
                ),
                ("classifier", clone(base_lr)),
            ]
        ),
    }


def run_dummy_baseline(x_train, y_train, x_val, y_val, seed: int) -> Dict[str, Dict[str, float]]:
    strategies = ["most_frequent", "stratified"]
    result = {}

    for strategy in strategies:
        dummy = DummyClassifier(strategy=strategy, random_state=seed)
        metrics, _ = evaluate_model(dummy, x_train, y_train, x_val, y_val)
        result[strategy] = metrics

    return result


def choose_best_model(base_metrics: Dict[str, Dict[str, float]]) -> str:
    return max(base_metrics, key=lambda key: base_metrics[key]["macro_f1"])


def run_tuning(
    model_name: str,
    base_pipeline: Pipeline,
    x_train: pd.Series,
    y_train: np.ndarray,
    seed: int,
    n_iter: int,
    n_jobs: int,
    cv_splits: int,
) -> RandomizedSearchCV:
    param_distributions = {
        "vectorizer__ngram_range": [(1, 1), (1, 2)],
        "vectorizer__min_df": [2, 3, 5],
        "vectorizer__max_df": [0.9, 0.95, 1.0],
        "vectorizer__max_features": [50_000, 80_000, 120_000],
        "classifier__C": np.logspace(-2, 1.2, 20),
        "classifier__class_weight": [None, "balanced"],
        "classifier__max_iter": [700, 900, 1100],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        cv=cv,
        n_jobs=n_jobs,
        random_state=seed,
        verbose=1,
    )

    print(f"Starting RandomizedSearchCV for {model_name} ...")
    search.fit(x_train, y_train)
    return search


def maybe_subsample_for_tuning(
    x_train: pd.Series,
    y_train: np.ndarray,
    tuning_sample_size: int,
    seed: int,
) -> Tuple[pd.Series, np.ndarray]:
    if tuning_sample_size <= 0 or tuning_sample_size >= len(x_train):
        return x_train, y_train

    indices = np.arange(len(x_train))
    sampled_indices, _ = train_test_split(
        indices,
        train_size=tuning_sample_size,
        random_state=seed,
        stratify=y_train,
    )
    sampled_indices = np.sort(sampled_indices)
    x_sub = x_train.iloc[sampled_indices]
    y_sub = y_train[sampled_indices]
    return x_sub, y_sub


def top_confusions(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str], top_k: int = 10):
    cm = confusion_matrix(y_true, y_pred)
    rows = []
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if i == j or cm[i, j] == 0:
                continue
            rows.append(
                {
                    "true_label": class_names[i],
                    "predicted_label": class_names[j],
                    "count": int(cm[i, j]),
                }
            )
    rows.sort(key=lambda row: row["count"], reverse=True)
    return rows[:top_k]


def run_error_analysis(
    model,
    split: SplitData,
    label_encoder: LabelEncoder,
    report_dir: Path,
    top_k: int = 10,
) -> Dict[str, object]:
    ensure_dir(report_dir)
    y_pred = model.predict(split.x_test)

    report_dict = classification_report(
        split.y_test,
        y_pred,
        target_names=list(label_encoder.classes_),
        output_dict=True,
        zero_division=0,
    )

    confusion_rows = top_confusions(split.y_test, y_pred, list(label_encoder.classes_), top_k=top_k)

    wrong_mask = y_pred != split.y_test
    wrong_df = pd.DataFrame(
        {
            "text": split.x_test.to_numpy()[wrong_mask],
            "true_label": label_encoder.inverse_transform(split.y_test[wrong_mask]),
            "pred_label": label_encoder.inverse_transform(y_pred[wrong_mask]),
        }
    )
    wrong_df["text_preview"] = wrong_df["text"].str.slice(0, 300)
    wrong_df = wrong_df.drop(columns=["text"])
    wrong_df.to_csv(report_dir / "misclassified_examples.csv", index=False)

    report_lines = [
        "# Error analysis",
        "",
        f"Total test errors: {int(wrong_mask.sum())}",
        "",
        "## Top confusion pairs",
        "",
    ]
    for row in confusion_rows:
        report_lines.append(
            f"- `{row['true_label']}` -> `{row['predicted_label']}`: {row['count']}"
        )

    (report_dir / "error_analysis.md").write_text("\n".join(report_lines), encoding="utf-8")

    return {
        "classification_report": report_dict,
        "top_confusions": confusion_rows,
        "test_errors": int(wrong_mask.sum()),
    }


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def resolve_dataset_path(args: argparse.Namespace) -> Path:
    if args.dataset_path is not None:
        return args.dataset_path
    if args.skip_download:
        expected_name = DATASET_URLS[args.dataset_version].rsplit("/", maxsplit=1)[-1]
        dataset_path = args.data_dir / expected_name
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Dataset not found at {dataset_path}. Remove --skip-download or set --dataset-path."
            )
        return dataset_path
    return ensure_dataset(args.data_dir, args.dataset_version)


def load_or_prepare_processed_sample(
    args: argparse.Namespace,
    artifacts_dir: Path,
    metrics_dir: Path,
) -> pd.DataFrame:
    sample_output_path = artifacts_dir / "sample_100k.csv"
    processed_output_path = artifacts_dir / "sample_100k_processed.csv"

    if args.preprocessed_sample_path is not None:
        print(f"Using preprocessed sample: {args.preprocessed_sample_path}")
        processed_df = pd.read_csv(args.preprocessed_sample_path)
        ensure_columns(processed_df, ["processed_text", "topic"], "preprocessed sample")
        processed_df = sanitize_processed_df(processed_df)
        maybe_warn_sample_size(args.sample_size, len(processed_df), "preprocessed sample")
        processed_df.to_csv(processed_output_path, index=False)
        save_json(
            {
                "mode": "preprocessed_sample_path",
                "path": str(args.preprocessed_sample_path),
                "rows": int(len(processed_df)),
            },
            metrics_dir / "sample_source.json",
        )
        return processed_df

    if args.sample_path is not None:
        print(f"Using sampled data from: {args.sample_path}")
        sampled_df = pd.read_csv(args.sample_path)
        ensure_columns(sampled_df, ["title", "text", "topic"], "sampled data")
        sampled_df = sanitize_sample_df(sampled_df)
        maybe_warn_sample_size(args.sample_size, len(sampled_df), "sampled data")
    else:
        dataset_path = resolve_dataset_path(args)
        print(f"Using dataset: {dataset_path}")
        topic_counts = count_topics(dataset_path)
        targets = allocate_targets(topic_counts, args.sample_size)
        sampled_df = stratified_reservoir_sample(
            dataset_path=dataset_path,
            targets=targets,
            seed=args.seed,
        )
        sampled_df = sanitize_sample_df(sampled_df)
        maybe_warn_sample_size(args.sample_size, len(sampled_df), "sampled data")

    sampled_df.to_csv(sample_output_path, index=False)
    processed_df = apply_preprocessing(sampled_df)
    processed_df = sanitize_processed_df(processed_df)
    maybe_warn_sample_size(args.sample_size, len(processed_df), "preprocessed sample")
    processed_df.to_csv(processed_output_path, index=False)
    save_json(
        {
            "mode": "dataset_sampling" if args.sample_path is None else "sample_path",
            "path": str(args.sample_path) if args.sample_path is not None else None,
            "rows": int(len(processed_df)),
        },
        metrics_dir / "sample_source.json",
    )
    return processed_df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITMO NLP HW1 training pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--dataset-version", type=str, default="v1.1", choices=sorted(DATASET_URLS))
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tuning-iters", type=int, default=20)
    parser.add_argument(
        "--tuning-sample-size",
        type=int,
        default=50_000,
        help="Use a stratified subset of train split for CV tuning. <=0 disables subsampling.",
    )
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--cv-splits", type=int, default=3)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--sample-path",
        type=Path,
        default=None,
        help="Optional path to sampled CSV with columns title,text,topic.",
    )
    parser.add_argument(
        "--preprocessed-sample-path",
        type=Path,
        default=None,
        help="Optional path to preprocessed sampled CSV with columns processed_text,topic.",
    )
    parser.add_argument(
        "--skip-base-eval",
        action="store_true",
        help="Skip training/evaluating both base models and use --best-base-model directly for tuning.",
    )
    parser.add_argument(
        "--best-base-model",
        type=str,
        default="count_logreg",
        choices=BASE_MODEL_NAMES,
        help="Base model to tune when --skip-base-eval is used.",
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Optional explicit path to lenta dataset (.csv.gz or .csv.bz2)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cv_splits < 2:
        raise ValueError("--cv-splits must be >= 2")
    set_seed(args.seed)

    artifacts_dir: Path = args.artifacts_dir
    metrics_dir = artifacts_dir / "metrics"
    reports_dir = artifacts_dir / "reports"
    models_dir = artifacts_dir / "models"
    ensure_dir(metrics_dir)
    ensure_dir(reports_dir)
    ensure_dir(models_dir)

    processed_df = load_or_prepare_processed_sample(
        args=args,
        artifacts_dir=artifacts_dir,
        metrics_dir=metrics_dir,
    )

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(processed_df["topic"].to_numpy())
    x = processed_df["processed_text"]

    split = split_data(texts=x, labels=y, seed=args.seed)

    split_stats = {
        "train_size": int(len(split.x_train)),
        "val_size": int(len(split.x_val)),
        "test_size": int(len(split.x_test)),
        "num_classes": int(len(label_encoder.classes_)),
    }
    save_json(split_stats, metrics_dir / "split_stats.json")

    baseline_metrics = run_dummy_baseline(
        split.x_train,
        split.y_train,
        split.x_val,
        split.y_val,
        seed=args.seed,
    )
    save_json(baseline_metrics, metrics_dir / "dummy_baseline.json")

    pipelines = build_base_pipelines(args.seed)
    if args.skip_base_eval:
        best_base_name = args.best_base_model
        save_json(
            {
                "skipped": True,
                "selected_model": best_base_name,
            },
            metrics_dir / "base_models_val.json",
        )
    else:
        base_metrics: Dict[str, Dict[str, float]] = {}
        for model_name, pipeline in pipelines.items():
            metrics, _ = evaluate_model(
                pipeline,
                split.x_train,
                split.y_train,
                split.x_val,
                split.y_val,
            )
            base_metrics[model_name] = metrics
        save_json(base_metrics, metrics_dir / "base_models_val.json")
        best_base_name = choose_best_model(base_metrics)

    best_base_pipeline = pipelines[best_base_name]
    print(f"Best base model on val: {best_base_name}")

    x_trainval = pd.concat([split.x_train, split.x_val], axis=0)
    y_trainval = np.concatenate([split.y_train, split.y_val])

    x_tune, y_tune = maybe_subsample_for_tuning(
        split.x_train,
        split.y_train,
        tuning_sample_size=args.tuning_sample_size,
        seed=args.seed,
    )

    tuning_data_stats = {
        "train_size_full": int(len(split.x_train)),
        "train_size_for_tuning": int(len(x_tune)),
    }
    save_json(tuning_data_stats, metrics_dir / "tuning_data_stats.json")

    search = run_tuning(
        model_name=best_base_name,
        base_pipeline=best_base_pipeline,
        x_train=x_tune,
        y_train=y_tune,
        seed=args.seed,
        n_iter=args.tuning_iters,
        n_jobs=args.n_jobs,
        cv_splits=args.cv_splits,
    )

    tuning_results = {
        "best_score_cv_macro_f1": float(search.best_score_),
        "best_params": search.best_params_,
    }
    save_json(tuning_results, metrics_dir / "tuning_results.json")

    final_model = search.best_estimator_
    final_model.fit(x_trainval, y_trainval)

    y_test_pred = final_model.predict(split.x_test)
    final_test_metrics = compute_metrics(split.y_test, y_test_pred)
    save_json(final_test_metrics, metrics_dir / "final_test_metrics.json")

    dump(final_model, models_dir / "best_pipeline.joblib")
    dump(label_encoder, models_dir / "label_encoder.joblib")

    error_analysis = run_error_analysis(
        model=final_model,
        split=split,
        label_encoder=label_encoder,
        report_dir=reports_dir,
        top_k=10,
    )
    save_json(error_analysis, metrics_dir / "error_analysis.json")

    preprocessing_notes = {
        "pipeline": [
            "merge title and text",
            "lowercase",
            "remove non-alphanumeric chars (including punctuation)",
            "normalize spaces",
        ],
        "motivation": (
            "This preprocessing is intentionally lightweight for speed on 100k documents. "
            "Vectorizers with n-grams capture lexical patterns, while avoiding expensive "
            "lemmatization keeps runtime practical."
        ),
    }
    save_json(preprocessing_notes, metrics_dir / "preprocessing_notes.json")

    print("Pipeline finished successfully.")
    print(f"Final test metrics: {final_test_metrics}")


if __name__ == "__main__":
    main()
