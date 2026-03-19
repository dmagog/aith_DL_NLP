#!/usr/bin/env python3
"""End-to-end pipeline for HW2."""

from __future__ import annotations

import argparse
import json
import random
import re
import zipfile
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from corus import load_lenta, load_lenta2
from gensim.models import KeyedVectors, Word2Vec
from joblib import dump
from pymorphy3 import MorphAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from tqdm import tqdm

try:
    from navec import Navec
except ImportError:  # pragma: no cover - optional dependency is handled at runtime
    Navec = None

DATASET_URLS = {
    "v1.0": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.0/lenta-ru-news.csv.gz",
    "v1.1": "https://github.com/yutkin/Lenta.Ru-News-Dataset/releases/download/v1.1/lenta-ru-news.csv.bz2",
}

NAVEC_URL = (
    "https://storage.yandexcloud.net/natasha-navec/packs/"
    "navec_news_v1_1B_250K_300d_100q.tar"
)
RUSVECTORES_URL = "http://vectors.nlpl.eu/repository/20/184.zip"

NON_ALNUM_RE = re.compile(r"[^0-9a-zA-Zа-яА-ЯёЁ]+")
MULTISPACE_RE = re.compile(r"\s+")
ALLOWED_EMBEDDINGS = ("custom_w2v", "navec", "rusvectores")

PYMORPHY_TO_UPOS = {
    "NOUN": "NOUN",
    "ADJF": "ADJ",
    "ADJS": "ADJ",
    "COMP": "ADJ",
    "VERB": "VERB",
    "INFN": "VERB",
    "PRTF": "ADJ",
    "PRTS": "ADJ",
    "GRND": "VERB",
    "NUMR": "NUM",
    "ADVB": "ADV",
    "NPRO": "PRON",
    "PRED": "ADV",
    "PREP": "ADP",
    "CONJ": "CCONJ",
    "PRCL": "PART",
    "INTJ": "INTJ",
}


@dataclass(frozen=True)
class SplitData:
    train_texts: pd.Series
    val_texts: pd.Series
    test_texts: pd.Series
    train_labels: np.ndarray
    val_labels: np.ndarray
    test_labels: np.ndarray


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def ensure_columns(df: pd.DataFrame, required_columns: Sequence[str], context: str) -> None:
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{context}: missing required columns: {missing}")


def maybe_warn_sample_size(expected: int, actual: int, label: str) -> None:
    if actual != expected:
        print(f"Warning: {label} has {actual} rows, expected {expected}. Continuing with available rows.")


def save_json(data: Dict, path: Path) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> None:
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0))
    ensure_dir(destination.parent)

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
    url = DATASET_URLS[version]
    dataset_path = data_dir / url.rsplit("/", maxsplit=1)[-1]
    if dataset_path.exists() and dataset_path.stat().st_size > 0:
        return dataset_path
    print(f"Dataset not found, downloading {version} to {dataset_path} ...")
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


def stratified_reservoir_sample(dataset_path: Path, targets: Dict[str, int], seed: int) -> pd.DataFrame:
    rng = random.Random(seed)
    seen_by_topic: Counter = Counter()
    reservoirs: Dict[str, List[Dict[str, str]]] = {topic: [] for topic in targets}

    for record in tqdm(iterate_records(dataset_path), desc=f"Sampling {sum(targets.values())}"):
        if not is_valid_record(record):
            continue

        topic = record.topic
        if topic not in targets:
            continue

        target_size = targets[topic]
        if target_size <= 0:
            continue

        seen_by_topic[topic] += 1
        row = {"title": record.title, "text": record.text, "topic": topic}
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
            raise RuntimeError(f"Topic '{topic}' has only {len(rows)} sampled rows, expected {expected}.")
        sampled_rows.extend(rows)

    sampled_df = pd.DataFrame(sampled_rows)
    return sampled_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


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


def filter_rare_topics(processed_df: pd.DataFrame, min_count: int) -> Tuple[pd.DataFrame, Dict[str, object]]:
    topic_counts = processed_df["topic"].value_counts().sort_index()
    kept_topics = topic_counts[topic_counts >= min_count].index
    dropped_topics = topic_counts[topic_counts < min_count]

    filtered_df = processed_df[processed_df["topic"].isin(kept_topics)].reset_index(drop=True)
    report = {
        "min_count_for_split": int(min_count),
        "rows_before": int(len(processed_df)),
        "rows_after": int(len(filtered_df)),
        "classes_before": int(topic_counts.shape[0]),
        "classes_after": int(filtered_df["topic"].nunique()) if not filtered_df.empty else 0,
        "dropped_topics": {str(topic): int(count) for topic, count in dropped_topics.items()},
    }
    return filtered_df, report


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
    sample_output_path = artifacts_dir / "sample.csv"
    processed_output_path = artifacts_dir / "sample_processed.csv"

    if args.preprocessed_sample_path is not None:
        print(f"Using preprocessed sample: {args.preprocessed_sample_path}")
        processed_df = pd.read_csv(args.preprocessed_sample_path)
        ensure_columns(processed_df, ["processed_text", "topic"], "preprocessed sample")
        processed_df = sanitize_processed_df(processed_df)
        if args.sample_size > 0:
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
        if args.sample_size > 0:
            maybe_warn_sample_size(args.sample_size, len(sampled_df), "sampled data")
    else:
        dataset_path = resolve_dataset_path(args)
        print(f"Using dataset: {dataset_path}")
        if args.sample_size > 0:
            topic_counts = count_topics(dataset_path)
            targets = allocate_targets(topic_counts, args.sample_size)
            sampled_df = stratified_reservoir_sample(dataset_path=dataset_path, targets=targets, seed=args.seed)
        else:
            rows = []
            for record in tqdm(iterate_records(dataset_path), desc="Reading full dataset"):
                if not is_valid_record(record):
                    continue
                rows.append({"title": record.title, "text": record.text, "topic": record.topic})
            sampled_df = pd.DataFrame(rows)
        sampled_df = sanitize_sample_df(sampled_df)
        if args.sample_size > 0:
            maybe_warn_sample_size(args.sample_size, len(sampled_df), "sampled data")

    sampled_df.to_csv(sample_output_path, index=False)
    processed_df = apply_preprocessing(sampled_df)
    processed_df = sanitize_processed_df(processed_df)
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


def split_data(texts: pd.Series, labels: np.ndarray, seed: int) -> SplitData:
    train_texts, temp_texts, train_labels, temp_labels = train_test_split(
        texts,
        labels,
        test_size=0.4,
        random_state=seed,
        stratify=labels,
    )

    val_texts, test_texts, val_labels, test_labels = train_test_split(
        temp_texts,
        temp_labels,
        test_size=0.5,
        random_state=seed,
        stratify=temp_labels,
    )

    return SplitData(
        train_texts=train_texts.reset_index(drop=True),
        val_texts=val_texts.reset_index(drop=True),
        test_texts=test_texts.reset_index(drop=True),
        train_labels=train_labels,
        val_labels=val_labels,
        test_labels=test_labels,
    )


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def tokenize_texts(texts: Sequence[str]) -> List[List[str]]:
    return [text.split() for text in texts]


def build_logreg(seed: int, c: float, max_iter: int) -> LogisticRegression:
    return LogisticRegression(
        solver="lbfgs",
        C=c,
        max_iter=max_iter,
        random_state=seed,
    )


def train_word2vec(
    sentences: Sequence[Sequence[str]],
    seed: int,
    vector_size: int,
    window: int,
    min_count: int,
    sg: int,
    negative: int,
    epochs: int,
    workers: int,
) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        epochs=epochs,
        workers=workers,
        seed=seed,
    )
    return model


def safe_most_similar(model, positive: str, topn: int = 5):
    if positive not in model.key_to_index:
        return None
    return [(word, float(score)) for word, score in model.most_similar(positive=positive, topn=topn)]


def safe_doesnt_match(model, words: Sequence[str]):
    if any(word not in model.key_to_index for word in words):
        return None
    return model.doesnt_match(list(words))


def intrinsic_evaluation(model) -> Dict[str, object]:
    most_similar_queries = ["политика", "экономика", "спорт", "президент", "рубль"]
    doesnt_match_queries = [
        ["футбол", "хоккей", "теннис", "президент"],
        ["рубль", "доллар", "евро", "шайба"],
        ["министр", "депутат", "сенатор", "гол"],
    ]

    return {
        "most_similar": {
            query: safe_most_similar(model, query) for query in most_similar_queries
        },
        "doesnt_match": {
            ", ".join(words): safe_doesnt_match(model, words) for words in doesnt_match_queries
        },
    }


@lru_cache(maxsize=300_000)
def normalize_rusvectores_token(token: str) -> str | None:
    if not token:
        return None
    parse = MORPH.parse(token)[0]
    pos = PYMORPHY_TO_UPOS.get(parse.tag.POS)
    if pos is None:
        return None
    return f"{parse.normal_form}_{pos}"


MORPH = MorphAnalyzer()


def build_rusvectores_token_lists(token_lists: Sequence[Sequence[str]]) -> List[List[str]]:
    normalized_docs: List[List[str]] = []
    for tokens in tqdm(token_lists, desc="Normalizing tokens for RusVectores"):
        normalized = [norm for token in tokens if (norm := normalize_rusvectores_token(token)) is not None]
        normalized_docs.append(normalized)
    return normalized_docs


def load_navec_model(path: Path):
    if Navec is None:
        raise ImportError("Package 'navec' is not installed. Run: python3 -m pip install navec")
    return Navec.load(str(path))


def ensure_navec_model(path: Path) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    print(f"Navec model not found, downloading to {path} ...")
    download_file(NAVEC_URL, path)
    return path


def ensure_rusvectores_model(zip_path: Path) -> Path:
    if not zip_path.exists() or zip_path.stat().st_size == 0:
        print(f"RusVectores model not found, downloading to {zip_path} ...")
        download_file(RUSVECTORES_URL, zip_path)

    extract_dir = zip_path.with_suffix("")
    model_bin_path = extract_dir / "model.bin"
    if model_bin_path.exists() and model_bin_path.stat().st_size > 0:
        return model_bin_path

    ensure_dir(extract_dir)
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(extract_dir)
    if not model_bin_path.exists():
        raise FileNotFoundError(f"Expected model.bin in extracted RusVectores archive at {model_bin_path}")
    return model_bin_path


def mean_pool_vectors(token_lists: Sequence[Sequence[str]], keyed_vectors) -> Tuple[np.ndarray, Dict[str, float]]:
    vector_size = get_vector_size(keyed_vectors)
    vectors = np.zeros((len(token_lists), vector_size), dtype=np.float32)
    non_empty_docs = 0
    covered_tokens = 0
    total_tokens = 0

    for index, tokens in enumerate(tqdm(token_lists, desc="Mean pooling embeddings")):
        doc_vectors = []
        for token in tokens:
            total_tokens += 1
            if has_embedding(keyed_vectors, token):
                covered_tokens += 1
                doc_vectors.append(get_embedding_vector(keyed_vectors, token))
        if doc_vectors:
            non_empty_docs += 1
            vectors[index] = np.mean(np.asarray(doc_vectors, dtype=np.float32), axis=0)

    coverage = {
        "token_coverage": float(covered_tokens / total_tokens) if total_tokens else 0.0,
        "non_empty_doc_ratio": float(non_empty_docs / len(token_lists)) if token_lists else 0.0,
    }
    return vectors, coverage


def tfidf_weighted_vectors(
    train_tokens: Sequence[Sequence[str]],
    eval_tokens: Sequence[Sequence[str]],
    keyed_vectors,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    vectorizer = TfidfVectorizer(
        analyzer=lambda x: x,
        lowercase=False,
        token_pattern=None,
        min_df=2,
    )
    train_matrix = vectorizer.fit_transform(train_tokens)
    eval_matrix = vectorizer.transform(eval_tokens)
    feature_names = np.asarray(vectorizer.get_feature_names_out())

    vector_size = get_vector_size(keyed_vectors)

    def build_from_matrix(matrix) -> Tuple[np.ndarray, Dict[str, float]]:
        vectors = np.zeros((matrix.shape[0], vector_size), dtype=np.float32)
        non_empty_docs = 0
        covered_features = 0
        total_features = 0

        for row_index in tqdm(range(matrix.shape[0]), desc="TF-IDF pooling embeddings"):
            row = matrix.getrow(row_index)
            if row.nnz == 0:
                continue

            weighted_sum = np.zeros(vector_size, dtype=np.float32)
            weight_total = 0.0
            for feature_index, weight in zip(row.indices, row.data):
                total_features += 1
                token = feature_names[feature_index]
                if not has_embedding(keyed_vectors, token):
                    continue
                covered_features += 1
                weighted_sum += get_embedding_vector(keyed_vectors, token) * weight
                weight_total += float(weight)

            if weight_total > 0.0:
                non_empty_docs += 1
                vectors[row_index] = weighted_sum / weight_total

        coverage = {
            "feature_coverage": float(covered_features / total_features) if total_features else 0.0,
            "non_empty_doc_ratio": float(non_empty_docs / matrix.shape[0]) if matrix.shape[0] else 0.0,
        }
        return vectors, coverage

    train_vectors, train_coverage = build_from_matrix(train_matrix)
    eval_vectors, eval_coverage = build_from_matrix(eval_matrix)
    return train_vectors, eval_vectors, {"train": train_coverage, "eval": eval_coverage}


def fit_and_eval_logreg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[LogisticRegression, Dict[str, float]]:
    model = build_logreg(seed=seed, c=c, max_iter=max_iter)
    model.fit(x_train, y_train)
    predictions = model.predict(x_eval)
    return model, compute_metrics(y_eval, predictions)


def get_vector_size(model) -> int:
    if hasattr(model, "vector_size"):
        return int(model.vector_size)
    if hasattr(model, "pq") and hasattr(model.pq, "dim"):
        return int(model.pq.dim)
    raise AttributeError(f"Unsupported embedding model type: {type(model)!r}")


def has_embedding(model, token: str) -> bool:
    try:
        return token in model
    except TypeError:
        return False


def get_embedding_vector(model, token: str) -> np.ndarray:
    return np.asarray(model[token], dtype=np.float32)


def evaluate_embedding_source(
    source_name: str,
    train_tokens: Sequence[Sequence[str]],
    eval_tokens: Sequence[Sequence[str]],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    keyed_vectors,
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[LogisticRegression, Dict[str, object]]:
    train_vectors, train_coverage = mean_pool_vectors(train_tokens, keyed_vectors)
    eval_vectors, eval_coverage = mean_pool_vectors(eval_tokens, keyed_vectors)
    model, metrics = fit_and_eval_logreg(
        x_train=train_vectors,
        y_train=y_train,
        x_eval=eval_vectors,
        y_eval=y_eval,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    return model, {
        "source": source_name,
        "metrics": metrics,
        "coverage": {
            "train": train_coverage,
            "eval": eval_coverage,
        },
    }


def evaluate_tfidf_variant(
    source_name: str,
    train_tokens: Sequence[Sequence[str]],
    eval_tokens: Sequence[Sequence[str]],
    y_train: np.ndarray,
    y_eval: np.ndarray,
    keyed_vectors,
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[LogisticRegression, Dict[str, object]]:
    train_vectors, eval_vectors, coverage = tfidf_weighted_vectors(
        train_tokens=train_tokens,
        eval_tokens=eval_tokens,
        keyed_vectors=keyed_vectors,
    )
    model, metrics = fit_and_eval_logreg(
        x_train=train_vectors,
        y_train=y_train,
        x_eval=eval_vectors,
        y_eval=y_eval,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    return model, {
        "source": source_name,
        "metrics": metrics,
        "coverage": coverage,
    }


def retrain_and_test_mean_pool(
    train_tokens: Sequence[Sequence[str]],
    test_tokens: Sequence[Sequence[str]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    keyed_vectors,
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[LogisticRegression, Dict[str, object]]:
    train_vectors, train_coverage = mean_pool_vectors(train_tokens, keyed_vectors)
    test_vectors, test_coverage = mean_pool_vectors(test_tokens, keyed_vectors)
    model, metrics = fit_and_eval_logreg(
        x_train=train_vectors,
        y_train=y_train,
        x_eval=test_vectors,
        y_eval=y_test,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    return model, {"metrics": metrics, "coverage": {"train": train_coverage, "test": test_coverage}}


def retrain_and_test_tfidf(
    train_tokens: Sequence[Sequence[str]],
    test_tokens: Sequence[Sequence[str]],
    y_train: np.ndarray,
    y_test: np.ndarray,
    keyed_vectors,
    seed: int,
    c: float,
    max_iter: int,
) -> Tuple[LogisticRegression, Dict[str, object]]:
    train_vectors, test_vectors, coverage = tfidf_weighted_vectors(
        train_tokens=train_tokens,
        eval_tokens=test_tokens,
        keyed_vectors=keyed_vectors,
    )
    model, metrics = fit_and_eval_logreg(
        x_train=train_vectors,
        y_train=y_train,
        x_eval=test_vectors,
        y_eval=y_test,
        seed=seed,
        c=c,
        max_iter=max_iter,
    )
    return model, {"metrics": metrics, "coverage": {"train": coverage["train"], "test": coverage["eval"]}}


def parse_embedding_sources(raw_value: str) -> List[str]:
    sources = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = [item for item in sources if item not in ALLOWED_EMBEDDINGS]
    if unknown:
        raise ValueError(f"Unknown embedding sources: {unknown}. Allowed: {ALLOWED_EMBEDDINGS}")
    if not sources:
        raise ValueError("At least one embedding source must be selected.")
    return sources


def build_pretrained_models(
    selected_sources: Sequence[str],
    embeddings_dir: Path,
    navec_path_arg: Path | None,
    rusvectores_path_arg: Path | None,
):
    models = {}

    if "navec" in selected_sources:
        navec_path = navec_path_arg or embeddings_dir / "navec_news_v1_1B_250K_300d_100q.tar"
        navec_path = ensure_navec_model(navec_path)
        models["navec"] = load_navec_model(navec_path)

    if "rusvectores" in selected_sources:
        rusvectores_zip = rusvectores_path_arg or embeddings_dir / "news_upos_skipgram_300_5_2019.zip"
        model_bin_path = ensure_rusvectores_model(rusvectores_zip)
        models["rusvectores"] = KeyedVectors.load_word2vec_format(
            str(model_bin_path),
            binary=True,
            unicode_errors="ignore",
        )

    return models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITMO NLP HW2 training pipeline")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts_hw2/run_default"))
    parser.add_argument("--dataset-version", type=str, default="v1.1", choices=sorted(DATASET_URLS))
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--sample-size", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument("--sample-path", type=Path, default=None)
    parser.add_argument("--preprocessed-sample-path", type=Path, default=None)
    parser.add_argument("--embedding-sources", type=str, default="custom_w2v,navec,rusvectores")
    parser.add_argument("--navec-path", type=Path, default=None)
    parser.add_argument("--rusvectores-path", type=Path, default=None)
    parser.add_argument("--w2v-vector-size", type=int, default=300)
    parser.add_argument("--w2v-window", type=int, default=5)
    parser.add_argument("--w2v-min-count", type=int, default=3)
    parser.add_argument("--w2v-sg", type=int, default=1)
    parser.add_argument("--w2v-negative", type=int, default=10)
    parser.add_argument("--w2v-epochs", type=int, default=12)
    parser.add_argument("--w2v-workers", type=int, default=1)
    parser.add_argument("--logreg-c", type=float, default=1.0)
    parser.add_argument("--logreg-max-iter", type=int, default=1000)
    parser.add_argument(
        "--min-topic-count-for-split",
        type=int,
        default=5,
        help="Drop topics with fewer examples than this before stratified 60/20/20 split.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    selected_sources = parse_embedding_sources(args.embedding_sources)

    artifacts_dir = args.artifacts_dir
    metrics_dir = artifacts_dir / "metrics"
    models_dir = artifacts_dir / "models"
    ensure_dir(metrics_dir)
    ensure_dir(models_dir)

    processed_df = load_or_prepare_processed_sample(
        args=args,
        artifacts_dir=artifacts_dir,
        metrics_dir=metrics_dir,
    )
    processed_df, filtering_report = filter_rare_topics(
        processed_df=processed_df,
        min_count=args.min_topic_count_for_split,
    )
    save_json(filtering_report, metrics_dir / "label_filtering.json")
    if filtering_report["rows_after"] == 0:
        raise ValueError("No rows left after filtering rare topics.")
    if filtering_report["dropped_topics"]:
        print(f"Dropped rare topics before split: {filtering_report['dropped_topics']}")

    label_to_id = {label: index for index, label in enumerate(sorted(processed_df["topic"].unique()))}
    y = processed_df["topic"].map(label_to_id).to_numpy()
    split = split_data(processed_df["processed_text"], y, seed=args.seed)

    split_stats = {
        "train_size": int(len(split.train_texts)),
        "val_size": int(len(split.val_texts)),
        "test_size": int(len(split.test_texts)),
        "num_classes": int(len(label_to_id)),
        "sample_size": int(len(processed_df)),
    }
    save_json(split_stats, metrics_dir / "split_stats.json")

    train_tokens = tokenize_texts(split.train_texts.tolist())
    val_tokens = tokenize_texts(split.val_texts.tolist())
    test_tokens = tokenize_texts(split.test_texts.tolist())

    word2vec_model = train_word2vec(
        sentences=train_tokens,
        seed=args.seed,
        vector_size=args.w2v_vector_size,
        window=args.w2v_window,
        min_count=args.w2v_min_count,
        sg=args.w2v_sg,
        negative=args.w2v_negative,
        epochs=args.w2v_epochs,
        workers=args.w2v_workers,
    )
    word2vec_model.save(str(models_dir / "custom_word2vec.model"))

    intrinsic_eval = intrinsic_evaluation(word2vec_model.wv)
    intrinsic_eval["hyperparameters"] = {
        "vector_size": args.w2v_vector_size,
        "window": args.w2v_window,
        "min_count": args.w2v_min_count,
        "sg": args.w2v_sg,
        "negative": args.w2v_negative,
        "epochs": args.w2v_epochs,
        "workers": args.w2v_workers,
    }
    save_json(intrinsic_eval, metrics_dir / "intrinsic_eval.json")

    pretrained_models = build_pretrained_models(
        selected_sources=selected_sources,
        embeddings_dir=args.data_dir / "embeddings",
        navec_path_arg=args.navec_path,
        rusvectores_path_arg=args.rusvectores_path,
    )

    rusvectores_train_tokens = None
    rusvectores_val_tokens = None
    rusvectores_test_tokens = None
    if "rusvectores" in selected_sources:
        rusvectores_train_tokens = build_rusvectores_token_lists(train_tokens)
        rusvectores_val_tokens = build_rusvectores_token_lists(val_tokens)
        rusvectores_test_tokens = build_rusvectores_token_lists(test_tokens)

    validation_results: Dict[str, Dict[str, object]] = {}
    if "custom_w2v" in selected_sources:
        model, result = evaluate_embedding_source(
            source_name="custom_w2v",
            train_tokens=train_tokens,
            eval_tokens=val_tokens,
            y_train=split.train_labels,
            y_eval=split.val_labels,
            keyed_vectors=word2vec_model.wv,
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        validation_results["custom_w2v"] = result

    if "navec" in selected_sources:
        model, result = evaluate_embedding_source(
            source_name="navec",
            train_tokens=train_tokens,
            eval_tokens=val_tokens,
            y_train=split.train_labels,
            y_eval=split.val_labels,
            keyed_vectors=pretrained_models["navec"],
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        validation_results["navec"] = result

    if "rusvectores" in selected_sources:
        model, result = evaluate_embedding_source(
            source_name="rusvectores",
            train_tokens=rusvectores_train_tokens,
            eval_tokens=rusvectores_val_tokens,
            y_train=split.train_labels,
            y_eval=split.val_labels,
            keyed_vectors=pretrained_models["rusvectores"],
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        validation_results["rusvectores"] = result

    save_json(validation_results, metrics_dir / "validation_results.json")

    best_base_source = max(
        validation_results,
        key=lambda source_name: validation_results[source_name]["metrics"]["macro_f1"],
    )
    save_json(
        {
            "best_base_source": best_base_source,
            "best_base_metrics": validation_results[best_base_source]["metrics"],
        },
        metrics_dir / "best_base_source.json",
    )

    if best_base_source == "custom_w2v":
        best_train_tokens = train_tokens
        best_val_tokens = val_tokens
        best_test_tokens = test_tokens
        best_keyed_vectors = word2vec_model.wv
    elif best_base_source == "navec":
        best_train_tokens = train_tokens
        best_val_tokens = val_tokens
        best_test_tokens = test_tokens
        best_keyed_vectors = pretrained_models["navec"]
    else:
        best_train_tokens = rusvectores_train_tokens
        best_val_tokens = rusvectores_val_tokens
        best_test_tokens = rusvectores_test_tokens
        best_keyed_vectors = pretrained_models["rusvectores"]

    _, tfidf_val_result = evaluate_tfidf_variant(
        source_name=best_base_source,
        train_tokens=best_train_tokens,
        eval_tokens=best_val_tokens,
        y_train=split.train_labels,
        y_eval=split.val_labels,
        keyed_vectors=best_keyed_vectors,
        seed=args.seed,
        c=args.logreg_c,
        max_iter=args.logreg_max_iter,
    )
    save_json(tfidf_val_result, metrics_dir / "tfidf_validation_result.json")

    trainval_texts = pd.concat([split.train_texts, split.val_texts], axis=0).reset_index(drop=True)
    trainval_labels = np.concatenate([split.train_labels, split.val_labels])
    trainval_tokens = tokenize_texts(trainval_texts.tolist())

    test_results: Dict[str, Dict[str, object]] = {}

    if "custom_w2v" in selected_sources:
        trainval_w2v_model = train_word2vec(
            sentences=trainval_tokens,
            seed=args.seed,
            vector_size=args.w2v_vector_size,
            window=args.w2v_window,
            min_count=args.w2v_min_count,
            sg=args.w2v_sg,
            negative=args.w2v_negative,
            epochs=args.w2v_epochs,
            workers=args.w2v_workers,
        )
        trainval_w2v_model.save(str(models_dir / "custom_word2vec_trainval.model"))
        model, result = retrain_and_test_mean_pool(
            train_tokens=trainval_tokens,
            test_tokens=test_tokens,
            y_train=trainval_labels,
            y_test=split.test_labels,
            keyed_vectors=trainval_w2v_model.wv,
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        dump(model, models_dir / "logreg_custom_w2v.joblib")
        test_results["custom_w2v"] = result

    if "navec" in selected_sources:
        model, result = retrain_and_test_mean_pool(
            train_tokens=trainval_tokens,
            test_tokens=test_tokens,
            y_train=trainval_labels,
            y_test=split.test_labels,
            keyed_vectors=pretrained_models["navec"],
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        dump(model, models_dir / "logreg_navec.joblib")
        test_results["navec"] = result

    trainval_rusvectores_tokens = None
    if "rusvectores" in selected_sources:
        trainval_rusvectores_tokens = build_rusvectores_token_lists(trainval_tokens)
        model, result = retrain_and_test_mean_pool(
            train_tokens=trainval_rusvectores_tokens,
            test_tokens=rusvectores_test_tokens,
            y_train=trainval_labels,
            y_test=split.test_labels,
            keyed_vectors=pretrained_models["rusvectores"],
            seed=args.seed,
            c=args.logreg_c,
            max_iter=args.logreg_max_iter,
        )
        dump(model, models_dir / "logreg_rusvectores.joblib")
        test_results["rusvectores"] = result

    if best_base_source == "custom_w2v":
        final_best_vectors = trainval_w2v_model.wv
        final_best_train_tokens = trainval_tokens
        final_best_test_tokens = test_tokens
    elif best_base_source == "navec":
        final_best_vectors = pretrained_models["navec"]
        final_best_train_tokens = trainval_tokens
        final_best_test_tokens = test_tokens
    else:
        final_best_vectors = pretrained_models["rusvectores"]
        final_best_train_tokens = trainval_rusvectores_tokens
        final_best_test_tokens = rusvectores_test_tokens

    tfidf_model, tfidf_test_result = retrain_and_test_tfidf(
        train_tokens=final_best_train_tokens,
        test_tokens=final_best_test_tokens,
        y_train=trainval_labels,
        y_test=split.test_labels,
        keyed_vectors=final_best_vectors,
        seed=args.seed,
        c=args.logreg_c,
        max_iter=args.logreg_max_iter,
    )
    dump(tfidf_model, models_dir / f"logreg_tfidf_{best_base_source}.joblib")
    test_results[f"tfidf_weighted_{best_base_source}"] = tfidf_test_result

    save_json(test_results, metrics_dir / "test_results.json")

    run_notes = {
        "preprocessing": [
            "merge title and text",
            "lowercase",
            "remove non-alphanumeric chars",
            "normalize spaces",
        ],
        "word2vec_design": {
            "algorithm": "skip-gram" if args.w2v_sg == 1 else "cbow",
            "vector_size": args.w2v_vector_size,
            "window": args.w2v_window,
            "min_count": args.w2v_min_count,
            "negative": args.w2v_negative,
            "epochs": args.w2v_epochs,
            "reasoning": [
                "300 dimensions is a standard compromise between quality and model size.",
                "window=5 balances topical and local context.",
                "min_count=3 filters noisy rare tokens without losing too much vocabulary.",
                "skip-gram usually works better than cbow on semantic similarity for medium corpora.",
                "workers=1 keeps the training reproducible.",
            ],
        },
        "logreg_design": {
            "solver": "lbfgs",
            "C": args.logreg_c,
            "max_iter": args.logreg_max_iter,
            "reasoning": [
                "Document vectors are dense and low-dimensional, so lbfgs is stable here.",
                "The same classifier setup is used for all embeddings to keep the comparison fair.",
            ],
        },
        "rusvectores_preprocessing": (
            "RusVectores lookup uses pymorphy3-based normalization to lemma_UPOS because the chosen "
            "pretrained model stores lemmatized and POS-tagged entries."
        ),
    }
    save_json(run_notes, metrics_dir / "run_notes.json")

    print("HW2 pipeline finished successfully.")
    print(f"Best base source on validation: {best_base_source}")
    print(f"Validation metrics: {validation_results[best_base_source]['metrics']}")


if __name__ == "__main__":
    main()
