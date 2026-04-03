#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import re
import textwrap
import time
from collections import Counter
from dataclasses import asdict, dataclass, field, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from corus import load_lenta2
from sklearn.feature_extraction.text import CountVectorizer
from tqdm import tqdm

try:
    from pymorphy3 import MorphAnalyzer
except ImportError:  # pragma: no cover - optional until lemmatization is requested.
    MorphAnalyzer = None


os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё-]{1,}")
SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://\S+|www\.\S+")

RUSSIAN_STOPWORDS = {
    "а", "без", "более", "больше", "будет", "будто", "бы", "был", "была",
    "были", "было", "быть", "в", "вам", "вас", "вдруг", "ведь", "во", "вот",
    "впрочем", "все", "всего", "всех", "вы", "где", "да", "даже", "два", "для",
    "до", "его", "ее", "если", "есть", "еще", "же", "за", "здесь", "и", "из",
    "или", "им", "их", "к", "как", "какая", "какой", "когда", "конечно",
    "кто", "куда", "ли", "либо", "между", "меня", "мне", "может", "можно",
    "мой", "моя", "мы", "на", "над", "надо", "наконец", "нас", "не", "него",
    "нее", "нет", "ни", "нибудь", "никогда", "ним", "них", "но", "ну", "о",
    "об", "однако", "он", "она", "они", "оно", "опять", "от", "по", "под",
    "после", "потом", "потому", "почти", "при", "про", "раз", "разве", "с",
    "сам", "свое", "свою", "себе", "себя", "сейчас", "со", "совсем", "так",
    "также", "такой", "там", "тебя", "тем", "теперь", "то", "тогда", "того",
    "тоже", "только", "том", "тот", "три", "тут", "ты", "у", "уж", "уже",
    "хоть", "чего", "чей", "чем", "через", "что", "чтобы", "чуть", "эта",
    "эти", "это", "этого", "этой", "этом", "этот", "я",
}

NEWS_STOPWORDS = {
    "года", "году", "год", "день", "дня", "дней", "месяц", "месяца", "месяцев",
    "неделя", "недели", "сегодня", "завтра", "вчера", "сообщить", "сообщать",
    "заявить", "отметить", "рассказать", "добавить", "пояснить", "подчеркнуть",
    "сказать", "стать", "стало", "ранее", "позже", "первый", "второй", "третий",
    "который", "которая", "которые", "свой", "своя", "свои", "человек", "люди",
    "москва", "россиянин", "россияне", "российский", "российская", "российские",
    "тысяча", "миллион", "миллиард", "фото", "видео", "источник",
}

DEFAULT_EXCLUDED_TOPICS = {"Библиотека"}

PRESET_CONFIGS: dict[str, dict[str, Any]] = {
    "baseline_multilingual": {
        "encoder_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "lemmatize": False,
        "vectorizer_min_df": 1,
        "vectorizer_max_df": 1.0,
        "ngram_min": 1,
        "ngram_max": 1,
        "umap_n_neighbors": 15,
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "hdbscan_min_cluster_size": 120,
        "hdbscan_min_samples": 20,
        "top_n_words": 10,
        "nr_topics": "auto",
        "use_keybert": False,
        "use_mmr": False,
        "bm25_weighting": False,
        "max_seq_length": 256,
    },
    "lemmatized_multilingual": {
        "encoder_name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "lemmatize": True,
        "vectorizer_min_df": 1,
        "vectorizer_max_df": 1.0,
        "ngram_min": 1,
        "ngram_max": 2,
        "umap_n_neighbors": 20,
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "hdbscan_min_cluster_size": 100,
        "hdbscan_min_samples": 15,
        "top_n_words": 10,
        "nr_topics": "auto",
        "use_keybert": True,
        "use_mmr": True,
        "bm25_weighting": True,
        "max_seq_length": 256,
    },
    "english_ablation": {
        "encoder_name": "sentence-transformers/all-MiniLM-L6-v2",
        "lemmatize": True,
        "vectorizer_min_df": 1,
        "vectorizer_max_df": 1.0,
        "ngram_min": 1,
        "ngram_max": 2,
        "umap_n_neighbors": 20,
        "umap_n_components": 5,
        "umap_min_dist": 0.0,
        "hdbscan_min_cluster_size": 100,
        "hdbscan_min_samples": 15,
        "top_n_words": 10,
        "nr_topics": "auto",
        "use_keybert": True,
        "use_mmr": True,
        "bm25_weighting": True,
        "max_seq_length": 256,
    },
}


@dataclass
class CorpusSample:
    frame: pd.DataFrame
    stats: dict[str, Any]


@dataclass
class FitConfig:
    preset: str
    seed: int
    data_path: str
    output_dir: str
    sample_size: int
    max_records: int | None
    prepared_corpus_path: str | None
    max_text_chars: int
    min_document_chars: int
    excluded_topics: list[str] = field(default_factory=lambda: sorted(DEFAULT_EXCLUDED_TOPICS))
    encoder_name: str = PRESET_CONFIGS["lemmatized_multilingual"]["encoder_name"]
    batch_size: int = 32
    normalize_embeddings: bool = True
    lemmatize: bool = True
    vectorizer_min_df: int = 1
    vectorizer_max_df: float = 1.0
    ngram_min: int = 1
    ngram_max: int = 2
    max_features: int = 50000
    umap_n_neighbors: int = 20
    umap_n_components: int = 5
    umap_min_dist: float = 0.0
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: int = 15
    top_n_words: int = 10
    nr_topics: str | int | None = "auto"
    use_keybert: bool = True
    use_mmr: bool = True
    bm25_weighting: bool = True
    max_seq_length: int = 256
    documents_viz_sample_size: int = 2500
    distribution_doc_count: int = 3
    min_probability_to_show: float = 0.02


class RussianTokenizer:
    def __init__(self, *, lemmatize: bool, stopwords: set[str]) -> None:
        self.stopwords = stopwords
        self.lemmatize = lemmatize
        self.morph = None
        if lemmatize:
            if MorphAnalyzer is None:
                raise ImportError("pymorphy3 is required when --lemmatize is enabled")
            self.morph = MorphAnalyzer()

    def __call__(self, text: str) -> list[str]:
        tokens: list[str] = []
        for raw_token in TOKEN_RE.findall(text.lower()):
            token = raw_token.strip("-")
            if len(token) < 3:
                continue
            if self.morph is not None:
                token = self.morph.parse(token)[0].normal_form
            if token in self.stopwords:
                continue
            if token.isdigit():
                continue
            tokens.append(token)
        return tokens


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="HW4 pipeline for BERTopic on Lenta.ru news with Corus."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect-dataset", help="Inspect and sample the corpus")
    add_common_dataset_args(inspect_parser)
    inspect_parser.add_argument("--output-dir", default="artifacts_hw4/dataset_inspect")
    inspect_parser.add_argument("--preview-size", type=int, default=5)

    fit_parser = subparsers.add_parser("fit", help="Fit a BERTopic model")
    add_common_dataset_args(fit_parser)
    fit_parser.add_argument("--output-dir", default="artifacts_hw4/default_run")
    fit_parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CONFIGS),
        default="lemmatized_multilingual",
    )
    fit_parser.add_argument("--batch-size", type=int, default=32)
    fit_parser.add_argument("--normalize-embeddings", action=argparse.BooleanOptionalAction, default=True)
    fit_parser.add_argument("--encoder-name", default=None)
    fit_parser.add_argument("--lemmatize", action=argparse.BooleanOptionalAction, default=None)
    fit_parser.add_argument("--vectorizer-min-df", type=int, default=None)
    fit_parser.add_argument("--vectorizer-max-df", type=float, default=None)
    fit_parser.add_argument("--ngram-min", type=int, default=None)
    fit_parser.add_argument("--ngram-max", type=int, default=None)
    fit_parser.add_argument("--max-features", type=int, default=50000)
    fit_parser.add_argument("--umap-n-neighbors", type=int, default=None)
    fit_parser.add_argument("--umap-n-components", type=int, default=None)
    fit_parser.add_argument("--umap-min-dist", type=float, default=None)
    fit_parser.add_argument("--hdbscan-min-cluster-size", type=int, default=None)
    fit_parser.add_argument("--hdbscan-min-samples", type=int, default=None)
    fit_parser.add_argument("--top-n-words", type=int, default=None)
    fit_parser.add_argument("--nr-topics", default=None)
    fit_parser.add_argument("--use-keybert", action=argparse.BooleanOptionalAction, default=None)
    fit_parser.add_argument("--use-mmr", action=argparse.BooleanOptionalAction, default=None)
    fit_parser.add_argument("--bm25-weighting", action=argparse.BooleanOptionalAction, default=None)
    fit_parser.add_argument("--max-seq-length", type=int, default=None)
    fit_parser.add_argument("--documents-viz-sample-size", type=int, default=2500)
    fit_parser.add_argument("--distribution-doc-count", type=int, default=3)
    fit_parser.add_argument("--min-probability-to-show", type=float, default=0.02)

    compare_parser = subparsers.add_parser("compare-runs", help="Compare metrics across saved runs")
    compare_parser.add_argument("--root-dir", default="artifacts_hw4")
    compare_parser.add_argument("--output-path", default="artifacts_hw4/run_summary.md")

    return parser.parse_args()


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data-path", default="../data/lenta-ru-news.csv.bz2")
    parser.add_argument("--sample-size", type=int, default=15000)
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--prepared-corpus-path", default=None)
    parser.add_argument("--max-text-chars", type=int, default=1200)
    parser.add_argument("--min-document-chars", type=int, default=80)
    parser.add_argument("--exclude-topic", action="append", dest="excluded_topics", default=None)
    parser.add_argument("--seed", type=int, default=42)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def now_iso() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat()


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_json(path: str | Path, payload: Any) -> None:
    Path(path).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    text = URL_RE.sub(" ", text)
    text = text.replace("\xa0", " ").replace("\u200b", " ")
    text = SPACE_RE.sub(" ", text).strip()
    return text


def compose_document(title: str | None, text: str | None, max_text_chars: int) -> str:
    clean_title = normalize_text(title)
    clean_text = normalize_text(text)
    if max_text_chars > 0:
        clean_text = clean_text[:max_text_chars].rstrip()
    if clean_title and clean_text:
        return f"{clean_title}. {clean_text}"
    return clean_title or clean_text


def build_row(record: Any, *, max_text_chars: int, min_document_chars: int, excluded_topics: set[str]) -> tuple[dict[str, Any] | None, str | None]:
    topic = (record.topic or "").strip()
    if topic in excluded_topics:
        return None, "excluded_topic"

    document = compose_document(record.title, record.text, max_text_chars=max_text_chars)
    if len(document) < min_document_chars:
        return None, "too_short"

    return {
        "url": record.url,
        "title": normalize_text(record.title),
        "text": normalize_text(record.text),
        "topic": topic or None,
        "tags": normalize_text(record.tags),
        "date": str(record.date) if record.date is not None else None,
        "document": document,
        "document_length": len(document),
        "title_length": len(normalize_text(record.title)),
        "text_length": len(normalize_text(record.text)),
    }, None


def sample_corpus(
    *,
    data_path: str,
    sample_size: int,
    seed: int,
    max_text_chars: int,
    min_document_chars: int,
    excluded_topics: set[str],
    max_records: int | None = None,
) -> CorpusSample:
    rng = random.Random(seed)
    sample_rows: list[dict[str, Any]] = []
    kept_records = 0
    scanned_records = 0
    filtered_records = Counter()
    topic_counter = Counter()
    document_lengths: list[int] = []

    iterator = load_lenta2(data_path)
    for raw_index, record in enumerate(tqdm(iterator, desc="Scanning Lenta corpus"), start=1):
        if max_records is not None and raw_index > max_records:
            break
        scanned_records = raw_index

        row, reason = build_row(
            record,
            max_text_chars=max_text_chars,
            min_document_chars=min_document_chars,
            excluded_topics=excluded_topics,
        )
        if row is None:
            filtered_records[reason or "unknown"] += 1
            continue

        topic_counter[row["topic"] or "UNKNOWN"] += 1
        document_lengths.append(row["document_length"])
        kept_records += 1
        if len(sample_rows) < sample_size:
            sample_rows.append(row)
            continue

        replace_index = rng.randint(0, kept_records - 1)
        if replace_index < sample_size:
            sample_rows[replace_index] = row

    frame = pd.DataFrame(sample_rows)
    if not frame.empty and "date" in frame.columns:
        frame = frame.sort_values(["date", "url"], na_position="last").reset_index(drop=True)

    length_series = pd.Series(document_lengths, dtype=float) if document_lengths else pd.Series(dtype=float)
    stats = {
        "data_path": data_path,
        "seed": seed,
        "requested_sample_size": sample_size,
        "sample_size": int(len(frame)),
        "max_records": max_records,
        "scanned_records": scanned_records,
        "kept_records_after_filters": kept_records,
        "filtered_records": dict(filtered_records),
        "excluded_topics": sorted(excluded_topics),
        "topic_counts_after_filters_top20": dict(topic_counter.most_common(20)),
        "topic_count_after_filters": int(len(topic_counter)),
        "document_length_stats": (
            length_series.describe(percentiles=[0.1, 0.5, 0.9, 0.99]).to_dict()
            if not length_series.empty
            else {}
        ),
    }
    return CorpusSample(frame=frame, stats=stats)


def inspect_dataset(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    output_dir = ensure_dir(args.output_dir)
    excluded_topics = set(args.excluded_topics or DEFAULT_EXCLUDED_TOPICS)
    sample = sample_corpus(
        data_path=args.data_path,
        sample_size=args.sample_size,
        seed=args.seed,
        max_text_chars=args.max_text_chars,
        min_document_chars=args.min_document_chars,
        excluded_topics=excluded_topics,
        max_records=args.max_records,
    )

    corpus_path = output_dir / "prepared_corpus_sample.csv"
    preview_path = output_dir / "dataset_preview.txt"
    info_path = output_dir / "dataset_info.json"

    sample.frame.to_csv(corpus_path, index=False)

    preview_rows = sample.frame.head(args.preview_size).to_dict(orient="records")
    preview_text = []
    for index, row in enumerate(preview_rows, start=1):
        preview_text.append(
            textwrap.dedent(
                f"""\
                [{index}]
                title: {row['title']}
                topic: {row['topic']}
                tags: {row['tags']}
                url: {row['url']}
                document: {row['document'][:600]}
                """
            ).strip()
        )
    preview_path.write_text("\n\n".join(preview_text), encoding="utf-8")

    payload = {
        "created_at": now_iso(),
        **sample.stats,
        "prepared_corpus_path": str(corpus_path),
        "preview_path": str(preview_path),
    }
    write_json(info_path, payload)
    print(f"Saved dataset inspection artifacts to {output_dir}")


def import_topic_stack() -> dict[str, Any]:
    try:
        from bertopic import BERTopic
        from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
    except ImportError as exc:  # pragma: no cover - depends on local env.
        raise SystemExit(
            "BERTopic stack is missing. Install dz4/requirements.txt before running `fit`."
        ) from exc

    return {
        "BERTopic": BERTopic,
        "ClassTfidfTransformer": ClassTfidfTransformer,
        "HDBSCAN": HDBSCAN,
        "KeyBERTInspired": KeyBERTInspired,
        "MaximalMarginalRelevance": MaximalMarginalRelevance,
        "SentenceTransformer": SentenceTransformer,
        "UMAP": UMAP,
    }


def resolve_fit_config(args: argparse.Namespace) -> FitConfig:
    preset_values = PRESET_CONFIGS[args.preset].copy()

    def pick(name: str, default: Any) -> Any:
        cli_value = getattr(args, name)
        return default if cli_value is None else cli_value

    nr_topics = pick("nr_topics", preset_values["nr_topics"])
    if isinstance(nr_topics, str) and nr_topics.isdigit():
        nr_topics = int(nr_topics)
    elif nr_topics == "none":
        nr_topics = None

    return FitConfig(
        preset=args.preset,
        seed=args.seed,
        data_path=args.data_path,
        output_dir=args.output_dir,
        sample_size=args.sample_size,
        max_records=args.max_records,
        prepared_corpus_path=args.prepared_corpus_path,
        max_text_chars=args.max_text_chars,
        min_document_chars=args.min_document_chars,
        excluded_topics=sorted(set(args.excluded_topics or DEFAULT_EXCLUDED_TOPICS)),
        encoder_name=pick("encoder_name", preset_values["encoder_name"]),
        batch_size=args.batch_size,
        normalize_embeddings=args.normalize_embeddings,
        lemmatize=pick("lemmatize", preset_values["lemmatize"]),
        vectorizer_min_df=pick("vectorizer_min_df", preset_values["vectorizer_min_df"]),
        vectorizer_max_df=pick("vectorizer_max_df", preset_values["vectorizer_max_df"]),
        ngram_min=pick("ngram_min", preset_values["ngram_min"]),
        ngram_max=pick("ngram_max", preset_values["ngram_max"]),
        max_features=args.max_features,
        umap_n_neighbors=pick("umap_n_neighbors", preset_values["umap_n_neighbors"]),
        umap_n_components=pick("umap_n_components", preset_values["umap_n_components"]),
        umap_min_dist=pick("umap_min_dist", preset_values["umap_min_dist"]),
        hdbscan_min_cluster_size=pick("hdbscan_min_cluster_size", preset_values["hdbscan_min_cluster_size"]),
        hdbscan_min_samples=pick("hdbscan_min_samples", preset_values["hdbscan_min_samples"]),
        top_n_words=pick("top_n_words", preset_values["top_n_words"]),
        nr_topics=nr_topics,
        use_keybert=pick("use_keybert", preset_values["use_keybert"]),
        use_mmr=pick("use_mmr", preset_values["use_mmr"]),
        bm25_weighting=pick("bm25_weighting", preset_values["bm25_weighting"]),
        max_seq_length=pick("max_seq_length", preset_values["max_seq_length"]),
        documents_viz_sample_size=args.documents_viz_sample_size,
        distribution_doc_count=args.distribution_doc_count,
        min_probability_to_show=args.min_probability_to_show,
    )


def prepare_corpus_for_fit(config: FitConfig, run_dir: Path) -> pd.DataFrame:
    corpus_path = run_dir / "prepared_corpus.csv"
    inspection_dir = ensure_dir(run_dir / "dataset_inspect")

    if config.prepared_corpus_path is not None:
        prepared_path = Path(config.prepared_corpus_path)
        sample_frame = pd.read_csv(prepared_path)
        sample_frame.to_csv(corpus_path, index=False)
        write_json(
            inspection_dir / "dataset_info.json",
            {
                "created_at": now_iso(),
                "data_path": config.data_path,
                "seed": config.seed,
                "requested_sample_size": config.sample_size,
                "sample_size": int(len(sample_frame)),
                "max_records": config.max_records,
                "prepared_corpus_path": str(corpus_path),
                "source_prepared_corpus_path": str(prepared_path),
                "note": "Loaded from an externally prepared corpus sample.",
            },
        )
        preview_text = "\n\n".join(
            [
                textwrap.dedent(
                    f"""\
                    [{index}]
                    title: {row.get('title', '')}
                    topic: {row.get('topic', '')}
                    document: {str(row.get('document', ''))[:600]}
                    """
                ).strip()
                for index, row in enumerate(sample_frame.head(5).to_dict(orient="records"), start=1)
            ]
        )
        (inspection_dir / "dataset_preview.txt").write_text(preview_text, encoding="utf-8")
        return sample_frame

    sample = sample_corpus(
        data_path=config.data_path,
        sample_size=config.sample_size,
        seed=config.seed,
        max_text_chars=config.max_text_chars,
        min_document_chars=config.min_document_chars,
        excluded_topics=set(config.excluded_topics),
        max_records=config.max_records,
    )

    sample.frame.to_csv(corpus_path, index=False)
    write_json(
        inspection_dir / "dataset_info.json",
        {
            "created_at": now_iso(),
            **sample.stats,
            "prepared_corpus_path": str(corpus_path),
        },
    )
    preview_text = "\n\n".join(
        [
            textwrap.dedent(
                f"""\
                [{index}]
                title: {row['title']}
                topic: {row['topic']}
                document: {row['document'][:600]}
                """
            ).strip()
            for index, row in enumerate(sample.frame.head(5).to_dict(orient="records"), start=1)
        ]
    )
    (inspection_dir / "dataset_preview.txt").write_text(preview_text, encoding="utf-8")
    return sample.frame


def build_vectorizer(config: FitConfig) -> tuple[CountVectorizer, RussianTokenizer]:
    stopwords = set(RUSSIAN_STOPWORDS) | set(NEWS_STOPWORDS)
    tokenizer = RussianTokenizer(lemmatize=config.lemmatize, stopwords=stopwords)
    vectorizer = CountVectorizer(
        tokenizer=tokenizer,
        token_pattern=None,
        lowercase=False,
        min_df=config.vectorizer_min_df,
        max_df=config.vectorizer_max_df,
        ngram_range=(config.ngram_min, config.ngram_max),
        max_features=config.max_features,
    )
    return vectorizer, tokenizer


def compute_embeddings(
    sentence_transformer_cls: Any,
    docs: list[str],
    config: FitConfig,
) -> tuple[Any, np.ndarray]:
    model = sentence_transformer_cls(config.encoder_name, device="cpu")
    if hasattr(model, "max_seq_length"):
        model.max_seq_length = config.max_seq_length
    embeddings = model.encode(
        docs,
        batch_size=config.batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=config.normalize_embeddings,
    )
    return model, embeddings


def build_topic_model(config: FitConfig, embedding_model: Any, stack: dict[str, Any], vectorizer: CountVectorizer) -> Any:
    UMAP = stack["UMAP"]
    HDBSCAN = stack["HDBSCAN"]
    BERTopic = stack["BERTopic"]
    ClassTfidfTransformer = stack["ClassTfidfTransformer"]
    KeyBERTInspired = stack["KeyBERTInspired"]
    MaximalMarginalRelevance = stack["MaximalMarginalRelevance"]

    umap_model = UMAP(
        n_neighbors=config.umap_n_neighbors,
        n_components=config.umap_n_components,
        min_dist=config.umap_min_dist,
        metric="cosine",
        random_state=config.seed,
    )
    hdbscan_model = HDBSCAN(
        min_cluster_size=config.hdbscan_min_cluster_size,
        min_samples=config.hdbscan_min_samples,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )
    ctfidf_model = ClassTfidfTransformer(
        bm25_weighting=config.bm25_weighting,
        reduce_frequent_words=True,
    )

    representation_model = None
    if config.use_keybert or config.use_mmr:
        representation_model = {}
        if config.use_keybert:
            representation_model["Main"] = KeyBERTInspired(top_n_words=config.top_n_words)
        if config.use_mmr:
            representation_model["MMR"] = MaximalMarginalRelevance(diversity=0.3)

    return BERTopic(
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer,
        ctfidf_model=ctfidf_model,
        representation_model=representation_model,
        calculate_probabilities=True,
        nr_topics=config.nr_topics,
        top_n_words=config.top_n_words,
        verbose=True,
    )


def max_probabilities(probabilities: Any) -> list[float | None]:
    if probabilities is None:
        return [None]
    if isinstance(probabilities, np.ndarray):
        if probabilities.ndim == 1:
            return probabilities.astype(float).tolist()
        return probabilities.max(axis=1).astype(float).tolist()
    return [None]


def extract_topic_keywords(topic_model: Any, top_n_words: int) -> dict[str, list[dict[str, float]]]:
    output: dict[str, list[dict[str, float]]] = {}
    for topic_id, words in topic_model.get_topics().items():
        if words is None:
            continue
        output[str(topic_id)] = [
            {"word": word, "weight": float(weight)}
            for word, weight in words[:top_n_words]
        ]
    return output


def compute_topic_diversity(topic_model: Any, top_n_words: int) -> float:
    all_words: list[str] = []
    for topic_id, words in topic_model.get_topics().items():
        if topic_id == -1 or not words:
            continue
        all_words.extend([word for word, _ in words[:top_n_words]])
    if not all_words:
        return 0.0
    return len(set(all_words)) / len(all_words)


def compute_umass_coherence(topic_model: Any, tokenized_docs: list[list[str]], top_n_words: int) -> float | None:
    from gensim.corpora import Dictionary
    from gensim.models.coherencemodel import CoherenceModel

    dictionary = Dictionary(tokenized_docs)
    if len(dictionary) == 0:
        return None

    corpus = [dictionary.doc2bow(tokens) for tokens in tokenized_docs]
    topics: list[list[str]] = []
    for topic_id, words in topic_model.get_topics().items():
        if topic_id == -1 or not words:
            continue
        filtered = [word for word, _ in words[:top_n_words] if word in dictionary.token2id]
        if len(filtered) >= 2:
            topics.append(filtered)

    if not topics:
        return None

    coherence = CoherenceModel(
        topics=topics,
        corpus=corpus,
        dictionary=dictionary,
        coherence="u_mass",
    )
    return float(coherence.get_coherence())


def save_plot(fig: Any, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn")


def save_documents_scatter(doc_info: pd.DataFrame, path: Path) -> None:
    import plotly.express as px

    plot_frame = doc_info.copy()
    plot_frame["topic_label"] = plot_frame["topic"].astype(str)
    plot_frame["hover_title"] = plot_frame["title"].fillna("")
    plot_frame["hover_preview"] = plot_frame["document"].fillna("").str.slice(0, 220)

    fig = px.scatter(
        plot_frame,
        x="x_2d",
        y="y_2d",
        color="topic_label",
        hover_data={
            "topic_label": True,
            "hover_title": True,
            "hover_preview": True,
            "x_2d": False,
            "y_2d": False,
        },
        title="Documents in 2D topic space",
    )
    fig.update_traces(marker={"size": 6, "opacity": 0.7})
    fig.update_layout(legend_title_text="Topic")
    save_plot(fig, path)


def pick_selected_doc_indices(doc_info: pd.DataFrame, count: int) -> list[int]:
    chosen: list[int] = []
    ranked_topics = (
        doc_info.loc[doc_info["topic"] != -1]
        .groupby("topic")
        .size()
        .sort_values(ascending=False)
        .index
        .tolist()
    )
    for topic_id in ranked_topics:
        candidates = (
            doc_info.loc[doc_info["topic"] == topic_id]
            .sort_values("probability", ascending=False, na_position="last")
            .index
            .tolist()
        )
        for idx in candidates:
            if idx not in chosen:
                chosen.append(idx)
                break
        if len(chosen) >= count:
            break
    return chosen


def fit_model(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    config = resolve_fit_config(args)
    run_dir = ensure_dir(config.output_dir)
    ensure_dir(run_dir / "visualizations")

    corpus_df = prepare_corpus_for_fit(config, run_dir)
    docs = corpus_df["document"].tolist()
    if not docs:
        raise SystemExit("Prepared corpus is empty after filtering.")

    stack = import_topic_stack()
    vectorizer, tokenizer = build_vectorizer(config)

    start_time = time.time()
    embedding_model, embeddings = compute_embeddings(stack["SentenceTransformer"], docs, config)
    effective_config = config
    topic_model = build_topic_model(effective_config, embedding_model, stack, vectorizer)
    try:
        topics, probabilities = topic_model.fit_transform(docs, embeddings)
    except ValueError as exc:
        if effective_config.nr_topics == "auto" and "0 sample(s)" in str(exc):
            effective_config = replace(effective_config, nr_topics=None)
            topic_model = build_topic_model(effective_config, embedding_model, stack, vectorizer)
            topics, probabilities = topic_model.fit_transform(docs, embeddings)
        else:
            raise
    reduced_2d = stack["UMAP"](
        n_neighbors=max(10, min(effective_config.umap_n_neighbors, 30)),
        n_components=2,
        min_dist=0.0,
        metric="cosine",
        random_state=effective_config.seed,
    ).fit_transform(embeddings)
    elapsed_seconds = time.time() - start_time

    run_config_payload = asdict(config)
    run_config_payload["effective_nr_topics"] = effective_config.nr_topics
    run_config_payload["fit_completed_at"] = now_iso()
    write_json(run_dir / "run_config.json", run_config_payload)

    topic_info = topic_model.get_topic_info()
    topic_info.to_csv(run_dir / "topic_info.csv", index=False)
    write_json(run_dir / "topic_keywords.json", extract_topic_keywords(topic_model, effective_config.top_n_words))

    probability_values = max_probabilities(probabilities)
    if len(probability_values) == 1 and len(docs) > 1:
        probability_values = probability_values * len(docs)

    doc_info = corpus_df.copy()
    if "topic" in doc_info.columns:
        doc_info = doc_info.rename(columns={"topic": "ground_truth_topic"})
    doc_info["topic"] = topics
    doc_info["probability"] = probability_values
    doc_info["x_2d"] = reduced_2d[:, 0]
    doc_info["y_2d"] = reduced_2d[:, 1]
    doc_info.to_csv(run_dir / "document_topics.csv", index=False)

    tokenized_docs = [tokenizer(doc) for doc in tqdm(docs, desc="Tokenizing for coherence")]
    topic_diversity = compute_topic_diversity(topic_model, effective_config.top_n_words)
    umass_coherence = compute_umass_coherence(topic_model, tokenized_docs, effective_config.top_n_words)

    valid_topic_sizes = topic_info.loc[topic_info["Topic"] != -1, "Count"]
    outlier_share = float((doc_info["topic"] == -1).mean())
    metrics = {
        "created_at": now_iso(),
        "preset": effective_config.preset,
        "encoder_name": effective_config.encoder_name,
        "effective_nr_topics": effective_config.nr_topics,
        "n_documents": int(len(doc_info)),
        "n_topics_excluding_outliers": int((topic_info["Topic"] != -1).sum()),
        "outlier_share": outlier_share,
        "median_topic_size": float(valid_topic_sizes.median()) if not valid_topic_sizes.empty else None,
        "largest_topic_size": int(valid_topic_sizes.max()) if not valid_topic_sizes.empty else None,
        "topic_diversity_top_n": topic_diversity,
        "umass_coherence_top_n": umass_coherence,
        "runtime_seconds": elapsed_seconds,
    }
    write_json(run_dir / "metrics.json", metrics)

    representative_docs = topic_model.get_representative_docs()
    write_json(run_dir / "representative_docs.json", representative_docs)

    if (topic_info["Topic"] != -1).any():
        try:
            save_plot(
                topic_model.visualize_barchart(top_n_topics=16, n_words=effective_config.top_n_words),
                run_dir / "visualizations" / "topic_barchart.html",
            )
        except Exception as exc:  # pragma: no cover - plotting can fail in some environments.
            print(f"Skipping topic_barchart.html: {exc}")

    try:
        viz_sample_size = min(effective_config.documents_viz_sample_size, len(doc_info))
        viz_sample = doc_info.sample(viz_sample_size, random_state=effective_config.seed).sort_index()
        save_documents_scatter(viz_sample, run_dir / "visualizations" / "documents_2d.html")
    except Exception as exc:  # pragma: no cover
        print(f"Skipping documents_2d.html: {exc}")

    selected_doc_indices = pick_selected_doc_indices(doc_info, effective_config.distribution_doc_count)
    selected_docs_payload = []
    if isinstance(probabilities, np.ndarray) and probabilities.ndim == 2:
        for index in selected_doc_indices:
            row = doc_info.iloc[index]
            selected_docs_payload.append(
                {
                    "index": int(index),
                    "title": row["title"],
                    "topic": int(row["topic"]),
                    "probability": row["probability"],
                    "document_preview": row["document"][:1000],
                }
            )
            try:
                save_plot(
                    topic_model.visualize_distribution(
                        probabilities[index],
                        min_probability=effective_config.min_probability_to_show,
                    ),
                    run_dir / "visualizations" / f"distribution_doc_{index}.html",
                )
            except Exception as exc:  # pragma: no cover
                print(f"Skipping distribution_doc_{index}.html: {exc}")
    write_json(run_dir / "selected_documents.json", selected_docs_payload)

    print(f"Saved run artifacts to {run_dir}")


def compare_runs(args: argparse.Namespace) -> None:
    root_dir = Path(args.root_dir)
    rows: list[dict[str, Any]] = []
    for metrics_path in sorted(root_dir.glob("*/metrics.json")):
        run_dir = metrics_path.parent
        config_path = run_dir / "run_config.json"
        if not config_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        config = json.loads(config_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "run_name": run_dir.name,
                "preset": config.get("preset"),
                "encoder_name": metrics.get("encoder_name"),
                "n_documents": metrics.get("n_documents"),
                "n_topics": metrics.get("n_topics_excluding_outliers"),
                "outlier_share": metrics.get("outlier_share"),
                "topic_diversity": metrics.get("topic_diversity_top_n"),
                "umass_coherence": metrics.get("umass_coherence_top_n"),
                "runtime_seconds": metrics.get("runtime_seconds"),
                "practical_candidate": bool((metrics.get("n_topics_excluding_outliers") or 0) >= 5),
            }
        )

    if not rows:
        raise SystemExit(f"No metrics.json files found under {root_dir}")

    frame = pd.DataFrame(rows).sort_values(
        ["practical_candidate", "umass_coherence", "topic_diversity", "outlier_share"],
        ascending=[False, False, False, True],
        na_position="last",
    )
    frame.to_csv(root_dir / "run_comparison.csv", index=False)

    lines = [
        "# HW4 BERTopic Runs",
        "",
        frame.to_markdown(index=False),
    ]
    output_path = Path(args.output_path)
    ensure_dir(output_path.parent)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved comparison report to {output_path}")


def main() -> None:
    args = parse_args()
    if args.command == "inspect-dataset":
        inspect_dataset(args)
        return
    if args.command == "fit":
        fit_model(args)
        return
    if args.command == "compare-runs":
        compare_runs(args)
        return
    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
