"""Подготовка обучающей выборки из корпуса Lenta.ru."""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
from corus import load_lenta2

EXCLUDED_TOPICS = {"Библиотека"}
MIN_TEXT_CHARS = 300
MIN_TITLE_CHARS = 10
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class SampleConfig:
    sample_size: int = 12000
    eval_size: int = 500
    seed: int = 42
    text_char_limit: int = 600


def _normalize(text: str) -> str:
    return WHITESPACE_RE.sub(" ", text or "").strip()


def _iter_records(path: Path) -> Iterable[dict]:
    for rec in load_lenta2(str(path)):
        title = _normalize(rec.title)
        text = _normalize(rec.text)
        topic = (rec.topic or "").strip()
        if topic in EXCLUDED_TOPICS:
            continue
        if len(title) < MIN_TITLE_CHARS or len(text) < MIN_TEXT_CHARS:
            continue
        yield {"title": title, "text": text, "topic": topic, "url": rec.url}


def reservoir_sample(path: Path, sample_size: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    reservoir: list[dict] = []
    for i, rec in enumerate(_iter_records(path)):
        if i < sample_size:
            reservoir.append(rec)
        else:
            j = rng.randint(0, i)
            if j < sample_size:
                reservoir[j] = rec
    rng.shuffle(reservoir)
    return reservoir


def build_dataframe(records: list[dict], text_char_limit: int) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    df["text"] = df["text"].str.slice(0, text_char_limit)
    return df


def format_training_text(title: str, text: str) -> str:
    """Единый формат и для обучения, и для inference-промпта корзинки."""
    return f"Заголовок: {title}\n\nТекст: {text}"


def split_train_eval(df: pd.DataFrame, eval_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    eval_df = shuffled.iloc[:eval_size].reset_index(drop=True)
    train_df = shuffled.iloc[eval_size:].reset_index(drop=True)
    return train_df, eval_df


def prepare_lenta_sample(corpus_path: Path, cfg: SampleConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = reservoir_sample(corpus_path, cfg.sample_size, cfg.seed)
    df = build_dataframe(records, cfg.text_char_limit)
    return split_train_eval(df, cfg.eval_size, cfg.seed)


def save_sample(train_df: pd.DataFrame, eval_df: pd.DataFrame, out_dir: Path, cfg: SampleConfig) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    eval_df.to_csv(out_dir / "eval.csv", index=False)
    info = {
        "sample_size": cfg.sample_size,
        "eval_size": cfg.eval_size,
        "train_rows": len(train_df),
        "eval_rows": len(eval_df),
        "seed": cfg.seed,
        "text_char_limit": cfg.text_char_limit,
        "excluded_topics": sorted(EXCLUDED_TOPICS),
        "min_text_chars": MIN_TEXT_CHARS,
        "min_title_chars": MIN_TITLE_CHARS,
        "topics_in_train": train_df["topic"].value_counts().to_dict(),
    }
    (out_dir / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )
