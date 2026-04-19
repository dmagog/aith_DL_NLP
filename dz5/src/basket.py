"""Тестовая корзинка промптов и сериализация генераций в markdown."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# Заголовки в стиле Lenta.ru, покрывающие разные тематики;
# взяты как нейтральные синтетические формулировки, чтобы не зависеть
# от конкретных строк корпуса.
BASKET_TITLES: list[str] = [
    "Минфин рассказал о новых правилах налогообложения для самозанятых",
    "Сборная России по хоккею назвала состав на ближайший турнир",
    "В Москве открылась выставка работ современных уральских художников",
    "Учёные сообщили о новой методике ранней диагностики диабета",
    "Эксперты оценили влияние ключевой ставки на ипотечный рынок",
    "Жители Калининграда пожаловались на резкое подорожание проезда",
    "В аэропорту Домодедово задержали несколько международных рейсов",
    "Госдума рассмотрела законопроект о цифровых сервисах для пенсионеров",
    "Российские разработчики представили новый сервис распознавания речи",
    "В Якутии впервые за десятилетие зафиксировали редкий природный феномен",
]


@dataclass(frozen=True)
class GenerationConfig:
    max_new_tokens: int = 160
    do_sample: bool = True
    temperature: float = 0.8
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    seed: int = 42


def format_prompt(title: str) -> str:
    return f"Заголовок: {title}\n\nТекст:"


def run_basket(
    model: "PreTrainedModel",
    tokenizer: "PreTrainedTokenizerBase",
    titles: list[str] = BASKET_TITLES,
    gen_cfg: GenerationConfig = GenerationConfig(),
) -> list[dict]:
    import torch

    torch.manual_seed(gen_cfg.seed)
    model.eval()
    device = next(model.parameters()).device

    items: list[dict] = []
    for title in titles:
        prompt = format_prompt(title)
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=gen_cfg.max_new_tokens,
                do_sample=gen_cfg.do_sample,
                temperature=gen_cfg.temperature,
                top_p=gen_cfg.top_p,
                repetition_penalty=gen_cfg.repetition_penalty,
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded = tokenizer.decode(out[0], skip_special_tokens=True)
        generation = decoded[len(prompt):].strip()
        items.append({"title": title, "prompt": prompt, "generation": generation})
    return items


def save_basket_md(items: list[dict], out_path: Path, header: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    blocks = [f"# {header}\n"]
    for i, item in enumerate(items, start=1):
        blocks.append(f"## {i}. {item['title']}\n")
        blocks.append(item["generation"].strip() + "\n")
    out_path.write_text("\n".join(blocks), encoding="utf-8")
