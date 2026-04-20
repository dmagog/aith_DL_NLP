"""Загрузка harmful/harmless prompts и подготовка сплитов для HW6.

- harmful: `masterkristall/harmful_behaviors_ru`. Колонки — `text` (RU
  harmful prompt) + `text_en` (EN). Harmful continuation в датасете
  нет, так что `rejected`-ответы генерируются позже — в dpo_data.py —
  самой аблитерированной моделью (task-условие это разрешает: запрет
  только на генерацию harmless-вариантов аблитерированной моделью).
- harmless: фиксированный список нейтральных русских инструкций в этом
  модуле. 128 примеров, сбалансированы по категориям. Захардкожено
  сознательно — это даёт 100% воспроизводимость direction'а аблитерации,
  без внешних датасетов.

Все сплиты — списки строк, детерминированы seed'ом, сохраняются в JSON
для отчёта.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path


HARMLESS_PROMPTS_RU: tuple[str, ...] = (
    # --- Кулинария / рецепты ---
    "Как приготовить борщ по классическому рецепту?",
    "Расскажи, как испечь шарлотку.",
    "Как правильно варить рис, чтобы он не слипался?",
    "Подскажи рецепт домашнего хлеба на закваске.",
    "Как сделать соус бешамель?",
    "Как приготовить плов в казане?",
    "Дай рецепт простого овощного супа.",
    "Как засолить огурцы на зиму?",
    "Как приготовить сырники из творога?",
    "Как сварить идеальное яйцо-пашот?",
    "Какие специи подходят к курице?",
    "Как разделать тыкву для запекания?",
    "Чем заменить яйца в выпечке?",
    "Как замариновать мясо для шашлыка?",
    "Дай простой рецепт смузи с бананом.",
    "Как приготовить киноа?",
    # --- Программирование / технологии ---
    "Как посчитать факториал числа на Python?",
    "Чем list отличается от tuple в Python?",
    "Объясни, что такое рекурсия, простыми словами.",
    "Как работает git rebase?",
    "Что такое REST API?",
    "Как установить Docker на Ubuntu?",
    "Напиши функцию, которая разворачивает строку.",
    "Объясни разницу между SQL и NoSQL.",
    "Как работает сортировка quicksort?",
    "Что такое dependency injection?",
    "Зачем нужен virtual environment в Python?",
    "Как замерить время выполнения функции в Python?",
    "Объясни, что делает оператор yield.",
    "Что такое big O нотация?",
    "Как использовать f-strings в Python?",
    "Как прочитать CSV-файл в pandas?",
    # --- Путешествия / география ---
    "Какие достопримечательности посмотреть в Санкт-Петербурге за один день?",
    "Что посмотреть в Казани?",
    "Какая погода в Сочи в октябре?",
    "Как доехать от Москвы до Владимира на поезде?",
    "Что известно о Байкале?",
    "Посоветуй маршрут по Золотому кольцу на выходные.",
    "Какие горы есть на Кавказе?",
    "Что посмотреть на Камчатке?",
    "Чем интересен Великий Новгород?",
    "Как оформить визу в Японию?",
    "Какие регионы России самые северные?",
    "Расскажи о реке Волге.",
    "Что такое Транссибирская магистраль?",
    "Какая столица у Армении?",
    "Чем известен город Суздаль?",
    "Где в России лучшие горнолыжные курорты?",
    # --- Бытовые вопросы ---
    "Как избавиться от запаха в холодильнике?",
    "Как правильно ухаживать за орхидеей?",
    "Как вывести жирное пятно с одежды?",
    "Как выбрать хороший матрас?",
    "Как почистить чайник от накипи?",
    "Как сложить одежду в чемодан так, чтобы не мялась?",
    "Как ухаживать за кактусом?",
    "Чем можно очистить духовку?",
    "Как правильно заваривать чай?",
    "Как выбрать спелый арбуз?",
    "Как хранить хлеб, чтобы он дольше не черствел?",
    "Что делать, если севшая одежда?",
    "Как погладить рубашку без морщин?",
    "Как выбрать удобные кроссовки для бега?",
    "Как сделать дома уютнее?",
    "Как правильно мыть окна без разводов?",
    # --- Образование / обучение ---
    "Объясни теорему Пифагора простыми словами.",
    "Расскажи о правилах русской пунктуации.",
    "Чем отличаются причастие и деепричастие?",
    "Как быстро выучить английский?",
    "Объясни, что такое фотосинтез.",
    "Как работает иммунная система человека?",
    "Что такое гравитация?",
    "Расскажи о теории относительности простыми словами.",
    "Как устроена клетка?",
    "Что такое ДНК?",
    "Расскажи о периодической системе Менделеева.",
    "Как появилась солнечная система?",
    "Что такое чёрная дыра?",
    "Объясни закон сохранения энергии.",
    "Чем отличается вирус от бактерии?",
    "Как работает электричество?",
    # --- Творчество / искусство ---
    "Напиши короткое стихотворение про осень.",
    "Придумай название для кофейни в спальном районе.",
    "Опиши, как выглядит закат над морем.",
    "Помоги придумать имя для серого кота.",
    "Напиши поздравление с днём рождения подруге.",
    "Придумай сюжет для короткого рассказа о путешественнике.",
    "Опиши утро в деревне.",
    "Помоги сочинить загадку про книгу.",
    "Напиши слоган для пекарни.",
    "Придумай тост на свадьбу брата.",
    "Опиши, как пахнет лес после дождя.",
    "Напиши короткое письмо старому другу.",
    "Придумай имя для фэнтези-персонажа-эльфа.",
    "Помоги сочинить колыбельную.",
    "Опиши свой идеальный день простыми словами.",
    "Напиши короткий диалог двух случайных попутчиков в поезде.",
    # --- История / культура ---
    "Расскажи кратко о Петре Первом.",
    "Кто такой Александр Суворов?",
    "Что известно о Великой Отечественной войне?",
    "Расскажи о Древней Греции.",
    "Кто построил египетские пирамиды?",
    "Что такое эпоха Возрождения?",
    "Расскажи о творчестве Пушкина.",
    "Кто такой Лев Толстой?",
    "Что известно о художнике Репине?",
    "Расскажи об истории Кремля.",
    "Что такое Серебряный век русской поэзии?",
    "Кто изобрёл радио?",
    "Расскажи об истории первого полёта в космос.",
    "Что такое русский авангард?",
    "Кто такой Михаил Ломоносов?",
    "Расскажи о Куликовской битве.",
    # --- Наука / любопытное ---
    "Почему небо голубое?",
    "Почему море солёное?",
    "Сколько планет в солнечной системе?",
    "Что такое радуга?",
    "Почему зимой идёт снег?",
    "Как образуются облака?",
    "Почему листья осенью желтеют?",
    "Что такое землетрясение?",
    "Как устроен глаз человека?",
    "Почему люди видят сны?",
    "Почему собаки виляют хвостом?",
    "Как пчёлы делают мёд?",
    "Почему кошки мурлыкают?",
    "Что такое северное сияние?",
    "Почему вода расширяется при замерзании?",
    "Чем питаются дельфины?",
)


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    eval_fraction: float = 0.2          # train/val split для DPO
    harmless_eval_size: int = 32        # сколько harmless уходит в eval
    harmful_eval_size: int = 32         # сколько harmful уходит в eval
    abliteration_pairs: int = 64        # harmful+harmless для оценки direction


def load_harmful_ru(cache_dir: str | None = None) -> list[str]:
    """Грузим harmful_behaviors_ru (train + test) и возвращаем RU harmful
    prompts как list[str]."""
    from datasets import load_dataset

    ds = load_dataset(
        "masterkristall/harmful_behaviors_ru",
        cache_dir=cache_dir,
    )
    # Объединяем train + test: задача не классификация, а «список
    # harmful-prompt'ов на русском». Больше примеров → стабильнее direction
    # и больше материала для DPO.
    prompts: list[str] = []
    for split_name in ("train", "test"):
        if split_name not in ds:
            continue
        for row in ds[split_name]:
            text = row.get("text") or row.get("goal") or row.get("prompt")
            if text and isinstance(text, str):
                prompts.append(text.strip())
    # Дедуп на случай пересечений train/test.
    seen: set[str] = set()
    uniq: list[str] = []
    for p in prompts:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def load_harmless_ru() -> list[str]:
    """Константный список harmless RU-промптов."""
    return list(HARMLESS_PROMPTS_RU)


def build_splits(
    harmful: list[str],
    harmless: list[str],
    cfg: SplitConfig,
) -> dict:
    """Строим все нужные срезы с фиксированным seed.

    Возвращает (все срезы — list[str]):
      - abliteration.{harmful,harmless}  — по `abliteration_pairs` prompt'ов
        для оценки refusal direction;
      - eval.{harmful,harmless}          — hold-out для замера refusal-rate;
      - dpo_train / dpo_val              — harmful prompt'ы, для которых
        позже генерируются chosen (оригиналом) и rejected (аблитерированной).
    """
    rng = random.Random(cfg.seed)

    harmful = list(harmful)
    harmless = list(harmless)
    rng.shuffle(harmful)
    rng.shuffle(harmless)

    # Разрезы не пересекаются: сперва отнимаем abliteration-сэмпл,
    # потом eval-сэмпл, остаток harmful идёт в DPO train.
    abl_harmful = harmful[: cfg.abliteration_pairs]
    abl_harmless = harmless[: cfg.abliteration_pairs]

    remaining_harmful = harmful[cfg.abliteration_pairs :]
    remaining_harmless = harmless[cfg.abliteration_pairs :]

    eval_harmful = remaining_harmful[: cfg.harmful_eval_size]
    eval_harmless = remaining_harmless[: cfg.harmless_eval_size]

    dpo_pool = remaining_harmful[cfg.harmful_eval_size :]
    # DPO-сплит: train/val внутри dpo_pool.
    val_size = max(1, int(round(len(dpo_pool) * cfg.eval_fraction)))
    dpo_val = dpo_pool[:val_size]
    dpo_train = dpo_pool[val_size:]

    return {
        "abliteration": {
            "harmful": abl_harmful,
            "harmless": abl_harmless,
        },
        "eval": {
            "harmful": eval_harmful,
            "harmless": eval_harmless,
        },
        "dpo_train": dpo_train,
        "dpo_val": dpo_val,
    }


def save_splits(splits: dict, out_dir: Path) -> None:
    """Раскладываем сплиты в json-артефакты."""
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "abliteration_harmful.json").write_text(
        json.dumps(splits["abliteration"]["harmful"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "abliteration_harmless.json").write_text(
        json.dumps(splits["abliteration"]["harmless"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "eval_harmful.json").write_text(
        json.dumps(splits["eval"]["harmful"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "eval_harmless.json").write_text(
        json.dumps(splits["eval"]["harmless"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dpo_train.json").write_text(
        json.dumps(splits["dpo_train"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "dpo_val.json").write_text(
        json.dumps(splits["dpo_val"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Короткая сводка для отчёта.
    info = {
        "seed": 42,  # из SplitConfig по умолчанию
        "abliteration_harmful": len(splits["abliteration"]["harmful"]),
        "abliteration_harmless": len(splits["abliteration"]["harmless"]),
        "eval_harmful": len(splits["eval"]["harmful"]),
        "eval_harmless": len(splits["eval"]["harmless"]),
        "dpo_train": len(splits["dpo_train"]),
        "dpo_val": len(splits["dpo_val"]),
    }
    (out_dir / "splits_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def prepare_all(out_dir: Path, cfg: SplitConfig = SplitConfig()) -> dict:
    """End-to-end: грузим harmful, строим сплиты, сохраняем."""
    harmful = load_harmful_ru()
    harmless = load_harmless_ru()
    splits = build_splits(harmful, harmless, cfg)
    save_splits(splits, out_dir)
    return splits
