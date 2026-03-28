# ITMO AITH: DL+NLP HW3

Независимое решение домашней работы 3 по русскоязычному NER на `factRuEval-2016`.

Практический план экспериментов в этом репозитории такой:

1. Проверить и стабилизировать baseline на небольшом русскоязычном encoder'е.
2. Перенести ту же конфигурацию на более сильный checkpoint, если время обучения на текущей машине приемлемо.
3. Добавить `MLM -> NER` и сравнить `random`, `whole_word`, `entity` masking.
4. Добавить pseudo-labeling на внешнем русскоязычном корпусе и сравнить с gold-only baseline.

Такой порядок выбран потому, что текущая среда работает без CUDA/MPS, а значит сначала нужно получить корректный и воспроизводимый baseline, а уже потом запускать более дорогие по времени эксперименты.

Репозиторий содержит воспроизводимый пайплайн для трех сценариев:

- baseline fine-tuning энкодера на NER;
- дополнительное MLM-дообучение (`random`, `whole_word`, `entity`) перед NER;
- pseudo-labeling на внешнем русскоязычном корпусе и финальное дообучение на gold-разметке.

## Текущие результаты

Актуальная сводка лежит в `artifacts_hw3/run_summary.md`. На текущей `cpu-only` конфигурации лучшие test F1 получились такими:

- `baseline_rubert_tiny2_e1`: `0.7367`
- `ner_from_mlm_rubert_tiny2_whole_word_e1`: `0.7487`
- `ner_from_mlm_rubert_tiny2_entity_e1`: `0.7721`
- `ner_with_synthetic_titles_10k`: `0.8251`
- `ner_from_entity_mlm_with_synthetic_titles_10k`: `0.8328`

То есть на текущих прогонах лучший результат даёт комбинация `entity MLM + synthetic headlines`.

## Структура

- `src/hw3_pipeline.py` - CLI для всех этапов экспериментов;
- `src/hw3/data.py` - загрузка датасетов и подготовка признаков;
- `src/hw3/training.py` - обучение и оценка NER;
- `src/hw3/mlm.py` - коллаторы и запуск MLM;
- `src/hw3/pseudo_label.py` - генерация синтетической разметки teacher-моделью;
- `src/hw3/reporting.py` - сводка результатов нескольких запусков;
- `artifacts_hw3/<run_name>/...` - локальные артефакты экспериментов.

## Установка

```bash
python3 -m pip install -r requirements.txt
```

## Базовый прогон

```bash
python3 src/hw3_pipeline.py train-ner \
  --model-name-or-path DeepPavlov/rubert-base-cased \
  --dataset-name gusevski/factrueval2016 \
  --output-dir artifacts_hw3/baseline_rubert \
  --num-train-epochs 5 \
  --learning-rate 3e-5 \
  --seed 42
```

Команда:

- загружает `factRuEval`;
- разворачивает нестандартный вложенный формат `data -> [{tokens, ner_tags, ...}]` в обычные split'ы;
- находит колонки с токенами и метками;
- считает качество модели до обучения;
- дообучает `AutoModelForTokenClassification`;
- сохраняет метрики и предсказания после обучения.

Для быстрой отладки поддерживаются лимиты:

```bash
--max-train-samples 256 --max-validation-samples 256 --max-test-samples 256
```

## MLM -> NER

1. Дообучить encoder в MLM-режиме:

```bash
python3 src/hw3_pipeline.py train-mlm \
  --model-name-or-path DeepPavlov/rubert-base-cased \
  --dataset-name gusevski/factrueval2016 \
  --output-dir artifacts_hw3/mlm_whole_word \
  --masking-strategy whole_word \
  --num-train-epochs 3 \
  --seed 42
```

2. Использовать полученный чекпойнт как инициализацию для NER:

```bash
python3 src/hw3_pipeline.py train-ner \
  --model-name-or-path artifacts_hw3/mlm_whole_word/checkpoint-best \
  --dataset-name gusevski/factrueval2016 \
  --output-dir artifacts_hw3/ner_from_mlm \
  --num-train-epochs 5 \
  --seed 42
```

`masking-strategy` поддерживает:

- `random` - стандартное случайное маскирование;
- `whole_word` - маскирование целых слов;
- `entity` - маскирование токенов, входящих в размеченные сущности.

## Pseudo-labeling

```bash
python3 src/hw3_pipeline.py pseudo-label \
  --teacher-model r1char9/ner-rubert-tiny-news \
  --source-dataset-name data-silence/lenta.ru_2-extended \
  --source-split 'train[:10000]' \
  --text-column title \
  --max-samples 10000 \
  --batch-size 16 \
  --output-path artifacts_hw3/pseudo_lenta_titles_10k.jsonl
```

В этой конфигурации teacher-модель работает по заголовкам `Lenta.ru`. Это осознанный компромисс для `cpu-only` среды: у заголовков высокая плотность сущностей, а время разметки 10k текстов остаётся разумным.

Для сдачи в git включены только лёгкие артефакты:

- итоговый ноутбук;
- исходный код пайплайна;
- `requirements.txt`;
- сводка `run_summary.md`;
- `metrics.json` и `run_config.json` для ключевых запусков;
- synthetic-корпус `pseudo_lenta_titles_10k.jsonl`.

Тяжёлые чекпойнты моделей и служебные файлы `Trainer` в репозиторий не добавляются.

Затем synthetic-набор можно смешать с gold-данными:

```bash
python3 src/hw3_pipeline.py train-ner \
  --model-name-or-path cointegrated/rubert-tiny2 \
  --dataset-name gusevski/factrueval2016 \
  --synthetic-path artifacts_hw3/pseudo_lenta_titles_10k.jsonl \
  --output-dir artifacts_hw3/ner_with_synthetic \
  --seed 42
```

## Воспроизводимость

- используйте фиксированный `--seed 42`;
- все промежуточные метрики сохраняются в JSON;
- все выходы Trainer пишутся в отдельный `output-dir`.

## Сравнение запусков

```bash
python3 src/hw3_pipeline.py compare-runs \
  --root-dir artifacts_hw3 \
  --output-path artifacts_hw3/run_summary.md
```
