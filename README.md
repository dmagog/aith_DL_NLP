# ITMO AITH: DL+NLP

Решение задачи классификации новостей Lenta.ru по топикам (`title + text -> topic`), включающее:
- репрезентативную стратифицированную выборку на 100k текстов;
- dummy-бейзлайн;
- `LogisticRegression + CountVectorizer`;
- `LogisticRegression + TfidfVectorizer`;
- подбор гиперпараметров на кросс-валидации;
- финальную оценку на test и анализ ошибок.

Основной финальный прогон для сдачи:
- `artifacts/run_v2`
- test `accuracy = 0.8135`
- test `macro-F1 = 0.7040`

Дополнительный прогон для сравнения:
- `artifacts/run_v8_fast`
- даёт немного лучшую `accuracy`, но худший `macro-F1`, поэтому не выбран как основной финальный результат

## Установка

```bash
python3 -m pip install -r requirements.txt
```

## Полный запуск пайплайна

```bash
python3 src/hw1_pipeline.py \
  --dataset-version v1.1 \
  --sample-size 100000 \
  --tuning-iters 20 \
  --tuning-sample-size 50000 \
  --cv-splits 3 \
  --seed 42 \
  --n-jobs 1
```

Если датасет уже скачан локально:

```bash
python3 src/hw1_pipeline.py \
  --dataset-path data/lenta-ru-news.csv.bz2 \
  --sample-size 100000 \
  --tuning-iters 20 \
  --tuning-sample-size 50000 \
  --cv-splits 3 \
  --seed 42 \
  --n-jobs 1
```

## Быстрый тюнинг из кэшированного сэмпла

Если уже подготовлен `artifacts/run_v2/sample_100k_processed.csv`, можно
пропустить загрузку датасета, семплирование и предобработку и запустить только
обучение и тюнинг:

```bash
python3 src/hw1_pipeline.py \
  --preprocessed-sample-path artifacts/run_v2/sample_100k_processed.csv \
  --sample-size 100000 \
  --skip-base-eval \
  --best-base-model count_logreg \
  --tuning-iters 6 \
  --tuning-sample-size 15000 \
  --cv-splits 2 \
  --n-jobs 1 \
  --seed 42 \
  --artifacts-dir artifacts/run_v8_fast
```

## Ноутбук

`hw1.ipynb` — это воспроизводимый отчётный ноутбук. Он не пересчитывает весь
пайплайн обучения заново, а читает подготовленные артефакты из
`artifacts/run_v2` и показывает итоговые метрики, анализ ошибок и выводы.

Проверка полного выполнения:

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace hw1.ipynb
```

## Структура проекта

- `src/hw1_pipeline.py` - основной пайплайн обучения и оценки
- `hw1.ipynb` - отчётный ноутбук по подтверждённому прогону
- `results_summary.md` - краткая сводка по выполненным прогонам
- `tasklist.md` - итеративный план и текущий статус
- `artifacts/<run_name>/metrics/*.json` - метрики конкретного прогона
- `artifacts/<run_name>/reports/error_analysis.md` - отчёт по анализу ошибок
- `artifacts/<run_name>/reports/misclassified_examples.csv` - примеры ошибочных предсказаний
- `artifacts/<run_name>/models/*.joblib` - сохранённые артефакты модели

