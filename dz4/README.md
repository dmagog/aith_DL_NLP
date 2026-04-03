# ITMO AITH: DL+NLP HW4

Решение домашней работы 4 по тематическому моделированию новостей `Lenta.ru` с
использованием `Corus` и `BERTopic`.

## Что реализовано

- загрузка корпуса через `corus.load_lenta2()` из локального `bz2`-архива;
- потоковая подготовка корпуса с reservoir sampling, чтобы не держать весь датасет в памяти;
- лёгкая предобработка под `BERTopic`: очистка whitespace/URL, отбрасывание слишком коротких документов, исключение архивного топика `Библиотека`, ограничение длины текста для CPU-only среды;
- несколько пресетов пайплайна для абляций:
  - `baseline_multilingual`;
  - `lemmatized_multilingual`;
  - `english_ablation`;
- настраиваемые компоненты `BERTopic`: encoder, `UMAP`, `HDBSCAN`, `CountVectorizer`, `ClassTfidfTransformer`, `KeyBERTInspired`/`MMR`;
- автоматический экспорт:
  - `topic_info.csv`;
  - `document_topics.csv`;
  - `topic_keywords.json`;
  - `metrics.json`;
  - HTML-визуализаций.
- отдельные сравнительные и итоговые артефакты в `artifacts_hw4/`, включая
  формальную сводку `run_summary.md` и итоговый аналитический отчёт
  `final_report.md`.

## Итоговый результат

На текущей `cpu-only` конфигурации наиболее полезным по совокупности метрик и
интерпретируемости оказался прогон `baseline_multilingual_3k`:

- `10` тем без учёта outliers;
- `Topic Diversity@10 = 0.91`;
- `UMass Coherence@10 = -3.2434`;
- содержательные темы по экономике, судам, спорту, внешней политике, big tech.

Более крупные `6k` full-text прогоны улучшали coherence численно, но
схлопывались в `2` сверхкрупные темы. Это зафиксировано как важное ограничение
постановки и использовано в итоговом выборе модели.

## Почему такой дизайн

Для русскоязычного корпуса здесь выбран базовый рабочий путь:

1. `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` как encoder.
   Он уже sentence-level, нормально работает на русском и заметно дешевле по CPU,
   чем полный `ruBERT`. Для sanity-check оставлен `english_ablation`.
2. `UMAP` вместо `PCA/t-SNE` для кластеризации.
   В `BERTopic` это де-факто стандарт: он лучше сохраняет локальную структуру
   соседства и даёт удобное пространство для `HDBSCAN`.
3. `HDBSCAN` вместо `KMeans`.
   Число тем заранее неизвестно, а новостной корпус содержит шум и выбросы. Для
   такой постановки полезно уметь оставлять часть документов в `-1`.
4. Лемматизация используется только на шаге `CountVectorizer/c-TF-IDF`, а не в
   encoder.
   Для семантических эмбеддингов агрессивная нормализация обычно не нужна, но
   для топ-токенов она уменьшает морфологическую разреженность.

## Структура

- `src/hw4_pipeline.py` - CLI-пайплайн;
- `hw4.ipynb` - ноутбук, который читает артефакты и собирает отчёт;
- `requirements.txt` - зависимости;
- `artifacts_hw4/<run_name>/...` - локальные артефакты прогонов.

## Установка

Запускать команды ниже предполагается из каталога `dz4/`.

```bash
python3 -m pip install -r requirements.txt
```

## Быстрый осмотр данных

```bash
python3 src/hw4_pipeline.py inspect-dataset \
  --output-dir artifacts_hw4/dataset_inspect \
  --sample-size 5000 \
  --max-records 50000 \
  --seed 42
```

Эта команда:

- читает `../data/lenta-ru-news.csv.bz2` через `Corus`;
- применяет те же фильтры, что и основной пайплайн;
- сохраняет `dataset_info.json`, `dataset_preview.txt` и небольшой CSV-сэмпл.

Для честного сравнения нескольких конфигураций удобно сначала один раз
подготовить фиксированный sample, а потом передавать его через
`--prepared-corpus-path`.

Пример:

```bash
python3 src/hw4_pipeline.py inspect-dataset \
  --output-dir artifacts_hw4/compare_sample_3k \
  --sample-size 3000 \
  --seed 42
```

## Отчётный прогон

Рекомендуемый пресет для основного эксперимента:

```bash
python3 src/hw4_pipeline.py fit \
  --preset lemmatized_multilingual \
  --output-dir artifacts_hw4/lemmatized_multilingual_15k \
  --sample-size 15000 \
  --seed 42
```

Для воспроизводимого сравнения на одном и том же sample:

```bash
python3 src/hw4_pipeline.py fit \
  --preset baseline_multilingual \
  --prepared-corpus-path artifacts_hw4/compare_sample_3k/prepared_corpus_sample.csv \
  --output-dir artifacts_hw4/baseline_multilingual_3k \
  --sample-size 3000 \
  --hdbscan-min-cluster-size 80 \
  --hdbscan-min-samples 15 \
  --seed 42
```

Для быстрой абляции на уменьшенной подвыборке можно менять `--preset`:

```bash
python3 src/hw4_pipeline.py fit \
  --preset baseline_multilingual \
  --output-dir artifacts_hw4/baseline_multilingual_8k \
  --sample-size 8000 \
  --seed 42
```

```bash
python3 src/hw4_pipeline.py fit \
  --preset english_ablation \
  --output-dir artifacts_hw4/english_ablation_8k \
  --sample-size 8000 \
  --seed 42
```

## Сравнение прогонов

```bash
python3 src/hw4_pipeline.py compare-runs \
  --root-dir artifacts_hw4 \
  --output-path artifacts_hw4/run_summary.md
```

## Визуализации

После `fit` в `artifacts_hw4/<run_name>/visualizations/` сохраняются:

- `topic_barchart.html` - топ-токены по темам;
- `documents_2d.html` - документы и их темы в 2D;
- `distribution_doc_<idx>.html` - распределение тем по выборочным документам.

## Ноутбук

Ноутбук [hw4.ipynb](/Users/georgijmamarin/Desktop/Oplimp/dl_nlp_course/dz_1/dz4/hw4.ipynb)
не переобучает модель, а читает артефакты из `artifacts_hw4/`.

Проверка воспроизводимости:

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace hw4.ipynb
```

Дополнительный итоговый текстовый отчёт:

- [artifacts_hw4/final_report.md](/Users/georgijmamarin/Desktop/Oplimp/dl_nlp_course/dz_1/dz4/artifacts_hw4/final_report.md)
