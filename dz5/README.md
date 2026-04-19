# ITMO AITH: DL+NLP HW5

Решение домашней работы 5 — QLoRA-дообучение `Qwen/Qwen2.5-1.5B` на
новостном корпусе `Lenta.ru` с профилированием `torch.profiler` и
замером до/после на нескольких русскоязычных метриках.

## Что реализовано

- подготовка сэмпла из `Lenta.Ru-News-Dataset` (потоковый `corus.load_lenta2`,
  reservoir sampling с фиксированным seed, фильтры по длине и топику);
- 4-bit QLoRA-дообучение через `bitsandbytes` + `peft.LoraConfig` на LoRA-рангах
  `r=16, α=32` по всем линейным слоям attention и MLP;
- baseline/post-train замер на одном и том же сплите:
  - perplexity на 200 документах eval;
  - `lm-evaluation-harness` на `xnli_ru` и `xwinograd_ru`;
  - кастомный basket из 10 новостных заголовков для качественного сравнения генерации;
- отдельный короткий прогон под `torch.profiler` (CPU+CUDA, `record_shapes`,
  `profile_memory`) с сохранением `profiler_summary.md` и trace-файлов;
- end-to-end раннер `src/run_full.py` со всеми стадиями, skip-флагами и
  auto-resume c последнего чекпойнта HF Trainer;
- Colab-ноутбук `colab_train.ipynb`, который запускает полный прогон на
  бесплатном T4 и упаковывает артефакты в zip;
- отчётный `hw5.ipynb`, который читает зафиксированные артефакты и не
  требует GPU.

## Итоговый результат

| Метрика | before | after | Δ |
|---|---|---|---|
| Perplexity (200 док. eval) | 6.378 | 4.861 | **−23.8%** |
| `xnli_ru` accuracy (limit=200) | 0.445 | 0.465 | +0.020 (±0.035) |
| `xwinograd_ru` accuracy (limit=200) | 0.630 | 0.620 | −0.010 (±0.034) |

Основной эффект QLoRA — доменная адаптация: в basket до обучения модель
постоянно сваливается с русских заголовков в английский перевод и
инструктивный стиль, после — уверенно генерит связные новостные абзацы в
стиле Lenta, сохраняя структуру «Заголовок → Текст». Harness-метрики
держатся в пределах stderr — catastrophic forgetting не случилось.

Подробный разбор, профайлер-таблицы и сравнение basket-ов — в
[hw5.ipynb](hw5.ipynb).

## Почему такой дизайн

1. **Qwen2.5-1.5B base**. Подходит под 0.5B–1.5B, умеет русский без
   инструкт-финетюна, влезает в 4-bit на T4 вместе с LoRA-градиентами.
2. **QLoRA 4-bit NF4 + double-quant, fp16 compute**. На T4 (sm75) `bf16`
   недоступен, поэтому compute dtype — `fp16`. NF4+double-quant экономит
   достаточно VRAM, чтобы остался запас под активации и оптимизатор.
3. **LoRA по всем линейным**. Таргеты `q/k/v/o_proj` + `gate/up/down_proj`.
   При `r=16` обучаемых параметров ~1.2% от модели — хватает на domain
   adaptation, при этом LoRA low-rank защищает общие способности.
4. **`paged_adamw_8bit` + gradient checkpointing**. Без этого комбо train
   не помещается в 16 ГБ VRAM при `batch=2, seq=384, grad-accum=4`.
5. **Полный прогон на Colab, ноутбук на артефактах**. Free-tier T4 иногда
   обрывает сессию. Поэтому:
   - профайлер — отдельная короткая стадия (его артефакт сохраняется ещё
     до запуска длинного train);
   - HF Trainer сохраняет чекпойнты каждые 100 шагов;
   - `run_full.py` автоматически подхватывает последний чекпойнт при
     повторном запуске.

## Структура

- `src/data.py` — sampling и подготовка сплитов из Lenta;
- `src/basket.py` — 10 контрольных заголовков и генератор;
- `src/evaluate.py` — обёртки над `lm-eval-harness` и perplexity;
- `src/train.py` — QLoRA-train и `torch.profiler`-стадия;
- `src/run_full.py` — end-to-end пайплайн;
- `colab_train.ipynb` — раннер для Colab T4;
- `hw5.ipynb` — отчётный ноутбук (читает артефакты);
- `artifacts_hw5/` — зафиксированные артефакты прогона (включая
  LoRA-адаптер ~74 МБ);
- `requirements.txt` — зависимости (`transformers`, `peft`, `bitsandbytes`,
  `lm-eval`, `corus`).

## Установка

Все команды из каталога `dz5/`.

Для **ноутбука-отчёта** GPU не нужен и `bitsandbytes` можно не ставить:

```bash
python3 -m pip install pandas matplotlib jupyter nbformat
```

Для **локального запуска `run_full.py`** нужна CUDA-машина:

```bash
python3 -m pip install -r requirements.txt
```

## Полный прогон (Colab T4)

Самый простой путь — открыть `colab_train.ipynb` в Colab, выбрать `Runtime
→ Change runtime type → T4 GPU` и выполнить ячейки сверху вниз. Ноутбук
клонирует ветку `hw_5`, ставит зависимости, качает `lenta-ru-news.csv.bz2`,
запускает `python -m src.run_full` и упаковывает `artifacts_hw5/` в zip на
скачивание. При обрыве сессии перезапуск автоматически пропустит уже
готовые стадии и продолжит с последнего чекпойнта train.

## Локальный запуск (CUDA-машина)

```bash
python3 -m src.run_full \
  --corpus-path lenta-ru-news.csv.bz2 \
  --out artifacts_hw5 \
  --sample-size 12000 \
  --eval-size 500 \
  --harness-limit 200 \
  --perplexity-samples 200 \
  --seed 42
```

Полезные флаги: `--skip-baseline`, `--skip-profile`, `--skip-train`,
`--skip-post` — пропустить отдельные стадии, если их артефакты уже на
диске. `--num-train-epochs` и `--max-seq-length` — override дефолтов из
`TrainConfig`.

## Ноутбук-отчёт

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace hw5.ipynb
```

Ноутбук читает `artifacts_hw5/*.json`, `artifacts_hw5/*.md` и рисует
сравнение before/after. GPU и `bitsandbytes` не требуются.
