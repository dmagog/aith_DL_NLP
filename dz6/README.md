# ITMO AITH: DL+NLP HW6

Решение HW6 — аблитерация (Arditi et al. 2024) + QLoRA-DPO модели
`Qwen/Qwen2.5-1.5B-Instruct` на русскоязычных harmful-промптах из
`masterkristall/harmful_behaviors_ru`. Цель — сначала механически
«выключить» refusal-направление модификацией весов, потом дообучить
DPO-адаптер, чтобы модель сохранила общую полезность, но не возвращала
refusal на harmful-запросы.

## Что реализовано

- подготовка harmful + harmless-сплитов (abliteration / eval / DPO-train /
  DPO-val) с фиксированным seed, harmless — зашитый список нейтральных
  RU-инструкций;
- **abliteration**: снятие активаций residual stream на последнем токене,
  расчёт refusal-направления по слоям (`mean(harmful) − mean(harmless)`),
  ablation-подбор лучшего слоя, weight orthogonalization по
  `embed_tokens`, `o_proj`, `down_proj`;
- **DPO-данные**: self-generated пары, где `chosen` — ответ оригинальной
  `Qwen2.5-Instruct`, `rejected` — ответ аблитерированной модели;
- **QLoRA-DPO**: 4-bit NF4 + LoRA `r=16, α=32` на всех линейных,
  `beta=0.1`, `paged_adamw_8bit`, gradient checkpointing, effective
  batch = 4, lr=5e-5, 1 эпоха;
- 3 eval-прогона (pretrained / abliterated / dpo) — refusal-rate по
  regex-эвристике на harmful и harmless подмножествах, одинаковый
  `GenConfig` (greedy, 192 токенов) для честного сравнения;
- `src/run_full.py` — end-to-end с skip-флагами и идемпотентным resume'ом;
- `colab_train.ipynb` — раннер для бесплатного Colab T4;
- `hw6.ipynb` — отчётный ноутбук, читает зафиксированные артефакты.

## Итоговый результат

Seed=42, eval-split — 32 harmful + 32 harmless промптов, greedy
(`max_new_tokens=192`), refusal определяется regex-эвристикой по RU+EN
паттернам. Полный прогон — на домашнем RTX 2070 8 GB (Windows).

| Модель          | refusal_rate harmful | refusal_rate harmless |
| --------------- | -------------------- | --------------------- |
| Qwen pretrained | 0.906                | 0.000                 |
| + abliterated   | **0.031**            | 0.000                 |
| + DPO           | 0.938                | **0.719**             |

**Что работает.** Abliteration по Arditi et al. отрабатывает как обещано:
refusal на harmful падает с 90.6 % до 3.1 % при нулевом refusal на
harmless. Направление выбрано на слое 13 из 28 (Qwen2.5-1.5B имеет
28 слоёв, hidden_dim=1536). Weight orthogonalization применена к
`embed_tokens`, всем 28 `o_proj` и всем 28 `down_proj`.

**Что не работает как хотелось.** QLoRA-DPO (96 шагов, effective batch=4,
β=0.1, lr=5e-5, 339 train-пар) восстанавливает refusal на harmful
(до 93.8 %) — но ломает harmless: refusal-rate вырастает с 0 % до 71.9 %.
Модель обобщает до «отказывать на всё»: обучающие пары `chosen` =
оригинальный Qwen (обычно отказ), `rejected` = аблитерированный Qwen
(отвечает по существу) — все пришли из harmful-сигнала, harmless-пар в
датасете не было. Плюс ответы на harmful после DPO выглядят как
«Sorry, but I can't assist with that» — узко заученный паттерн,
перекрывающий и русскоязычные harmless-вопросы. `train_loss=0.064` на
96 шагах подтверждает переобучение на этом узком сигнале.

**Как чинить (за рамками ДЗ, но очевидно).** Добавить harmless-пары в
DPO-train с `chosen` = ответ по существу / `rejected` = отказ
(двусторонний сигнал). Либо KTO/IPO вместо чистого DPO. Либо
KL-регуляризация к оригиналу на harmless-сэмплах. Пайплайн идемпотентен —
можно перезапустить только DPO-стадии после расширения
`dpo_train_pairs.json`.

Подробный разбор (график подбора слоя, кривая DPO-лосса, примеры
генераций до/после) — в [hw6.ipynb](hw6.ipynb).

## Почему такой дизайн

1. **Qwen2.5-1.5B-Instruct база**. Маленькая, хорошо знает русский, уже
   refusal-trained — есть что ломать. На T4 (sm75) / RTX 2070 8GB
   умещается в 4-bit вместе с DPO-градиентами при batch=2.
2. **Abliteration по Arditi et al.**. Дёшевая (≈10 мин на T4) операция:
   активации собираются один раз, weight orth одноразово. После неё
   модель сохраняется как обычный HF-чекпойнт — без runtime-хуков.
3. **Regex-refusal вместо LLM-Judge**. ДЗ явно допускает «простой
   вариант», а regex-эвристика по RU+EN паттернам воспроизводима и не
   требует внешнего API. Паттерны зафиксированы в `src/evaluate.py`.
4. **Self-generated DPO-пары**. ДЗ допускает как открытый датасет, так и
   собственный. Self-generated надёжнее: `rejected` — реально плохой
   ответ именно *этой* аблитерированной модели, а не абстрактный harmful
   текст. Генерация в один проход, greedy, 128 токенов.
5. **QLoRA-DPO, а не full-parameter DPO**. Модель после DPO используется
   как аблитерированная база + LoRA-адаптер. LoRA сохраняется отдельно
   (`artifacts_hw6/dpo_adapter/`, ~70 МБ при r=16 на всех линейных).
6. **HF push опционален**. Аблитерированная модель детерминированно
   воспроизводима из `src/abliterate.py` при том же seed (`42`), DPO
   берёт её с диска, не с Hub. Реальный прогон идёт с `--skip-push`;
   выложить на Hub — отдельная ручная операция при наличии write-
   токена, весь остальной пайплайн от неё не зависит.
7. **Идемпотентные стадии**. Бесплатный Colab и SSH-сессии обрываются —
   каждая стадия проверяет свой метрик-файл и пропускается, если
   артефакт уже готов. Полный прогон можно перезапустить сколько
   угодно раз.

## Структура

- `src/data.py` — сэмплинг harmful + harmless, сплиты, сохранение в
  `artifacts_hw6/sample/`;
- `src/evaluate.py` — refusal regex, `GenConfig`, `run_eval`;
- `src/abliterate.py` — активации, направление, подбор слоя,
  weight orthogonalization;
- `src/dpo_data.py` — генерация `chosen`/`rejected` пар;
- `src/dpo_train.py` — QLoRA-DPO (TRL `DPOTrainer`);
- `src/run_full.py` — end-to-end;
- `colab_train.ipynb` — раннер для Colab T4;
- `hw6.ipynb` — отчётный ноутбук;
- `artifacts_hw6/` — зафиксированные артефакты прогона (метрики,
  log_history, eval-сэмплы, DPO-пары, LoRA-адаптер ~70 МБ;
  полные fp16-веса моделей в репо не лежат — abliterated
  детерминированно воспроизводится из `src/abliterate.py`);
- `requirements.txt` — зависимости.

## Установка

Из каталога `dz6/`.

**Только отчётный ноутбук** (GPU не нужен):

```bash
python3 -m pip install pandas matplotlib jupyter nbformat
```

**Полный прогон** (CUDA-машина):

```bash
python3 -m pip install -r requirements.txt
```

Верхние границы версий (`transformers<5`, `trl<1`, `datasets<4`,
`huggingface_hub<1`) важны — свежие мажоры ломают DPO-API, на котором
построен `src/dpo_train.py`.

## Полный прогон

### Colab T4

`Runtime → Change runtime type → T4 GPU`, открыть `colab_train.ipynb`,
выполнить сверху вниз. Ноутбук клонирует ветку, ставит зависимости,
запускает `python -m src.run_full`, упаковывает `artifacts_hw6/` в zip.
При обрыве сессии идемпотентный resume подхватит с последней готовой
стадии.

### Локально на своём GPU

```bash
python3 -m src.run_full \
    --out artifacts_hw6 \
    --model Qwen/Qwen2.5-1.5B-Instruct \
    --skip-push \
    --seed 42
```

Полезные флаги: `--skip-prepare`, `--skip-eval-pretrained`,
`--skip-abliterate`, `--skip-eval-abliterated`, `--skip-push`,
`--skip-dpo-data`, `--skip-dpo-train`, `--skip-eval-dpo` — пропустить
стадии, если их артефакты уже на диске. Push на HF Hub требует write-
токена, для сдачи не нужен — пайплайн идёт с `--skip-push`. Если
понадобится, добавьте `--hf-repo-id <ns>/<name>` и уберите `--skip-push`.

На RTX 2070 8 GB (Windows) полный прогон занимает около 2.5–3 часов;
доминанта — три eval-прохода и DPO-fit.

## Ноутбук-отчёт

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace hw6.ipynb
```

Ноутбук читает `artifacts_hw6/*.json`, `artifacts_hw6/abliteration_*`,
`artifacts_hw6/dpo_log_history.json` и рисует сравнение до/после и
примеры генераций. GPU и `bitsandbytes` не требуются.
