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
- **abliteration**: снятие активаций residual stream на last-token, расчёт
  refusal direction по слоям (`mean(harmful) − mean(harmless)`), runtime-
  ablation для выбора лучшего слоя, weight orthogonalization по
  `embed_tokens`, `o_proj`, `down_proj`;
- **DPO-данные**: self-generated пары, где `chosen` — ответ оригинальной
  `Qwen2.5-Instruct`, `rejected` — ответ аблитерированной модели;
- **QLoRA-DPO**: 4-bit NF4 + LoRA `r=16, α=32` на всех линейных,
  `beta=0.1`, `paged_adamw_8bit`, gradient checkpointing, effective
  batch = 4, lr=5e-5, 1 эпоха;
- 3 eval-прогона (pretrained / abliterated / dpo) — refusal-rate по
  regex-эвристике на harmful и harmless подмножествах, одинаковый
  `GenConfig` (greedy, 192 токенов) для честного сравнения;
- `src/run_full.py` — end-to-end с skip-флагами и idempotent-resume'ом;
- `scripts/*.ps1` — Scheduled Task `DZ6Full` для remote Windows-машины
  (RTX 2070 8GB), живёт независимо от SSH-сессии;
- [REMOTE_GPU.md](REMOTE_GPU.md) — инструкция по удалённому прогону;
- `colab_train.ipynb` — fallback для free-tier T4;
- `hw6.ipynb` — отчётный ноутбук, читает зафиксированные артефакты.

## Итоговый результат

*TBD — заполнится после прогона.*

| Модель          | refusal_rate harmful | refusal_rate harmless |
| --------------- | -------------------- | --------------------- |
| Qwen pretrained | —                    | —                     |
| + abliterated   | —                    | —                     |
| + DPO           | —                    | —                     |

Подробный разбор (layer-search, DPO loss curve, примеры генераций, ссылка
на HF Hub) — в [hw6.ipynb](hw6.ipynb).

## Почему такой дизайн

1. **Qwen2.5-1.5B-Instruct база**. Маленькая, хорошо знает русский, уже
   refusal-trained — есть что ломать. На T4 (sm75) / RTX 2070 8GB
   умещается в 4-bit вместе с DPO-градиентами при batch=2.
2. **Abliteration по Arditi et al.**. Дёшевая (≈10 мин на T4) операция:
   активации собираются один раз, weight orth одноразово. После неё
   модель сохраняется как обычный HF-чекпойнт — без runtime-хуков.
3. **Regex-refusal вместо LLM-Judge**. ДЗ явно допускает «простой
   вариант», а regex-эвристика по RU+EN паттернам reproducible и не
   требует внешнего API. Паттерны зафиксированы в `src/evaluate.py`.
4. **Self-generated DPO-пары**. ДЗ допускает как открытый датасет, так и
   собственный. Self-generated надёжнее: `rejected` — реально плохой
   ответ именно *этой* аблитерированной модели, а не абстрактный harmful
   текст. Генерим в один проход, greedy, 128 токенов.
5. **QLoRA-DPO, а не full-parameter DPO**. Модель после DPO используется
   как аблитерированная-база + LoRA-адаптер. LoRA сохраняется отдельно
   (~20 МБ) и коммитится в репо.
6. **HF push опционален**. Аблитерированная модель детерминированно
   воспроизводима из `src/abliterate.py` при том же seed (`42`), DPO
   берёт её с диска, не с Hub. Реальный прогон идёт с `--skip-push`;
   выложить на Hub — отдельная ручная операция при наличии write-
   токена, весь остальной пайплайн от неё не зависит.
7. **Windows Scheduled Task, а не nohup**. На remote Windows без sshd-as-
   service длинный процесс из SSH-сессии умирает при разрыве. Scheduled
   Task живёт независимо, лог пишется построчно, всё мониторится по SSH
   через `status_dz6_full.ps1` / `tail_dz6_full.ps1`.
8. **Идемпотентные стадии**. Free-tier Colab/SSH обрывается — каждый
   stage проверяет свой метрик-файл и пропускается, если артефакт уже
   готов. Полный прогон можно перезапустить сколько угодно раз.

## Структура

- `src/data.py` — сэмплинг harmful + harmless, сплиты, сохранение в
  `artifacts_hw6/sample/`;
- `src/evaluate.py` — refusal regex, `GenConfig`, `run_eval`;
- `src/abliterate.py` — активации, direction, layer search, weight orth;
- `src/dpo_data.py` — генерация `chosen`/`rejected` пар;
- `src/dpo_train.py` — QLoRA-DPO (TRL `DPOTrainer`);
- `src/run_full.py` — end-to-end;
- `colab_train.ipynb` — раннер для Colab T4;
- `scripts/` — PS-скрипты Scheduled Task'а DZ6Full (см. REMOTE_GPU.md);
- `hw6.ipynb` — отчётный ноутбук;
- `artifacts_hw6/` — зафиксированные артефакты прогона (метрики,
  log_history, eval-сэмплы, LoRA-адаптер DPO ~20 МБ; полные fp16-веса
  моделей в репо не лежат — abliterated детерминированно
  воспроизводится из `src/abliterate.py`);
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

### Вариант A — удалённый GPU (RTX 2070 8GB)

Основной путь. См. подробную инструкцию в [REMOTE_GPU.md](REMOTE_GPU.md).
После разовой настройки (conda env + Scheduled Task):

```powershell
# из C:\...\aith_dl_nlp_hw6\dz6\scripts
powershell -NoProfile -File .\start_dz6_full.ps1
powershell -NoProfile -File .\tail_dz6_full.ps1 -Follow
```

Лог пишется в `artifacts_hw6/logs/run_full.log`, контроль task'а —
`status_dz6_full.ps1`. На RTX 2070 8GB полный прогон ~2.5–3 часа
(доминанта — три eval-прохода и DPO-fit).

### Вариант B — Colab T4 (fallback)

`Runtime → Change runtime type → T4 GPU`, открыть `colab_train.ipynb`,
выполнить сверху вниз. Ноутбук клонирует `hw_6`, ставит зависимости,
запускает `python -m src.run_full`, упаковывает `artifacts_hw6/` в zip.
При обрыве сессии idempotent-resume подхватит с последней готовой
стадии.

### Вариант C — локально на своём GPU

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
понадобится, добавь `--hf-repo-id <ns>/<name>` и убери `--skip-push`.

## Ноутбук-отчёт

```bash
python3 -m jupyter nbconvert --to notebook --execute --inplace hw6.ipynb
```

Ноутбук читает `artifacts_hw6/*.json`, `artifacts_hw6/abliteration_*`,
`artifacts_hw6/dpo_log_history.json` и рисует сравнение before/after и
примеры генераций. GPU и `bitsandbytes` не требуются.
