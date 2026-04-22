# Удалённый GPU для DZ6 (RTX 2070 8 GB)

Краткая памятка: как гонять полный пайплайн HW6 (abliteration + DPO)
на домашнем компьютере через Tailscale SSH + Windows Task Scheduler.
Нужно, потому что Colab free-tier срывает сессию на середине прогона,
а локальный T4 мы теряли по 2-3 раза подряд.

## 1. Инфраструктура

| Узел             | Значение                                                |
| ---------------- | ------------------------------------------------------- |
| Tailscale IP     | `100.121.5.55`                                          |
| SSH user         | `georgy`                                                |
| SSH key          | `~/.ssh/id_ed25519_dz5_gpu` (переиспользован с DZ5)     |
| Remote OS        | Windows 11, PowerShell 5.1                              |
| Remote Python    | `C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe`    |
| Remote repo path | `C:\Users\Georgy\Source\aith_dl_nlp_hw6` (ветка `hw_6`) |
| Артефакты + HF   | `P:\dz6-hw6\` (на C: не хватает места)                  |
| GPU              | NVIDIA GeForce RTX 2070, 8 GB, CUDA 12.6 driver         |

На C: у нас ~9 GB свободно, а только fp16-чекпойнт Qwen2.5-1.5B весит
~3 GB плюс HF-cache самой базовой модели — ещё ~3-4 GB. Поэтому
артефакты пайплайна и HF-кэш уехали на P: (`artifacts_hw6/`,
`hf_cache/`). Репозиторий с кодом остаётся на C:. Разделение прошито в
`scripts/run_dz6_full.ps1` через `$workRoot = "P:\dz6-hw6"` и env-
переменные `HF_HOME` / `TRANSFORMERS_CACHE`.

Подключение:

```bash
ssh -i ~/.ssh/id_ed25519_dz5_gpu georgy@100.121.5.55
```

Предупреждение про post-quantum KEX можно игнорировать.

## 2. Одноразовая подготовка окружения

Всё это уже сделано; шаги — на случай пересоздания.

### 2.1. Клонирование репозитория

```powershell
cd C:\Users\Georgy\Source
git clone https://github.com/dmagog/aith_DL_NLP.git aith_dl_nlp_hw6
cd aith_dl_nlp_hw6
git checkout hw_6
```

### 2.2. Conda env + зависимости

```powershell
conda create -y -n dz6-hw6 python=3.11
C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe -m pip install -U pip
C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe -m pip install `
    --index-url https://download.pytorch.org/whl/cu121 torch
C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe -m pip install `
    -r C:\Users\Georgy\Source\aith_dl_nlp_hw6\dz6\requirements.txt
```

Проверка:

```powershell
C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe -c `
    "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Ожидаем `2.5.1+cu121 True NVIDIA GeForce RTX 2070`.

### 2.3. HF token

В системных переменных уже должен быть `HF_TOKEN`. Проверить:

```powershell
[Environment]::GetEnvironmentVariable("HF_TOKEN", "User")
```

Если пусто — прописать:

```powershell
[Environment]::SetEnvironmentVariable("HF_TOKEN", "<токен>", "User")
```

После установки переменной нужно перелогиниться или открыть новый
PowerShell, чтобы Scheduled Task её увидел.

## 3. Scheduled Task DZ6Full

Все скрипты лежат в `dz6/scripts/`.

| Файл                           | Что делает                                         |
| ------------------------------ | -------------------------------------------------- |
| `run_dz6_full.ps1`             | Runner: активирует env, вызывает `src.run_full`    |
| `register_dz6_full_task.ps1`   | Создаёт / обновляет Scheduled Task `DZ6Full`       |
| `start_dz6_full.ps1`           | Запускает task вручную                             |
| `status_dz6_full.ps1`          | Печатает состояние task'а + хвост лога             |
| `tail_dz6_full.ps1`            | `tail` / `tail -f` по `run_full.log`               |
| `stop_dz6_full.ps1`            | Останавливает task и добивает stray python'ы      |
| `unregister_dz6_full_task.ps1` | Снимает регистрацию                                |

### 3.1. Регистрация

Один раз:

```powershell
cd C:\Users\Georgy\Source\aith_dl_nlp_hw6\dz6\scripts
powershell -NoProfile -ExecutionPolicy Bypass -File .\register_dz6_full_task.ps1
```

### 3.2. Старт / контроль из PowerShell на машине

```powershell
# старт
powershell -NoProfile -File .\start_dz6_full.ps1
# статус + последние 20 строк лога
powershell -NoProfile -File .\status_dz6_full.ps1
# follow — как tail -f
powershell -NoProfile -File .\tail_dz6_full.ps1 -Follow
# остановить
powershell -NoProfile -File .\stop_dz6_full.ps1
```

### 3.3. Управление через SSH с macOS

На macOS все кавычки — двойные вокруг `powershell -Command`, внутри
одиночные (иначе zsh съест). Путь к скриптам — в одну строку.

```bash
RPATH='C:\Users\Georgy\Source\aith_dl_nlp_hw6\dz6\scripts'
SSH="ssh -i ~/.ssh/id_ed25519_dz5_gpu georgy@100.121.5.55"

# старт
$SSH "powershell -NoProfile -File $RPATH\\start_dz6_full.ps1"
# статус + tail 40
$SSH "powershell -NoProfile -File $RPATH\\status_dz6_full.ps1 -Tail 40"
# стрим лога (Ctrl+C чтобы отцепиться)
$SSH "powershell -NoProfile -File $RPATH\\tail_dz6_full.ps1 -Follow"
# стоп
$SSH "powershell -NoProfile -File $RPATH\\stop_dz6_full.ps1"
```

## 4. Выгрузка артефактов

После завершения прогона артефакты лежат в
`P:\dz6-hw6\artifacts_hw6\`. HF-cache (`P:\dz6-hw6\hf_cache\`) не
тянем — он восстанавливается `transformers`'ом сам. Также не тянем
полные fp16-веса `abliterated_model` и `dpo_model`: аблитерированная
база уже на HF Hub (`dmagog/Qwen2.5-1.5B-Instruct-ru-abliterated`), а
для DPO нам нужен только LoRA-адаптер.

На ремоте:

```powershell
cd P:\dz6-hw6
Compress-Archive `
    -Path artifacts_hw6 `
    -DestinationPath artifacts_hw6.zip `
    -Force
```

Если zip получается слишком большой (>200 MB) — сначала вырежьте из
`artifacts_hw6\` папки `abliterated_model\` и `dpo_model\` (полные fp16-
снэпшоты модели, в репозиторий нам нужен только LoRA-адаптер
`dpo_adapter\`):

```powershell
Remove-Item -Recurse -Force artifacts_hw6\abliterated_model
Remove-Item -Recurse -Force artifacts_hw6\dpo_model
```

На локалке:

```bash
scp -i ~/.ssh/id_ed25519_dz5_gpu \
    georgy@100.121.5.55:'P:/dz6-hw6/artifacts_hw6.zip' \
    /Users/georgijmamarin/Desktop/Oplimp/dl_nlp_course/dz_1/dz6/
unzip -o artifacts_hw6.zip
```

## 5. Диагностика

### VRAM занят / CUDA OOM

```powershell
nvidia-smi
```

Типичный потребитель — DZ5-шный `T5Gemma`. Его мы глушим через свой
scheduled task (`Stop-ScheduledTask -TaskName DZ5T5Gemma`) и ждём,
пока `nvidia-smi` покажет <500 MiB.

### Почему `run_full.log` пустой после старта

Scheduled Task долго стартует первый раз (прогрев WinRM-like stack).
Через 20-30 секунд в логе появляются строчки `[begin ...]` и env-
диагностика. Если через минуту пусто — `Get-ScheduledTaskInfo` покажет
`LastTaskResult != 0`, там и ищите причину (обычно — отсутствующий
python.exe или битый `PYTHONPATH`).

### Кириллица в логе съедается

В `run_dz6_full.ps1` уже прокинуты `PYTHONUTF8=1`, `PYTHONIOENCODING=utf-8`,
`PYTHONUNBUFFERED=1`. Если всё равно mojibake — проверьте кодировку
терминала, которым читаете: `chcp 65001` перед `Get-Content`.

## 6. Почему именно так

- **Scheduled Task вместо nohup / Start-Process**: на Windows без
  установленного sshd-as-service долгоживущий процесс, стартованный
  из SSH-сессии, умирает при разрыве соединения (child inherits job).
  Scheduled Task живёт независимо.
- **Прямой путь к python.exe**: `conda activate` внутри non-interactive
  PowerShell неустойчив — проще вызвать `python.exe` из env'а напрямую.
- **`IgnoreNew` для MultipleInstances**: исключает случайный двойной
  запуск, если руку дёрнуло и ты нажал Start-ScheduledTask дважды.
- **`-RestartCount 0`**: если упали — хочу разобраться, а не
  автоматом перезапускать кривой код.
