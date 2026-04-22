# Раннер, который вызывается Windows Scheduled Task'ом DZ6Full.
# Активирует conda-окружение dz6-hw6 (не через conda activate, а прямым
# вызовом python.exe из env'а — так не зависим от профиля оболочки),
# прокидывает UTF-8 env-переменные, запускает src.run_full и пишет
# stdout+stderr в run_full.log.
#
# Артефакты и HF-кэш лежат на P:\dz6-hw6\ — на C: на нашей remote-машине
# свободно ~9 GB, а только fp16-чекпойнт Qwen2.5-1.5B весит ~3 GB, плюс
# HF-cache самой базовой модели — ещё ~3-4 GB. На C: не помещаемся.
$ErrorActionPreference = "Continue"

$here   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$dz6    = Split-Path -Parent $here   # ...\aith_dl_nlp_hw6\dz6
$python = "C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe"

# --- Рабочие каталоги на P: ---
$workRoot  = "P:\dz6-hw6"
$artifacts = Join-Path $workRoot "artifacts_hw6"
$hfCache   = Join-Path $workRoot "hf_cache"
$logDir    = Join-Path $artifacts "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
New-Item -ItemType Directory -Force -Path $hfCache | Out-Null
$log = Join-Path $logDir "run_full.log"

# --- Env для python'а ---
$env:HF_HOME            = $hfCache
$env:TRANSFORMERS_CACHE = $hfCache
$env:PYTHONUTF8         = "1"
$env:PYTHONIOENCODING   = "utf-8"
$env:PYTHONUNBUFFERED   = "1"

# --- UTF-8 для pipeline'а ---
# PowerShell 5.1 по умолчанию читает stdout native-команд как OEM и
# пишет Tee-Object в UTF-16LE. Из-за этого лог при SSH/tail показывает
# «пробелы между буквами». Форсим UTF-8 на обоих концах.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8
$OutputEncoding           = [System.Text.Encoding]::UTF8

Set-Location $dz6

Add-Content -Path $log -Value ("[begin {0}]" -f (Get-Date -Format o)) -Encoding UTF8
Add-Content -Path $log -Value ("[python {0}]" -f $python)             -Encoding UTF8
Add-Content -Path $log -Value ("[artifacts {0}]" -f $artifacts)       -Encoding UTF8
Add-Content -Path $log -Value ("[hf_home {0}]" -f $env:HF_HOME)       -Encoding UTF8

# Tee-Object в PS5 не умеет -Encoding, поэтому пишем в лог вручную через
# Add-Content -Encoding UTF8 и одновременно дублируем на stdout (который
# в Scheduled Task никто не читает, но пригодится при ручном запуске).
& $python -u -m src.run_full `
    --out $artifacts `
    --model "Qwen/Qwen2.5-1.5B-Instruct" `
    --hf-repo-id "dmagog/Qwen2.5-1.5B-Instruct-ru-abliterated" `
    --seed 42 2>&1 | ForEach-Object {
        $line = [string]$_
        [Console]::Out.WriteLine($line)
        Add-Content -Path $log -Value $line -Encoding UTF8
    }

$exit = $LASTEXITCODE
Add-Content -Path $log -Value ("[end {0}, exit={1}]" -f (Get-Date -Format o), $exit) -Encoding UTF8
exit $exit
