# Раннер, который вызывается Windows Scheduled Task'ом DZ6Full.
# Активирует conda-окружение dz6-hw6 (не через conda activate, а прямым
# вызовом python.exe из env'а — так не зависим от профиля оболочки),
# прокидывает UTF-8 env-переменные, запускает src.run_full и пишет
# stdout+stderr в artifacts_hw6\logs\run_full.log.
$ErrorActionPreference = "Continue"

$here   = Split-Path -Parent $MyInvocation.MyCommand.Definition
$dz6    = Split-Path -Parent $here   # ...\aith_dl_nlp_hw6\dz6
$python = "C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe"

$logDir = Join-Path $dz6 "artifacts_hw6\logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$log = Join-Path $logDir "run_full.log"

$env:PYTHONUTF8       = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUNBUFFERED = "1"

Set-Location $dz6

Add-Content -Path $log -Value ("[begin {0}]" -f (Get-Date -Format o))
Add-Content -Path $log -Value ("[python {0}]" -f $python)
Add-Content -Path $log -Value ("[cwd    {0}]" -f (Get-Location).Path)

& $python -u -m src.run_full `
    --out artifacts_hw6 `
    --model "Qwen/Qwen2.5-1.5B-Instruct" `
    --hf-repo-id "dmagog/Qwen2.5-1.5B-Instruct-ru-abliterated" `
    --seed 42 2>&1 | Tee-Object -FilePath $log -Append

$exit = $LASTEXITCODE
Add-Content -Path $log -Value ("[end {0}, exit={1}]" -f (Get-Date -Format o), $exit)
exit $exit
