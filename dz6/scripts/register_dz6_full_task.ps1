# Регистрирует (или обновляет) Windows Scheduled Task 'DZ6Full'.
# Триггер — вручную через Start-ScheduledTask. Action вызывает
# run_dz6_full.ps1, тот активирует dz6-hw6 env и гоняет src.run_full.
# Лимит выполнения — 6 часов, чтобы случайно зависший процесс не висел
# сутками; наш полный прогон умещается в ~1.5 часа.

$taskName = "DZ6Full"
$here     = Split-Path -Parent $MyInvocation.MyCommand.Definition
$runner   = Join-Path $here "run_dz6_full.ps1"

if (-not (Test-Path $runner)) {
    Write-Error "runner not found: $runner"
    exit 1
}

$action = New-ScheduledTaskAction `
    -Execute "powershell.exe" `
    -Argument ("-NoProfile -ExecutionPolicy Bypass -File `"{0}`"" -f $runner)

$principal = New-ScheduledTaskPrincipal `
    -UserId $env:USERNAME `
    -LogonType Interactive `
    -RunLevel Highest

$settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries `
    -MultipleInstances IgnoreNew `
    -ExecutionTimeLimit (New-TimeSpan -Hours 6) `
    -RestartCount 0

Register-ScheduledTask `
    -TaskName $taskName `
    -Action $action `
    -Principal $principal `
    -Settings $settings `
    -Force | Out-Null

Write-Host ("Registered task '{0}'." -f $taskName)
Write-Host ("Runner:   {0}" -f $runner)
Write-Host ("Start:    Start-ScheduledTask -TaskName {0}" -f $taskName)
Write-Host ("Status:   powershell -NoProfile -File {0}\status_dz6_full.ps1" -f $here)
Write-Host ("Tail log: powershell -NoProfile -File {0}\tail_dz6_full.ps1" -f $here)
