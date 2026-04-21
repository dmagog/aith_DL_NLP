# Снимает регистрацию Scheduled Task DZ6Full. Используй, когда прогон
# окончательно закончен и задача больше не нужна.
$taskName = "DZ6Full"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host ("task '{0}' not registered" -f $taskName)
    exit 0
}

Unregister-ScheduledTask -TaskName $taskName -Confirm:$false
Write-Host ("unregistered task '{0}'" -f $taskName)
