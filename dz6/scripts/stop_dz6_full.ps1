# Экстренная остановка DZ6Full. Stop-ScheduledTask убивает процесс,
# который запустил task. Если python отвязался (редко, но бывает) —
# добивается Get-Process по python.exe из dz6-hw6 env'а.
$taskName = "DZ6Full"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if ($task) {
    Stop-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
    Write-Host ("stopped scheduled task '{0}'" -f $taskName)
} else {
    Write-Host ("task '{0}' not registered" -f $taskName)
}

$envPython = "C:\Users\Georgy\miniconda3\envs\dz6-hw6\python.exe"
$stray = Get-Process -Name python -ErrorAction SilentlyContinue |
    Where-Object { $_.Path -eq $envPython }
if ($stray) {
    Write-Host ("killing {0} stray python processes from dz6-hw6 env" -f $stray.Count)
    $stray | Stop-Process -Force
} else {
    Write-Host "no stray dz6-hw6 python processes"
}
