# Стартует Scheduled Task DZ6Full и печатает его состояние после запуска.
$taskName = "DZ6Full"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Error ("task '{0}' not registered. Run register_dz6_full_task.ps1 first." -f $taskName)
    exit 1
}

Start-ScheduledTask -TaskName $taskName
Start-Sleep -Seconds 3
Get-ScheduledTask -TaskName $taskName | Select-Object TaskName, State | Format-Table -AutoSize
Get-ScheduledTaskInfo -TaskName $taskName |
    Select-Object LastRunTime, LastTaskResult, NumberOfMissedRuns | Format-List
