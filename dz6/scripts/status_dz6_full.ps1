# Показывает состояние Scheduled Task DZ6Full + хвост лога.
param([int]$Tail = 20)

$taskName = "DZ6Full"
$task = Get-ScheduledTask -TaskName $taskName -ErrorAction SilentlyContinue
if (-not $task) {
    Write-Host ("task '{0}' not registered" -f $taskName)
    exit 1
}

$info = Get-ScheduledTaskInfo -TaskName $taskName
Write-Host ("State:       {0}" -f $task.State)
Write-Host ("LastRunTime: {0}" -f $info.LastRunTime)
Write-Host ("LastResult:  {0}" -f $info.LastTaskResult)
Write-Host ("NextRunTime: {0}" -f $info.NextRunTime)

$here    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$dz6     = Split-Path -Parent $here
$logPath = Join-Path $dz6 "artifacts_hw6\logs\run_full.log"

if (Test-Path $logPath) {
    $size = (Get-Item $logPath).Length
    Write-Host ""
    Write-Host ("--- last {0} lines of {1} ({2} bytes) ---" -f $Tail, $logPath, $size)
    Get-Content $logPath -Tail $Tail
} else {
    Write-Host ""
    Write-Host ("log not found yet: {0}" -f $logPath)
}
