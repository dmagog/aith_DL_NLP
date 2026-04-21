# Показывает хвост лога run_full.log. С ключом -Follow — стримит
# новые строки, как `tail -f` (обычный Get-Content -Wait). Удобно
# запускать локально через ssh, чтобы видеть прогресс прогона.
param(
    [int]$Tail = 40,
    [switch]$Follow
)

$here    = Split-Path -Parent $MyInvocation.MyCommand.Definition
$dz6     = Split-Path -Parent $here
$logPath = Join-Path $dz6 "artifacts_hw6\logs\run_full.log"

if (-not (Test-Path $logPath)) {
    Write-Host ("log not found yet: {0}" -f $logPath)
    exit 1
}

if ($Follow) {
    Get-Content $logPath -Tail $Tail -Wait
} else {
    Get-Content $logPath -Tail $Tail
}
