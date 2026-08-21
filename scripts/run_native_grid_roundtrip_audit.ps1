[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\AI_FYP",
    [string]$DataDir = "",
    [string]$OutputRoot = ""
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $ProjectRoot "Data\ACDC\database\training"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "experiment_outputs"
}

$DataDir = [System.IO.Path]::GetFullPath($DataDir)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $OutputRoot "native_grid_roundtrip_$Timestamp"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "ProjectRoot does not exist: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "ACDC data directory does not exist: $DataDir"
}

Set-Location -LiteralPath $ProjectRoot
Write-Host "=== Native-grid label round-trip audit ==="
python "src\26_native_grid_roundtrip_audit.py" `
    --project-root $ProjectRoot `
    --data-dir $DataDir `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Native-grid round-trip audit failed; do not report a result from this run."
}

$Summary = Join-Path $OutputDir "native_label_roundtrip_summary.json"
if (-not (Test-Path -LiteralPath $Summary -PathType Leaf)) {
    throw "Audit returned success but the summary JSON is missing: $Summary"
}

Write-Host ""
Write-Host "NATIVE-GRID ROUND-TRIP AUDIT COMPLETE" -ForegroundColor Green
Write-Host "Result directory: $OutputDir"
Write-Host "Send this file for final numeric interpretation: $Summary"
