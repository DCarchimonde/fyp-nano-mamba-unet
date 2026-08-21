[CmdletBinding()]
param(
    [string]$ProjectRoot = "D:\AI_FYP",
    [string]$DataDir = "",
    [string]$ExperimentOutput = "",
    [string]$OutputRoot = "",
    [int]$BatchSize = 1
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$ProjectRoot = [System.IO.Path]::GetFullPath($ProjectRoot)
if ([string]::IsNullOrWhiteSpace($DataDir)) {
    $DataDir = Join-Path $ProjectRoot "Data\ACDC\database\training"
}
if ([string]::IsNullOrWhiteSpace($ExperimentOutput)) {
    $ExperimentOutput = Join-Path $ProjectRoot "experiment_outputs\rigorous_patient_split"
}
if ([string]::IsNullOrWhiteSpace($OutputRoot)) {
    $OutputRoot = Join-Path $ProjectRoot "experiment_outputs"
}

$DataDir = [System.IO.Path]::GetFullPath($DataDir)
$ExperimentOutput = [System.IO.Path]::GetFullPath($ExperimentOutput)
$OutputRoot = [System.IO.Path]::GetFullPath($OutputRoot)
$Checkpoint = Join-Path $ExperimentOutput "checkpoints\best_NanoMambaUNet.pth"
$Timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$OutputDir = Join-Path $OutputRoot "spatiotemporal_cine_$Timestamp"
$ZipPath = "$OutputDir.zip"

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "ProjectRoot does not exist: $ProjectRoot"
}
if (-not (Test-Path -LiteralPath $DataDir -PathType Container)) {
    throw "ACDC data directory does not exist: $DataDir"
}
if (-not (Test-Path -LiteralPath $Checkpoint -PathType Leaf)) {
    throw "Nano-Mamba checkpoint does not exist: $Checkpoint"
}
if ($BatchSize -lt 1) {
    throw "BatchSize must be at least one."
}

Set-Location -LiteralPath $ProjectRoot
$env:MPLCONFIGDIR = Join-Path $env:TEMP "nano_mamba_matplotlib"
New-Item -ItemType Directory -Force -Path $env:MPLCONFIGDIR | Out-Null

Write-Host "=== Spatio-temporal analysis preflight ==="
python -c "import torch, monai, nibabel, numpy, matplotlib; print('Python dependencies: PASS'); print('PyTorch:', torch.__version__); print('MONAI:', monai.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
if ($LASTEXITCODE -ne 0) {
    throw "Python dependency preflight failed in the active environment."
}

python -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 2)"
if ($LASTEXITCODE -ne 0) {
    throw "CUDA is required for the final 20-patient cine run, but it is unavailable. Activate the nanomamba environment."
}

Write-Host ""
Write-Host "=== Full 20-patient cine analysis ==="
$Arguments = @(
    "src\23_spatiotemporal_cine_analysis.py",
    "--project-root", $ProjectRoot,
    "--data-dir", $DataDir,
    "--checkpoint", $Checkpoint,
    "--output-dir", $OutputDir,
    "--model", "NanoMambaUNet",
    "--device", "cuda",
    "--batch-size", $BatchSize.ToString()
)
python @Arguments
if ($LASTEXITCODE -ne 0) {
    throw "Spatio-temporal cine analysis failed. Read the final error; no result should be reported from this run."
}

$SummaryPath = Join-Path $OutputDir "summary.json"
if (-not (Test-Path -LiteralPath $SummaryPath -PathType Leaf)) {
    throw "Analysis returned success but summary.json is missing."
}
$Summary = Get-Content -LiteralPath $SummaryPath -Raw | ConvertFrom-Json
if ($Summary.status -ne "complete_validation_analysis") {
    throw "Expected complete_validation_analysis, received: $($Summary.status)"
}
if ([int]$Summary.patients -ne 20) {
    throw "Expected 20 validation patients, received: $($Summary.patients)"
}

Compress-Archive -Path (Join-Path $OutputDir "*") -DestinationPath $ZipPath -CompressionLevel Optimal
if (-not (Test-Path -LiteralPath $ZipPath -PathType Leaf)) {
    throw "Result ZIP was not created: $ZipPath"
}

Write-Host ""
Write-Host "SPATIO-TEMPORAL ANALYSIS COMPLETE" -ForegroundColor Green
Write-Host "Patients: $($Summary.patients)"
Write-Host "Frames: $($Summary.frames)"
Write-Host "Result directory: $OutputDir"
Write-Host "Upload this ZIP to ChatGPT: $ZipPath"
