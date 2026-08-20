<#
.SYNOPSIS
Collects the original rigorous-experiment evidence into a small closure ZIP.

.DESCRIPTION
Checkpoint weights are hashed and inspected but are not copied into the ZIP.
Use the confirmation switches only when the current environment/checkpoint set
is the one used for the reported experiment.  Otherwise the strict validator
will correctly keep the corresponding evidence gap open.
#>

[CmdletBinding()]
param(
    [string]$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [string]$ExperimentOutput = "",
    [string]$TrainingCommand = "",
    [switch]$ConfirmHistoricalEnvironment,
    [switch]$ConfirmCheckpointSet
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ([string]::IsNullOrWhiteSpace($ExperimentOutput)) {
    $ExperimentOutput = Join-Path $ProjectRoot "experiment_outputs\rigorous_patient_split"
}
if (-not (Test-Path -LiteralPath $ExperimentOutput -PathType Container)) {
    throw "Experiment output directory does not exist: $ExperimentOutput"
}

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$stage = Join-Path $ExperimentOutput "p2_closure_$stamp"
$zipPath = "$stage.zip"
New-Item -ItemType Directory -Path $stage | Out-Null

$coreFiles = @(
    "summary_metrics.csv",
    "summary_metrics.json",
    "patient_split_seed42.json",
    "data_discovery_report.json"
)
foreach ($name in $coreFiles) {
    $source = Join-Path $ExperimentOutput $name
    if (-not (Test-Path -LiteralPath $source -PathType Leaf)) {
        throw "Required aggregate evidence is missing: $source"
    }
    Copy-Item -LiteralPath $source -Destination $stage
}

$models = @(
    "UNet3D",
    "NanoMambaUNet",
    "Ablation_NoMamba_UNet",
    "Ablation_HalfMamba_UNet",
    "AttentionUNet",
    "SegResNet16"
)
foreach ($model in $models) {
    foreach ($prefix in @("per_case", "training_log")) {
        $name = "${prefix}_${model}.csv"
        $source = Join-Path $ExperimentOutput $name
        if (Test-Path -LiteralPath $source -PathType Leaf) {
            Copy-Item -LiteralPath $source -Destination $stage
        } else {
            Write-Warning "Missing original artefact: $name"
        }
    }
}

$python = "python"
$checkpointArgs = @(
    (Join-Path $ProjectRoot "scripts\p2_checkpoint_manifest.py"),
    "--checkpoint-dir", (Join-Path $ExperimentOutput "checkpoints"),
    "--output", (Join-Path $stage "checkpoint_manifest.json")
)
if ($ConfirmCheckpointSet) {
    $checkpointArgs += "--historical-checkpoint-set-confirmed"
}
& $python @checkpointArgs
if ($LASTEXITCODE -notin @(0, 2)) {
    throw "Checkpoint manifest command failed with exit code $LASTEXITCODE"
}

$environmentArgs = @(
    (Join-Path $ProjectRoot "scripts\p2_environment_capture.py"),
    "--output", (Join-Path $stage "environment.json")
)
if ($ConfirmHistoricalEnvironment) {
    $environmentArgs += "--historical-environment-confirmed"
}
& $python @environmentArgs
if ($LASTEXITCODE -ne 0) {
    throw "Environment capture failed with exit code $LASTEXITCODE"
}

$transcriptPath = Join-Path $stage "run_transcript.txt"
if ([string]::IsNullOrWhiteSpace($TrainingCommand)) {
    @(
        "Historical training command: UNCONFIRMED",
        "Historical working directory: UNCONFIRMED",
        "Replace these lines only from the original terminal/PyCharm record or a reliable contemporaneous note."
    ) | Set-Content -LiteralPath $transcriptPath -Encoding UTF8
} else {
    @(
        "Historical training command (user confirmed): $TrainingCommand",
        "Historical working directory (user confirmed): $ProjectRoot"
    ) | Set-Content -LiteralPath $transcriptPath -Encoding UTF8
}

$hashLines = Get-ChildItem -LiteralPath $stage -File | Sort-Object Name | ForEach-Object {
    $hash = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    "$hash  $($_.Name)"
}
$hashLines | Set-Content -LiteralPath (Join-Path $stage "SHA256SUMS.txt") -Encoding ASCII

$auditScript = Join-Path $ProjectRoot "src\22_p2_evidence_audit.py"
& $python $auditScript --evidence-dir $stage --output (Join-Path $stage "evidence_audit_report.json") --strict-closure
$auditExit = $LASTEXITCODE

Compress-Archive -Path (Join-Path $stage "*") -DestinationPath $zipPath -CompressionLevel Optimal
Write-Host "Closure bundle: $zipPath"
if ($auditExit -eq 0) {
    Write-Host "Strict closure audit: PASS"
} else {
    Write-Warning "Strict closure audit is still incomplete (exit $auditExit). Read evidence_audit_report.json and console warnings."
}
