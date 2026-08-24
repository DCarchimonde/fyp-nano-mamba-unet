param(
    [int]$ExpectedPages = 95
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$P2RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$P2SourceDir = Join-Path $P2RepoRoot "paper_write\Universiti_Malaya_Thesis_Template"
$P2BuildDir = Join-Path ([System.IO.Path]::GetTempPath()) ("nano-mamba-thesis-" + [guid]::NewGuid().ToString("N"))
$P2TexmfVar = Join-Path $P2BuildDir "texmf-var"
$P2TexmfConfig = Join-Path $P2BuildDir "texmf-config"
$P2TexmfHome = Join-Path $P2BuildDir "texmf-home"

New-Item -ItemType Directory -Force -Path $P2BuildDir, $P2TexmfVar, $P2TexmfConfig, $P2TexmfHome | Out-Null

foreach ($P2Command in @("pdflatex", "makeglossaries")) {
    if (-not (Get-Command $P2Command -ErrorAction SilentlyContinue)) {
        throw "Required command is unavailable: $P2Command"
    }
}

$P2ConflictMarkers = Get-ChildItem -Path $P2SourceDir -File -Include "*.tex", "*.cls" -Recurse |
    Select-String -Pattern '^(<<<<<<<|=======|>>>>>>>)'
if ($P2ConflictMarkers) {
    $P2ConflictMarkers | ForEach-Object { Write-Error $_.ToString() }
    throw "Unresolved Git conflict marker found in thesis sources."
}

$env:TEXMFVAR = $P2TexmfVar
$env:TEXMFCONFIG = $P2TexmfConfig
$env:TEXMFHOME = $P2TexmfHome

Push-Location $P2SourceDir
try {
    $P2PassOneLog = Join-Path $P2BuildDir "pdflatex-1.log"
    & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$P2BuildDir" thesis.tex *> $P2PassOneLog
    if ($LASTEXITCODE -ne 0) {
        throw "pdflatex pass 1 failed. See $P2PassOneLog"
    }

    Push-Location $P2BuildDir
    try {
        $P2GlossaryLog = Join-Path $P2BuildDir "makeglossaries.log"
        # Windows PowerShell 5.1 promotes native stderr output to an ErrorRecord
        # when ErrorActionPreference is Stop.  makeglossaries legitimately writes
        # a warning there when the main glossary is empty even though the acronym
        # glossary is generated successfully, so judge the command by its exit code.
        $P2SavedErrorActionPreference = $ErrorActionPreference
        $P2GlossaryExitCode = 1
        try {
            $ErrorActionPreference = "Continue"
            & makeglossaries thesis *> $P2GlossaryLog
            $P2GlossaryExitCode = $LASTEXITCODE
        }
        finally {
            $ErrorActionPreference = $P2SavedErrorActionPreference
        }
        if ($P2GlossaryExitCode -ne 0) {
            throw "makeglossaries failed. See $P2GlossaryLog"
        }
    }
    finally {
        Pop-Location
    }

    foreach ($P2Pass in 2..4) {
        $P2PassLog = Join-Path $P2BuildDir ("pdflatex-{0}.log" -f $P2Pass)
        & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$P2BuildDir" thesis.tex *> $P2PassLog
        if ($LASTEXITCODE -ne 0) {
            throw "pdflatex pass $P2Pass failed. See $P2PassLog"
        }
    }
}
finally {
    Pop-Location
}

$P2FinalLog = Join-Path $P2BuildDir "thesis.log"
$P2RejectedPatterns = @(
    "undefined citations",
    "There were undefined references",
    "Label(s) may have changed",
    "Overfull \hbox",
    "Overfull \vbox",
    "No \printgloss"
)
$P2AcceptanceFailures = Select-String -Path $P2FinalLog -SimpleMatch -Pattern $P2RejectedPatterns
if ($P2AcceptanceFailures) {
    $P2AcceptanceFailures | ForEach-Object { Write-Error $_.ToString() }
    throw "Thesis acceptance check failed. Build retained at $P2BuildDir"
}

$P2Pdf = Join-Path $P2BuildDir "thesis.pdf"
$P2PageLine = Select-String -Path $P2FinalLog -Pattern '\((\d+) pages?,' | Select-Object -Last 1
if (-not $P2PageLine) {
    throw "Could not read the PDF page count from the final LaTeX log. Build retained at $P2BuildDir"
}
[void]($P2PageLine.Line -match '\((\d+) pages?,')
$P2PageCount = [int]$Matches[1]
if ($P2PageCount -ne $ExpectedPages) {
    throw "Thesis acceptance check failed: expected $ExpectedPages pages, got $P2PageCount. Build retained at $P2BuildDir"
}

Write-Output ("Pages: {0}" -f $P2PageCount)
Write-Output ("File size: {0} bytes" -f (Get-Item $P2Pdf).Length)
if (Get-Command "pdfinfo" -ErrorAction SilentlyContinue) {
    & pdfinfo $P2Pdf | Where-Object { $_ -match '^(Page size|PDF version):' }
}
Get-FileHash -Algorithm SHA256 $P2Pdf

$P2CanonicalPdf = Join-Path $P2SourceDir "thesis.pdf"
Copy-Item -Force $P2Pdf $P2CanonicalPdf

Write-Output "Canonical thesis written to $P2CanonicalPdf"
Write-Output "Build logs retained at $P2BuildDir"
