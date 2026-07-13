param(
    [string]$MainFile = "main.tex"
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$BuildDir = Join-Path $Root "build"
$LogDir = Join-Path $Root "logs"
$LogPath = Join-Path $LogDir "LATEX_BUILD_LOG.md"
New-Item -ItemType Directory -Force -Path $BuildDir, $LogDir | Out-Null

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue
$timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"

if (-not $pdflatex -or -not $bibtex) {
    @"
# LaTeX Build Log

Last attempted: $timestamp

Status: not_run_missing_tex_toolchain

pdflatex available: $([bool]$pdflatex)

bibtex available: $([bool]$bibtex)

No PDF was generated because a complete TeX toolchain was not available on PATH.
"@ | Set-Content -Path $LogPath -Encoding UTF8
    Write-Host "LaTeX build skipped: pdflatex or bibtex not found. See $LogPath"
    exit 0
}

Push-Location $Root
try {
    Copy-Item -Path $MainFile -Destination $BuildDir -Force
    Copy-Item -Path "declarations.tex" -Destination $BuildDir -Force
    Copy-Item -Path "bib" -Destination $BuildDir -Recurse -Force
    Copy-Item -Path "figures" -Destination $BuildDir -Recurse -Force
    Copy-Item -Path "tables" -Destination $BuildDir -Recurse -Force
    Push-Location $BuildDir
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass1.log"
    & bibtex ([IO.Path]::GetFileNameWithoutExtension($MainFile)) | Tee-Object -FilePath "bibtex.log"
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass2.log"
    & pdflatex -interaction=nonstopmode $MainFile | Tee-Object -FilePath "pdflatex_pass3.log"
    Pop-Location
    @"
# LaTeX Build Log

Last attempted: $timestamp

Status: attempted

Output directory: paper/ieee_tcss/build

Review pdflatex_pass*.log and bibtex.log for warnings/errors.
"@ | Set-Content -Path $LogPath -Encoding UTF8
}
finally {
    while ((Get-Location).Path -ne $Root) { Pop-Location }
    Pop-Location
}
