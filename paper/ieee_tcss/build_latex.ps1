param(
    [string]$MainFile = "main.tex"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptRoot = if ($PSScriptRoot) { $PSScriptRoot } else { Split-Path -Parent $MyInvocation.MyCommand.Path }
$BuildDir = Join-Path $ScriptRoot "build"
$LogDir = Join-Path $ScriptRoot "logs"
$LogPath = Join-Path $LogDir "LATEX_BUILD_LOG.md"
$MainPath = Join-Path $ScriptRoot $MainFile
$MainStem = [IO.Path]::GetFileNameWithoutExtension($MainFile)
$PdfPath = Join-Path $BuildDir "$MainStem.pdf"

New-Item -ItemType Directory -Force -Path $BuildDir, $LogDir | Out-Null

function Write-LatexBuildLog {
    param(
        [string]$Status,
        [string[]]$Lines
    )

    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss K"
    $content = @(
        "# LaTeX Build Log",
        "",
        "Last attempted: $timestamp",
        "",
        "Status: $Status",
        "",
        $Lines
    )
    $flatContent = foreach ($item in $content) {
        if ($item -is [System.Array]) {
            foreach ($nested in $item) {
                [string]$nested
            }
        }
        else {
            [string]$item
        }
    }
    $TempLogPath = Join-Path $LogDir "build_latex_log_write.tmp"
    try {
        $flatContent | Set-Content -LiteralPath $TempLogPath -Encoding UTF8
        Move-Item -LiteralPath $TempLogPath -Destination $LogPath -Force
    }
    catch {
        $FallbackLogPath = Join-Path $LogDir "build_latex_log_fallback.md"
        try {
            $flatContent | Set-Content -LiteralPath $FallbackLogPath -Encoding UTF8
            Write-Warning "Could not replace LATEX_BUILD_LOG.md; wrote fallback log to $FallbackLogPath. Error: $($_.Exception.Message)"
        }
        catch {
            Write-Warning "Could not write LaTeX build log. Error: $($_.Exception.Message)"
        }
    }
}

function Invoke-LatexStep {
    param(
        [string]$Name,
        [string]$Command,
        [string[]]$Arguments,
        [string]$LogFile
    )

    Push-Location $BuildDir
    try {
        $output = & $Command @Arguments 2>&1
        $exitCode = $LASTEXITCODE
        $output | Tee-Object -FilePath $LogFile
        return [pscustomobject]@{
            Name = $Name
            ExitCode = $exitCode
            LogFile = $LogFile
        }
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $MainPath)) {
    Write-LatexBuildLog -Status "failed_missing_main_tex" -Lines @(
        "Main file not found: $MainPath",
        "No source files were modified."
    )
    Write-Host "LaTeX build failed: main file not found. See $LogPath"
    exit 1
}

$pdflatex = Get-Command pdflatex -ErrorAction SilentlyContinue
$bibtex = Get-Command bibtex -ErrorAction SilentlyContinue

if (-not $pdflatex -or -not $bibtex) {
    Write-LatexBuildLog -Status "not_run_missing_tex_toolchain" -Lines @(
        "pdflatex available: $([bool]$pdflatex)",
        "",
        "bibtex available: $([bool]$bibtex)",
        "",
        "PDF generated: False",
        "",
        "No PDF was generated because a complete TeX toolchain was not available on PATH.",
        "",
        "Next steps: install MiKTeX or TeX Live, ensure pdflatex and bibtex are on PATH, then rerun ./build_latex.ps1 from paper/ieee_tcss.",
        "",
        "No source files were modified."
    )
    Write-Host "LaTeX build skipped: pdflatex or bibtex not found. See $LogPath"
    exit 0
}

Copy-Item -LiteralPath $MainPath -Destination $BuildDir -Force
Copy-Item -LiteralPath (Join-Path $ScriptRoot "declarations.tex") -Destination $BuildDir -Force
Copy-Item -LiteralPath (Join-Path $ScriptRoot "bib") -Destination $BuildDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ScriptRoot "figures") -Destination $BuildDir -Recurse -Force
Copy-Item -LiteralPath (Join-Path $ScriptRoot "tables") -Destination $BuildDir -Recurse -Force

$steps = @()
$steps += Invoke-LatexStep -Name "pdflatex_pass1" -Command $pdflatex.Source -Arguments @("-interaction=nonstopmode", "-halt-on-error", $MainFile) -LogFile "pdflatex_pass1.log"
$steps += Invoke-LatexStep -Name "bibtex" -Command $bibtex.Source -Arguments @($MainStem) -LogFile "bibtex.log"
$steps += Invoke-LatexStep -Name "pdflatex_pass2" -Command $pdflatex.Source -Arguments @("-interaction=nonstopmode", "-halt-on-error", $MainFile) -LogFile "pdflatex_pass2.log"
$steps += Invoke-LatexStep -Name "pdflatex_pass3" -Command $pdflatex.Source -Arguments @("-interaction=nonstopmode", "-halt-on-error", $MainFile) -LogFile "pdflatex_pass3.log"

$combinedLog = Get-ChildItem -LiteralPath $BuildDir -Filter "*.log" | ForEach-Object {
    Get-Content -LiteralPath $_.FullName -Raw
} | Out-String

$warnings = @()
if ($combinedLog -match "LaTeX Warning|Package .* Warning") {
    $warnings += "LaTeX/package warnings present; inspect build/*.log."
}
if ($combinedLog -match "undefined references|Citation .* undefined|undefined citations") {
    $warnings += "Unresolved references or citations detected."
}
if ($combinedLog -match "Overfull \\\\hbox|Overfull \\\\vbox") {
    $warnings += "Overfull boxes detected."
}
if (-not $warnings) {
    $warnings += "No warning summary detected by the lightweight log scan."
}

$failedSteps = $steps | Where-Object { $_.ExitCode -ne 0 }
$pdfExists = Test-Path -LiteralPath $PdfPath
$status = if (-not $failedSteps -and $pdfExists) { "succeeded" } else { "failed_see_build_logs" }

$stepLines = $steps | ForEach-Object { "- $($_.Name): exit $($_.ExitCode), log build/$($_.LogFile)" }
$warningLines = $warnings | ForEach-Object { "- $_" }

Write-LatexBuildLog -Status $status -Lines @(
    "Output directory: paper/ieee_tcss/build",
    "",
    "PDF path: $PdfPath",
    "",
    "PDF generated: $pdfExists",
    "",
    "Tool paths:",
    "- pdflatex: $($pdflatex.Source)",
    "- bibtex: $($bibtex.Source)",
    "",
    "Build steps:",
    $stepLines,
    "",
    "Warning summary:",
    $warningLines,
    "",
    "No source files were modified."
)

if ($status -eq "succeeded") {
    Write-Host "LaTeX build succeeded. PDF: $PdfPath"
    exit 0
}

Write-Host "LaTeX build failed or produced no PDF. See $LogPath and build logs."
exit 1
