param(
    [string]$ImageDir = "trackingbmp",
    [string]$Pattern = "*.bmp",
    [ValidateSet("timestamp", "name", "mtime")]
    [string]$SortBy = "timestamp",
    [string]$InitialRois = "tracking_results\drawn_rois.json",
    [string]$OutDir = "tracking_results",
    [int]$MaxFrames = 0,
    [int]$ProgressEvery = 10,
    [double]$TrackScale = 1,
    [switch]$NoVideo
)

$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
$Python = "python"
if (Test-Path -LiteralPath $VenvPython) {
    try {
        & $VenvPython --version > $null 2>&1
        if ($LASTEXITCODE -eq 0) {
            $Python = $VenvPython
        }
    }
    catch {
        $Python = "python"
    }
}

Set-Location $ProjectRoot

$CliArgs = @(
    "track_main_tube_sequence.py",
    "--image-dir", $ImageDir,
    "--pattern", $Pattern,
    "--sort-by", $SortBy,
    "--initial-rois", $InitialRois,
    "--out-dir", $OutDir,
    "--max-frames", [string]$MaxFrames,
    "--no-window",
    "--review-mode", "continuous",
    "--pause-on", "none",
    "--progress-every", [string]$ProgressEvery,
    "--track-scale", [string]$TrackScale
)

if ($NoVideo) {
    $CliArgs += "--no-video"
}

& $Python @CliArgs
exit $LASTEXITCODE
