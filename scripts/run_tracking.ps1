param(
    [string]$ImageDir = "testbmp",
    [string]$Pattern = "*.bmp",
    [ValidateSet("timestamp", "name", "mtime")]
    [string]$SortBy = "timestamp",
    [string]$InitialRois,
    [string]$OutDir,
    [int]$MaxFrames = 0,
    [int]$ProgressEvery = 10,
    [double]$TrackScale = 1,
    [switch]$NoVideo,
    [switch]$Diagnostics,
    [switch]$Interactive,
    [ValidateSet("left", "right")]
    [string]$Camera
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
    "--max-frames", [string]$MaxFrames,
    "--review-mode", "continuous",
    "--pause-on", "none",
    "--progress-every", [string]$ProgressEvery,
    "--track-scale", [string]$TrackScale
)

if ($Camera) {
    $CliArgs += @("--camera", $Camera.ToLowerInvariant())
}
if ($PSBoundParameters.ContainsKey("OutDir")) {
    $CliArgs += @("--out-dir", $OutDir)
}

if ($Interactive) {
    if ($PSBoundParameters.ContainsKey("InitialRois")) {
        $CliArgs += @("--initial-rois", $InitialRois)
    } else {
        $CliArgs += "--draw-rois"
    }
} else {
    $CliArgs += "--no-window"
    if ($InitialRois) {
        $CliArgs += @("--initial-rois", $InitialRois)
    }
}

if ($NoVideo) {
    $CliArgs += "--no-video"
}
if ($Diagnostics) {
    $CliArgs += "--diagnostics"
}

& $Python @CliArgs
exit $LASTEXITCODE
