# Project layout

This workspace is organized around the sequence-tracking workflow.

## Daily-use files

- `track_main_tube_sequence.py`: main entry for tracking ROIs through BMP sequences.
- `detect_main_tube.py`: single-frame tube detector reused by the tracker.
- `trackingbmp/`: input BMP frame sequence. Kept at the root because it is the large primary dataset.
- `tracking_results/`: current working output folder. It keeps `drawn_rois.json`, `sequence_centerlines.json`, and timestamped review videos such as `tracking_review_20260608_191500.mp4`.
- `scripts/run_tracking.ps1`: short wrapper for the long tracking command.
- `docs/`: algorithm notes and historical planning notes.

## Archived files

- `archive/tracking_runs/`: historical experiment outputs such as smoke runs, probes, and offset-fix checks.
- `archive/legacy_main_tube_detect/`: older single-image detection snapshot with its own output folder.
- `archive/root_outputs/`: loose old output files that used to sit in the project root.
- `archive/incomplete_downloads/`: zero-byte or interrupted download placeholders.
- `archive/third_party/`: old third-party code snapshots that are not used by the current tracker.

## Standard command

From `F:\tracking_test`, run:

```powershell
.\scripts\run_tracking.ps1
```

That is equivalent to:

```powershell
python track_main_tube_sequence.py --image-dir trackingbmp --pattern "*.bmp" --sort-by timestamp --initial-rois tracking_results\drawn_rois.json --out-dir tracking_results --no-window --review-mode continuous --pause-on none --progress-every 10 --track-scale 1
```

Default generated outputs stay under `tracking_results/`. Review videos use timestamped names by default, so repeated runs do not overwrite earlier videos.

Useful variants:

```powershell
.\scripts\run_tracking.ps1 -NoVideo
.\scripts\run_tracking.ps1 -OutDir tracking_results_new
.\scripts\run_tracking.ps1 -MaxFrames 20 -NoVideo -OutDir tracking_results_probe
.\scripts\run_tracking.ps1 -ProgressEvery 1
```
