# Project layout

This workspace is organized around the sequence-tracking workflow.

## Daily-use files

- `track_main_tube_sequence.py`: main entry for tracking ROIs through BMP sequences.
- `detect_main_tube.py`: single-frame tube detector reused by the tracker.
- `testbmp/`: sample BMP sequence used by the launcher.
- `run_testbmp_live.cmd`: Windows launcher for selecting pipes and viewing results.
- `scripts/run_tracking.ps1`: passes the selected input folder and options to the sequence tracker.
- `tracking_results/runs/<side>_camera/<timestamp>/`: both videos, per-frame measurements, and any newly drawn ROIs for one run.
- `tracking_results/archive/`: previous experiment outputs, grouped by their original run.
- `docs/`: algorithm notes and historical planning notes.

## Running on another folder

From the project root in PowerShell, supply the folder containing the sequence images.

```powershell
.\run_testbmp_live.cmd -Camera right -ImageDir "D:\images\sequence 2"
```

The default input pattern is `*.bmp`. For another format, add the matching pattern, such as `-Pattern "*.png"`. Double-clicking the launcher uses the sample sequence.
