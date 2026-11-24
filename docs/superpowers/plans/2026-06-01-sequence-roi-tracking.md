# Sequence ROI Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an independent sequence tracker that propagates first-frame ROIs and writes one JSON file containing centerline summaries for every frame.

**Architecture:** Keep `detect_main_tube.py` as the single-frame detector. Add `track_main_tube_sequence.py` as a separate orchestration layer that tracks ROI polygons between frames with LK optical flow and affine RANSAC, then calls the existing detector on each propagated ROI.

**Tech Stack:** Python, OpenCV, NumPy, existing `detect_main_tube` module, `unittest`.

---

### Task 1: Output Contract Tests

**Files:**
- Create: `test_track_main_tube_sequence.py`
- Create: `track_main_tube_sequence.py`

- [ ] **Step 1: Write failing tests**

Add tests that assert the sequence output contains one top-level `frames` list, each frame has `image`, `num_rois`, and `centerlines`, and centerline entries do not include sampling arrays such as `left_boundary`, `right_boundary`, or `centerline_points`.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest test_track_main_tube_sequence.py`

Expected: failure because `track_main_tube_sequence` does not exist.

- [ ] **Step 3: Implement summary helpers**

Create helper functions that convert existing detector results through `detect_main_tube.make_centerline_summary()` and wrap all frame summaries into a single payload.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m unittest test_track_main_tube_sequence.py`

Expected: all tests pass.

### Task 2: ROI Tracking

**Files:**
- Modify: `track_main_tube_sequence.py`
- Modify: `test_track_main_tube_sequence.py`

- [ ] **Step 1: Write failing tests**

Add synthetic-image tests for propagating a square ROI through a known translation. Assert that polygon vertices move by the expected offset.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest test_track_main_tube_sequence.py`

Expected: failure because ROI propagation is not implemented.

- [ ] **Step 3: Implement optical-flow propagation**

Use `cv2.goodFeaturesToTrack()` inside each ROI mask, track points with `cv2.calcOpticalFlowPyrLK()`, estimate an affine transform with `cv2.estimateAffinePartial2D(..., method=cv2.RANSAC)`, and apply it to each ROI polygon. If there are too few valid points, keep the previous ROI and mark tracking as failed.

- [ ] **Step 4: Run tests and verify pass**

Run: `python -m unittest test_track_main_tube_sequence.py`

Expected: all tests pass.

### Task 3: CLI and Smoke Verification

**Files:**
- Modify: `track_main_tube_sequence.py`
- Modify: `test_track_main_tube_sequence.py`

- [ ] **Step 1: Write failing CLI parse tests**

Assert `--images`, `--initial-rois`, and `--out-json` are accepted, and fewer than two images are rejected.

- [ ] **Step 2: Run tests and verify failure**

Run: `python -m unittest test_track_main_tube_sequence.py`

Expected: failure because CLI parsing is not implemented.

- [ ] **Step 3: Implement CLI**

Implement `parse_args()`, `run_sequence()`, and `main()`. The CLI reads the first ROI JSON, processes every image, and writes only the combined centerline summary JSON.

- [ ] **Step 4: Run all tests and smoke command**

Run: `python -m unittest`

Run: `python track_main_tube_sequence.py --images pic/1.bmp pic/2.bmp --initial-rois output/1_draw/drawn_rois.json --out-json output/sequence_centerlines.json`

Expected: tests pass and the smoke command writes `output/sequence_centerlines.json`.
