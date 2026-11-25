"""Small local review bundles, with 1-based frame numbers matching the videos."""

import csv
import hashlib
import io
import json
import platform
import tempfile
from collections import Counter
from itertools import zip_longest
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import cv2
import numpy as np

import detect_main_tube as dmt


METRICS = ("confidence", "tube_width_px", "continuity", "edge_contrast", "wall_pair_coverage",
           "width_ratio", "lateral_shift_px")
MOTION = ("method", "status", "failure_reason", "num_features", "num_inliers",
          "track_center_dx", "track_center_dy", "update_status")
MAX_SAMPLES = 32


def add_evidence(frame, detections, exposure):
    frame["exposure"] = exposure
    by_id = {d["roi_idx"]: d for d in detections}
    for item in frame["centerlines"]:
        raw = by_id[item["roi_idx"]]
        item["candidate"] = {k: raw[k] for k in METRICS if k in raw}
        if item["status"] == "ok":
            for key in ("left_boundary", "right_boundary"):
                line = raw[key]
                item[key] = [line[0], line[-1]]


def _summarize(frames):
    stats = {}
    for n, frame in enumerate(frames, 1):
        for item in frame["centerlines"]:
            idx = item["roi_idx"]
            if idx not in stats:
                stats[idx] = dict(ok=0, lost=0, loss_runs=0, first_loss=None, first_recovery=None,
                                  longest_loss=None, reasons=Counter(), weakest_ok=None)
            s = stats[idx]
            s[item["status"]] += 1
            if item["status"] == "lost":
                if "start" not in s:
                    s["start"] = n
                    s["loss_runs"] += 1
                    s["first_loss"] = s["first_loss"] or n
                s["reasons"][item.get("failure_reason", "unknown")] += 1
                longest = s["longest_loss"]
                if longest is None or n - s["start"] > longest[1] - longest[0]:
                    s["longest_loss"] = [s["start"], n]
            else:
                if s.pop("start", None) is not None and s["first_recovery"] is None:
                    s["first_recovery"] = n
                coverage = item.get("wall_pair_coverage")
                if coverage is not None and (s["weakest_ok"] is None or coverage < s["weakest_ok"][1]):
                    s["weakest_ok"] = [n, coverage]
    events, routine = [], []
    for idx, s in stats.items():
        s.pop("start", None)
        if s["first_loss"]:
            a, b = s["longest_loss"]
            numbers = [s["first_loss"], s["first_recovery"], a - 1, a, (a + b) // 2, b, b + 1]
            events.append([(n, idx) for n in dict.fromkeys(numbers) if n and 1 <= n <= len(frames)])
        weakest = s["weakest_ok"]
        numbers = [1, weakest[0] if weakest else 1, len(frames), (len(frames) + 1) // 2]
        routine.append([(n, idx) for n in dict.fromkeys(numbers)])
    selected = {}
    for groups, limit in ((events, MAX_SAMPLES // 2), (routine, MAX_SAMPLES), (events, MAX_SAMPLES)):
        for batch in zip_longest(*groups):
            for key in batch:
                if key and len(selected) < limit:
                    selected[key] = None
    return stats, sorted(selected)


def _review_crop(gray, item):
    vertices = np.asarray(item["vertices"], dtype=float)
    padding = np.maximum(8, np.ptp(vertices, axis=0) * .12)
    lo = np.maximum(0, np.floor(vertices.min(axis=0) - padding).astype(int))
    hi = np.minimum(gray.shape[1::-1], np.ceil(vertices.max(axis=0) + padding).astype(int) + 1)
    if np.any(hi <= lo):
        return None
    crop = gray[lo[1]:hi[1], lo[0]:hi[0]]
    size = np.maximum(1, np.rint((hi - lo) * min(1, 320 / max(crop.shape))).astype(int))
    raw = cv2.cvtColor(cv2.resize(crop, tuple(size), interpolation=cv2.INTER_AREA), cv2.COLOR_GRAY2BGR)
    marked = raw.copy()

    def project(points):
        return np.rint((np.asarray(points) - lo) * size / (hi - lo)).astype(np.int32)

    cv2.polylines(marked, [project(vertices)], True, (120, 120, 120), 1, cv2.LINE_AA)
    if item["status"] == "ok":
        for key, color in dmt.TUBE_LINES:
            line = item[key] if key != "centerline" else np.reshape(item[key]["endpoints"], (2, 2))
            cv2.polylines(marked, [project(line)], False, color, 1, cv2.LINE_AA)
    ok, encoded = cv2.imencode(".jpg", np.hstack([raw, marked]), [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        raise OSError("Could not encode diagnostic crop")
    return encoded.tobytes()


def write_diagnostics(path, frames, initial_rois, planned_frames, args):
    stats, samples = _summarize(frames)
    settings = {k: v for k, v in vars(args).items() if not k.startswith("_") and
                (isinstance(v, (int, float, bool)) or k in {"roi_update_mode", "exposure_guard", "sort_by",
                                                            "min_roi_update_confidence"})}
    source = hashlib.sha256()
    for name in ("detect_main_tube.py", "track_main_tube_sequence.py"):
        source.update(Path(__file__).with_name(name).read_text(encoding="utf-8").encode())
    report = dict(format_version=1, frames_processed=len(frames), frames_planned=planned_frames,
                  python=platform.python_version(), opencv=cv2.__version__, numpy=np.__version__,
                  tracker_sha256=source.hexdigest(), settings=settings, initial_rois=initial_rois,
                  by_roi=stats, samples=[], sample_errors=[])
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with ZipFile(tmp_path, "w", compression=ZIP_DEFLATED, compresslevel=3) as z:
            with z.open("measurements.csv", "w", force_zip64=True) as entry, io.TextIOWrapper(entry, encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(["frame", "roi", "status", "reason", *["candidate_" + k for k in METRICS],
                                 *["motion_" + k for k in MOTION], "mean_gray", "saturated_frac",
                                 "center_x1", "center_y1", "center_x2", "center_y2"])
                for n, frame in enumerate(frames, 1):
                    for item in frame["centerlines"]:
                        candidate = item["candidate"]
                        motion = item.get("tracking", {})
                        line = (item.get("centerline") or {}).get("endpoints", [None] * 4)
                        writer.writerow([n, item["roi_idx"], item["status"], item.get("failure_reason"),
                                         *[candidate.get(k) for k in METRICS], *[motion.get(k) for k in MOTION],
                                         frame["exposure"]["mean"], frame["exposure"]["saturated_frac"], *line])
            previous_n = None
            for n, idx in samples:
                frame = frames[n - 1]
                if previous_n != n:
                    previous_n = n
                    try:
                        gray = dmt.read_image_unicode(frame["image"])
                    except (OSError, ValueError, cv2.error):
                        gray = None
                item = next(r for r in frame["centerlines"] if r["roi_idx"] == idx)
                jpeg = _review_crop(gray, item) if gray is not None else None
                if jpeg is None:
                    report["sample_errors"].append(dict(frame=n, roi=idx, reason="image or crop unavailable"))
                    continue
                name = f"samples/F{n:06d}_R{idx}_{item['status']}.jpg"
                z.writestr(name, jpeg)
                report["samples"].append(name)
            z.writestr("summary.json", json.dumps(report, ensure_ascii=False, indent=2))
        tmp_path.replace(path)
    finally:
        tmp_path.unlink(missing_ok=True)
