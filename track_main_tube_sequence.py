#!/usr/bin/env python
"""Track first-frame ROIs through an image sequence and summarize centerlines.

This module intentionally keeps sequence tracking separate from
detect_main_tube.py. The existing detector is reused as a black box after each
ROI polygon is propagated to the current frame.
"""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from tempfile import mkdtemp
from typing import Any, Sequence

import cv2
import numpy as np

import detect_main_tube as dmt
from review_window import ReviewWindow
from tracking_diagnostics import add_evidence, write_diagnostics
from tracking_videos import TrackingVideos, make_unique_output_path


DEFAULT_OUTPUT_JSON = "sequence_centerlines.json"
DEFAULT_OUTPUT_DIR = "tracking_results"
DEFAULT_OUTPUT_JSON_STEM = "sequence_centerlines"
DEFAULT_OUTPUT_JSON_SUFFIX = ".json"
DEFAULT_OUTPUT_VIDEO_STEM = "tracking_review"
DEFAULT_OUTPUT_VIDEO_SUFFIX = ".mp4"
CONFIDENCE_RANK = {"failed": 0, "low": 1, "medium": 2, "high": 3}
EXPOSURE_SATURATION_LEVEL = 250


def _leading_timestamp(path: Path) -> tuple[int, int | str, str]:
    match = re.match(r"^(\d+)", path.stem)
    if match:
        return (0, int(match.group(1)), path.name.lower())
    return (1, path.name.lower(), path.name.lower())


def collect_image_paths(image_dir: str | Path, pattern: str, sort_by: str = "timestamp") -> list[Path]:
    root = Path(image_dir)
    paths = [path for path in root.glob(pattern) if path.is_file()]
    if sort_by == "timestamp":
        paths.sort(key=_leading_timestamp)
    elif sort_by == "mtime":
        paths.sort(key=lambda path: (path.stat().st_mtime, path.name.lower()))
    elif sort_by == "name":
        paths.sort(key=lambda path: path.name.lower())
    else:
        raise ValueError(f"Unsupported sort mode: {sort_by}")
    return paths


def default_output_json_name(run_timestamp: str | None = None) -> str:
    timestamp = run_timestamp or make_output_timestamp()
    return f"{DEFAULT_OUTPUT_JSON_STEM}_{timestamp}{DEFAULT_OUTPUT_JSON_SUFFIX}"


def camera_from_path(path: str | Path) -> str | None:
    folders = {"trackingright": "right", "right_camera": "right", "right": "right",
               "trackingleft": "left", "trackingbmp": "left", "testbmp": "left",
               "left_camera": "left", "left": "left"}
    return next((folders[p.lower()] for p in reversed(Path(path).parts) if p.lower() in folders), None)


def resolve_output_json(
    out_json: str | Path | None,
    out_dir: str | Path | None,
    *,
    run_timestamp: str | None = None,
) -> Path:
    default_json = out_json is None
    if out_json:
        path = Path(out_json)
        if out_dir and not path.is_absolute():
            path = Path(out_dir) / path
    else:
        path = Path(out_dir or DEFAULT_OUTPUT_DIR) / default_output_json_name(run_timestamp)

    if not path.is_absolute():
        path = Path.cwd() / path
    return make_unique_output_path(path) if default_json else path


def make_output_timestamp(now: datetime | None = None) -> str:
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def default_output_video_name(run_timestamp: str | None = None) -> str:
    timestamp = run_timestamp or make_output_timestamp()
    return f"{DEFAULT_OUTPUT_VIDEO_STEM}_{timestamp}{DEFAULT_OUTPUT_VIDEO_SUFFIX}"


def resolve_output_video(
    out_video: str | Path | None,
    out_dir: str | Path | None,
    *,
    run_timestamp: str | None = None,
) -> Path:
    default_video = out_video is None
    path = Path(out_video) if out_video else Path(out_dir or DEFAULT_OUTPUT_DIR) / default_output_video_name(run_timestamp)
    if out_video and out_dir and not path.is_absolute():
        path = Path(out_dir) / path
    if not path.is_absolute():
        path = Path.cwd() / path
    return make_unique_output_path(path) if default_video else path


def _tracking_text(tracking: list[dict[str, Any]]) -> str:
    if not tracking:
        return "ROI motion: initial"
    ok_count = sum(1 for item in tracking if item.get("status") == "ok")
    return f"ROI motion: {ok_count}/{len(tracking)} ok"


def _needs_pause(results: list[dict[str, Any]], pause_on: str) -> bool:
    if pause_on == "none":
        return False
    if pause_on == "failed":
        return any(result.get("status") != "ok" for result in results)
    if pause_on == "low":
        return any(result.get("status") != "ok" or result.get("confidence") == "low" for result in results)
    return False


def make_review_frame(
    gray: np.ndarray,
    results: list[dict[str, Any]],
    *,
    image_path: str | Path,
    frame_index: int,
    total_frames: int,
    tracking: list[dict[str, Any]],
    video_scale: float = 1.0,
    review_polygons=None,
) -> np.ndarray:
    frame = dmt.draw_overlay(gray, results, review_polygons=review_polygons)
    ok_count = sum(1 for result in results if result.get("status") == "ok")
    header = (
        f"{frame_index + 1}/{total_frames}  {Path(image_path).name}  "
        f"visible: {ok_count}/{len(results)}   lost: {len(results) - ok_count}   {_tracking_text(tracking)}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(frame, header, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    return _scale_review_frame(frame, video_scale)


def _scale_review_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = max(1, int(round(frame.shape[1] * scale)))
    height = max(1, int(round(frame.shape[0] * scale)))
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def _emit_progress(
    frame_index: int,
    total_frames: int,
    image_path: str | Path,
    results: list[dict[str, Any]],
    progress_every: int,
) -> None:
    if progress_every <= 0:
        return
    frame_number = frame_index + 1
    if frame_number != total_frames and frame_number % progress_every != 0:
        return
    ok_count = sum(1 for result in results if result.get("status") == "ok")
    print(
        f"[{frame_number}/{total_frames}] {Path(image_path).name} ok={ok_count}/{len(results)}",
        flush=True,
    )


def make_frame_centerline_summary(
    image_path: str | Path,
    rois_source: str | Path,
    results: list[dict[str, Any]],
    tracking: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    summary = dmt.make_centerline_summary(image_path, rois_source, results)
    tracking_by_idx = {int(item.get("roi_idx", 0)): item for item in tracking or []}
    results_by_idx = {int(item.get("roi_idx", 0)): item for item in results}

    for item in summary.get("centerlines", []):
        roi_idx = int(item.get("roi_idx", 0))
        result = results_by_idx.get(roi_idx, {})
        if result.get("vertices") is not None:
            item["vertices"] = _roi_copy(result)["vertices"]
        if result.get("failure_reason") and "failure_reason" not in item:
            item["failure_reason"] = result["failure_reason"]
        if result.get("fallback_reason"):
            item["fallback_reason"] = result["fallback_reason"]
        for key in (
            "wall_pair_coverage",
            "centerline_guard_status",
            "centerline_lateral_jump",
            "centerline_lateral_jump_limit",
            "centerline_original_source",
        ):
            if key in result:
                item[key] = result[key]
        if roi_idx in tracking_by_idx:
            item["tracking"] = dict(tracking_by_idx[roi_idx])

    return summary


def make_sequence_centerline_payload(
    frames: list[dict[str, Any]],
    initial_rois: str | Path,
) -> dict[str, Any]:
    return {
        "initial_rois": str(initial_rois),
        "num_frames": len(frames),
        "frames": frames,
    }


def _roi_feature_mask(shape: tuple[int, int], vertices: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    pts = vertices.round().astype(np.int32)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def _detect_roi_features(gray: np.ndarray, vertices: np.ndarray, max_corners: int) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    height, width = gray.shape[:2]
    block_size = 5
    padding = block_size // 2 + 2
    x0 = max(0, int(np.floor(float(points[:, 0].min()))) - padding)
    y0 = max(0, int(np.floor(float(points[:, 1].min()))) - padding)
    x1 = min(width, int(np.ceil(float(points[:, 0].max()))) + 1 + padding)
    y1 = min(height, int(np.ceil(float(points[:, 1].max()))) + 1 + padding)
    if x1 <= x0 or y1 <= y0:
        return np.empty((0, 2), dtype=np.float32)

    offset = np.asarray([x0, y0], dtype=np.float32)
    crop = gray[y0:y1, x0:x1]
    mask = _roi_feature_mask(crop.shape, points - offset)
    points = cv2.goodFeaturesToTrack(
        crop,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=block_size,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32) + offset


def _template_match_roi_transform(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    vertices: np.ndarray,
    max_roi_jump: float | None,
    *,
    min_score: float = 0.40,
    context: bool = False,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    if context:
        center, (a, b), angle = cv2.minAreaRect(points)
        pad = min(a, b)
        points = cv2.boxPoints((center, (a + 2 * pad, b + 2 * pad), angle))
    height, width = prev_gray.shape[:2]
    x0 = max(0, int(np.floor(float(points[:, 0].min()))))
    y0 = max(0, int(np.floor(float(points[:, 1].min()))))
    x1 = min(width, int(np.ceil(float(points[:, 0].max()))) + 1)
    y1 = min(height, int(np.ceil(float(points[:, 1].max()))) + 1)
    template_width = x1 - x0
    template_height = y1 - y0
    if template_width < 8 or template_height < 8:
        return None, {
            "status": "failed",
            "method": "template",
            "failure_reason": "roi template too small",
        }

    margin = int(np.ceil(max_roi_jump if max_roi_jump is not None else max(template_width, template_height)))
    margin = max(8, margin)
    sx0 = max(0, x0 - margin)
    sy0 = max(0, y0 - margin)
    sx1 = min(next_gray.shape[1], x1 + margin)
    sy1 = min(next_gray.shape[0], y1 + margin)
    search = next_gray[sy0:sy1, sx0:sx1]
    if search.shape[0] < template_height or search.shape[1] < template_width:
        return None, {
            "status": "failed",
            "method": "template",
            "failure_reason": "template search area too small",
        }

    template = prev_gray[y0:y1, x0:x1]
    mask = np.zeros((template_height, template_width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(points - np.asarray([x0, y0], dtype=np.float32)).astype(np.int32)], 255)
    if int(np.count_nonzero(mask)) < 16:
        mask = np.full_like(mask, 255)

    try:
        scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED, mask=mask)
    except cv2.error:
        scores = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
    scores = np.nan_to_num(scores, nan=-1.0, posinf=-1.0, neginf=-1.0)
    _, max_value, _, max_loc = cv2.minMaxLoc(scores)
    score = float(max_value)
    if score < min_score:
        return None, {
            "status": "failed",
            "method": "template",
            "template_score": score,
            "failure_reason": "template match score too low",
        }

    # A featureless wall gives the same match at many offsets. Moving the
    # search box to an arbitrary maximum can prevent a visible tube returning.
    ys, xs = np.nonzero(scores >= score - 1e-6)
    if np.ptp(xs) > 2 or np.ptp(ys) > 2:
        return None, {"status": "failed", "method": "template", "template_score": score,
                      "failure_reason": "ambiguous template position"}

    dx = float(sx0 + max_loc[0] - x0)
    dy = float(sy0 + max_loc[1] - y0)
    matrix = np.asarray([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32)
    return matrix, {
        "status": "ok",
        "method": "template",
        "template_score": round(score, 6),
        "num_features": 0,
        "num_tracked": 0,
        "num_inliers": 0,
    }


def _estimate_roi_transform(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    vertices: np.ndarray,
    max_corners: int,
    min_track_points: int,
    max_forward_backward_error: float,
    max_roi_jump: float | None = None,
    lk_window_size: int = 31,
    lk_max_level: int = 4,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prev_points = _detect_roi_features(prev_gray, vertices, max_corners)
    if len(prev_points) < min_track_points:
        template_matrix, template_info = _template_match_roi_transform(prev_gray, next_gray, vertices, max_roi_jump)
        if template_matrix is not None:
            return template_matrix, {**template_info, "lk_failure_reason": "not enough features in roi"}
        return None, {
            "status": "failed",
            "method": "lk",
            "num_features": int(len(prev_points)),
            "num_tracked": 0,
            "num_inliers": 0,
            "failure_reason": "not enough features in roi",
            "template_failure_reason": template_info.get("failure_reason"),
        }

    lk_window_size = max(5, int(lk_window_size))
    if lk_window_size % 2 == 0:
        lk_window_size += 1
    # LK builds image pyramids on every call. Keep the displacement allowance
    # and the coarsest window's support around the ROI, in both frames.
    flow_prev, flow_next = prev_gray, next_gray
    origin = np.zeros(2, dtype=np.float32)
    if max_roi_jump is not None:
        height, width = prev_gray.shape
        stride = 1 << min(int(lk_max_level), max(height, width).bit_length())
        margin = int(np.ceil(max_roi_jump)) + (lk_window_size // 2 + 3) * stride
        lo = np.maximum(0, (np.floor(prev_points.min(axis=0)).astype(int) - margin) // stride * stride)
        hi = np.ceil(prev_points.max(axis=0)).astype(int) + margin + 1
        region = np.s_[lo[1]:hi[1], lo[0]:hi[0]]
        origin = lo.astype(np.float32)
        flow_prev, flow_next = prev_gray[region], next_gray[region]
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        flow_prev,
        flow_next,
        (prev_points - origin).reshape(-1, 1, 2),
        None,
        winSize=(lk_window_size, lk_window_size),
        maxLevel=int(lk_max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if next_points is None or status is None:
        template_matrix, template_info = _template_match_roi_transform(prev_gray, next_gray, vertices, max_roi_jump)
        if template_matrix is not None:
            return template_matrix, {**template_info, "lk_failure_reason": "optical flow failed"}
        return None, {
            "status": "failed",
            "method": "lk",
            "num_features": int(len(prev_points)),
            "num_tracked": 0,
            "num_inliers": 0,
            "failure_reason": "optical flow failed",
            "template_failure_reason": template_info.get("failure_reason"),
        }

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        flow_next,
        flow_prev,
        next_points,
        None,
        winSize=(lk_window_size, lk_window_size),
        maxLevel=int(lk_max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    prev_flat = prev_points.reshape(-1, 2)
    next_flat = next_points.reshape(-1, 2) + origin
    good = status.reshape(-1).astype(bool)
    if back_points is not None and back_status is not None:
        back_flat = back_points.reshape(-1, 2) + origin
        fb_error = np.linalg.norm(back_flat - prev_flat, axis=1)
        good &= back_status.reshape(-1).astype(bool)
        good &= fb_error <= max_forward_backward_error

    tracked_prev = prev_flat[good]
    tracked_next = next_flat[good]
    if len(tracked_prev) < min_track_points:
        template_matrix, template_info = _template_match_roi_transform(prev_gray, next_gray, vertices, max_roi_jump)
        if template_matrix is not None:
            return template_matrix, {**template_info, "lk_failure_reason": "not enough valid tracked points"}
        return None, {
            "status": "failed",
            "method": "lk",
            "num_features": int(len(prev_points)),
            "num_tracked": int(len(tracked_prev)),
            "num_inliers": 0,
            "failure_reason": "not enough valid tracked points",
            "template_failure_reason": template_info.get("failure_reason"),
        }

    matrix, inliers = cv2.estimateAffinePartial2D(
        tracked_prev,
        tracked_next,
        method=cv2.RANSAC,
        ransacReprojThreshold=3.0,
        maxIters=2000,
        confidence=0.99,
        refineIters=10,
    )
    if matrix is not None and inliers is not None and np.any(inliers):
        keep = inliers.ravel().astype(bool)
        tracked_prev, tracked_next = tracked_prev[keep], tracked_next[keep]
    inlier_count = len(tracked_prev)
    # A cluster at one tube end measures local translation, but extrapolating
    # its scale and rotation can move the opposite wall out of the search box.
    footprint = np.linalg.norm(np.ptp(tracked_prev, axis=0))
    if matrix is None or footprint < np.linalg.norm(np.ptp(vertices, axis=0)) / 3:
        delta = np.median(tracked_next - tracked_prev, axis=0)
        matrix = np.asarray([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]], dtype=np.float32)

    if inlier_count < max(8, .55 * len(prev_points)):
        center, (a, b), _ = cv2.minAreaRect(vertices)
        shift = matrix[:, :2] @ center + matrix[:, 2] - center
        zoom = np.linalg.norm(matrix[0, :2])
        # Clusters on wires near opposite ends can imply a large zoom or
        # slide along a nearly textureless pipe. Include its surroundings
        # to distinguish translations that match the pipe equally well.
        if not .90 <= zoom <= 1.10 or np.linalg.norm(shift) > max(8., .35 * min(a, b)):
            return _template_match_roi_transform(prev_gray, next_gray, vertices, max_roi_jump, context=True)

    return matrix.astype(np.float32), {
        "status": "ok",
        "method": "lk",
        "num_features": int(len(prev_points)),
        "num_tracked": int(len(tracked_prev)),
        "num_inliers": inlier_count,
    }


def _transform_vertices(vertices: np.ndarray, matrix: np.ndarray) -> list[list[float]]:
    transformed = cv2.transform(vertices.reshape(1, -1, 2).astype(np.float32), matrix)[0]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in transformed]


def _roi_vertices_center(vertices: np.ndarray) -> np.ndarray:
    return np.asarray(vertices, dtype=np.float32).reshape(-1, 2).mean(axis=0)


def _roi_vertices_span(vertices: np.ndarray) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    return points.max(axis=0) - points.min(axis=0)


def _tracked_roi_rejection_reason(
    previous_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    max_roi_jump: float | None,
    *,
    roi_kind: str = "tracked",
) -> str | None:
    candidate = np.asarray(candidate_vertices, dtype=np.float32).reshape(-1, 2)
    if not np.all(np.isfinite(candidate)):
        return f"non-finite {roi_kind} roi"

    if max_roi_jump is not None:
        jump = float(np.linalg.norm(_roi_vertices_center(candidate) - _roi_vertices_center(previous_vertices)))
        if jump > max_roi_jump:
            return f"unreasonable {roi_kind} roi"

        previous_span = _roi_vertices_span(previous_vertices)
        candidate_span = _roi_vertices_span(candidate)
        max_span = np.maximum(previous_span * 8.0, previous_span + 2.0 * float(max_roi_jump))
        max_span = np.minimum(max_span, 30000.0)
        if np.any(candidate_span > max_span):
            return f"unreasonable {roi_kind} roi"

    return None


def _guard_group_motion_outliers(
    tracked_rois: list[dict[str, Any]],
    tracking: list[dict[str, Any]],
    previous_vertices_by_idx: dict[int, np.ndarray],
    *,
    min_abs_dy: float,
    max_dy_deviation: float,
) -> None:
    if min_abs_dy <= 0 or max_dy_deviation <= 0:
        return

    dy_values: list[float] = []
    for item in tracking:
        if item.get("status") != "ok":
            continue
        dy = item.get("track_center_dy")
        if dy is None:
            continue
        dy_float = float(dy)
        if np.isfinite(dy_float):
            dy_values.append(dy_float)

    if len(dy_values) < 3:
        return

    median_dy = float(np.median(np.asarray(dy_values, dtype=np.float32)))
    rois_by_idx = {int(roi["roi_idx"]): roi for roi in tracked_rois}
    for item in tracking:
        if item.get("status") != "ok":
            continue
        dy = item.get("track_center_dy")
        if dy is None:
            continue
        dy_float = float(dy)
        if not np.isfinite(dy_float):
            continue
        if abs(dy_float) < float(min_abs_dy):
            continue
        if abs(dy_float - median_dy) <= float(max_dy_deviation):
            continue

        roi_idx = int(item.get("roi_idx", 0))
        previous_vertices = previous_vertices_by_idx.get(roi_idx)
        tracked_roi = rois_by_idx.get(roi_idx)
        if previous_vertices is None or tracked_roi is None:
            continue

        tracked_roi["vertices"] = [
            [round(float(x), 3), round(float(y), 3)]
            for x, y in np.asarray(previous_vertices, dtype=np.float32).reshape(-1, 2)
        ]
        item.update(
            {
                "status": "failed",
                "failure_reason": "inconsistent group motion",
                "group_motion_median_dy": round(median_dy, 3),
                "group_motion_outlier_dy": round(dy_float, 3),
            }
        )


def track_rois_between_frames(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    roi_payload: dict[str, Any],
    *,
    max_corners: int = 80,
    min_track_points: int = 3,
    max_forward_backward_error: float = 1.5,
    track_scale: float = 0.5,
    max_roi_jump: float | None = None,
    lk_window_size: int = 31,
    lk_max_level: int = 4,
    track_group_outlier_min_dy: float = 0.0,
    track_group_outlier_dy_deviation: float = 45.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if track_scale <= 0:
        raise ValueError("track_scale must be positive.")

    if track_scale < 1.0:
        scaled_width = max(1, int(round(prev_gray.shape[1] * track_scale)))
        scaled_height = max(1, int(round(prev_gray.shape[0] * track_scale)))
        prev_track = cv2.resize(prev_gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        next_track = cv2.resize(next_gray, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
        coord_scale = float(track_scale)
    else:
        prev_track = prev_gray
        next_track = next_gray
        coord_scale = 1.0

    normalized = dmt.normalize_roi_payload(roi_payload)
    tracked_rois: list[dict[str, Any]] = []
    tracking: list[dict[str, Any]] = []
    previous_vertices_by_idx: dict[int, np.ndarray] = {}

    for roi in normalized.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        vertices = np.asarray(roi["vertices"], dtype=np.float32)
        previous_vertices_by_idx[roi_idx] = vertices.copy()
        track_vertices = vertices * coord_scale
        matrix, info = _estimate_roi_transform(
            prev_track,
            next_track,
            track_vertices,
            max_corners=max_corners,
            min_track_points=min_track_points,
            max_forward_backward_error=max_forward_backward_error,
            max_roi_jump=float(max_roi_jump) * coord_scale if max_roi_jump is not None else None,
            lk_window_size=lk_window_size,
            lk_max_level=lk_max_level,
        )
        if matrix is None:
            tracked_vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in vertices]
        else:
            scaled_tracked = cv2.transform(track_vertices.reshape(1, -1, 2).astype(np.float32), matrix)[0]
            restored = scaled_tracked / coord_scale
            center_delta = _roi_vertices_center(restored) - _roi_vertices_center(vertices)
            info = {
                **info,
                "track_center_dx": round(float(center_delta[0]), 3),
                "track_center_dy": round(float(center_delta[1]), 3),
            }
            rejection_reason = _tracked_roi_rejection_reason(vertices, restored, max_roi_jump)
            if rejection_reason is not None:
                tracked_vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in vertices]
                info = {
                    **info,
                    "status": "failed",
                    "failure_reason": rejection_reason,
                }
            else:
                tracked_vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in restored]

        tracked_rois.append({"roi_idx": roi_idx, "vertices": tracked_vertices})
        tracking.append({"roi_idx": roi_idx, "track_scale": float(coord_scale), **info})

    _guard_group_motion_outliers(
        tracked_rois,
        tracking,
        previous_vertices_by_idx,
        min_abs_dy=track_group_outlier_min_dy,
        max_dy_deviation=track_group_outlier_dy_deviation,
    )

    return {"num_rois": len(tracked_rois), "rois": tracked_rois}, tracking


def _roi_copy(roi: dict[str, Any]) -> dict[str, Any]:
    vertices = np.asarray(roi["vertices"], dtype=np.float32)
    return {
        "roi_idx": int(roi["roi_idx"]),
        "vertices": [[round(float(x), 3), round(float(y), 3)] for x, y in vertices],
    }


def _roi_center(roi: dict[str, Any]) -> np.ndarray:
    vertices = np.asarray(roi["vertices"], dtype=np.float32)
    return vertices.mean(axis=0)


def _roi_short_span(vertices: Sequence[Sequence[float]]) -> float:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    axes = _roi_axes(points)
    if axes is None:
        return float("nan")

    center, _, short_axis = axes
    short_proj = (points - center) @ short_axis
    return float(short_proj.max() - short_proj.min())


def _roi_long_span(vertices: Sequence[Sequence[float]]) -> float:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    axes = _roi_axes(points)
    if axes is None:
        return float("nan")

    center, long_axis, _ = axes
    long_proj = (points - center) @ long_axis
    return float(long_proj.max() - long_proj.min())


def _format_vertices(vertices: np.ndarray) -> list[list[float]]:
    return [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(vertices, dtype=np.float32)]


def _roi_axes(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    if len(points) < 3:
        return None

    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float32)
    sides: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        delta = box[(i + 1) % 4] - box[i]
        length = float(np.linalg.norm(delta))
        if length > 1e-6:
            sides.append((length, delta / length))
    if not sides:
        return None

    sides.sort(key=lambda item: item[0], reverse=True)
    long_axis = sides[0][1].astype(np.float32)
    short_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float32)
    return points.mean(axis=0), long_axis, short_axis


def _preserve_search_extent(roi, width, reference):
    """Keep both walls in the search after many small flow-scale errors."""
    initial_width, initial_length = reference
    points = np.asarray(roi["vertices"], dtype=np.float32)
    axes = _roi_axes(points)
    if axes is None:
        return
    center, along, across = axes
    xy = points - center
    length, breadth = np.ptp(xy @ along), np.ptp(xy @ across)
    if min(length, breadth) < 1:
        return
    # Allow foreshortening and zoom. Scale the original selected length by
    # observed pipe width, independent of the number of processed frames.
    long_gain = max(1., .85 * initial_length * width / initial_width / length)
    short_gain = max(1., 1.30 * width / breadth)
    if long_gain > 1.001 or short_gain > 1.001:
        points = center + (xy @ along)[:, None] * along * long_gain + (xy @ across)[:, None] * across * short_gain
        roi["vertices"] = _format_vertices(points)


def _ensure_min_roi_short_span(vertices: list[list[float]], min_short_span: float) -> list[list[float]]:
    if min_short_span <= 0:
        return vertices

    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    axes = _roi_axes(points)
    if axes is None:
        return vertices

    center, long_axis, short_axis = axes
    rel = points - center
    long_proj = rel @ long_axis
    short_proj = rel @ short_axis
    long_min = float(long_proj.min())
    long_max = float(long_proj.max())
    short_min = float(short_proj.min())
    short_max = float(short_proj.max())
    short_span = short_max - short_min
    if short_span >= min_short_span:
        return vertices

    short_mid = 0.5 * (short_min + short_max)
    half_short = 0.5 * float(min_short_span)
    short_min = short_mid - half_short
    short_max = short_mid + half_short
    expanded = np.asarray(
        [
            center + long_axis * long_min + short_axis * short_min,
            center + long_axis * long_min + short_axis * short_max,
            center + long_axis * long_max + short_axis * short_max,
            center + long_axis * long_max + short_axis * short_min,
        ],
        dtype=np.float32,
    )
    return _format_vertices(expanded)


def _cap_roi_growth_to_reference(
    candidate_vertices: list[list[float]],
    reference_vertices: list[list[float]],
    *,
    max_growth: float | None,
    min_short_span: float,
) -> list[list[float]]:
    if max_growth is None or max_growth <= 0:
        return candidate_vertices

    candidate = np.asarray(candidate_vertices, dtype=np.float32).reshape(-1, 2)
    reference = np.asarray(reference_vertices, dtype=np.float32).reshape(-1, 2)
    axes = _roi_axes(candidate)
    if axes is None:
        return candidate_vertices

    center, long_axis, short_axis = axes
    candidate_rel = candidate - center
    reference_rel = reference - center

    candidate_long = candidate_rel @ long_axis
    candidate_short = candidate_rel @ short_axis
    reference_long = reference_rel @ long_axis
    reference_short = reference_rel @ short_axis

    long_min = float(candidate_long.min())
    long_max = float(candidate_long.max())
    short_min = float(candidate_short.min())
    short_max = float(candidate_short.max())
    long_mid = 0.5 * (long_min + long_max)
    short_mid = 0.5 * (short_min + short_max)

    candidate_long_span = max(0.0, long_max - long_min)
    candidate_short_span = max(0.0, short_max - short_min)
    reference_long_span = max(0.0, float(reference_long.max() - reference_long.min()))
    reference_short_span = max(0.0, float(reference_short.max() - reference_short.min()))

    capped_long_span = min(candidate_long_span, reference_long_span * float(max_growth))
    capped_short_span = min(candidate_short_span, reference_short_span * float(max_growth))
    capped_short_span = max(capped_short_span, max(0.0, float(min_short_span)))
    if capped_long_span <= 1e-6 or capped_short_span <= 1e-6:
        return candidate_vertices

    if (
        abs(capped_long_span - candidate_long_span) <= 1e-3
        and abs(capped_short_span - candidate_short_span) <= 1e-3
    ):
        return candidate_vertices

    long_min = long_mid - 0.5 * capped_long_span
    long_max = long_mid + 0.5 * capped_long_span
    short_min = short_mid - 0.5 * capped_short_span
    short_max = short_mid + 0.5 * capped_short_span
    capped = np.asarray(
        [
            center + long_axis * long_min + short_axis * short_min,
            center + long_axis * long_min + short_axis * short_max,
            center + long_axis * long_max + short_axis * short_max,
            center + long_axis * long_max + short_axis * short_min,
        ],
        dtype=np.float32,
    )
    return _format_vertices(capped)


def _confidence_meets(value: str | None, minimum: str) -> bool:
    return CONFIDENCE_RANK.get(str(value or "high"), 0) >= CONFIDENCE_RANK[str(minimum)]


def measure_exposure_stats(gray: np.ndarray, *, sample_stride: int = 8) -> dict[str, float]:
    stride = max(1, int(sample_stride))
    sample = np.asarray(gray)[::stride, ::stride]
    if sample.size == 0:
        return {"mean": float("nan"), "p99": float("nan"), "saturated_frac": float("nan")}

    sample_f = sample.astype(np.float32, copy=False)
    return {
        "mean": round(float(sample_f.mean()), 3),
        "p99": round(float(np.percentile(sample_f, 99)), 3),
        "saturated_frac": round(float(np.mean(sample_f >= EXPOSURE_SATURATION_LEVEL)), 6),
    }


def exposure_suspect_info(
    previous_stats: dict[str, float] | None,
    current_stats: dict[str, float],
    *,
    mean_jump: float,
    saturation_jump: float,
    bright_mean: float,
    saturation_frac: float,
) -> tuple[bool, dict[str, float | str]]:
    previous_mean = float(previous_stats.get("mean", float("nan"))) if previous_stats else float("nan")
    previous_saturated = (
        float(previous_stats.get("saturated_frac", float("nan"))) if previous_stats else float("nan")
    )
    current_mean = float(current_stats.get("mean", float("nan")))
    current_p99 = float(current_stats.get("p99", float("nan")))
    current_saturated = float(current_stats.get("saturated_frac", float("nan")))

    mean_delta = current_mean - previous_mean if np.isfinite(previous_mean) and np.isfinite(current_mean) else 0.0
    saturation_delta = (
        current_saturated - previous_saturated
        if np.isfinite(previous_saturated) and np.isfinite(current_saturated)
        else 0.0
    )
    near_clip = np.isfinite(current_p99) and current_p99 >= 254.0
    jump_suspect = near_clip and (
        mean_delta >= float(mean_jump) or saturation_delta >= float(saturation_jump)
    )
    absolute_suspect = (
        near_clip
        and np.isfinite(current_mean)
        and np.isfinite(current_saturated)
        and current_mean >= float(bright_mean)
        and current_saturated >= float(saturation_frac)
    )
    suspected = bool(jump_suspect or absolute_suspect)
    return suspected, {
        "exposure_status": "suspect" if suspected else "normal",
        "exposure_mean": round(current_mean, 3),
        "exposure_mean_delta": round(float(mean_delta), 3),
        "exposure_p99": round(current_p99, 3),
        "exposure_saturated_frac": round(current_saturated, 6),
        "exposure_saturated_delta": round(float(saturation_delta), 6),
    }


def correct_roi_from_detection(
    predicted_roi: dict[str, Any],
    result: dict[str, Any] | None,
    *,
    padding_frac: float,
    length_padding_frac: float,
    max_roi_jump: float,
    min_update_confidence: str = "low",
    min_corrected_roi_short_span: float = 0.0,
    max_corrected_roi_growth: float | None = None,
    min_corrected_roi_long_span_ratio: float = 0.0,
    min_corrected_roi_long_span_drop: float = 0.0,
    anchor_roi: dict[str, Any] | None = None,
    max_anchor_drift: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    roi = _roi_copy(predicted_roi)
    roi_idx = int(roi["roi_idx"])
    info: dict[str, Any] = {"roi_idx": roi_idx}

    if result is None:
        return roi, {**info, "update_status": "missing-detection"}
    if result.get("status") != "ok":
        return roi, {**info, "update_status": "detection-failed"}
    if not _confidence_meets(result.get("confidence"), min_update_confidence):
        return roi, {
            **info,
            "update_status": "low-confidence-detection",
            "detection_confidence": result.get("confidence", "failed"),
        }

    corrected_vertices = dmt.roi_vertices_from_detected_boundaries(
        result,
        padding_frac=padding_frac,
        length_padding_frac=length_padding_frac,
    )
    if corrected_vertices is None:
        return roi, {**info, "update_status": "correction-unavailable"}

    corrected_vertices = _ensure_min_roi_short_span(corrected_vertices, min_corrected_roi_short_span)
    corrected_vertices = _cap_roi_growth_to_reference(
        corrected_vertices,
        roi["vertices"],
        max_growth=max_corrected_roi_growth,
        min_short_span=min_corrected_roi_short_span,
    )
    reference_long_span = _roi_long_span(roi["vertices"])
    corrected_long_span = _roi_long_span(corrected_vertices)
    long_span_drop = reference_long_span - corrected_long_span
    if (
        min_corrected_roi_long_span_ratio > 0
        and np.isfinite(reference_long_span)
        and np.isfinite(corrected_long_span)
        and reference_long_span > 1e-6
        and corrected_long_span / reference_long_span < float(min_corrected_roi_long_span_ratio)
        and long_span_drop >= float(min_corrected_roi_long_span_drop)
    ):
        return roi, {
            **info,
            "update_status": "rejected-corrected-roi-shrink",
            "roi_long_span_before": round(float(reference_long_span), 3),
            "roi_long_span_after": round(float(corrected_long_span), 3),
            "roi_long_span_ratio": round(float(corrected_long_span / reference_long_span), 6),
            "roi_long_span_drop": round(float(long_span_drop), 3),
        }

    corrected_roi = {"roi_idx": roi_idx, "vertices": corrected_vertices}
    jump = float(np.linalg.norm(_roi_center(corrected_roi) - _roi_center(roi)))
    if jump > max_roi_jump:
        return roi, {**info, "update_status": "rejected-roi-jump", "roi_center_jump": jump}

    rejection_reason = _tracked_roi_rejection_reason(
        np.asarray(roi["vertices"], dtype=np.float32),
        np.asarray(corrected_vertices, dtype=np.float32),
        max_roi_jump,
        roi_kind="corrected",
    )
    if rejection_reason is not None:
        return roi, {
            **info,
            "update_status": "rejected-corrected-roi",
            "rejection_reason": rejection_reason,
            "roi_center_jump": jump,
        }

    if anchor_roi is not None and max_anchor_drift is not None and max_anchor_drift > 0:
        anchor = _roi_copy(anchor_roi)
        anchor_drift = float(np.linalg.norm(_roi_center(corrected_roi) - _roi_center(anchor)))
        if anchor_drift > max_anchor_drift:
            return anchor, {
                **info,
                "update_status": "rejected-anchor-drift",
                "roi_center_jump": jump,
                "anchor_center_drift": anchor_drift,
            }

    return corrected_roi, {**info, "update_status": "detect-corrected", "roi_center_jump": jump}


def correct_rois_from_detections(
    predicted_payload: dict[str, Any],
    results: list[dict[str, Any]],
    *,
    padding_frac: float,
    length_padding_frac: float,
    max_roi_jump: float,
    min_update_confidence: str = "low",
    min_corrected_roi_short_span: float = 0.0,
    max_corrected_roi_growth: float | None = None,
    min_corrected_roi_long_span_ratio: float = 0.0,
    min_corrected_roi_long_span_drop: float = 0.0,
    anchor_payload: dict[str, Any] | None = None,
    max_anchor_drift: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = dmt.normalize_roi_payload(predicted_payload)
    results_by_idx = {int(result.get("roi_idx", 0)): result for result in results}
    anchors_by_idx = (
        {int(roi["roi_idx"]): roi for roi in dmt.normalize_roi_payload(anchor_payload).get("rois", [])}
        if anchor_payload is not None
        else {}
    )
    corrected_rois: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    for roi in normalized.get("rois", []):
        corrected_roi, update = correct_roi_from_detection(
            roi,
            results_by_idx.get(int(roi["roi_idx"])),
            padding_frac=padding_frac,
            length_padding_frac=length_padding_frac,
            max_roi_jump=max_roi_jump,
            min_update_confidence=min_update_confidence,
            min_corrected_roi_short_span=min_corrected_roi_short_span,
            max_corrected_roi_growth=max_corrected_roi_growth,
            min_corrected_roi_long_span_ratio=min_corrected_roi_long_span_ratio,
            min_corrected_roi_long_span_drop=min_corrected_roi_long_span_drop,
            anchor_roi=anchors_by_idx.get(int(roi["roi_idx"])),
            max_anchor_drift=max_anchor_drift,
        )
        corrected_rois.append(corrected_roi)
        updates.append(update)

    return {"num_rois": len(corrected_rois), "rois": corrected_rois}, updates


def guard_exposure_shrunken_rois(
    predicted_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    updates: list[dict[str, Any]],
    *,
    exposure_active: bool,
    exposure_info: dict[str, float | str],
    shrink_ratio_threshold: float,
    shrink_pixels_threshold: float,
    min_shrunk_rois: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predicted_rois = {
        int(roi["roi_idx"]): _roi_copy(roi)
        for roi in dmt.normalize_roi_payload(predicted_payload).get("rois", [])
    }
    corrected = dmt.normalize_roi_payload(corrected_payload)

    shrunk_by_idx: dict[int, dict[str, float]] = {}
    for roi in corrected.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        predicted_roi = predicted_rois.get(roi_idx)
        if predicted_roi is None:
            continue

        reference_short = _roi_short_span(predicted_roi["vertices"])
        corrected_short = _roi_short_span(roi["vertices"])
        if not np.isfinite(reference_short) or not np.isfinite(corrected_short) or reference_short <= 1e-6:
            continue

        short_drop = reference_short - corrected_short
        short_ratio = corrected_short / reference_short
        if short_ratio <= float(shrink_ratio_threshold) and short_drop >= float(shrink_pixels_threshold):
            shrunk_by_idx[roi_idx] = {
                "roi_short_span_before": reference_short,
                "roi_short_span_after": corrected_short,
                "roi_short_span_ratio": short_ratio,
                "roi_short_span_drop": short_drop,
            }

    should_guard = bool(shrunk_by_idx) and (
        bool(exposure_active) or len(shrunk_by_idx) >= max(1, int(min_shrunk_rois))
    )
    if not should_guard:
        return corrected, [dict(update) for update in updates]

    guarded_rois: list[dict[str, Any]] = []
    for roi in corrected.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        guarded_rois.append(predicted_rois[roi_idx] if roi_idx in shrunk_by_idx else _roi_copy(roi))

    guarded_updates: list[dict[str, Any]] = []
    for update in updates:
        update_copy = dict(update)
        roi_idx = int(update_copy["roi_idx"])
        shrink = shrunk_by_idx.get(roi_idx)
        if shrink is not None:
            update_copy = {
                **update_copy,
                "update_status": "rejected-exposure-shrink",
                "exposure_rejected_update_status": update_copy.get("update_status", "unknown"),
                "exposure_guard_status": "roi-width-shrink",
                **{key: round(float(value), 6) for key, value in shrink.items()},
                **exposure_info,
            }
        guarded_updates.append(update_copy)

    return {"num_rois": len(guarded_rois), "rois": guarded_rois}, guarded_updates


def _roi_span_stats(roi: dict[str, Any]) -> dict[str, float]:
    vertices = roi.get("vertices", [])
    return {
        "long_span": _roi_long_span(vertices),
        "short_span": _roi_short_span(vertices),
    }


def update_roi_span_history(
    history: dict[int, dict[str, float]],
    roi_payload: dict[str, Any],
) -> dict[int, dict[str, float]]:
    updated = {roi_idx: dict(values) for roi_idx, values in history.items()}
    for roi in dmt.normalize_roi_payload(roi_payload).get("rois", []):
        roi_idx = int(roi["roi_idx"])
        stats = _roi_span_stats(roi)
        long_span = stats["long_span"]
        short_span = stats["short_span"]
        if not np.isfinite(long_span) or not np.isfinite(short_span):
            continue
        previous = updated.get(roi_idx)
        if previous is None or long_span > float(previous.get("long_span", 0.0)):
            updated[roi_idx] = {
                "long_span": float(long_span),
                "short_span": float(short_span),
            }
    return updated


def guard_long_span_collapsed_rois(
    predicted_payload: dict[str, Any],
    corrected_payload: dict[str, Any],
    updates: list[dict[str, Any]],
    span_history: dict[int, dict[str, float]],
    *,
    long_span_ratio_threshold: float,
    long_span_drop_threshold: float,
    short_span_ratio_threshold: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if long_span_ratio_threshold <= 0 or long_span_drop_threshold < 0 or short_span_ratio_threshold <= 0:
        return dmt.normalize_roi_payload(corrected_payload), [dict(update) for update in updates]

    predicted_rois = {
        int(roi["roi_idx"]): _roi_copy(roi)
        for roi in dmt.normalize_roi_payload(predicted_payload).get("rois", [])
    }
    corrected = dmt.normalize_roi_payload(corrected_payload)

    collapsed_by_idx: dict[int, dict[str, float]] = {}
    for roi in corrected.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        history_stats = span_history.get(roi_idx)
        if history_stats is None:
            continue
        history_long = float(history_stats.get("long_span", float("nan")))
        history_short = float(history_stats.get("short_span", float("nan")))
        current_stats = _roi_span_stats(roi)
        current_long = current_stats["long_span"]
        current_short = current_stats["short_span"]
        if (
            not np.isfinite(history_long)
            or not np.isfinite(history_short)
            or not np.isfinite(current_long)
            or not np.isfinite(current_short)
            or history_long <= 1e-6
            or history_short <= 1e-6
        ):
            continue

        long_ratio = current_long / history_long
        short_ratio = current_short / history_short
        long_drop = history_long - current_long
        if (
            long_ratio < float(long_span_ratio_threshold)
            and long_drop >= float(long_span_drop_threshold)
            and short_ratio < float(short_span_ratio_threshold)
        ):
            collapsed_by_idx[roi_idx] = {
                "roi_history_long_span": history_long,
                "roi_long_span_after": current_long,
                "roi_long_span_ratio": long_ratio,
                "roi_long_span_drop": long_drop,
                "roi_history_short_span": history_short,
                "roi_short_span_after": current_short,
                "roi_short_span_ratio": short_ratio,
            }

    if not collapsed_by_idx:
        return corrected, [dict(update) for update in updates]

    guarded_rois: list[dict[str, Any]] = []
    for roi in corrected.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        guarded_rois.append(predicted_rois[roi_idx] if roi_idx in collapsed_by_idx else _roi_copy(roi))

    guarded_updates: list[dict[str, Any]] = []
    for update in updates:
        update_copy = dict(update)
        roi_idx = int(update_copy["roi_idx"])
        collapse = collapsed_by_idx.get(roi_idx)
        if collapse is not None:
            update_copy = {
                **update_copy,
                "update_status": "rejected-long-span-history-shrink",
                "span_guard_rejected_update_status": update_copy.get("update_status", "unknown"),
                **{key: round(float(value), 6) for key, value in collapse.items()},
            }
        guarded_updates.append(update_copy)

    return {"num_rois": len(guarded_rois), "rois": guarded_rois}, guarded_updates


def _apply_roi_vertices_to_results(results: list[dict[str, Any]], roi_payload: dict[str, Any]) -> None:
    rois_by_idx = {int(roi["roi_idx"]): roi for roi in roi_payload.get("rois", [])}
    for result in results:
        roi = rois_by_idx.get(int(result.get("roi_idx", 0)))
        if roi is not None:
            result["vertices"] = _roi_copy(roi)["vertices"]


def _merge_roi_update_info(tracking: list[dict[str, Any]], updates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    updates_by_idx = {int(update["roi_idx"]): update for update in updates}
    merged: list[dict[str, Any]] = []
    for item in tracking:
        roi_idx = int(item.get("roi_idx", 0))
        update = updates_by_idx.get(roi_idx, {})
        merged.append({**item, **{k: v for k, v in update.items() if k != "roi_idx"}})
    return merged


def _lost_measurement(result: dict[str, Any], reason: str) -> dict[str, Any]:
    lost = dmt._failed_roi_result(result["roi_idx"], result["vertices"], reason)
    lost.update(status="lost", confidence="lost")
    if "wall_pair_coverage" in result:
        lost["wall_pair_coverage"] = result["wall_pair_coverage"]
    return lost


def assess_visibility(
    results: list[dict[str, Any]],
    candidates: dict[int, dict[str, Any] | None],
    previous_rois: dict[str, Any],
    predicted_rois: dict[str, Any],
    max_lateral_jump: float,
    record_metrics: bool = False,
) -> list[dict[str, Any]]:
    """Only publish current bilateral measurements. Confirm reacquisition on a second frame."""
    old_centers = {r["roi_idx"]: _roi_center(r) for r in previous_rois["rois"]}
    new_centers = {r["roi_idx"]: _roi_center(r) for r in predicted_rois["rois"]}
    visible = []
    for result in results:
        idx = result["roi_idx"]
        if result.get("status") != "ok":
            reason = result.get("failure_reason", "no wall pair")
        elif result.get("confidence") not in {"high", "medium"}:
            reason = "weak wall detection"
        elif result["wall_pair_coverage"] < 0.20:
            reason = "insufficient simultaneous wall contrast"
        else:
            reason = None

        if reason is not None:
            candidates[idx] = None
        else:
            if idx in candidates:
                prev = candidates[idx]
                if prev is None:
                    reason = "awaiting second wall measurement"
                else:
                    # Compensate for search-window motion before checking the lateral shift.
                    line = np.asarray(result["centerline"])
                    direction = line[-1] - line[0]
                    normal = np.array([-direction[1], direction[0]]) / max(np.linalg.norm(direction), 1e-6)
                    motion = new_centers[idx] - old_centers[idx]
                    shift = line.mean(axis=0) - np.mean(prev["centerline"], axis=0) - motion
                    width_ratio = result["tube_width_px"] / prev["tube_width_px"]
                    if record_metrics:
                        result["width_ratio"] = round(width_ratio, 4)
                        result["lateral_shift_px"] = round(float(shift @ normal), 3)
                    if not 2 / 3 <= width_ratio <= 1.5:
                        reason = "inconsistent tube width"
                    elif max_lateral_jump > 0 and abs(shift @ normal) > max_lateral_jump:
                        reason = "inconsistent lateral position"
            # Keep a strong but unconfirmed candidate for one-frame confirmation only.
            candidates[idx] = result
        visible.append(_lost_measurement(result, reason) if reason else result)
    return visible


def reject_invalid_updates(results, updates, candidates, confirmed_widths=None):
    rejected = {u["roi_idx"]: u for u in updates if u["update_status"].startswith("rejected-")}
    for i, result in enumerate(results):
        idx = result["roi_idx"]
        if idx in rejected:
            update = rejected[idx]
            reason = update["update_status"]
            width = (confirmed_widths or {}).get(idx)
            if width and result["status"] == "ok":
                ratio = result["tube_width_px"] / width
                if reason == "rejected-exposure-shrink" and .78 <= ratio <= 1.28:
                    continue
                # Only the search window would shrink. Publish the shorter
                # measured segment if both fitted walls still establish its width.
                if (reason == "rejected-corrected-roi-shrink" and .9 <= ratio <= 1.1
                        and update["roi_long_span_ratio"] >= .60
                        and result.get("segment_wall_pair_coverage", result["wall_pair_coverage"]) >= .50):
                    height = result["rectified_size"]["height"]
                    if all(result.get(key, {}).get("support_rows", 0) >= height / 3
                           for key in ("left_line", "right_line")):
                        continue
            if result["status"] == "ok":
                results[i] = _lost_measurement(result, reason)
            candidates[idx] = None


def load_initial_roi_payload(
    args: argparse.Namespace,
    first_image_path: str | Path,
    out_dir: str | Path,
) -> tuple[dict[str, Any], Path]:
    if getattr(args, "draw_rois", False):
        color_image = dmt.read_image_unicode(first_image_path, cv2.IMREAD_COLOR)
        roi_payload = dmt.normalize_roi_payload(dmt.draw_rois_interactive(first_image_path, color_image))
        if not roi_payload.get("rois"):
            raise ValueError("No ROI was drawn.")
        rois_source = Path(out_dir) / "drawn_rois.json"
        dmt.save_json(rois_source, roi_payload)
        return roi_payload, rois_source

    camera = getattr(args, "camera", None) or camera_from_path(Path(first_image_path).parent)
    if not args.initial_rois:
        first = Path(first_image_path).resolve()
        root = Path(DEFAULT_OUTPUT_DIR)
        candidates = sorted((root / "runs" / f"{camera}_camera").glob("*/drawn_rois.json"), reverse=True)
        if camera == "left" and first.parent.name.lower() == "testbmp":
            candidates.insert(0, root / "drawn_rois.json")
        for candidate in candidates:
            if candidate.is_file():
                saved = dmt.load_roi_json(candidate)
                if saved.get("image") and Path(saved["image"]).resolve() == first:
                    print(f"Using saved ROIs: {candidate}", flush=True)
                    return dmt.normalize_roi_payload(saved), candidate
        raise ValueError("No saved ROIs for this camera and first image. "
                         "Pass -InitialRois or use -Interactive to select the pipes.")
    rois_source = Path(args.initial_rois)
    payload = dmt.normalize_roi_payload(dmt.load_roi_json(rois_source))
    source_camera = camera_from_path(rois_source.parent) or camera_from_path(payload.get("image") or "")
    if source_camera and camera and source_camera != camera:
        raise ValueError(f"{camera} camera input has {source_camera} camera ROIs. "
                         "Select its drawn_rois.json with -InitialRois or draw new ROIs with -Interactive.")
    return payload, rois_source


def run_sequence(args: argparse.Namespace) -> dict[str, Any]:
    if args.image_dir:
        image_paths = collect_image_paths(args.image_dir, args.pattern, args.sort_by)
    else:
        image_paths = [Path(path) for path in args.images]
    max_frames = getattr(args, "max_frames", 0)
    if max_frames:
        image_paths = image_paths[: int(max_frames)]
    if len(image_paths) < 2:
        raise ValueError("At least two ordered frames are required.")

    run_timestamp = getattr(args, "run_timestamp", None) or make_output_timestamp()
    saved_rois = None
    if not getattr(args, "draw_rois", False):
        saved_rois = load_initial_roi_payload(args, image_paths[0], args.out_dir or DEFAULT_OUTPUT_DIR)
    if args.out_dir is None and not args.out_json and not args.out_video:
        camera = getattr(args, "camera", None) or camera_from_path(image_paths[0].parent)
        if camera is None and saved_rois:
            camera = camera_from_path(saved_rois[1].parent) or camera_from_path(saved_rois[0].get("image") or "")
        if camera is None:
            raise ValueError("Unknown camera folder. Pass -Camera left/right to the launcher "
                             "(--camera left/right in Python), or specify an output directory.")
        run_root = Path(DEFAULT_OUTPUT_DIR) / "runs" / f"{camera}_camera"
        run_dir = run_root / f"{run_timestamp}_{datetime.now().microsecond // 1000:03d}"
        try:
            run_dir.mkdir(parents=True)
        except FileExistsError:
            run_dir = Path(mkdtemp(prefix=f"{run_timestamp}_", dir=run_root))
        args.out_dir = run_dir
        print(f"Output folder: {args.out_dir}", flush=True)
    out_json = resolve_output_json(args.out_json, args.out_dir, run_timestamp=run_timestamp)
    args._resolved_out_json = out_json
    args._run_timestamp = run_timestamp
    roi_payload, rois_source = saved_rois or load_initial_roi_payload(args, image_paths[0], out_json.parent)
    config = dmt.DetectionConfig(
        min_width_frac=args.min_width_frac,
        max_width_frac=args.max_width_frac,
        boundary_samples=args.boundary_samples,
        centerline_foreground_correction=getattr(args, "centerline_foreground_correction", True),
    )

    reviewer: ReviewWindow | None = None
    out_video = resolve_output_video(args.out_video, args.out_dir, run_timestamp=run_timestamp)
    writer = (
        TrackingVideos(out_video, args.fps, roi_payload["rois"], image_paths)
        if args.video else None
    )
    args._resolved_out_video = out_video
    if writer is not None:
        args._resolved_roi_video = writer.roi_path

    prev_gray = dmt.read_image_unicode(image_paths[0], cv2.IMREAD_GRAYSCALE)
    previous_exposure_stats = measure_exposure_stats(prev_gray)
    exposure_hold_remaining = 0
    diagnostics = getattr(args, "diagnostics", False)
    initial_rois = roi_payload["rois"]
    planned_frames = len(image_paths)
    raw_results, _ = dmt.detect_all(prev_gray, roi_payload, config, debug_panels_enabled=False)
    candidates: dict[int, dict[str, Any] | None] = {}
    jump_limit = getattr(args, "centerline_lateral_jump", 7.0)
    results = assess_visibility(raw_results, candidates, roi_payload, roi_payload, jump_limit, diagnostics)
    if args.roi_update_mode == "detect-corrected":
        roi_payload, roi_updates = correct_rois_from_detections(
            roi_payload,
            results,
            padding_frac=args.roi_padding_frac,
            length_padding_frac=args.roi_length_padding_frac,
            max_roi_jump=args.max_roi_jump,
            min_update_confidence=getattr(args, "min_roi_update_confidence", "low"),
            min_corrected_roi_short_span=getattr(args, "min_corrected_roi_short_span", 0.0),
            max_corrected_roi_growth=getattr(args, "max_corrected_roi_growth", None),
            min_corrected_roi_long_span_ratio=getattr(args, "min_corrected_roi_long_span_ratio", 0.0),
            min_corrected_roi_long_span_drop=getattr(args, "min_corrected_roi_long_span_drop", 0.0),
        )
        reject_invalid_updates(results, roi_updates, candidates)
        _apply_roi_vertices_to_results(results, roi_payload)
    anchor_roi_payload = dmt.normalize_roi_payload(roi_payload)
    confirmed_widths = {r["roi_idx"]: r["tube_width_px"] for r in results if r["status"] == "ok"}
    search_references = {r["roi_idx"]: (confirmed_widths[r["roi_idx"]], _roi_long_span(r["vertices"]))
                         for r in roi_payload["rois"] if r["roi_idx"] in confirmed_widths}
    roi_span_history = update_roi_span_history({}, roi_payload)
    frames = [make_frame_centerline_summary(image_paths[0], rois_source, results)]
    if diagnostics:
        add_evidence(frames[-1], raw_results, previous_exposure_stats)
    _emit_progress(0, len(image_paths), image_paths[0], results, getattr(args, "progress_every", 0))
    review_polygons = {}
    overlay = (
        make_review_frame(prev_gray, results, image_path=image_paths[0], frame_index=0,
                          total_frames=len(image_paths), tracking=[], review_polygons=review_polygons)
        if args.window or args.video else None
    )
    display_frame = overlay if args.window else None
    video_frame = _scale_review_frame(overlay, args.video_scale) if args.video else None

    try:
        if writer is not None and video_frame is not None:
            writer.write(video_frame, prev_gray, results, 0)
        if args.window and display_frame is not None:
            reviewer = ReviewWindow(
                display_frame.shape[:2],
                max_width=args.window_width,
                max_height=args.window_height,
                review_mode=args.review_mode,
                wait_ms=args.wait_ms,
            )
            if not reviewer.show(display_frame, force_pause=_needs_pause(results, args.pause_on)):
                image_paths = image_paths[:1]

        for frame_index, image_path in enumerate(image_paths[1:], start=1):
            next_gray = dmt.read_image_unicode(image_path, cv2.IMREAD_GRAYSCALE)
            current_exposure_stats = measure_exposure_stats(next_gray)
            exposure_suspected, exposure_info = exposure_suspect_info(
                previous_exposure_stats,
                current_exposure_stats,
                mean_jump=getattr(args, "exposure_mean_jump", 25.0),
                saturation_jump=getattr(args, "exposure_saturation_jump", 0.15),
                bright_mean=getattr(args, "exposure_bright_mean", 175.0),
                saturation_frac=getattr(args, "exposure_saturation_frac", 0.48),
            )
            exposure_guard_enabled = getattr(args, "exposure_guard", "shrink") != "none"
            exposure_active = exposure_guard_enabled and (exposure_suspected or exposure_hold_remaining > 0)
            previous_roi_payload = roi_payload
            predicted_roi_payload, tracking = track_rois_between_frames(
                prev_gray,
                next_gray,
                previous_roi_payload,
                max_corners=args.max_corners,
                min_track_points=args.min_track_points,
                max_forward_backward_error=args.max_forward_backward_error,
                track_scale=args.track_scale,
                max_roi_jump=args.max_roi_jump,
                lk_window_size=getattr(args, "lk_window_size", 31),
                lk_max_level=getattr(args, "lk_max_level", 4),
                track_group_outlier_min_dy=getattr(args, "track_group_outlier_min_dy", 0.0),
                track_group_outlier_dy_deviation=getattr(args, "track_group_outlier_dy_deviation", 45.0),
            )
            for roi in predicted_roi_payload["rois"]:
                idx = roi["roi_idx"]
                roi["expected_width"] = confirmed_widths.get(idx)
                if args.roi_update_mode == "detect-corrected" and idx in search_references:
                    _preserve_search_extent(roi, confirmed_widths[idx], search_references[idx])
            raw_results, _ = dmt.detect_all(next_gray, predicted_roi_payload, config, debug_panels_enabled=False)
            results = assess_visibility(raw_results, candidates, previous_roi_payload, predicted_roi_payload, jump_limit, diagnostics)
            if args.roi_update_mode == "detect-corrected":
                roi_payload, roi_updates = correct_rois_from_detections(
                    predicted_roi_payload,
                    results,
                    padding_frac=args.roi_padding_frac,
                    length_padding_frac=args.roi_length_padding_frac,
                    max_roi_jump=args.max_roi_jump,
                    min_update_confidence=getattr(args, "min_roi_update_confidence", "low"),
                    min_corrected_roi_short_span=getattr(args, "min_corrected_roi_short_span", 0.0),
                    max_corrected_roi_growth=getattr(args, "max_corrected_roi_growth", None),
                    min_corrected_roi_long_span_ratio=getattr(args, "min_corrected_roi_long_span_ratio", 0.0),
                    min_corrected_roi_long_span_drop=getattr(args, "min_corrected_roi_long_span_drop", 0.0),
                    anchor_payload=anchor_roi_payload,
                    max_anchor_drift=getattr(args, "max_anchor_drift", None),
                )
                if exposure_guard_enabled:
                    roi_payload, roi_updates = guard_exposure_shrunken_rois(
                        predicted_roi_payload,
                        roi_payload,
                        roi_updates,
                        exposure_active=exposure_active,
                        exposure_info=exposure_info,
                        shrink_ratio_threshold=getattr(args, "exposure_shrink_ratio", 0.93),
                        shrink_pixels_threshold=getattr(args, "exposure_shrink_pixels", 5.0),
                        min_shrunk_rois=getattr(args, "exposure_min_shrunk_rois", 2),
                    )
                    exposure_rejected = any(
                        update.get("update_status") == "rejected-exposure-shrink"
                        for update in roi_updates
                    )
                    if exposure_rejected:
                        exposure_active = True
                roi_payload, roi_updates = guard_long_span_collapsed_rois(
                    predicted_roi_payload,
                    roi_payload,
                    roi_updates,
                    roi_span_history,
                    long_span_ratio_threshold=getattr(args, "history_long_span_ratio", 0.60),
                    long_span_drop_threshold=getattr(args, "history_long_span_drop", 80.0),
                    short_span_ratio_threshold=getattr(args, "history_short_span_ratio", 0.85),
                )
                reject_invalid_updates(results, roi_updates, candidates, confirmed_widths)
                _apply_roi_vertices_to_results(results, roi_payload)
                tracking = _merge_roi_update_info(tracking, roi_updates)
            else:
                roi_payload = predicted_roi_payload
            held_widths = {t["roi_idx"] for t in tracking if t.get("update_status") in
                           {"rejected-exposure-shrink", "rejected-corrected-roi-shrink"}}
            for result in results:
                if result["status"] == "ok" and result["roi_idx"] not in held_widths:
                    idx = result["roi_idx"]
                    confirmed_widths[idx] = result["tube_width_px"]
                    if idx not in search_references:
                        search_references[idx] = (confirmed_widths[idx], _roi_long_span(result["vertices"]))
            roi_span_history = update_roi_span_history(roi_span_history, roi_payload)
            frames.append(
                make_frame_centerline_summary(
                    image_path,
                    "tracked_from_previous_frame",
                    results,
                    tracking=tracking,
                )
            )
            if diagnostics:
                add_evidence(frames[-1], raw_results, current_exposure_stats)
            _emit_progress(frame_index, len(image_paths), image_path, results, getattr(args, "progress_every", 0))
            overlay = (
                make_review_frame(next_gray, results, image_path=image_path, frame_index=frame_index,
                                  total_frames=len(image_paths), tracking=tracking, review_polygons=review_polygons)
                if reviewer is not None or writer is not None else None
            )
            display_frame = overlay if reviewer is not None else None
            video_frame = _scale_review_frame(overlay, args.video_scale) if writer is not None else None
            if writer is not None and video_frame is not None:
                writer.write(video_frame, next_gray, results, frame_index)
            if reviewer is not None and display_frame is not None and not reviewer.show(display_frame, force_pause=_needs_pause(results, args.pause_on)):
                prev_gray = next_gray
                break
            if exposure_guard_enabled:
                exposure_rejected = any(
                    (item.get("update_status") == "rejected-exposure-shrink")
                    for item in tracking
                )
                if exposure_suspected or exposure_rejected:
                    exposure_hold_remaining = max(0, int(getattr(args, "exposure_hold_frames", 8)))
                elif exposure_hold_remaining > 0:
                    exposure_hold_remaining -= 1
            previous_exposure_stats = current_exposure_stats
            prev_gray = next_gray
    finally:
        if writer is not None:
            writer.release()
        if reviewer is not None:
            reviewer.close()

    payload = make_sequence_centerline_payload(frames, rois_source)
    dmt.save_json(out_json, payload)
    if diagnostics:
        zip_path = make_unique_output_path(out_json.with_name(out_json.stem + "_diagnostics.zip"))
        write_diagnostics(zip_path, frames, initial_rois, planned_frames, args)
        print(f"Diagnostics: {zip_path}")
    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Track first-frame ROIs through a sequence and output centerlines.")
    image_source = parser.add_mutually_exclusive_group(required=True)
    image_source.add_argument("--images", nargs="+", help="Ordered input image paths.")
    image_source.add_argument("--image-dir", help="Input image directory. Files are read only and never moved or deleted.")
    parser.add_argument("--pattern", default="*.bmp", help="Image filename pattern used with --image-dir.")
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="Process only the first N ordered frames. Use 0 to process all frames.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["timestamp", "name", "mtime"],
        default="timestamp",
        help="Ordering for --image-dir. timestamp sorts by leading digits in each filename.",
    )
    roi_source = parser.add_mutually_exclusive_group()
    roi_source.add_argument("--initial-rois", help="First-frame ROI JSON path.")
    roi_source.add_argument("--draw-rois", action="store_true", help="Draw first-frame ROIs interactively.")
    parser.add_argument("--out-json", default=None, help="Combined centerline JSON output path.")
    parser.add_argument("--out-video", default=None, help="Review video output path.")
    parser.add_argument("--camera", choices=("left", "right"), help="Camera for the default run directory.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for generated outputs. Defaults to tracking_results/runs/<camera>_camera/<run>. "
            "Relative paths are resolved from the current working directory."
        ),
    )
    parser.set_defaults(window=True, video=True)
    parser.add_argument("--no-window", dest="window", action="store_false", help="Disable real-time review window.")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Disable review video recording.")
    parser.add_argument("--diagnostics", action="store_true", help="Save a shareable local ZIP with measurements and sampled ROI crops.")
    parser.add_argument(
        "--jupyter-friendly",
        action="store_true",
        help="Disable OpenCV windows/video, run continuously, never pause on failures, and print per-frame progress.",
    )
    parser.add_argument("--fps", type=float, default=10.0, help="Output review video frame rate.")
    parser.add_argument("--video-scale", type=float, default=0.5, help="Scale factor for review video frames.")
    parser.add_argument("--window-width", type=int, default=1400, help="Maximum review window canvas width.")
    parser.add_argument("--window-height", type=int, default=900, help="Maximum review window canvas height.")
    parser.add_argument("--wait-ms", type=int, default=30, help="Continuous review delay per frame in milliseconds.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Print one progress line every N frames. Use 0 to disable progress output.",
    )
    parser.add_argument(
        "--review-mode",
        choices=["step", "continuous"],
        default="step",
        help="step waits for n/space every frame; continuous advances until p/q or pause-on triggers.",
    )
    parser.add_argument(
        "--pause-on",
        choices=["none", "failed", "low"],
        default="failed",
        help="In continuous mode, pause when detection quality reaches this condition.",
    )
    parser.add_argument("--min-width-frac", type=float, default=0.28, help="Minimum tube width fraction for detector.")
    parser.add_argument("--max-width-frac", type=float, default=0.92, help="Maximum tube width fraction for detector.")
    parser.add_argument("--boundary-samples", type=int, default=60, help="Internal detector sampling count.")
    parser.add_argument(
        "--centerline-foreground-correction",
        dest="centerline_foreground_correction",
        action="store_true",
        default=True,
        help="Shift centerlines toward the foreground/body profile when the foreground span is reliable.",
    )
    parser.add_argument(
        "--no-centerline-foreground-correction",
        dest="centerline_foreground_correction",
        action="store_false",
        help="Keep centerlines at the midpoint of detected left/right boundaries.",
    )
    parser.add_argument(
        "--centerline-lateral-jump",
        type=float,
        default=7.0,
        help="Require a second wall measurement after a motion-compensated lateral jump exceeding this many pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--track-scale",
        type=float,
        default=None,
        help="Scale used only for optical-flow tracking. Detection and JSON stay in original coordinates.",
    )
    parser.add_argument("--max-corners", type=int, default=80, help="Maximum LK feature points per ROI.")
    parser.add_argument("--min-track-points", type=int, default=3, help="Minimum valid tracked points per ROI.")
    parser.add_argument(
        "--max-forward-backward-error",
        type=float,
        default=1.5,
        help="Maximum LK forward-backward error in pixels.",
    )
    parser.add_argument(
        "--lk-window-size",
        type=int,
        default=31,
        help="LK optical-flow search window size. Larger values tolerate larger per-frame motion.",
    )
    parser.add_argument(
        "--lk-max-level",
        type=int,
        default=4,
        help="LK optical-flow pyramid levels. Larger values tolerate larger per-frame motion.",
    )
    parser.add_argument(
        "--track-group-outlier-min-dy",
        type=float,
        default=0.0,
        help="Opt in to group outlier rejection for jointly moving targets. Default 0 tracks each ROI independently.",
    )
    parser.add_argument(
        "--track-group-outlier-dy-deviation",
        type=float,
        default=45.0,
        help="Reject ROI tracking when its vertical motion differs from the group median by more than this many pixels.",
    )
    parser.add_argument(
        "--roi-update-mode",
        choices=["flow", "detect-corrected"],
        default="detect-corrected",
        help="flow keeps LK/RANSAC ROI propagation; detect-corrected rebuilds each ROI from detected tube boundaries.",
    )
    parser.add_argument(
        "--min-roi-update-confidence",
        choices=["low", "medium", "high"],
        default="medium",
        help="Minimum detector confidence allowed to update ROI state in detect-corrected mode.",
    )
    parser.add_argument(
        "--min-corrected-roi-short-span",
        type=float,
        default=40.0,
        help="Minimum short-axis span in pixels for ROIs rebuilt from detector boundaries.",
    )
    parser.add_argument(
        "--max-corrected-roi-growth",
        type=float,
        default=1.0,
        help="Maximum corrected ROI long/short-axis growth relative to the predicted ROI. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-corrected-roi-long-span-ratio",
        type=float,
        default=0.78,
        help="Reject corrected ROIs whose long-axis span collapses below this ratio of the predicted ROI. Use 0 to disable.",
    )
    parser.add_argument(
        "--min-corrected-roi-long-span-drop",
        type=float,
        default=45.0,
        help="Minimum long-axis span drop in pixels required before corrected ROI shrink rejection triggers.",
    )
    parser.add_argument(
        "--history-long-span-ratio",
        type=float,
        default=0.60,
        help="Reject corrected ROIs whose long-axis span falls below this ratio of the best accepted history span.",
    )
    parser.add_argument(
        "--history-long-span-drop",
        type=float,
        default=80.0,
        help="Minimum long-axis drop from best accepted history span before history span rejection can trigger.",
    )
    parser.add_argument(
        "--history-short-span-ratio",
        type=float,
        default=0.85,
        help="Require the corrected ROI short-axis span to fall below this history ratio before history span rejection triggers.",
    )
    parser.add_argument(
        "--max-anchor-drift",
        type=float,
        default=0.0,
        help="Maximum corrected ROI center drift from the first-frame anchor ROI in pixels. Default 0 disables this for moving targets.",
    )
    parser.add_argument(
        "--roi-padding-frac",
        type=float,
        default=0.35,
        help="Extra ROI width fraction added after detection correction.",
    )
    parser.add_argument(
        "--exposure-guard",
        choices=["none", "shrink"],
        default="shrink",
        help="Reject sudden ROI shrink updates during suspected overexposure.",
    )
    parser.add_argument(
        "--exposure-mean-jump",
        type=float,
        default=25.0,
        help="Frame mean-gray increase that marks a possible exposure jump.",
    )
    parser.add_argument(
        "--exposure-saturation-jump",
        type=float,
        default=0.15,
        help="Frame saturated-pixel fraction increase that marks a possible exposure jump.",
    )
    parser.add_argument(
        "--exposure-bright-mean",
        type=float,
        default=175.0,
        help="Absolute mean-gray threshold for sustained overexposure.",
    )
    parser.add_argument(
        "--exposure-saturation-frac",
        type=float,
        default=0.48,
        help="Absolute saturated-pixel fraction threshold for sustained overexposure.",
    )
    parser.add_argument(
        "--exposure-hold-frames",
        type=int,
        default=8,
        help="Frames to keep exposure guard active after a jump or guarded shrink.",
    )
    parser.add_argument(
        "--exposure-shrink-ratio",
        type=float,
        default=0.93,
        help="Corrected ROI short-span ratio below which an exposure shrink is rejected.",
    )
    parser.add_argument(
        "--exposure-shrink-pixels",
        type=float,
        default=5.0,
        help="Corrected ROI short-span pixel drop required before exposure shrink rejection.",
    )
    parser.add_argument(
        "--exposure-min-shrunk-rois",
        type=int,
        default=2,
        help="Number of simultaneous ROI shrinks that can confirm overexposure without image-stat triggers.",
    )
    parser.add_argument(
        "--roi-length-padding-frac",
        type=float,
        default=0.10,
        help="Extra ROI length fraction added at each end after detection correction.",
    )
    parser.add_argument(
        "--max-roi-jump",
        type=float,
        default=200.0,
        help="Maximum allowed corrected ROI center jump in original-image pixels.",
    )
    args = parser.parse_args(argv)
    if args.jupyter_friendly:
        args.window = False
        args.video = False
        args.review_mode = "continuous"
        args.pause_on = "none"
        if args.progress_every == 0:
            args.progress_every = 1
    if args.track_scale is None:
        args.track_scale = 0.25 if args.jupyter_friendly else 0.5
    if args.images is not None and len(args.images) < 2:
        parser.error("--images must contain at least two ordered frames.")
    if args.max_frames < 0:
        parser.error("--max-frames must be non-negative.")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    if args.track_scale <= 0:
        parser.error("--track-scale must be positive.")
    if args.centerline_lateral_jump < 0:
        parser.error("--centerline-lateral-jump must be non-negative.")
    if args.lk_window_size < 5:
        parser.error("--lk-window-size must be at least 5.")
    if args.lk_max_level < 0:
        parser.error("--lk-max-level must be non-negative.")
    if args.track_group_outlier_min_dy < 0:
        parser.error("--track-group-outlier-min-dy must be non-negative.")
    if args.track_group_outlier_dy_deviation < 0:
        parser.error("--track-group-outlier-dy-deviation must be non-negative.")
    if args.roi_padding_frac < 0:
        parser.error("--roi-padding-frac must be non-negative.")
    if args.roi_length_padding_frac < 0:
        parser.error("--roi-length-padding-frac must be non-negative.")
    if args.min_corrected_roi_short_span < 0:
        parser.error("--min-corrected-roi-short-span must be non-negative.")
    if args.max_corrected_roi_growth < 0:
        parser.error("--max-corrected-roi-growth must be non-negative.")
    if not 0 <= args.min_corrected_roi_long_span_ratio <= 1:
        parser.error("--min-corrected-roi-long-span-ratio must be between 0 and 1.")
    if args.min_corrected_roi_long_span_drop < 0:
        parser.error("--min-corrected-roi-long-span-drop must be non-negative.")
    if not 0 <= args.history_long_span_ratio <= 1:
        parser.error("--history-long-span-ratio must be between 0 and 1.")
    if args.history_long_span_drop < 0:
        parser.error("--history-long-span-drop must be non-negative.")
    if not 0 <= args.history_short_span_ratio <= 1:
        parser.error("--history-short-span-ratio must be between 0 and 1.")
    if args.max_anchor_drift < 0:
        parser.error("--max-anchor-drift must be non-negative.")
    if args.exposure_mean_jump < 0:
        parser.error("--exposure-mean-jump must be non-negative.")
    if args.exposure_saturation_jump < 0:
        parser.error("--exposure-saturation-jump must be non-negative.")
    if args.exposure_bright_mean < 0:
        parser.error("--exposure-bright-mean must be non-negative.")
    if not 0 <= args.exposure_saturation_frac <= 1:
        parser.error("--exposure-saturation-frac must be between 0 and 1.")
    if args.exposure_hold_frames < 0:
        parser.error("--exposure-hold-frames must be non-negative.")
    if not 0 < args.exposure_shrink_ratio <= 1:
        parser.error("--exposure-shrink-ratio must be in (0, 1].")
    if args.exposure_shrink_pixels < 0:
        parser.error("--exposure-shrink-pixels must be non-negative.")
    if args.exposure_min_shrunk_rois < 1:
        parser.error("--exposure-min-shrunk-rois must be at least 1.")
    if args.max_roi_jump <= 0:
        parser.error("--max-roi-jump must be positive.")
    return args


def main() -> int:
    args = parse_args()
    payload = run_sequence(args)
    out_json = getattr(args, "_resolved_out_json", None)
    if out_json is None:
        out_json = resolve_output_json(args.out_json, args.out_dir)
    print(f"Processed {payload['num_frames']} frames.")
    print(f"Results: {out_json}")
    if args.video:
        print(f"Global video: {args._resolved_out_video}")
        print(f"ROI video: {args._resolved_roi_video}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
