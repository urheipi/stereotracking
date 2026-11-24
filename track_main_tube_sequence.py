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
from typing import Any, Sequence

import cv2
import numpy as np

import detect_main_tube as dmt


DEFAULT_OUTPUT_JSON = "sequence_centerlines.json"
DEFAULT_OUTPUT_DIR = "tracking_results"
DEFAULT_OUTPUT_JSON_STEM = "sequence_centerlines"
DEFAULT_OUTPUT_JSON_SUFFIX = ".json"
DEFAULT_OUTPUT_VIDEO_STEM = "tracking_review"
DEFAULT_OUTPUT_VIDEO_SUFFIX = ".mp4"
WINDOW_NAME = "Main Tube Sequence Review"
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


def make_unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No available output filename near {path}")


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
        return "track: initial"
    ok_count = sum(1 for item in tracking if item.get("status") == "ok")
    return f"track: {ok_count}/{len(tracking)} ok"


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
) -> np.ndarray:
    frame = dmt.draw_overlay(gray, results)
    ok_count = sum(1 for result in results if result.get("status") == "ok")
    header = (
        f"{frame_index + 1}/{total_frames}  {Path(image_path).name}  "
        f"detect: {ok_count}/{len(results)} ok  {_tracking_text(tracking)}"
    )
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 0), -1)
    cv2.putText(frame, header, (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.78, (255, 255, 255), 2, cv2.LINE_AA)

    if video_scale != 1.0:
        width = max(1, int(round(frame.shape[1] * video_scale)))
        height = max(1, int(round(frame.shape[0] * video_scale)))
        frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
    return frame


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


class ReviewWindow:
    def __init__(
        self,
        image_shape: tuple[int, int],
        *,
        max_width: int,
        max_height: int,
        review_mode: str,
        wait_ms: int,
    ) -> None:
        height, width = image_shape
        self.viewport = dmt.RoiViewport.for_image(width, height, max_width=max_width, max_height=max_height)
        self.review_mode = review_mode
        self.wait_ms = max(1, int(wait_ms))
        self.continuous = review_mode == "continuous"
        self.dragging = False
        self.drag_start: tuple[int, int] | None = None
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.viewport.canvas_width, self.viewport.canvas_height)
        cv2.setMouseCallback(WINDOW_NAME, self._on_mouse)

    def _on_mouse(self, event: int, x: int, y: int, flags: int, _param: Any) -> None:
        if event == cv2.EVENT_MOUSEWHEEL:
            self.viewport.zoom_at(x, y, 1.2 if flags > 0 else 1 / 1.2)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.dragging = True
            self.drag_start = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging and self.drag_start is not None:
            last_x, last_y = self.drag_start
            self.viewport.pan(x - last_x, y - last_y)
            self.drag_start = (x, y)
        elif event == cv2.EVENT_RBUTTONUP:
            self.dragging = False
            self.drag_start = None

    def _draw(self, frame: np.ndarray) -> None:
        cv2.imshow(WINDOW_NAME, self.viewport.render(frame))

    def show(self, frame: np.ndarray, *, force_pause: bool) -> bool:
        pause = force_pause or not self.continuous
        while True:
            self._draw(frame)
            key = cv2.waitKey(0 if pause else self.wait_ms) & 0xFF
            if key in (ord("q"), 27):
                return False
            if key in (ord("n"), ord(" "), 13):
                return True
            if key == ord("c"):
                self.continuous = True
                return True
            if key == ord("p"):
                self.continuous = False
                pause = True
                continue
            if key == ord("r"):
                self.viewport.reset()
                continue
            if not pause:
                return True

    def close(self) -> None:
        cv2.destroyWindow(WINDOW_NAME)


def _open_video_writer(path: Path, frame: np.ndarray, fps: float) -> cv2.VideoWriter:
    path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (frame.shape[1], frame.shape[0]))
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {path}")
    return writer


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
) -> tuple[np.ndarray | None, dict[str, Any]]:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
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
    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        prev_points.reshape(-1, 1, 2),
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
        next_gray,
        prev_gray,
        next_points,
        None,
        winSize=(lk_window_size, lk_window_size),
        maxLevel=int(lk_max_level),
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )

    prev_flat = prev_points.reshape(-1, 2)
    next_flat = next_points.reshape(-1, 2)
    good = status.reshape(-1).astype(bool)
    if back_points is not None and back_status is not None:
        back_flat = back_points.reshape(-1, 2)
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
    if matrix is None:
        delta = np.median(tracked_next - tracked_prev, axis=0)
        matrix = np.asarray([[1.0, 0.0, delta[0]], [0.0, 1.0, delta[1]]], dtype=np.float32)
        inlier_count = int(len(tracked_prev))
    else:
        inlier_count = int(inliers.sum()) if inliers is not None else int(len(tracked_prev))

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
    track_group_outlier_min_dy: float = 55.0,
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


def _translate_roi_vertices(vertices: list[list[float]], delta: np.ndarray) -> list[list[float]]:
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2) + delta.astype(np.float32)
    return _format_vertices(points)


def apply_group_vertical_motion_prior(
    roi_payload: dict[str, Any],
    anchor_payload: dict[str, Any],
    *,
    history_payload: dict[str, Any] | None = None,
    min_group_drift: float = 25.0,
    floor_frac: float = 0.85,
    tolerance: float = 6.0,
    history_tolerance: float = 2.0,
    max_adjustment: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    normalized = dmt.normalize_roi_payload(roi_payload)
    anchors = {
        int(roi["roi_idx"]): roi
        for roi in dmt.normalize_roi_payload(anchor_payload).get("rois", [])
    }
    history = (
        {
            int(roi["roi_idx"]): roi
            for roi in dmt.normalize_roi_payload(history_payload).get("rois", [])
        }
        if history_payload is not None
        else {}
    )

    drifts: list[float] = []
    for roi in normalized.get("rois", []):
        anchor = anchors.get(int(roi["roi_idx"]))
        if anchor is None:
            continue
        drift = float(_roi_center(roi)[1] - _roi_center(anchor)[1])
        if np.isfinite(drift):
            drifts.append(drift)

    if not drifts:
        return normalized, []

    group_drift = float(np.median(np.asarray(drifts, dtype=np.float32)))
    if group_drift <= float(min_group_drift):
        return normalized, []

    target_drift = group_drift * max(0.0, float(floor_frac))
    max_shift = float(max_adjustment) if max_adjustment is not None else None
    adjusted_rois: list[dict[str, Any]] = []
    updates: list[dict[str, Any]] = []

    for roi in normalized.get("rois", []):
        roi_copy = _roi_copy(roi)
        roi_idx = int(roi_copy["roi_idx"])
        anchor = anchors.get(roi_idx)
        if anchor is None:
            adjusted_rois.append(roi_copy)
            continue

        current_drift = float(_roi_center(roi_copy)[1] - _roi_center(anchor)[1])
        shift_y = 0.0
        reasons: list[str] = []

        drift_shift_y = target_drift - current_drift
        if drift_shift_y > float(tolerance):
            shift_y = max(shift_y, drift_shift_y)
            reasons.append("group-drift-floor")

        history_roi = history.get(roi_idx)
        if history_roi is not None:
            history_floor_y = float(_roi_center(history_roi)[1]) - float(history_tolerance)
            history_shift_y = history_floor_y - float(_roi_center(roi_copy)[1])
            if history_shift_y > 0:
                shift_y = max(shift_y, history_shift_y)
                reasons.append("history-floor")

        if shift_y <= 0:
            adjusted_rois.append(roi_copy)
            continue

        if max_shift is not None:
            shift_y = min(shift_y, max_shift)
        roi_copy["vertices"] = _translate_roi_vertices(roi_copy["vertices"], np.asarray([0.0, shift_y], dtype=np.float32))
        adjusted_rois.append(roi_copy)
        updates.append(
            {
                "roi_idx": roi_idx,
                "motion_prior_status": "group-vertical-adjusted",
                "motion_prior_shift_y": round(float(shift_y), 3),
                "motion_prior_group_drift_y": round(group_drift, 3),
                "motion_prior_current_drift_y": round(current_drift, 3),
                "motion_prior_target_drift_y": round(target_drift, 3),
                "motion_prior_reasons": reasons,
            }
        )

    return {"num_rois": len(adjusted_rois), "rois": adjusted_rois}, updates


def update_vertical_motion_history(
    history_payload: dict[str, Any],
    roi_payload: dict[str, Any],
) -> dict[str, Any]:
    history_by_idx = {
        int(roi["roi_idx"]): _roi_copy(roi)
        for roi in dmt.normalize_roi_payload(history_payload).get("rois", [])
    }
    for roi in dmt.normalize_roi_payload(roi_payload).get("rois", []):
        roi_copy = _roi_copy(roi)
        roi_idx = int(roi_copy["roi_idx"])
        history_roi = history_by_idx.get(roi_idx)
        if history_roi is None or float(_roi_center(roi_copy)[1]) > float(_roi_center(history_roi)[1]):
            history_by_idx[roi_idx] = roi_copy

    rois = [history_by_idx[roi_idx] for roi_idx in sorted(history_by_idx)]
    return {"num_rois": len(rois), "rois": rois}


def _translate_points(points: Any, delta: np.ndarray) -> list[list[float]]:
    translated = np.asarray(points, dtype=np.float32).reshape(-1, 2) + delta.astype(np.float32)
    return _format_vertices(translated)


def _centerline_array(result: dict[str, Any]) -> np.ndarray | None:
    centerline = np.asarray(result.get("centerline", []), dtype=np.float32)
    if centerline.ndim != 2 or len(centerline) < 2 or centerline.shape[1:] != (2,):
        return None
    if not np.all(np.isfinite(centerline)):
        return None
    return centerline


def guard_centerline_lateral_jumps(
    results: list[dict[str, Any]],
    previous_results_by_idx: dict[int, dict[str, Any]],
    previous_roi_payload: dict[str, Any],
    current_roi_payload: dict[str, Any],
    *,
    max_lateral_jump: float,
) -> list[dict[str, Any]]:
    if max_lateral_jump <= 0:
        return results

    previous_rois = {
        int(roi["roi_idx"]): roi
        for roi in dmt.normalize_roi_payload(previous_roi_payload).get("rois", [])
    }
    current_rois = {
        int(roi["roi_idx"]): roi
        for roi in dmt.normalize_roi_payload(current_roi_payload).get("rois", [])
    }

    guarded: list[dict[str, Any]] = []
    for result in results:
        roi_idx = int(result.get("roi_idx", 0))
        previous_result = previous_results_by_idx.get(roi_idx)
        previous_roi = previous_rois.get(roi_idx)
        current_roi = current_rois.get(roi_idx)
        current_centerline = _centerline_array(result)
        previous_centerline = _centerline_array(previous_result or {})
        if (
            result.get("status") != "ok"
            or previous_result is None
            or previous_roi is None
            or current_roi is None
            or current_centerline is None
            or previous_centerline is None
        ):
            guarded.append(result)
            continue

        delta = _roi_center(current_roi) - _roi_center(previous_roi)
        predicted_centerline = previous_centerline + delta.astype(np.float32)
        current_center = current_centerline.mean(axis=0)
        predicted_center = predicted_centerline.mean(axis=0)
        direction = current_centerline[-1] - current_centerline[0]
        direction_norm = float(np.linalg.norm(direction))
        if direction_norm <= 1e-6:
            guarded.append(result)
            continue

        normal = np.asarray([-direction[1], direction[0]], dtype=np.float32) / direction_norm
        lateral_jump = float(np.dot(current_center - predicted_center, normal))
        if abs(lateral_jump) <= float(max_lateral_jump):
            guarded.append(result)
            continue

        guarded.append(
            {
                **result,
                "centerline": _format_vertices(predicted_centerline),
                "centerline_guard_status": "temporal-lateral-clamped",
                "centerline_lateral_jump": round(lateral_jump, 3),
                "centerline_lateral_jump_limit": round(float(max_lateral_jump), 3),
                "centerline_original_source": result.get("centerline_source"),
                "centerline_source": "temporal-propagated",
            }
        )

    return guarded


def _trusted_detection_results(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    trusted: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.get("status") == "ok" and result.get("confidence") in {"medium", "high"}:
            trusted[int(result["roi_idx"])] = result
    return trusted


def _propagated_detection_results(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    propagated: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.get("status") == "ok" and result.get("confidence") == "tracked":
            propagated[int(result["roi_idx"])] = result
    return propagated


def _real_detection_results(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    real: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.get("status") == "ok" and result.get("confidence") != "tracked":
            real[int(result["roi_idx"])] = result
    return real


def apply_tracked_fallbacks(
    results: list[dict[str, Any]],
    previous_results_by_idx: dict[int, dict[str, Any]],
    previous_roi_payload: dict[str, Any],
    predicted_roi_payload: dict[str, Any],
    tracking: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    previous_rois = {int(roi["roi_idx"]): roi for roi in dmt.normalize_roi_payload(previous_roi_payload).get("rois", [])}
    predicted_rois = {int(roi["roi_idx"]): roi for roi in dmt.normalize_roi_payload(predicted_roi_payload).get("rois", [])}
    tracking_by_idx = {int(item.get("roi_idx", 0)): item for item in tracking}

    fallback_results: list[dict[str, Any]] = []
    for result in results:
        roi_idx = int(result.get("roi_idx", 0))
        previous_result = previous_results_by_idx.get(roi_idx)
        previous_roi = previous_rois.get(roi_idx)
        predicted_roi = predicted_rois.get(roi_idx)
        track_info = tracking_by_idx.get(roi_idx, {})
        stationary_prediction = (
            previous_roi is not None
            and predicted_roi is not None
            and float(np.linalg.norm(_roi_center(predicted_roi) - _roi_center(previous_roi))) <= 1e-3
        )
        if (
            result.get("status") == "ok"
            or previous_result is None
            or previous_roi is None
            or predicted_roi is None
            or (track_info.get("status") != "ok" and not stationary_prediction)
        ):
            fallback_results.append(result)
            continue

        delta = _roi_center(predicted_roi) - _roi_center(previous_roi)
        fallback = {
            **result,
            "status": "ok",
            "confidence": "tracked",
            "fallback_reason": result.get("failure_reason", "detection failed"),
            "vertices": _roi_copy(predicted_roi)["vertices"],
            "left_boundary": _translate_points(previous_result.get("left_boundary", []), delta),
            "right_boundary": _translate_points(previous_result.get("right_boundary", []), delta),
            "centerline": _translate_points(previous_result.get("centerline", []), delta),
        }
        fallback_results.append(fallback)

    return fallback_results


def apply_update_rejection_fallbacks(
    results: list[dict[str, Any]],
    updates: list[dict[str, Any]],
    previous_results_by_idx: dict[int, dict[str, Any]],
    previous_roi_payload: dict[str, Any],
    corrected_roi_payload: dict[str, Any],
) -> list[dict[str, Any]]:
    rejected_statuses = {
        "rejected-anchor-drift",
        "rejected-roi-jump",
        "rejected-corrected-roi",
        "rejected-corrected-roi-shrink",
        "rejected-exposure-shrink",
        "rejected-long-span-history-shrink",
    }
    rejected_by_idx = {
        int(update["roi_idx"]): update
        for update in updates
        if update.get("update_status") in rejected_statuses
    }
    previous_rois = {int(roi["roi_idx"]): roi for roi in dmt.normalize_roi_payload(previous_roi_payload).get("rois", [])}
    corrected_rois = {int(roi["roi_idx"]): roi for roi in dmt.normalize_roi_payload(corrected_roi_payload).get("rois", [])}

    fallback_results: list[dict[str, Any]] = []
    for result in results:
        roi_idx = int(result.get("roi_idx", 0))
        update = rejected_by_idx.get(roi_idx)
        previous_result = previous_results_by_idx.get(roi_idx)
        previous_roi = previous_rois.get(roi_idx)
        corrected_roi = corrected_rois.get(roi_idx)
        if update is None or previous_result is None or previous_roi is None or corrected_roi is None:
            fallback_results.append(result)
            continue

        delta = _roi_center(corrected_roi) - _roi_center(previous_roi)
        fallback_results.append(
            {
                **result,
                "status": "ok",
                "confidence": "tracked",
                "fallback_reason": update.get("update_status", "rejected-update"),
                "vertices": _roi_copy(corrected_roi)["vertices"],
                "left_boundary": _translate_points(previous_result.get("left_boundary", []), delta),
                "right_boundary": _translate_points(previous_result.get("right_boundary", []), delta),
                "centerline": _translate_points(previous_result.get("centerline", []), delta),
            }
        )

    return fallback_results


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

    rois_source = Path(args.initial_rois)
    return dmt.normalize_roi_payload(dmt.load_roi_json(rois_source)), rois_source


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
    out_json = resolve_output_json(args.out_json, args.out_dir, run_timestamp=run_timestamp)
    args._resolved_out_json = out_json
    args._run_timestamp = run_timestamp
    roi_payload, rois_source = load_initial_roi_payload(args, image_paths[0], out_json.parent)
    config = dmt.DetectionConfig(
        min_width_frac=args.min_width_frac,
        max_width_frac=args.max_width_frac,
        boundary_samples=args.boundary_samples,
        centerline_foreground_correction=getattr(args, "centerline_foreground_correction", True),
    )

    reviewer: ReviewWindow | None = None
    writer: cv2.VideoWriter | None = None
    out_video = resolve_output_video(args.out_video, args.out_dir, run_timestamp=run_timestamp)
    args._resolved_out_video = out_video

    prev_gray = dmt.read_image_unicode(image_paths[0], cv2.IMREAD_GRAYSCALE)
    previous_exposure_stats = measure_exposure_stats(prev_gray)
    exposure_hold_remaining = 0
    results, _ = dmt.detect_all(prev_gray, roi_payload, config)
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
        _apply_roi_vertices_to_results(results, roi_payload)
    anchor_roi_payload = dmt.normalize_roi_payload(roi_payload)
    vertical_motion_history_payload = dmt.normalize_roi_payload(roi_payload)
    roi_span_history = update_roi_span_history({}, roi_payload)
    last_detection_results_by_idx = _trusted_detection_results(results)
    frames = [make_frame_centerline_summary(image_paths[0], rois_source, results)]
    _emit_progress(0, len(image_paths), image_paths[0], results, getattr(args, "progress_every", 0))
    display_frame = (
        make_review_frame(
            prev_gray,
            results,
            image_path=image_paths[0],
            frame_index=0,
            total_frames=len(image_paths),
            tracking=[],
            video_scale=1.0,
        )
        if args.window
        else None
    )
    video_frame = (
        make_review_frame(
            prev_gray,
            results,
            image_path=image_paths[0],
            frame_index=0,
            total_frames=len(image_paths),
            tracking=[],
            video_scale=args.video_scale,
        )
        if args.video
        else None
    )

    try:
        if args.video and video_frame is not None:
            writer = _open_video_writer(out_video, video_frame, args.fps)
            writer.write(video_frame)
        if args.window and display_frame is not None:
            reviewer = ReviewWindow(
                display_frame.shape[:2],
                max_width=args.window_width,
                max_height=args.window_height,
                review_mode=args.review_mode,
                wait_ms=args.wait_ms,
            )
            if not reviewer.show(display_frame, force_pause=_needs_pause(results, args.pause_on)):
                payload = make_sequence_centerline_payload(frames, rois_source)
                dmt.save_json(out_json, payload)
                return payload

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
                track_group_outlier_min_dy=getattr(args, "track_group_outlier_min_dy", 55.0),
                track_group_outlier_dy_deviation=getattr(args, "track_group_outlier_dy_deviation", 45.0),
            )
            results, _ = dmt.detect_all(next_gray, predicted_roi_payload, config)
            results = apply_tracked_fallbacks(
                results,
                last_detection_results_by_idx,
                roi_payload,
                predicted_roi_payload,
                tracking,
            )
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
                results = apply_update_rejection_fallbacks(
                    results,
                    roi_updates,
                    last_detection_results_by_idx,
                    previous_roi_payload,
                    roi_payload,
                )
                _apply_roi_vertices_to_results(results, roi_payload)
                tracking = _merge_roi_update_info(tracking, roi_updates)
                if getattr(args, "vertical_motion_prior", "group") == "group":
                    prior_roi_payload, prior_updates = apply_group_vertical_motion_prior(
                        roi_payload,
                        anchor_roi_payload,
                        history_payload=vertical_motion_history_payload,
                        min_group_drift=getattr(args, "vertical_motion_min_drift", 25.0),
                        floor_frac=getattr(args, "vertical_motion_floor_frac", 0.85),
                        tolerance=getattr(args, "vertical_motion_tolerance", 6.0),
                        history_tolerance=getattr(args, "vertical_motion_history_tolerance", 2.0),
                        max_adjustment=args.max_roi_jump,
                    )
                    if prior_updates:
                        roi_payload = prior_roi_payload
                        tracking = _merge_roi_update_info(tracking, prior_updates)
                        results, _ = dmt.detect_all(next_gray, roi_payload, config)
                        results = apply_tracked_fallbacks(
                            results,
                            last_detection_results_by_idx,
                            previous_roi_payload,
                            roi_payload,
                            tracking,
                        )
                        results = apply_update_rejection_fallbacks(
                            results,
                            tracking,
                            last_detection_results_by_idx,
                            previous_roi_payload,
                            roi_payload,
                        )
                        _apply_roi_vertices_to_results(results, roi_payload)
            else:
                roi_payload = predicted_roi_payload
            results = guard_centerline_lateral_jumps(
                results,
                last_detection_results_by_idx,
                previous_roi_payload,
                roi_payload,
                max_lateral_jump=getattr(args, "centerline_lateral_jump", 7.0),
            )
            if getattr(args, "vertical_motion_prior", "group") == "group":
                vertical_motion_history_payload = update_vertical_motion_history(
                    vertical_motion_history_payload,
                    roi_payload,
                )
            roi_span_history = update_roi_span_history(roi_span_history, roi_payload)
            last_detection_results_by_idx.update(_trusted_detection_results(results))
            last_detection_results_by_idx.update(_propagated_detection_results(results))
            frames.append(
                make_frame_centerline_summary(
                    image_path,
                    "tracked_from_previous_frame",
                    results,
                    tracking=tracking,
                )
            )
            _emit_progress(frame_index, len(image_paths), image_path, results, getattr(args, "progress_every", 0))
            display_frame = (
                make_review_frame(
                    next_gray,
                    results,
                    image_path=image_path,
                    frame_index=frame_index,
                    total_frames=len(image_paths),
                    tracking=tracking,
                    video_scale=1.0,
                )
                if reviewer is not None
                else None
            )
            video_frame = (
                make_review_frame(
                    next_gray,
                    results,
                    image_path=image_path,
                    frame_index=frame_index,
                    total_frames=len(image_paths),
                    tracking=tracking,
                    video_scale=args.video_scale,
                )
                if writer is not None
                else None
            )
            if writer is not None and video_frame is not None:
                writer.write(video_frame)
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
    roi_source = parser.add_mutually_exclusive_group(required=True)
    roi_source.add_argument("--initial-rois", help="First-frame ROI JSON path.")
    roi_source.add_argument("--draw-rois", action="store_true", help="Draw first-frame ROIs interactively.")
    parser.add_argument("--out-json", default=None, help="Combined centerline JSON output path.")
    parser.add_argument("--out-video", default=None, help="Review video output path.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Directory for generated outputs. Defaults to tracking_results for generated filenames. "
            "Relative paths are resolved from the current working directory."
        ),
    )
    parser.set_defaults(window=True, video=True)
    parser.add_argument("--no-window", dest="window", action="store_false", help="Disable real-time review window.")
    parser.add_argument("--no-video", dest="video", action="store_false", help="Disable review video recording.")
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
        help="Clamp output centerlines to the previous-frame prediction when lateral centerline motion exceeds this many pixels. Use 0 to disable.",
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
        default=55.0,
        help="Minimum single-frame vertical ROI motion before group outlier rejection can trigger.",
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
        "--vertical-motion-prior",
        choices=["none", "group"],
        default="group",
        help="Use the group ROI downward trend to rescue ROIs whose local vertical tracking gets stuck.",
    )
    parser.add_argument(
        "--vertical-motion-min-drift",
        type=float,
        default=15.0,
        help="Minimum median group downward drift from the anchor before vertical motion prior is applied.",
    )
    parser.add_argument(
        "--vertical-motion-floor-frac",
        type=float,
        default=0.85,
        help="Minimum fraction of median group downward drift each ROI should keep once the prior is active.",
    )
    parser.add_argument(
        "--vertical-motion-tolerance",
        type=float,
        default=4.0,
        help="Allowed pixels below the vertical motion prior target before an ROI is nudged downward.",
    )
    parser.add_argument(
        "--vertical-motion-history-tolerance",
        type=float,
        default=2.0,
        help="Allowed pixels above each ROI's historical lowest center before the vertical prior restores it.",
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
    if args.vertical_motion_min_drift < 0:
        parser.error("--vertical-motion-min-drift must be non-negative.")
    if not 0 <= args.vertical_motion_floor_frac <= 1:
        parser.error("--vertical-motion-floor-frac must be between 0 and 1.")
    if args.vertical_motion_tolerance < 0:
        parser.error("--vertical-motion-tolerance must be non-negative.")
    if args.vertical_motion_history_tolerance < 0:
        parser.error("--vertical-motion-history-tolerance must be non-negative.")
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
