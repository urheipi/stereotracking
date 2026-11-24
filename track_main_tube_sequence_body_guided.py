#!/usr/bin/env python
"""Track first-frame ROIs through an image sequence and summarize centerlines.

This module intentionally keeps sequence tracking separate from
detect_main_tube.py. The existing detector is reused as a black box after each
ROI polygon is propagated to the current frame.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

import detect_main_tube_body_guided as dmt


DEFAULT_OUTPUT_JSON = "sequence_centerlines.json"
DEFAULT_OUTPUT_VIDEO = "tracking_review.mp4"
WINDOW_NAME = "Main Tube Sequence Review"
CONFIDENCE_RANK = {"failed": 0, "low": 1, "medium": 2, "high": 3}


def make_detection_config(args: argparse.Namespace) -> dmt.DetectionConfig:
    config_kwargs = {
        "min_width_frac": args.min_width_frac,
        "max_width_frac": args.max_width_frac,
        "boundary_samples": args.boundary_samples,
        "body_guidance": args.body_guidance,
        "body_min_contrast": args.body_min_contrast,
        "body_min_fraction": args.body_min_fraction,
        "body_score_weight": args.body_score_weight,
        "body_min_span_ratio": args.body_min_span_ratio,
        "body_max_span_ratio": args.body_max_span_ratio,
        "body_center_tolerance_frac": args.body_center_tolerance_frac,
        "body_outside_margin_frac": args.body_outside_margin_frac,
        "body_outside_penalty_weight": args.body_outside_penalty_weight,
        "body_hard_reject": args.body_hard_reject,
        "body_edge_gate_frac": args.body_edge_gate_frac,
        "body_adjustment_scale": args.body_adjustment_scale,
    }
    supported = getattr(dmt.DetectionConfig, "__dataclass_fields__", {})
    if supported:
        unsupported = sorted(set(config_kwargs) - set(supported))
        if unsupported:
            print(
                "Warning: detect_main_tube_body_guided.py does not support these tracker options: "
                + ", ".join(unsupported)
                + ". Copy the matching updated detector file to get the optimized body-guided behavior.",
                file=sys.stderr,
            )
        config_kwargs = {key: value for key, value in config_kwargs.items() if key in supported}
    return dmt.DetectionConfig(**config_kwargs)


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


def resolve_output_json(out_json: str | Path | None, out_dir: str | Path | None) -> Path:
    if out_json:
        path = Path(out_json)
        if out_dir and not path.is_absolute():
            path = Path(out_dir) / path
    else:
        path = Path(out_dir or ".") / DEFAULT_OUTPUT_JSON

    if not path.is_absolute():
        path = Path.cwd() / path
    return path


def resolve_output_video(out_video: str | Path | None, out_dir: str | Path | None) -> Path:
    path = Path(out_video) if out_video else Path(out_dir or ".") / DEFAULT_OUTPUT_VIDEO
    if out_video and out_dir and not path.is_absolute():
        path = Path(out_dir) / path
    if not path.is_absolute():
        path = Path.cwd() / path
    return path


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
) -> dict[str, Any]:
    return dmt.make_centerline_summary(image_path, rois_source, results)


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
    mask = _roi_feature_mask(gray.shape, vertices)
    points = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=max_corners,
        qualityLevel=0.01,
        minDistance=5,
        mask=mask,
        blockSize=5,
    )
    if points is None:
        return np.empty((0, 2), dtype=np.float32)
    return points.reshape(-1, 2).astype(np.float32)


def _estimate_roi_transform(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    vertices: np.ndarray,
    max_corners: int,
    min_track_points: int,
    max_forward_backward_error: float,
    initial_flow_delta: np.ndarray | None = None,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    prev_points = _detect_roi_features(prev_gray, vertices, max_corners)
    if len(prev_points) < min_track_points:
        return None, {
            "status": "failed",
            "num_features": int(len(prev_points)),
            "num_tracked": 0,
            "num_inliers": 0,
            "failure_reason": "not enough features in roi",
        }

    initial_next_points = None
    lk_flags = 0
    if initial_flow_delta is not None:
        delta = np.asarray(initial_flow_delta, dtype=np.float32).reshape(1, 2)
        if np.all(np.isfinite(delta)):
            initial_next_points = (prev_points + delta).reshape(-1, 1, 2).astype(np.float32)
            lk_flags = cv2.OPTFLOW_USE_INITIAL_FLOW

    next_points, status, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray,
        next_gray,
        prev_points.reshape(-1, 1, 2),
        initial_next_points,
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        flags=lk_flags,
    )
    if next_points is None or status is None:
        return None, {
            "status": "failed",
            "num_features": int(len(prev_points)),
            "num_tracked": 0,
            "num_inliers": 0,
            "failure_reason": "optical flow failed",
        }

    back_points, back_status, _ = cv2.calcOpticalFlowPyrLK(
        next_gray,
        prev_gray,
        next_points,
        None,
        winSize=(21, 21),
        maxLevel=3,
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
        return None, {
            "status": "failed",
            "num_features": int(len(prev_points)),
            "num_tracked": int(len(tracked_prev)),
            "num_inliers": 0,
            "failure_reason": "not enough valid tracked points",
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


def _tracking_window_bounds(
    image_shape: tuple[int, int],
    vertices: np.ndarray,
    padding: float,
) -> tuple[int, int, int, int] | None:
    height, width = image_shape
    points = np.asarray(vertices, dtype=np.float32).reshape(-1, 2)
    if points.size == 0 or not np.all(np.isfinite(points)):
        return None

    x0 = int(np.floor(float(points[:, 0].min()) - padding))
    y0 = int(np.floor(float(points[:, 1].min()) - padding))
    x1 = int(np.ceil(float(points[:, 0].max()) + padding))
    y1 = int(np.ceil(float(points[:, 1].max()) + padding))

    x0 = max(0, min(width, x0))
    y0 = max(0, min(height, y0))
    x1 = max(0, min(width, x1))
    y1 = max(0, min(height, y1))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def _estimate_template_translation(
    prev_window: np.ndarray,
    next_window: np.ndarray,
    vertices: np.ndarray,
    *,
    template_padding: float = 25.0,
    min_score: float = 0.35,
) -> tuple[np.ndarray | None, dict[str, Any]]:
    bounds = _tracking_window_bounds(prev_window.shape[:2], vertices, template_padding)
    if bounds is None:
        return None, {"template_match_status": "invalid-template"}

    x0, y0, x1, y1 = bounds
    template = prev_window[y0:y1, x0:x1]
    if template.shape[0] < 8 or template.shape[1] < 8:
        return None, {"template_match_status": "template-too-small"}
    if template.shape[0] > next_window.shape[0] or template.shape[1] > next_window.shape[1]:
        return None, {"template_match_status": "template-larger-than-search"}
    if float(template.std()) < 1.0:
        return None, {"template_match_status": "low-texture-template"}

    scores = cv2.matchTemplate(next_window, template, cv2.TM_CCOEFF_NORMED)
    if scores.size == 0 or not np.all(np.isfinite(scores)):
        return None, {"template_match_status": "invalid-score"}
    _, max_score, _, max_loc = cv2.minMaxLoc(scores)
    info: dict[str, Any] = {
        "template_match_status": "ok" if max_score >= min_score else "low-score",
        "template_match_score": round(float(max_score), 4),
    }
    if max_score < min_score:
        return None, info

    delta = np.asarray([max_loc[0] - x0, max_loc[1] - y0], dtype=np.float32)
    info["template_dx"] = round(float(delta[0]), 3)
    info["template_dy"] = round(float(delta[1]), 3)
    return delta, info


def track_rois_between_frames(
    prev_gray: np.ndarray,
    next_gray: np.ndarray,
    roi_payload: dict[str, Any],
    *,
    max_corners: int = 80,
    min_track_points: int = 3,
    max_forward_backward_error: float = 1.5,
    track_scale: float = 1.0,
    track_window_padding: float = 200.0,
    track_retry_window_padding: float = 800.0,
    max_roi_jump: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if track_scale <= 0 or not np.isfinite(track_scale):
        raise ValueError("track_scale must be a finite positive value.")
    if track_window_padding < 0 or not np.isfinite(track_window_padding):
        raise ValueError("track_window_padding must be a finite non-negative value.")
    if track_retry_window_padding < 0 or not np.isfinite(track_retry_window_padding):
        raise ValueError("track_retry_window_padding must be a finite non-negative value.")

    normalized = dmt.normalize_roi_payload(roi_payload)
    tracked_rois: list[dict[str, Any]] = []
    tracking: list[dict[str, Any]] = []
    attempt_paddings = [float(track_window_padding)]
    if track_retry_window_padding > track_window_padding:
        attempt_paddings.append(float(track_retry_window_padding))

    for roi in normalized.get("rois", []):
        roi_idx = int(roi["roi_idx"])
        vertices = np.asarray(roi["vertices"], dtype=np.float32)
        tracked_vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in vertices]
        info: dict[str, Any] = {
            "status": "failed",
            "num_features": 0,
            "num_tracked": 0,
            "num_inliers": 0,
            "failure_reason": "tracking not attempted",
        }

        for attempt_idx, window_padding in enumerate(attempt_paddings, start=1):
            bounds = _tracking_window_bounds(prev_gray.shape[:2], vertices, window_padding)
            if bounds is None:
                info = {
                    "status": "failed",
                    "num_features": 0,
                    "num_tracked": 0,
                    "num_inliers": 0,
                    "failure_reason": "invalid tracking window",
                    "track_attempt": attempt_idx,
                    "track_window_padding": float(window_padding),
                }
                continue

            x0, y0, x1, y1 = bounds
            prev_window = prev_gray[y0:y1, x0:x1]
            next_window = next_gray[y0:y1, x0:x1]
            window_offset = np.asarray([x0, y0], dtype=np.float32)
            window_vertices = vertices - window_offset
            if attempt_idx > 1:
                template_delta, template_info = _estimate_template_translation(
                    prev_window,
                    next_window,
                    window_vertices,
                )
            else:
                template_delta = None
                template_info = {"template_match_status": "not-used"}

            if track_scale < 1.0:
                scaled_width = max(1, int(round(prev_window.shape[1] * track_scale)))
                scaled_height = max(1, int(round(prev_window.shape[0] * track_scale)))
                prev_track = cv2.resize(prev_window, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
                next_track = cv2.resize(next_window, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
                coord_scale = float(track_scale)
            else:
                prev_track = prev_window
                next_track = next_window
                coord_scale = 1.0

            track_vertices = window_vertices * coord_scale
            initial_delta = template_delta * coord_scale if template_delta is not None else None
            matrix, estimate_info = _estimate_roi_transform(
                prev_track,
                next_track,
                track_vertices,
                max_corners=max_corners,
                min_track_points=min_track_points,
                max_forward_backward_error=max_forward_backward_error,
                initial_flow_delta=initial_delta,
            )
            info = {
                "track_attempt": attempt_idx,
                "track_scale": float(coord_scale),
                "track_window": [int(x0), int(y0), int(x1), int(y1)],
                "track_window_padding": float(window_padding),
                **template_info,
                **estimate_info,
            }
            if matrix is None:
                continue

            scaled_tracked = cv2.transform(track_vertices.reshape(1, -1, 2).astype(np.float32), matrix)[0]
            restored = scaled_tracked / coord_scale + window_offset
            rejection_reason = _tracked_roi_rejection_reason(vertices, restored, max_roi_jump)
            if rejection_reason is not None:
                info = {
                    **info,
                    "status": "failed",
                    "failure_reason": rejection_reason,
                }
                continue

            tracked_vertices = [[round(float(x), 3), round(float(y), 3)] for x, y in restored]
            break

        tracked_rois.append({"roi_idx": roi_idx, "vertices": tracked_vertices})
        tracking.append(
            {
                "roi_idx": roi_idx,
                **info,
            }
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
            anchor_roi=anchors_by_idx.get(int(roi["roi_idx"])),
            max_anchor_drift=max_anchor_drift,
        )
        corrected_rois.append(corrected_roi)
        updates.append(update)

    return {"num_rois": len(corrected_rois), "rois": corrected_rois}, updates


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


def _translate_points(points: Any, delta: np.ndarray) -> list[list[float]]:
    translated = np.asarray(points, dtype=np.float32).reshape(-1, 2) + delta.astype(np.float32)
    return _format_vertices(translated)


def _trusted_detection_results(results: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    trusted: dict[int, dict[str, Any]] = {}
    for result in results:
        if result.get("status") == "ok" and result.get("confidence") in {"medium", "high"}:
            trusted[int(result["roi_idx"])] = result
    return trusted


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
    rejected_statuses = {"rejected-anchor-drift", "rejected-roi-jump", "rejected-corrected-roi"}
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
    if len(image_paths) < 2:
        raise ValueError("At least two ordered frames are required.")

    out_json = resolve_output_json(args.out_json, args.out_dir)
    roi_payload, rois_source = load_initial_roi_payload(args, image_paths[0], out_json.parent)
    config = make_detection_config(args)

    reviewer: ReviewWindow | None = None
    writer: cv2.VideoWriter | None = None
    out_video = resolve_output_video(args.out_video, args.out_dir)

    prev_gray = dmt.read_image_unicode(image_paths[0], cv2.IMREAD_GRAYSCALE)
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
        )
        _apply_roi_vertices_to_results(results, roi_payload)
    anchor_roi_payload = dmt.normalize_roi_payload(roi_payload)
    last_detection_results_by_idx = _real_detection_results(results)
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
            previous_roi_payload = roi_payload
            predicted_roi_payload, tracking = track_rois_between_frames(
                prev_gray,
                next_gray,
                previous_roi_payload,
                max_corners=args.max_corners,
                min_track_points=args.min_track_points,
                max_forward_backward_error=args.max_forward_backward_error,
                track_scale=args.track_scale,
                track_window_padding=args.track_window_padding,
                track_retry_window_padding=args.track_retry_window_padding,
                max_roi_jump=args.max_roi_jump,
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
                    anchor_payload=anchor_roi_payload,
                    max_anchor_drift=getattr(args, "max_anchor_drift", None),
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
            else:
                roi_payload = predicted_roi_payload
            last_detection_results_by_idx.update(_real_detection_results(results))
            frames.append(make_frame_centerline_summary(image_path, "tracked_from_previous_frame", results))
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
        default=".",
        help="Directory for default output JSON. Relative paths are resolved from the current working directory.",
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
        "--body-guidance",
        dest="body_guidance",
        action="store_true",
        default=True,
        help="Use foreground/body evidence to re-rank detector edge-pair candidates.",
    )
    parser.add_argument(
        "--no-body-guidance",
        dest="body_guidance",
        action="store_false",
        help="Disable foreground/body candidate re-ranking in the detector.",
    )
    parser.add_argument(
        "--body-min-contrast",
        type=float,
        default=35.0,
        help="Minimum ROI gray dynamic range required for detector body guidance.",
    )
    parser.add_argument(
        "--body-min-fraction",
        type=float,
        default=0.50,
        help="Minimum dark foreground column fraction used to form detector body spans.",
    )
    parser.add_argument(
        "--body-score-weight",
        type=float,
        default=0.25,
        help="Soft weight of body score in detector candidate scoring.",
    )
    parser.add_argument(
        "--body-edge-gate-frac",
        type=float,
        default=0.90,
        help="Only detector candidates with this fraction of the best edge score can receive body-score boosts.",
    )
    parser.add_argument(
        "--body-adjustment-scale",
        type=float,
        default=0.50,
        help="Scale applied to detector body-score adjustment after edge gating.",
    )
    parser.add_argument(
        "--body-min-span-ratio",
        type=float,
        default=0.45,
        help="Minimum foreground span width relative to detector candidate tube width.",
    )
    parser.add_argument(
        "--body-max-span-ratio",
        type=float,
        default=1.12,
        help="Maximum foreground span width relative to detector candidate tube width.",
    )
    parser.add_argument(
        "--body-center-tolerance-frac",
        type=float,
        default=0.22,
        help="Allowed detector body center offset as a fraction of candidate tube width.",
    )
    parser.add_argument(
        "--body-outside-margin-frac",
        type=float,
        default=0.10,
        help="Outside band width used to penalize detector body leakage past candidate edges.",
    )
    parser.add_argument(
        "--body-outside-penalty-weight",
        type=float,
        default=0.35,
        help="Penalty weight when detector body evidence continues outside candidate edges.",
    )
    parser.add_argument(
        "--body-hard-reject",
        dest="body_hard_reject",
        action="store_true",
        default=False,
        help="Reject weak detector edge candidates with very poor body evidence.",
    )
    parser.add_argument(
        "--no-body-hard-reject",
        dest="body_hard_reject",
        action="store_false",
        help="Disable hard rejection by body evidence.",
    )
    parser.add_argument(
        "--track-scale",
        type=float,
        default=None,
        help="Scale used only for optical-flow tracking. Detection and JSON stay in original coordinates.",
    )
    parser.add_argument(
        "--track-window-padding",
        type=float,
        default=200.0,
        help="Original-image pixels added around each ROI for local optical-flow tracking.",
    )
    parser.add_argument(
        "--track-retry-window-padding",
        type=float,
        default=800.0,
        help="Larger local window used to retry only ROIs whose first optical-flow attempt failed.",
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
        "--max-anchor-drift",
        type=float,
        default=30.0,
        help="Maximum corrected ROI center drift from the first-frame anchor ROI in pixels. Use 0 to disable.",
    )
    parser.add_argument(
        "--roi-padding-frac",
        type=float,
        default=0.35,
        help="Extra ROI width fraction added after detection correction.",
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
        args.track_scale = 1.0
    if args.images is not None and len(args.images) < 2:
        parser.error("--images must contain at least two ordered frames.")
    if args.progress_every < 0:
        parser.error("--progress-every must be non-negative.")
    if args.track_scale <= 0 or not np.isfinite(args.track_scale):
        parser.error("--track-scale must be a finite positive value.")
    if args.track_window_padding < 0 or not np.isfinite(args.track_window_padding):
        parser.error("--track-window-padding must be a finite non-negative value.")
    if args.track_retry_window_padding < 0 or not np.isfinite(args.track_retry_window_padding):
        parser.error("--track-retry-window-padding must be a finite non-negative value.")
    if args.roi_padding_frac < 0:
        parser.error("--roi-padding-frac must be non-negative.")
    if args.roi_length_padding_frac < 0:
        parser.error("--roi-length-padding-frac must be non-negative.")
    if args.min_corrected_roi_short_span < 0:
        parser.error("--min-corrected-roi-short-span must be non-negative.")
    if args.max_corrected_roi_growth < 0:
        parser.error("--max-corrected-roi-growth must be non-negative.")
    if args.max_anchor_drift < 0:
        parser.error("--max-anchor-drift must be non-negative.")
    if args.max_roi_jump <= 0:
        parser.error("--max-roi-jump must be positive.")
    if args.body_min_contrast < 0:
        parser.error("--body-min-contrast must be non-negative.")
    if not 0 <= args.body_min_fraction <= 1:
        parser.error("--body-min-fraction must be between 0 and 1.")
    if args.body_score_weight < 0:
        parser.error("--body-score-weight must be non-negative.")
    if not 0 <= args.body_edge_gate_frac <= 1:
        parser.error("--body-edge-gate-frac must be between 0 and 1.")
    if args.body_adjustment_scale < 0:
        parser.error("--body-adjustment-scale must be non-negative.")
    if args.body_min_span_ratio <= 0:
        parser.error("--body-min-span-ratio must be positive.")
    if args.body_max_span_ratio < args.body_min_span_ratio:
        parser.error("--body-max-span-ratio must be greater than or equal to --body-min-span-ratio.")
    if args.body_center_tolerance_frac <= 0:
        parser.error("--body-center-tolerance-frac must be positive.")
    if args.body_outside_margin_frac < 0:
        parser.error("--body-outside-margin-frac must be non-negative.")
    if args.body_outside_penalty_weight < 0:
        parser.error("--body-outside-penalty-weight must be non-negative.")
    return args


def main() -> int:
    args = parse_args()
    payload = run_sequence(args)
    out_json = resolve_output_json(args.out_json, args.out_dir)
    print(f"Processed {payload['num_frames']} frames.")
    print(f"Results: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
