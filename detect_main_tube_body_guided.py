#!/usr/bin/env python
"""Detect the main tube inside polygon ROIs.

The detector rectifies each polygon ROI along its main axis, finds a pair of
long tube-side edges in the rectified image, then maps those lines back to the
original image.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np


MAX_RECTIFIED_DIM = 32000
MAX_RECTIFIED_PIXELS = 40_000_000
MIN_EDGE_PAIR_SEPARATION_PX = 8
LEFT_BOUNDARY_COLOR = (0, 0, 255)
RIGHT_BOUNDARY_COLOR = (0, 255, 0)
CENTERLINE_COLOR = (0, 255, 255)


@dataclass
class DetectionConfig:
    min_width_frac: float = 0.28
    max_width_frac: float = 0.92
    max_profile_peaks: int = 18
    boundary_samples: int = 60
    gaussian_ksize: int = 5
    centerline_foreground_correction: bool = False
    centerline_foreground_min_height: int = 650
    centerline_foreground_min_fraction: float = 0.50
    centerline_foreground_min_span_ratio: float = 0.45
    centerline_foreground_max_span_ratio: float = 0.90
    centerline_foreground_min_shift: float = 6.0
    centerline_foreground_max_shift_frac: float = 0.30
    body_guidance: bool = True
    body_min_contrast: float = 35.0
    body_min_fraction: float = 0.50
    body_score_weight: float = 0.25
    body_min_span_ratio: float = 0.45
    body_max_span_ratio: float = 1.12
    body_center_tolerance_frac: float = 0.22
    body_outside_margin_frac: float = 0.10
    body_outside_penalty_weight: float = 0.35
    body_hard_reject: bool = False
    body_edge_gate_frac: float = 0.90
    body_adjustment_scale: float = 0.50


@dataclass
class RectifyMeta:
    center: np.ndarray
    long_axis: np.ndarray
    short_axis: np.ndarray
    long_min: float
    short_min: float
    width: int
    height: int


@dataclass
class BoundaryLineModel:
    slope: float
    intercept: float
    support_rows: int
    median_residual: float

    def x_at(self, y: float) -> float:
        return self.slope * float(y) + self.intercept


@dataclass
class RoiViewport:
    image_width: int
    image_height: int
    canvas_width: int
    canvas_height: int
    fit_scale: float
    zoom: float = 1.0
    offset_x: float = 0.0
    offset_y: float = 0.0
    min_zoom: float = 1.0
    max_zoom: float = 16.0

    @classmethod
    def for_image(
        cls,
        image_width: int,
        image_height: int,
        max_width: int = 1400,
        max_height: int = 900,
    ) -> "RoiViewport":
        fit_scale = min(max_width / max(1, image_width), max_height / max(1, image_height), 1.0)
        canvas_width = max(1, int(round(image_width * fit_scale)))
        canvas_height = max(1, int(round(image_height * fit_scale)))
        return cls(image_width, image_height, canvas_width, canvas_height, fit_scale)

    @property
    def scale(self) -> float:
        return self.fit_scale * self.zoom

    def reset(self) -> None:
        self.zoom = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

    def _clamp_offset(self) -> None:
        visible_width = self.canvas_width / self.scale
        visible_height = self.canvas_height / self.scale
        max_x = max(0.0, self.image_width - visible_width)
        max_y = max(0.0, self.image_height - visible_height)
        self.offset_x = min(max(self.offset_x, 0.0), max_x)
        self.offset_y = min(max(self.offset_y, 0.0), max_y)

    def canvas_to_image(self, x: float, y: float) -> list[float]:
        image_x = self.offset_x + float(x) / self.scale
        image_y = self.offset_y + float(y) / self.scale
        image_x = min(max(image_x, 0.0), float(self.image_width - 1))
        image_y = min(max(image_y, 0.0), float(self.image_height - 1))
        return [round(image_x, 3), round(image_y, 3)]

    def image_to_canvas(self, points: list[list[float]]) -> np.ndarray:
        scaled = [
            [round((float(x) - self.offset_x) * self.scale), round((float(y) - self.offset_y) * self.scale)]
            for x, y in points
        ]
        return np.asarray(scaled, dtype=np.int32)

    def zoom_at(self, canvas_x: float, canvas_y: float, factor: float) -> None:
        anchor_x, anchor_y = self.canvas_to_image(canvas_x, canvas_y)
        self.zoom = min(max(self.zoom * factor, self.min_zoom), self.max_zoom)
        self.offset_x = anchor_x - float(canvas_x) / self.scale
        self.offset_y = anchor_y - float(canvas_y) / self.scale
        self._clamp_offset()

    def pan(self, delta_x: float, delta_y: float) -> None:
        self.offset_x -= float(delta_x) / self.scale
        self.offset_y -= float(delta_y) / self.scale
        self._clamp_offset()

    def render(self, image: np.ndarray) -> np.ndarray:
        transform = np.asarray(
            [
                [self.scale, 0.0, -self.offset_x * self.scale],
                [0.0, self.scale, -self.offset_y * self.scale],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            image,
            transform,
            (self.canvas_width, self.canvas_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )


def read_image_unicode(path: str | Path, flags: int = cv2.IMREAD_GRAYSCALE) -> np.ndarray:
    path = Path(path)
    data = np.frombuffer(path.read_bytes(), dtype=np.uint8)
    image = cv2.imdecode(data, flags)
    if image is None:
        raise ValueError(f"Could not decode image: {path}")
    return image


def write_image_unicode(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix if path.suffix else ".png"
    ok, encoded = cv2.imencode(ext, image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def load_roi_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def normalize_roi_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized_rois: list[dict[str, Any]] = []
    for fallback_idx, roi in enumerate(payload.get("rois", []), start=1):
        raw_vertices = roi.get("vertices")
        if raw_vertices is None:
            raw_vertices = roi.get("roi_points")
        if raw_vertices is None:
            raise ValueError(f"ROI {fallback_idx} must contain vertices or roi_points.")

        vertices: list[list[float]] = []
        for point_idx, point in enumerate(raw_vertices, start=1):
            if len(point) < 2:
                raise ValueError(f"ROI {fallback_idx} point {point_idx} must contain x and y.")
            vertices.append([float(point[0]), float(point[1])])
        if len(vertices) < 3:
            raise ValueError(f"ROI {fallback_idx} must contain at least 3 points.")

        normalized_rois.append(
            {
                "roi_idx": int(roi.get("roi_idx", fallback_idx)),
                "vertices": vertices,
            }
        )

    return {
        **payload,
        "num_rois": len(normalized_rois),
        "rois": normalized_rois,
    }


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _draw_roi_editor_canvas(
    image_bgr: np.ndarray,
    viewport: RoiViewport,
    saved_rois: list[list[list[float]]],
    current_points: list[list[float]],
) -> np.ndarray:
    canvas = viewport.render(image_bgr)
    for roi_idx, vertices in enumerate(saved_rois, start=1):
        pts = viewport.image_to_canvas(vertices)
        cv2.polylines(canvas, [pts], True, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(
            canvas,
            f"R{roi_idx}",
            tuple(pts.mean(axis=0).round().astype(int)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    if current_points:
        pts = viewport.image_to_canvas(current_points)
        for point in pts:
            cv2.circle(canvas, tuple(point), 4, (0, 0, 255), -1, cv2.LINE_AA)
        if len(pts) >= 2:
            cv2.polylines(canvas, [pts], False, (0, 0, 255), 2, cv2.LINE_AA)

    help_lines = [
        "Left click: add point",
        "Wheel: zoom",
        "Right drag: pan",
        "r: reset view",
        "n: finish current ROI",
        "u: undo last point",
        "q/Enter: run detection",
        "Esc: cancel",
        f"zoom: {viewport.zoom:.2f}x",
    ]
    y = 24
    for line in help_lines:
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(canvas, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 1, cv2.LINE_AA)
        y += 26
    return canvas


def _append_current_roi(
    saved_rois: list[list[list[float]]],
    current_points: list[list[float]],
) -> list[list[float]]:
    if len(current_points) < 3:
        raise ValueError("Current ROI must contain at least 3 points before it can be saved.")
    saved_rois.append([[float(x), float(y)] for x, y in current_points])
    return []


def draw_rois_interactive(image_path: str | Path, image_bgr: np.ndarray) -> dict[str, Any]:
    image_height, image_width = image_bgr.shape[:2]
    viewport = RoiViewport.for_image(image_width, image_height)
    window_name = "Draw ROIs"
    saved_rois: list[list[list[float]]] = []
    current_points: list[list[float]] = []
    dragging_view = False
    last_drag_pos = [0, 0]

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal dragging_view
        if event == cv2.EVENT_LBUTTONDOWN:
            current_points.append(viewport.canvas_to_image(x, y))
        elif event == cv2.EVENT_RBUTTONDOWN:
            dragging_view = True
            last_drag_pos[0] = x
            last_drag_pos[1] = y
        elif event == cv2.EVENT_MOUSEMOVE and dragging_view:
            viewport.pan(x - last_drag_pos[0], y - last_drag_pos[1])
            last_drag_pos[0] = x
            last_drag_pos[1] = y
        elif event == cv2.EVENT_RBUTTONUP:
            dragging_view = False
        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = cv2.getMouseWheelDelta(_flags) if hasattr(cv2, "getMouseWheelDelta") else _flags
            viewport.zoom_at(x, y, 1.25 if wheel_delta > 0 else 0.8)

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, viewport.canvas_width, viewport.canvas_height)
    cv2.setMouseCallback(window_name, on_mouse)
    print("Draw ROI: left click points, mouse wheel zoom, right drag pan, r reset, n finish ROI, u undo, q/Enter run detection, Esc cancel.")

    try:
        while True:
            canvas = _draw_roi_editor_canvas(image_bgr, viewport, saved_rois, current_points)
            cv2.imshow(window_name, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                try:
                    if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                        raise KeyboardInterrupt("ROI drawing window was closed.")
                except cv2.error as exc:
                    raise KeyboardInterrupt("ROI drawing window was closed.") from exc
                continue
            if key == ord("u"):
                if current_points:
                    current_points.pop()
                continue
            if key == ord("r"):
                viewport.reset()
                continue
            if key == ord("n"):
                try:
                    current_points = _append_current_roi(saved_rois, current_points)
                except ValueError as exc:
                    print(exc)
                continue
            if key in (ord("q"), 13, 10):
                if current_points:
                    current_points = _append_current_roi(saved_rois, current_points)
                break
            if key == 27:
                raise KeyboardInterrupt("ROI drawing cancelled.")
    finally:
        cv2.destroyWindow(window_name)

    return normalize_roi_payload(
        {
            "image": str(image_path),
            "rois": [
                {
                    "roi_idx": roi_idx,
                    "vertices": vertices,
                }
                for roi_idx, vertices in enumerate(saved_rois, start=1)
            ],
        }
    )


def _smooth_1d(values: np.ndarray, sigma: float) -> np.ndarray:
    row = np.asarray(values, dtype=np.float32).reshape(1, -1)
    return cv2.GaussianBlur(row, (0, 0), sigmaX=sigma).ravel()


def _basis_from_polygon(points: np.ndarray) -> RectifyMeta:
    points = np.asarray(points, dtype=np.float32)
    rect = cv2.minAreaRect(points)
    box = cv2.boxPoints(rect).astype(np.float32)

    sides: list[tuple[float, np.ndarray]] = []
    for i in range(4):
        p = box[i]
        q = box[(i + 1) % 4]
        delta = q - p
        sides.append((float(np.linalg.norm(delta)), delta))
    sides.sort(key=lambda item: item[0], reverse=True)

    long_axis = sides[0][1]
    long_axis = long_axis / (np.linalg.norm(long_axis) + 1e-6)
    short_axis = np.array([-long_axis[1], long_axis[0]], dtype=np.float32)
    center = np.asarray(rect[0], dtype=np.float32)

    rel = points - center
    long_projection = rel @ long_axis
    short_projection = rel @ short_axis

    margin = 4.0
    long_min = float(long_projection.min() - margin)
    long_max = float(long_projection.max() + margin)
    short_min = float(short_projection.min() - margin)
    short_max = float(short_projection.max() + margin)

    height = max(30, int(round(long_max - long_min)))
    width = max(20, int(round(short_max - short_min)))
    return RectifyMeta(center, long_axis, short_axis, long_min, short_min, width, height)


def rectify_polygon(gray: np.ndarray, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, RectifyMeta]:
    meta = _basis_from_polygon(points)
    if (
        not np.all(np.isfinite(meta.center))
        or not np.all(np.isfinite(meta.long_axis))
        or not np.all(np.isfinite(meta.short_axis))
        or meta.width <= 0
        or meta.height <= 0
        or meta.width >= MAX_RECTIFIED_DIM
        or meta.height >= MAX_RECTIFIED_DIM
        or meta.width * meta.height > MAX_RECTIFIED_PIXELS
    ):
        raise ValueError("invalid or oversized roi")

    xs = np.arange(meta.width, dtype=np.float32) + 0.5
    ys = np.arange(meta.height, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(xs, ys)

    map_x = (
        meta.center[0]
        + meta.long_axis[0] * (grid_y + meta.long_min)
        + meta.short_axis[0] * (grid_x + meta.short_min)
    )
    map_y = (
        meta.center[1]
        + meta.long_axis[1] * (grid_y + meta.long_min)
        + meta.short_axis[1] * (grid_x + meta.short_min)
    )
    crop = cv2.remap(gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    rel = points.astype(np.float32) - meta.center
    poly_x = rel @ meta.short_axis - meta.short_min
    poly_y = rel @ meta.long_axis - meta.long_min
    polygon = np.stack([poly_x, poly_y], axis=1).round().astype(np.int32)

    mask = np.zeros((meta.height, meta.width), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon], 255)
    return crop, mask, meta


def _find_profile_peaks(profile: np.ndarray, max_peaks: int) -> list[tuple[float, int]]:
    peaks: list[tuple[float, int]] = []
    for x in range(2, len(profile) - 2):
        if profile[x] >= profile[x - 1] and profile[x] >= profile[x + 1] and profile[x] > 0:
            peaks.append((float(profile[x]), x))
    peaks.sort(reverse=True)

    filtered: list[tuple[float, int]] = []
    min_gap = max(4, int(0.05 * len(profile)))
    for value, x in peaks:
        if all(abs(x - existing_x) >= min_gap for _, existing_x in filtered):
            filtered.append((value, x))
        if len(filtered) >= max_peaks:
            break
    return filtered


def _edge_evidence(crop: np.ndarray, mask: np.ndarray, config: DetectionConfig) -> tuple[np.ndarray, np.ndarray]:
    ksize = config.gaussian_ksize if config.gaussian_ksize % 2 == 1 else config.gaussian_ksize + 1
    blurred = cv2.GaussianBlur(crop, (ksize, ksize), 0)
    gx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3)

    abs_x = np.abs(gx)
    abs_y = np.abs(gy)
    orientation = abs_x / (abs_x + abs_y + 1e-3)
    evidence = abs_x * np.clip((orientation - 0.45) / 0.55, 0, 1)

    mask_f = (mask > 0).astype(np.float32)
    counts = mask_f.sum(axis=0)
    profile = (evidence * mask_f).sum(axis=0) / np.maximum(counts, 1)
    profile[counts < 0.30 * crop.shape[0]] = 0
    sigma = max(1.2, crop.shape[1] / 55.0)
    return evidence, _smooth_1d(profile, sigma=sigma)


def _boundary_support(evidence: np.ndarray, mask: np.ndarray, x: int, threshold: float) -> float:
    height, width = evidence.shape
    win = max(2, int(round(width * 0.025)))
    left = max(0, x - win)
    right = min(width, x + win + 1)
    edge_band = evidence[:, left:right]
    mask_band = mask[:, left:right] > 0
    valid_rows = mask_band.max(axis=1)
    if not np.any(valid_rows):
        return 0.0
    supported_rows = (edge_band.max(axis=1) > threshold) & valid_rows
    return float(supported_rows.sum() / valid_rows.sum())


def _foreground_column_profile(crop: np.ndarray, mask: np.ndarray, min_contrast: float = 35.0) -> np.ndarray | None:
    valid_pixels = crop[mask > 0]
    if valid_pixels.size == 0:
        return None

    p10, p35, p60, p90 = np.percentile(valid_pixels, [10, 35, 60, 90])
    if float(p90 - p10) < float(min_contrast):
        return None

    dark_threshold = float(max(p35 + 5.0, p60 - 3.0))
    mask_f = (mask > 0).astype(np.float32)
    counts = mask_f.sum(axis=0)
    dark_counts = ((crop.astype(np.float32) <= dark_threshold) * mask_f).sum(axis=0)
    profile = dark_counts / np.maximum(counts, 1.0)
    profile[counts < max(5.0, crop.shape[0] * 0.12)] = 0.0
    return _smooth_1d(profile.astype(np.float32), sigma=max(1.0, crop.shape[1] / 90.0))


def _clamp01(value: float) -> float:
    return float(min(1.0, max(0.0, value)))


def _mean_profile(profile: np.ndarray, start: int, end: int) -> float:
    width = int(profile.shape[0])
    start = int(np.clip(start, 0, width))
    end = int(np.clip(end, 0, width))
    if end <= start:
        return 0.0
    return float(np.mean(profile[start:end]))


def _foreground_span_around(profile: np.ndarray, center_x: int, min_fraction: float) -> tuple[int, int] | None:
    width = int(profile.shape[0])
    if width == 0:
        return None

    center_x = int(np.clip(center_x, 0, width - 1))
    if float(profile[center_x]) < min_fraction:
        nearby = np.arange(max(0, center_x - 4), min(width, center_x + 5), dtype=np.int32)
        valid = nearby[profile[nearby] >= min_fraction]
        if valid.size == 0:
            return None
        center_x = int(valid[np.argmin(np.abs(valid - center_x))])

    max_gap = max(3, int(round(width * 0.06)))

    left = center_x
    last_active = center_x
    gap = 0
    x = center_x - 1
    while x >= 0:
        if float(profile[x]) >= min_fraction:
            last_active = x
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        x -= 1
    left = last_active

    right = center_x
    last_active = center_x
    gap = 0
    x = center_x + 1
    while x < width:
        if float(profile[x]) >= min_fraction:
            last_active = x
            gap = 0
        else:
            gap += 1
            if gap > max_gap:
                break
        x += 1
    right = last_active

    if right - left < max(8, int(round(width * 0.12))):
        return None
    return left, right


def _body_alignment_score(
    foreground: np.ndarray | None,
    left_x: int,
    right_x: int,
    width: int,
    config: DetectionConfig,
) -> tuple[float, dict[str, Any]]:
    edge_width = int(right_x) - int(left_x)
    if foreground is None:
        return 0.5, {"available": False, "reason": "foreground profile unavailable"}
    if edge_width <= 0:
        return 0.0, {"available": True, "reason": "invalid candidate width"}

    edge_center = (float(left_x) + float(right_x)) / 2.0
    inside_strength = _mean_profile(foreground, int(left_x), int(right_x) + 1)
    inside_score = _clamp01((inside_strength - 0.25) / 0.45)

    margin = max(3, int(round(edge_width * max(0.0, float(config.body_outside_margin_frac)))))
    left_outside = _mean_profile(foreground, int(left_x) - margin, int(left_x))
    right_outside = _mean_profile(foreground, int(right_x) + 1, int(right_x) + 1 + margin)
    outside_strength = max(left_outside, right_outside)

    span = _foreground_span_around(
        foreground,
        int(round(edge_center)),
        min_fraction=float(config.body_min_fraction),
    )
    if span is None:
        score = _clamp01(0.40 * inside_score - float(config.body_outside_penalty_weight) * outside_strength)
        return score, {
            "available": True,
            "reason": "no foreground span around candidate center",
            "inside_strength": round(float(inside_strength), 3),
            "outside_strength": round(float(outside_strength), 3),
        }

    span_left, span_right = span
    span_width = max(0.0, float(span_right - span_left))
    span_ratio = span_width / max(1.0, float(edge_width))
    min_ratio = max(1e-6, float(config.body_min_span_ratio))
    max_ratio = max(min_ratio + 1e-6, float(config.body_max_span_ratio))
    if span_ratio < min_ratio:
        span_ratio_score = _clamp01(span_ratio / min_ratio)
    elif span_ratio > max_ratio:
        span_ratio_score = _clamp01(max_ratio / max(span_ratio, 1e-6))
    else:
        span_ratio_score = 1.0

    span_center = (float(span_left) + float(span_right)) / 2.0
    center_delta_frac = abs(span_center - edge_center) / max(1.0, float(edge_width))
    center_tolerance = max(1e-6, float(config.body_center_tolerance_frac))
    center_score = _clamp01(1.0 - center_delta_frac / center_tolerance)

    allowed_left = float(left_x - margin)
    allowed_right = float(right_x + margin)
    span_overrun = max(0.0, allowed_left - float(span_left), float(span_right) - allowed_right)
    containment_score = _clamp01(1.0 - span_overrun / max(1.0, float(edge_width) * 0.25))

    outside_penalty = float(config.body_outside_penalty_weight) * outside_strength
    score = _clamp01(
        0.35 * span_ratio_score
        + 0.30 * center_score
        + 0.25 * inside_score
        + 0.10 * containment_score
        - outside_penalty
    )

    return score, {
        "available": True,
        "reason": "ok",
        "span": [int(span_left), int(span_right)],
        "span_ratio": round(float(span_ratio), 3),
        "span_ratio_score": round(float(span_ratio_score), 3),
        "center_delta_frac": round(float(center_delta_frac), 3),
        "center_score": round(float(center_score), 3),
        "inside_strength": round(float(inside_strength), 3),
        "inside_score": round(float(inside_score), 3),
        "outside_strength": round(float(outside_strength), 3),
        "containment_score": round(float(containment_score), 3),
    }


def _nearest_peak_to_edge(
    peaks: list[tuple[float, int]],
    profile: np.ndarray,
    start_x: int,
    edge_x: int,
    side: str,
    margin: int,
    min_strength_ratio: float = 0.20,
) -> int | None:
    width = int(profile.shape[0])
    start_x = int(np.clip(start_x, 0, max(0, width - 1)))
    current_strength = float(profile[start_x]) if width else 0.0
    min_strength = max(1e-6, current_strength * float(min_strength_ratio))

    if side == "right":
        candidates = [x for value, x in peaks if start_x < x <= edge_x + margin and float(value) >= min_strength]
        return max(candidates) if candidates else None
    candidates = [x for value, x in peaks if edge_x - margin <= x < start_x and float(value) >= min_strength]
    return min(candidates) if candidates else None


def _edge_pair_metrics(
    left_x: int,
    right_x: int,
    profile: np.ndarray,
    evidence: np.ndarray,
    mask: np.ndarray,
    threshold: float,
    config: DetectionConfig,
    foreground: np.ndarray | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    width = int(profile.shape[0])
    tube_width = int(right_x) - int(left_x)
    width_frac = tube_width / max(1, width)
    if width_frac < 0.38:
        width_prior = width_frac / 0.38
    elif width_frac > 0.84:
        width_prior = max(0.55, (0.92 - width_frac) / 0.08)
    else:
        width_prior = 1.0

    left_strength = float(profile[int(np.clip(left_x, 0, width - 1))])
    right_strength = float(profile[int(np.clip(right_x, 0, width - 1))])
    left_support = _boundary_support(evidence, mask, left_x, threshold)
    right_support = _boundary_support(evidence, mask, right_x, threshold)
    continuity = 0.5 * (left_support + right_support)

    positive_profile = profile[profile > 0]
    edge_contrast = (
        (left_strength + right_strength) / (float(np.median(positive_profile)) + 1e-6)
        if positive_profile.size
        else 0.0
    )
    edge_score = (left_strength + right_strength) * (0.4 + continuity) * width_prior

    body_score = 0.5
    body_factor = 1.0
    body_debug: dict[str, Any] = {"available": False, "reason": "disabled"}
    if config.body_guidance:
        body_score, body_debug = _body_alignment_score(foreground, left_x, right_x, width, config)
        body_factor = 1.0 + float(config.body_score_weight) * float(config.body_adjustment_scale) * (float(body_score) - 0.5) * 2.0
        body_factor = max(0.35, float(body_factor))

    metrics = {
        "score": float(edge_score * body_factor),
        "edge_score": float(edge_score),
        "left_x": int(left_x),
        "right_x": int(right_x),
        "tube_width_px": int(tube_width),
        "tube_width_frac": float(width_frac),
        "left_strength": float(left_strength),
        "right_strength": float(right_strength),
        "continuity": float(continuity),
        "edge_contrast": float(edge_contrast),
        "body_score": float(body_score),
        "body_factor": float(body_factor),
        "body_available": bool(body_debug.get("available", False)),
        "body_reason": str(body_debug.get("reason", "")),
    }
    if body_debug.get("span") is not None:
        metrics["body_span"] = body_debug["span"]
    if body_debug.get("span_ratio") is not None:
        metrics["body_span_ratio"] = float(body_debug["span_ratio"])
    if body_debug.get("center_delta_frac") is not None:
        metrics["body_center_delta_frac"] = float(body_debug["center_delta_frac"])
    if body_debug.get("inside_strength") is not None:
        metrics["body_inside_strength"] = float(body_debug["inside_strength"])
    if body_debug.get("outside_strength") is not None:
        metrics["body_outside_strength"] = float(body_debug["outside_strength"])
    return metrics, body_debug


def _refine_pair_with_foreground_span(
    best: dict[str, Any],
    peaks: list[tuple[float, int]],
    profile: np.ndarray,
    evidence: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    config: DetectionConfig,
    threshold: float,
    foreground: np.ndarray | None = None,
) -> dict[str, Any]:
    if foreground is None:
        foreground = _foreground_column_profile(crop, mask, min_contrast=config.body_min_contrast)
    if foreground is None:
        return best

    width = crop.shape[1]
    left_x = int(best["left_x"])
    right_x = int(best["right_x"])
    center_x = int(round((left_x + right_x) / 2.0))
    span = _foreground_span_around(foreground, center_x, min_fraction=0.50)
    if span is None:
        return best

    span_left, span_right = span
    min_extra = max(5, int(round((right_x - left_x) * 0.10)))
    max_sep = max(12, int(config.max_width_frac * width))
    refined_left = left_x
    refined_right = right_x

    if left_x - span_left >= min_extra:
        peak_x = _nearest_peak_to_edge(peaks, profile, left_x, span_left, "left", margin=5)
        if peak_x is not None:
            refined_left = peak_x
    if span_right - right_x >= min_extra:
        peak_x = _nearest_peak_to_edge(peaks, profile, right_x, span_right, "right", margin=5)
        if peak_x is not None:
            refined_right = peak_x

    refined_width = refined_right - refined_left
    if refined_width <= 0 or refined_width > max_sep:
        return best
    if refined_left == left_x and refined_right == right_x:
        return best

    refined, _body_debug = _edge_pair_metrics(
        int(refined_left),
        int(refined_right),
        profile,
        evidence,
        mask,
        threshold,
        config,
        foreground,
    )
    if refined["continuity"] < 0.25 or refined["edge_contrast"] < 1.6:
        return best
    if refined["score"] < float(best.get("score", 0.0)) * 0.60:
        return best

    refined = {**best, **refined}
    refined["foreground_refine"] = {
        "from": [int(left_x), int(right_x)],
        "to": [int(refined_left), int(refined_right)],
        "span": [int(span_left), int(span_right)],
    }
    return refined


def find_edge_pair(crop: np.ndarray, mask: np.ndarray, config: DetectionConfig) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    height, width = crop.shape
    evidence, profile = _edge_evidence(crop, mask, config)
    peaks = _find_profile_peaks(profile, config.max_profile_peaks)
    foreground = (
        _foreground_column_profile(crop, mask, min_contrast=config.body_min_contrast)
        if config.body_guidance
        else None
    )

    valid_evidence = evidence[mask > 0]
    threshold = float(np.percentile(valid_evidence, 78)) if valid_evidence.size else 0.0
    min_sep = max(MIN_EDGE_PAIR_SEPARATION_PX, int(config.min_width_frac * width))
    max_sep = max(min_sep + 4, int(config.max_width_frac * width))

    candidates: list[dict[str, Any]] = []
    candidate_debug: list[dict[str, Any]] = []
    for i, (_strength_a, a) in enumerate(peaks):
        for _strength_b, b in peaks[i + 1 :]:
            left_x, right_x = sorted((a, b))
            tube_width = right_x - left_x
            if tube_width < min_sep or tube_width > max_sep:
                continue

            candidate, body_debug = _edge_pair_metrics(
                int(left_x),
                int(right_x),
                profile,
                evidence,
                mask,
                threshold,
                config,
                foreground,
            )
            if (
                config.body_hard_reject
                and foreground is not None
                and candidate["body_score"] < 0.15
                and candidate["edge_contrast"] < 1.8
                and candidate["continuity"] < 0.35
            ):
                candidate_debug.append(
                    {
                        "left_x": int(left_x),
                        "right_x": int(right_x),
                        "edge_score": round(float(candidate["edge_score"]), 3),
                        "body_score": round(float(candidate["body_score"]), 3),
                        "score": 0.0,
                        "rejected": "body-hard-reject",
                        "body_reason": str(body_debug.get("reason", "")),
                    }
                )
                continue
            candidates.append(candidate)

    peak_debug = [{"x": int(x), "value": round(float(value), 3)} for value, x in peaks[:10]]
    if candidates:
        edge_best_score = max(float(candidate["edge_score"]) for candidate in candidates)
        edge_gate_score = edge_best_score * float(config.body_edge_gate_frac)
        for candidate in candidates:
            edge_score = float(candidate["edge_score"])
            original_body_factor = float(candidate.get("body_factor", 1.0))
            if config.body_guidance and edge_score < edge_gate_score and original_body_factor > 1.0:
                candidate["body_factor"] = 1.0
                candidate["score"] = edge_score
                candidate["body_gate_status"] = "boost-blocked-by-edge-gate"
            elif config.body_guidance and edge_score < edge_gate_score:
                candidate["score"] = edge_score
                candidate["body_gate_status"] = "outside-edge-gate"
            else:
                candidate["score"] = edge_score * original_body_factor
                candidate["body_gate_status"] = "inside-edge-gate" if config.body_guidance else "disabled"
            candidate["edge_score_ratio"] = float(edge_score / max(edge_best_score, 1e-6))
            candidate_debug.append(
                {
                    "left_x": int(candidate["left_x"]),
                    "right_x": int(candidate["right_x"]),
                    "edge_score": round(float(candidate["edge_score"]), 3),
                    "edge_score_ratio": round(float(candidate["edge_score_ratio"]), 3),
                    "body_score": round(float(candidate["body_score"]), 3),
                    "body_factor": round(float(candidate["body_factor"]), 3),
                    "score": round(float(candidate["score"]), 3),
                    "body_gate_status": str(candidate.get("body_gate_status", "")),
                    "body_reason": str(candidate.get("body_reason", "")),
                }
            )
        best = max(candidates, key=lambda item: float(item["score"]))
    else:
        best = None
    candidate_debug.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    body_guidance_debug: dict[str, Any] = {
        "enabled": bool(config.body_guidance),
        "available": bool(foreground is not None),
        "edge_gate_frac": round(float(config.body_edge_gate_frac), 3),
        "adjustment_scale": round(float(config.body_adjustment_scale), 3),
    }
    if best is None:
        return {
            "status": "failed",
            "confidence": "failed",
            "score": 0.0,
            "candidate_peaks": peak_debug,
            "candidate_pairs": candidate_debug[:10],
            "body_guidance": body_guidance_debug,
            "failure_reason": "no valid pair of long side edges",
        }, profile, evidence

    best = _refine_pair_with_foreground_span(
        best,
        peaks,
        profile,
        evidence,
        crop,
        mask,
        config,
        threshold,
        foreground=foreground,
    )
    if config.body_guidance:
        body_score, body_debug = _body_alignment_score(foreground, int(best["left_x"]), int(best["right_x"]), width, config)
        best["body_score"] = float(body_score)
        best["body_available"] = bool(body_debug.get("available", False))
        best["body_reason"] = str(body_debug.get("reason", ""))
        if body_debug.get("span") is not None:
            best["body_span"] = body_debug["span"]
        if body_debug.get("span_ratio") is not None:
            best["body_span_ratio"] = float(body_debug["span_ratio"])
        if body_debug.get("center_delta_frac") is not None:
            best["body_center_delta_frac"] = float(body_debug["center_delta_frac"])
        if body_debug.get("inside_strength") is not None:
            best["body_inside_strength"] = float(body_debug["inside_strength"])
        if body_debug.get("outside_strength") is not None:
            best["body_outside_strength"] = float(body_debug["outside_strength"])
        body_guidance_debug.update(
            {
                "best_body_score": round(float(body_score), 3),
                "best_body_reason": str(body_debug.get("reason", "")),
            }
        )
        if body_debug.get("span") is not None:
            body_guidance_debug["best_body_span"] = body_debug["span"]
        if body_debug.get("span_ratio") is not None:
            body_guidance_debug["best_body_span_ratio"] = body_debug["span_ratio"]

    if (
        best["continuity"] >= 0.45
        and best["edge_contrast"] >= 2.2
        and 0.38 <= best["tube_width_frac"] <= 0.84
    ):
        confidence = "high"
    elif best["continuity"] >= 0.25 and best["edge_contrast"] >= 1.6:
        confidence = "medium"
    else:
        confidence = "low"

    best["status"] = "ok"
    best["confidence"] = confidence
    best["candidate_peaks"] = peak_debug
    best["candidate_pairs"] = candidate_debug[:10]
    best["body_guidance"] = body_guidance_debug
    return best, profile, evidence


def map_rectified_point(meta: RectifyMeta, x: float, y: float) -> list[float]:
    point = meta.center + meta.long_axis * (y + meta.long_min) + meta.short_axis * (x + meta.short_min)
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _fit_one_boundary_line(
    evidence: np.ndarray,
    mask: np.ndarray,
    expected_x: int,
    search_radius: int,
    min_strength: float,
) -> BoundaryLineModel | None:
    height, width = evidence.shape
    samples: list[tuple[float, float, float]] = []
    expected_x = int(np.clip(expected_x, 0, width - 1))

    for y in range(height):
        x0 = max(0, expected_x - search_radius)
        x1 = min(width, expected_x + search_radius + 1)
        if x1 <= x0:
            continue

        row_mask = mask[y, x0:x1] > 0
        if not np.any(row_mask):
            continue

        row_values = evidence[y, x0:x1].copy()
        row_values[~row_mask] = 0.0
        xs = np.arange(x0, x1, dtype=np.float32)
        sigma = max(2.0, search_radius / 2.0)
        spatial_weight = np.exp(-0.5 * ((xs - float(expected_x)) / sigma) ** 2)
        local_idx = int(np.argmax(row_values * spatial_weight))
        strength = float(row_values[local_idx])
        if strength <= min_strength:
            continue

        samples.append((float(y), float(x0 + local_idx), strength))

    min_samples = max(8, int(round(height * 0.18)))
    if len(samples) < min_samples:
        return None

    ys = np.asarray([sample[0] for sample in samples], dtype=np.float32)
    xs = np.asarray([sample[1] for sample in samples], dtype=np.float32)
    weights = np.sqrt(np.asarray([sample[2] for sample in samples], dtype=np.float32) + 1e-6)
    if float(np.ptp(ys)) < 4.0:
        return None

    slope, intercept = np.polyfit(ys, xs, 1, w=weights)
    predicted = slope * ys + intercept
    residual = np.abs(xs - predicted)
    median_residual = float(np.median(residual))
    mid_y = (height - 1) / 2.0
    if abs(float(slope * mid_y + intercept) - float(expected_x)) > max(4.0, search_radius * 0.55):
        return None
    if abs(float(slope)) > 0.45 or median_residual > max(2.5, search_radius * 0.35):
        return None

    return BoundaryLineModel(
        slope=float(slope),
        intercept=float(intercept),
        support_rows=len(samples),
        median_residual=median_residual,
    )


def _with_parallel_slope(model: BoundaryLineModel, slope: float, mid_y: float) -> BoundaryLineModel:
    return BoundaryLineModel(
        slope=float(slope),
        intercept=float(model.x_at(mid_y) - slope * mid_y),
        support_rows=model.support_rows,
        median_residual=model.median_residual,
    )


def _fit_boundary_lines(
    evidence: np.ndarray,
    mask: np.ndarray,
    left_x: int,
    right_x: int,
) -> tuple[BoundaryLineModel | None, BoundaryLineModel | None]:
    height, width = evidence.shape
    tube_width = max(1, int(right_x) - int(left_x))
    search_radius = max(6, min(int(round(tube_width * 0.45)), int(round(width * 0.24))))

    valid_evidence = evidence[mask > 0]
    positive_evidence = valid_evidence[valid_evidence > 0]
    if positive_evidence.size == 0:
        return None, None

    min_strength = float(np.percentile(positive_evidence, 70))
    left_model = _fit_one_boundary_line(evidence, mask, int(left_x), search_radius, min_strength)
    right_model = _fit_one_boundary_line(evidence, mask, int(right_x), search_radius, min_strength)
    if left_model is None or right_model is None:
        return None, None

    if abs(left_model.slope - right_model.slope) > 0.20:
        return None, None

    total_support = left_model.support_rows + right_model.support_rows
    parallel_slope = (
        left_model.slope * left_model.support_rows + right_model.slope * right_model.support_rows
    ) / max(1, total_support)
    mid_y = (height - 1) / 2.0
    return _with_parallel_slope(left_model, parallel_slope, mid_y), _with_parallel_slope(
        right_model,
        parallel_slope,
        mid_y,
    )


def _line_x(model: BoundaryLineModel | None, fallback_x: float, y: float, width: int) -> float:
    x = model.x_at(float(y)) if model is not None else float(fallback_x)
    return float(np.clip(x, 0.0, max(0.0, float(width - 1))))


def _band_has_mask(mask: np.ndarray, y: int, x: float) -> bool:
    width = mask.shape[1]
    center_x = int(round(x))
    band = mask[y, max(0, center_x - 2) : min(width, center_x + 3)]
    return bool(band.max(initial=0) > 0)


def _sample_boundaries(
    meta: RectifyMeta,
    mask: np.ndarray,
    left_x: int,
    right_x: int,
    count: int,
    left_model: BoundaryLineModel | None = None,
    right_model: BoundaryLineModel | None = None,
    center_offset_x: float = 0.0,
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    fallback_center_x = (left_x + right_x) / 2.0
    rows = np.arange(meta.height, dtype=np.int32)
    valid_rows: list[int] = []
    for y in rows:
        y_float = float(y)
        left_row_x = _line_x(left_model, float(left_x), y_float, meta.width)
        right_row_x = _line_x(right_model, float(right_x), y_float, meta.width)
        center_row_x = (left_row_x + right_row_x) / 2.0
        if (
            _band_has_mask(mask, int(y), left_row_x)
            or _band_has_mask(mask, int(y), right_row_x)
            or _band_has_mask(mask, int(y), center_row_x)
        ):
            valid_rows.append(int(y))

    if not valid_rows:
        valid_rows = [0, meta.height - 1]

    if len(valid_rows) > count:
        sample_rows = np.linspace(valid_rows[0], valid_rows[-1], count)
    else:
        sample_rows = np.asarray(valid_rows, dtype=np.float32)

    left = [
        map_rectified_point(meta, _line_x(left_model, float(left_x), float(y), meta.width), float(y))
        for y in sample_rows
    ]
    right = [
        map_rectified_point(meta, _line_x(right_model, float(right_x), float(y), meta.width), float(y))
        for y in sample_rows
    ]
    center = [
        map_rectified_point(
            meta,
            np.clip(
                (
                    (
                        _line_x(left_model, float(left_x), float(y), meta.width)
                        + _line_x(right_model, float(right_x), float(y), meta.width)
                    )
                    / 2.0
                    if left_model is not None and right_model is not None
                    else fallback_center_x
                )
                + float(center_offset_x),
                min(
                    _line_x(left_model, float(left_x), float(y), meta.width),
                    _line_x(right_model, float(right_x), float(y), meta.width),
                ),
                max(
                    _line_x(left_model, float(left_x), float(y), meta.width),
                    _line_x(right_model, float(right_x), float(y), meta.width),
                ),
            ),
            float(y),
        )
        for y in sample_rows
    ]
    return left, right, center


def _foreground_centerline_offset(
    crop: np.ndarray,
    mask: np.ndarray,
    left_x: int,
    right_x: int,
    config: DetectionConfig,
    left_model: BoundaryLineModel | None = None,
    right_model: BoundaryLineModel | None = None,
) -> tuple[float, dict[str, Any] | None]:
    if not config.centerline_foreground_correction:
        return 0.0, None
    if crop.shape[0] < int(config.centerline_foreground_min_height):
        return 0.0, None

    edge_width = float(int(right_x) - int(left_x))
    if edge_width <= MIN_EDGE_PAIR_SEPARATION_PX:
        return 0.0, None

    foreground = _foreground_column_profile(crop, mask, min_contrast=config.body_min_contrast)
    if foreground is None:
        return 0.0, None

    mid_y = (crop.shape[0] - 1) / 2.0
    edge_center_x = (
        (
            _line_x(left_model, float(left_x), mid_y, crop.shape[1])
            + _line_x(right_model, float(right_x), mid_y, crop.shape[1])
        )
        / 2.0
        if left_model is not None and right_model is not None
        else (float(left_x) + float(right_x)) / 2.0
    )
    span = _foreground_span_around(
        foreground,
        int(round(edge_center_x)),
        min_fraction=float(config.centerline_foreground_min_fraction),
    )
    if span is None:
        return 0.0, None

    span_left, span_right = span
    span_width = float(span_right - span_left)
    span_ratio = span_width / edge_width
    if span_ratio < float(config.centerline_foreground_min_span_ratio):
        return 0.0, None
    if span_ratio > float(config.centerline_foreground_max_span_ratio):
        return 0.0, None

    foreground_center_x = (float(span_left) + float(span_right)) / 2.0
    offset = foreground_center_x - float(edge_center_x)
    max_shift = edge_width * float(config.centerline_foreground_max_shift_frac)
    if abs(offset) < float(config.centerline_foreground_min_shift):
        return 0.0, None
    if abs(offset) > max_shift:
        return 0.0, None

    return float(offset), {
        "centerline_source": "foreground-body",
        "centerline_x_offset": round(float(offset), 3),
        "centerline_edge_center_x": round(float(edge_center_x), 3),
        "centerline_foreground_center_x": round(float(foreground_center_x), 3),
        "centerline_foreground_span": [int(span_left), int(span_right)],
        "centerline_foreground_span_ratio": round(float(span_ratio), 3),
    }


def centerline_geometry(centerline: list[list[float]]) -> dict[str, Any] | None:
    if len(centerline) < 2:
        return None

    x1, y1 = (float(centerline[0][0]), float(centerline[0][1]))
    x2, y2 = (float(centerline[-1][0]), float(centerline[-1][1]))
    dx = x2 - x1
    dy = y2 - y1
    norm = float(np.hypot(dx, dy))
    if norm <= 1e-9:
        return None

    moment = x1 * y2 - y1 * x2
    a = -dy / norm
    b = dx / norm
    c = -(a * x1 + b * y1)
    return {
        "endpoints": [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)],
        "plucker": [round(moment, 6), round(dx, 6), round(dy, 6)],
        "line_eq": {
            "A": round(a, 9),
            "B": round(b, 9),
            "C": round(c, 9),
        },
    }


def roi_vertices_from_detected_boundaries(
    result: dict[str, Any],
    *,
    padding_frac: float = 0.35,
    length_padding_frac: float = 0.10,
) -> list[list[float]] | None:
    if result.get("status") != "ok":
        return None

    left = np.asarray(result.get("left_boundary", []), dtype=np.float32)
    right = np.asarray(result.get("right_boundary", []), dtype=np.float32)
    if left.ndim != 2 or right.ndim != 2 or left.shape[1:] != (2,) or right.shape[1:] != (2,):
        return None

    count = min(len(left), len(right))
    if count < 2:
        return None
    left = left[:count]
    right = right[:count]
    centers = (left + right) / 2.0

    centerline = np.asarray(result.get("centerline", []), dtype=np.float32)
    axis_points = centerline if centerline.ndim == 2 and len(centerline) >= 2 and centerline.shape[1:] == (2,) else centers
    direction = axis_points[-1] - axis_points[0]
    direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        centered = centers - centers.mean(axis=0)
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
        direction = vh[0].astype(np.float32)
        direction_norm = float(np.linalg.norm(direction))
    if direction_norm <= 1e-6:
        return None
    long_axis = direction / direction_norm

    width_axis = np.asarray([-long_axis[1], long_axis[0]], dtype=np.float32)
    mean_width = (right - left).mean(axis=0)
    if float(np.dot(width_axis, mean_width)) < 0:
        width_axis = -width_axis

    origin = centers.mean(axis=0)
    all_points = np.vstack([left, right, centers])
    long_proj = (all_points - origin) @ long_axis
    width_proj = np.vstack([left, right]) - origin
    width_proj = width_proj @ width_axis

    long_min = float(long_proj.min())
    long_max = float(long_proj.max())
    width_min = float(width_proj.min())
    width_max = float(width_proj.max())

    long_span = max(0.0, long_max - long_min)
    width_span = max(0.0, width_max - width_min)
    if long_span <= 1e-6 or width_span <= 1e-6:
        return None

    long_pad = long_span * max(0.0, float(length_padding_frac))
    width_pad = width_span * max(0.0, float(padding_frac)) / 2.0
    long_min -= long_pad
    long_max += long_pad
    width_min -= width_pad
    width_max += width_pad

    vertices = [
        origin + long_axis * long_min + width_axis * width_min,
        origin + long_axis * long_min + width_axis * width_max,
        origin + long_axis * long_max + width_axis * width_max,
        origin + long_axis * long_max + width_axis * width_min,
    ]
    return [[round(float(x), 3), round(float(y), 3)] for x, y in vertices]


def make_centerline_summary(image_path: str | Path, rois_source: str | Path, results: list[dict[str, Any]]) -> dict[str, Any]:
    centerlines: list[dict[str, Any]] = []
    for result in results:
        item: dict[str, Any] = {
            "roi_idx": int(result["roi_idx"]),
            "status": result["status"],
            "confidence": result["confidence"],
            "centerline": centerline_geometry(result.get("centerline", [])),
        }
        if result["status"] != "ok" and result.get("failure_reason"):
            item["failure_reason"] = result["failure_reason"]
        centerlines.append(item)

    return {
        "image": str(image_path),
        "rois": str(rois_source),
        "num_rois": len(centerlines),
        "centerlines": centerlines,
    }


def _failed_roi_result(roi_idx: int, points: np.ndarray, failure_reason: str) -> dict[str, Any]:
    return {
        "roi_idx": int(roi_idx),
        "status": "failed",
        "confidence": "failed",
        "failure_reason": failure_reason,
        "vertices": [[round(float(x), 3), round(float(y), 3)] for x, y in np.asarray(points, dtype=np.float32)],
        "left_boundary": [],
        "right_boundary": [],
        "centerline": [],
    }


def detect_roi(gray: np.ndarray, roi: dict[str, Any], config: DetectionConfig | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    config = config or DetectionConfig()
    roi_idx = int(roi.get("roi_idx", 0))
    points = np.asarray(roi["vertices"], dtype=np.float32)

    try:
        crop, mask, meta = rectify_polygon(gray, points)
    except (ValueError, cv2.error):
        return _failed_roi_result(roi_idx, points, "invalid or oversized roi"), {}

    pair, profile, evidence = find_edge_pair(crop, mask, config)

    result: dict[str, Any] = {
        "roi_idx": roi_idx,
        "rectified_size": {"width": int(meta.width), "height": int(meta.height)},
        "vertices": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        **pair,
    }

    if pair["status"] == "ok":
        left_model, right_model = _fit_boundary_lines(
            evidence,
            mask,
            int(pair["left_x"]),
            int(pair["right_x"]),
        )
        if left_model is not None and right_model is not None:
            result["left_line"] = {
                "slope": round(left_model.slope, 6),
                "intercept": round(left_model.intercept, 3),
                "support_rows": int(left_model.support_rows),
                "median_residual": round(left_model.median_residual, 3),
            }
            result["right_line"] = {
                "slope": round(right_model.slope, 6),
                "intercept": round(right_model.intercept, 3),
                "support_rows": int(right_model.support_rows),
                "median_residual": round(right_model.median_residual, 3),
            }
        center_offset_x, centerline_info = _foreground_centerline_offset(
            crop,
            mask,
            int(pair["left_x"]),
            int(pair["right_x"]),
            config,
            left_model,
            right_model,
        )
        left, right, center = _sample_boundaries(
            meta,
            mask,
            int(pair["left_x"]),
            int(pair["right_x"]),
            config.boundary_samples,
            left_model,
            right_model,
            center_offset_x,
        )
        if centerline_info is not None:
            result.update(centerline_info)
        else:
            result["centerline_source"] = "edge-midpoint"
        result["left_boundary"] = left
        result["right_boundary"] = right
        result["centerline"] = center
    else:
        result["left_boundary"] = []
        result["right_boundary"] = []
        result["centerline"] = []

    debug = {
        "crop": crop,
        "mask": mask,
        "profile": profile,
        "foreground_profile": _foreground_column_profile(crop, mask, min_contrast=config.body_min_contrast)
        if config.body_guidance
        else None,
        "evidence": evidence,
        "meta": meta,
    }
    return result, debug


def draw_overlay(gray: np.ndarray, results: list[dict[str, Any]]) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    confidence_color = {
        "high": (0, 255, 0),
        "medium": (0, 200, 255),
        "low": (0, 0, 255),
        "failed": (0, 0, 255),
    }

    for result in results:
        vertices = np.asarray(result["vertices"], dtype=np.int32)
        color = confidence_color.get(result["confidence"], (0, 0, 255))
        cv2.polylines(overlay, [vertices], True, color, 2, cv2.LINE_AA)

        if result["status"] == "ok":
            for key, line_color in [
                ("left_boundary", LEFT_BOUNDARY_COLOR),
                ("right_boundary", RIGHT_BOUNDARY_COLOR),
                ("centerline", CENTERLINE_COLOR),
            ]:
                pts = np.asarray(result[key], dtype=np.float32).round().astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(overlay, [pts], False, line_color, 2, cv2.LINE_AA)

        label_point = vertices.mean(axis=0).round().astype(int)
        label = f"R{result['roi_idx']}:{result['confidence']}"
        cv2.putText(
            overlay,
            label,
            tuple(label_point),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )
    return overlay


def _make_failed_roi_debug_panel(result: dict[str, Any]) -> np.ndarray:
    panel = np.full((120, 230, 3), 245, dtype=np.uint8)
    title = f"ROI{result.get('roi_idx', '-')} failed"
    reason = str(result.get("failure_reason", "no debug crop"))[:30]
    cv2.putText(panel, title, (8, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 180), 1, cv2.LINE_AA)
    cv2.putText(panel, reason, (8, 68), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 60, 60), 1, cv2.LINE_AA)
    return panel


def make_roi_debug_panel(result: dict[str, Any], debug: dict[str, Any]) -> np.ndarray:
    if not {"crop", "mask", "profile"}.issubset(debug):
        return _make_failed_roi_debug_panel(result)

    crop = debug["crop"]
    mask = debug["mask"]
    profile = debug["profile"]
    foreground_profile = debug.get("foreground_profile")
    height, width = crop.shape

    panel = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    outside = mask == 0
    panel[outside] = (panel[outside] * 0.45 + np.array([90, 90, 90]) * 0.55).astype(np.uint8)

    if result["status"] == "ok":
        left_x = int(result["left_x"])
        right_x = int(result["right_x"])
        center_x = (left_x + right_x) // 2
        left_line = result.get("left_line")
        right_line = result.get("right_line")
        if left_line and right_line:
            def line_x(line: dict[str, Any], y: int) -> int:
                x = float(line["slope"]) * float(y) + float(line["intercept"])
                return int(round(np.clip(x, 0, width - 1)))

            left_top = line_x(left_line, 0)
            left_bottom = line_x(left_line, height - 1)
            right_top = line_x(right_line, 0)
            right_bottom = line_x(right_line, height - 1)
            center_top = int(round((left_top + right_top) / 2.0))
            center_bottom = int(round((left_bottom + right_bottom) / 2.0))
            cv2.line(panel, (left_top, 0), (left_bottom, height - 1), LEFT_BOUNDARY_COLOR, 2)
            cv2.line(panel, (right_top, 0), (right_bottom, height - 1), RIGHT_BOUNDARY_COLOR, 2)
            cv2.line(panel, (center_top, 0), (center_bottom, height - 1), CENTERLINE_COLOR, 1)
        else:
            cv2.line(panel, (left_x, 0), (left_x, height - 1), LEFT_BOUNDARY_COLOR, 2)
            cv2.line(panel, (right_x, 0), (right_x, height - 1), RIGHT_BOUNDARY_COLOR, 2)
            cv2.line(panel, (center_x, 0), (center_x, height - 1), CENTERLINE_COLOR, 1)

    title = (
        f"ROI{result['roi_idx']} {result['confidence']} "
        f"w={result.get('tube_width_px', '-')}"
    )
    cv2.putText(panel, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    plot_height = 80
    plot = np.full((plot_height, width, 3), 255, dtype=np.uint8)
    if result.get("body_span"):
        span_left, span_right = result["body_span"]
        span_left = int(np.clip(span_left, 0, width - 1))
        span_right = int(np.clip(span_right, 0, width - 1))
        if span_right > span_left:
            plot[:, span_left : span_right + 1] = (235, 255, 255)
    max_value = max(float(profile.max()), 1.0)
    for x, value in enumerate(profile):
        y = int(plot_height - 1 - value / max_value * (plot_height - 8))
        cv2.line(plot, (x, plot_height - 1), (x, y), (0, 0, 0), 1)
    if isinstance(foreground_profile, np.ndarray) and foreground_profile.shape[0] == width:
        pts = []
        for x, value in enumerate(foreground_profile):
            y = int(plot_height - 1 - _clamp01(float(value)) * (plot_height - 8))
            pts.append([x, y])
        if len(pts) >= 2:
            cv2.polylines(plot, [np.asarray(pts, dtype=np.int32)], False, (255, 120, 0), 1, cv2.LINE_AA)

    if result["status"] == "ok":
        cv2.line(plot, (int(result["left_x"]), 0), (int(result["left_x"]), plot_height - 1), LEFT_BOUNDARY_COLOR, 1)
        cv2.line(plot, (int(result["right_x"]), 0), (int(result["right_x"]), plot_height - 1), (0, 160, 0), 1)

    return np.vstack([panel, plot])


def make_debug_mosaic(panels: list[np.ndarray], target_width: int = 230, columns: int = 3) -> np.ndarray:
    resized: list[np.ndarray] = []
    for panel in panels:
        scale = target_width / panel.shape[1]
        target_height = max(120, int(round(panel.shape[0] * scale)))
        resized.append(cv2.resize(panel, (target_width, target_height), interpolation=cv2.INTER_AREA))

    rows: list[np.ndarray] = []
    for i in range(0, len(resized), columns):
        group = resized[i : i + columns]
        max_height = max(panel.shape[0] for panel in group)
        padded: list[np.ndarray] = []
        for panel in group:
            if panel.shape[0] < max_height:
                pad = np.full((max_height - panel.shape[0], panel.shape[1], 3), 245, dtype=np.uint8)
                panel = np.vstack([panel, pad])
            padded.append(panel)
        while len(padded) < columns:
            padded.append(np.full((max_height, target_width, 3), 245, dtype=np.uint8))
        rows.append(np.hstack(padded))
    return np.vstack(rows)


def clear_debug_outputs(debug_dir: Path) -> None:
    if not debug_dir.exists():
        return
    for pattern in ("roi_*_rectified.png", "all_rois_rectified_eval.png"):
        for path in debug_dir.glob(pattern):
            if path.is_file():
                path.unlink()


def detect_all(gray: np.ndarray, roi_payload: dict[str, Any], config: DetectionConfig) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    results: list[dict[str, Any]] = []
    debug_panels: list[np.ndarray] = []
    for roi in roi_payload.get("rois", []):
        result, debug = detect_roi(gray, roi, config)
        results.append(result)
        debug_panels.append(make_roi_debug_panel(result, debug))
    return results, debug_panels


def run_detection(args: argparse.Namespace) -> dict[str, Any]:
    image_path = Path(args.image)
    out_dir = Path(args.out_dir) if args.out_dir else Path(__file__).resolve().parent / "output"
    debug_dir = out_dir / "debug"

    gray = read_image_unicode(image_path, cv2.IMREAD_GRAYSCALE)
    if args.draw_rois:
        color_image = read_image_unicode(image_path, cv2.IMREAD_COLOR)
        roi_payload = draw_rois_interactive(image_path, color_image)
        if not roi_payload["rois"]:
            raise ValueError("No ROI was drawn.")
        rois_source = out_dir / "drawn_rois.json"
        save_json(rois_source, roi_payload)
    else:
        rois_source = Path(args.rois)
        roi_payload = normalize_roi_payload(load_roi_json(rois_source))

    config = DetectionConfig(
        min_width_frac=args.min_width_frac,
        max_width_frac=args.max_width_frac,
        boundary_samples=args.boundary_samples,
        body_guidance=args.body_guidance,
        body_min_contrast=args.body_min_contrast,
        body_min_fraction=args.body_min_fraction,
        body_score_weight=args.body_score_weight,
        body_min_span_ratio=args.body_min_span_ratio,
        body_max_span_ratio=args.body_max_span_ratio,
        body_center_tolerance_frac=args.body_center_tolerance_frac,
        body_outside_margin_frac=args.body_outside_margin_frac,
        body_outside_penalty_weight=args.body_outside_penalty_weight,
        body_hard_reject=args.body_hard_reject,
        body_edge_gate_frac=args.body_edge_gate_frac,
        body_adjustment_scale=args.body_adjustment_scale,
        centerline_foreground_correction=args.centerline_foreground_correction,
    )

    results, debug_panels = detect_all(gray, roi_payload, config)
    overlay = draw_overlay(gray, results)

    payload = {
        "image": str(image_path),
        "rois": str(rois_source),
        "num_rois": len(results),
        "results": results,
    }
    centerline_payload = make_centerline_summary(image_path, rois_source, results)
    save_json(out_dir / "results.json", payload)
    save_json(out_dir / "centerline_results.json", centerline_payload)
    write_image_unicode(out_dir / "overlay_detected.png", overlay)

    if args.debug:
        debug_dir.mkdir(parents=True, exist_ok=True)
        clear_debug_outputs(debug_dir)
        for result, panel in zip(results, debug_panels, strict=True):
            write_image_unicode(debug_dir / f"roi_{int(result['roi_idx']):02d}_rectified.png", panel)
        if debug_panels:
            write_image_unicode(debug_dir / "all_rois_rectified_eval.png", make_debug_mosaic(debug_panels))

    return payload


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Detect main tube boundaries inside polygon ROIs.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--rois", default=None, help="ROI JSON path with rois[].vertices or rois[].roi_points.")
    parser.add_argument("--draw-rois", action="store_true", help="Draw polygon ROIs interactively instead of loading JSON.")
    parser.add_argument("--out-dir", default=None, help="Output directory. Defaults to ./output beside this script.")
    parser.add_argument("--debug", action="store_true", help="Write per-ROI rectified debug images.")
    parser.add_argument("--min-width-frac", type=float, default=0.28, help="Minimum tube width as a fraction of rectified ROI width.")
    parser.add_argument("--max-width-frac", type=float, default=0.92, help="Maximum tube width as a fraction of rectified ROI width.")
    parser.add_argument("--boundary-samples", type=int, default=60, help="Number of mapped boundary points per ROI.")
    parser.add_argument("--centerline-foreground-correction", dest="centerline_foreground_correction", action="store_true", default=False, help="Shift centerlines toward the foreground/body profile. Disabled by default because occluders can bias it.")
    parser.add_argument("--no-centerline-foreground-correction", dest="centerline_foreground_correction", action="store_false", help="Keep centerlines at the midpoint of detected left/right boundaries.")
    parser.add_argument("--body-guidance", dest="body_guidance", action="store_true", default=True, help="Use foreground/body evidence to re-rank edge-pair candidates.")
    parser.add_argument("--no-body-guidance", dest="body_guidance", action="store_false", help="Disable foreground/body candidate re-ranking.")
    parser.add_argument("--body-min-contrast", type=float, default=35.0, help="Minimum ROI gray dynamic range required for body guidance.")
    parser.add_argument("--body-min-fraction", type=float, default=0.50, help="Minimum dark foreground column fraction used to form body spans.")
    parser.add_argument("--body-score-weight", type=float, default=0.25, help="Soft weight of body score in candidate scoring.")
    parser.add_argument("--body-edge-gate-frac", type=float, default=0.90, help="Only candidates with this fraction of the best edge score can receive body-score boosts.")
    parser.add_argument("--body-adjustment-scale", type=float, default=0.50, help="Scale applied to body-score adjustment after edge gating.")
    parser.add_argument("--body-min-span-ratio", type=float, default=0.45, help="Minimum foreground span width relative to candidate tube width.")
    parser.add_argument("--body-max-span-ratio", type=float, default=1.12, help="Maximum foreground span width relative to candidate tube width.")
    parser.add_argument("--body-center-tolerance-frac", type=float, default=0.22, help="Allowed body center offset as a fraction of candidate tube width.")
    parser.add_argument("--body-outside-margin-frac", type=float, default=0.10, help="Outside band width used to penalize body leakage past candidate edges.")
    parser.add_argument("--body-outside-penalty-weight", type=float, default=0.35, help="Penalty weight when body evidence continues outside candidate edges.")
    parser.add_argument("--body-hard-reject", dest="body_hard_reject", action="store_true", default=False, help="Reject weak edge candidates with very poor body evidence.")
    parser.add_argument("--no-body-hard-reject", dest="body_hard_reject", action="store_false", help="Disable hard rejection by body evidence.")
    args = parser.parse_args(argv)
    if args.draw_rois and args.rois:
        parser.error("--draw-rois cannot be used together with --rois.")
    if not args.draw_rois and not args.rois:
        parser.error("one of --rois or --draw-rois is required.")
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
    payload = run_detection(args)
    ok_count = sum(1 for item in payload["results"] if item["status"] == "ok")
    print(f"Processed {payload['num_rois']} ROI(s); {ok_count} ok.")
    print(f"Results: {Path(args.out_dir) / 'results.json' if args.out_dir else Path(__file__).resolve().parent / 'output' / 'results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
