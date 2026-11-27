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
WALL_COLOR = (255, 210, 0)
CENTERLINE_COLOR = (0, 255, 255)
TUBE_LINES = (("left_boundary", WALL_COLOR), ("right_boundary", WALL_COLOR),
              ("centerline", CENTERLINE_COLOR))


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
    support_y_min: float = 0.0
    support_y_max: float = 0.0

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
    supported_rows = ((edge_band > threshold) & mask_band).any(axis=1)
    return float(supported_rows.sum() / valid_rows.sum())


def _foreground_column_profile(crop: np.ndarray, mask: np.ndarray) -> np.ndarray | None:
    valid_pixels = crop[mask > 0]
    if valid_pixels.size == 0:
        return None

    p10, p35, p60, p90 = np.percentile(valid_pixels, [10, 35, 60, 90])
    if float(p90 - p10) < 35.0:
        return None

    dark_threshold = float(max(p35 + 5.0, p60 - 3.0))
    mask_f = (mask > 0).astype(np.float32)
    counts = mask_f.sum(axis=0)
    dark_counts = ((crop.astype(np.float32) <= dark_threshold) * mask_f).sum(axis=0)
    profile = dark_counts / np.maximum(counts, 1.0)
    profile[counts < max(5.0, crop.shape[0] * 0.12)] = 0.0
    return _smooth_1d(profile.astype(np.float32), sigma=max(1.0, crop.shape[1] / 90.0))


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


def _nearest_peak_to_edge(
    peaks: list[tuple[float, int]],
    start_x: int,
    edge_x: int,
    side: str,
    margin: int,
) -> int:
    if side == "right":
        candidates = [x for _value, x in peaks if start_x < x <= edge_x + margin]
        return max(candidates) if candidates else edge_x
    candidates = [x for _value, x in peaks if edge_x - margin <= x < start_x]
    return min(candidates) if candidates else edge_x


def _refine_pair_with_foreground_span(
    best: dict[str, Any],
    peaks: list[tuple[float, int]],
    profile: np.ndarray,
    crop: np.ndarray,
    mask: np.ndarray,
    config: DetectionConfig,
) -> dict[str, Any]:
    foreground = _foreground_column_profile(crop, mask)
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
        refined_left = _nearest_peak_to_edge(peaks, left_x, span_left, "left", margin=5)
    if span_right - right_x >= min_extra:
        refined_right = _nearest_peak_to_edge(peaks, right_x, span_right, "right", margin=5)

    refined_width = refined_right - refined_left
    if refined_width <= 0 or refined_width > max_sep:
        return best
    if refined_left == left_x and refined_right == right_x:
        return best

    refined = {**best}
    refined["left_x"] = int(refined_left)
    refined["right_x"] = int(refined_right)
    refined["tube_width_px"] = int(refined_width)
    refined["tube_width_frac"] = float(refined_width / max(1, width))
    refined["left_strength"] = float(profile[int(np.clip(refined_left, 0, width - 1))])
    refined["right_strength"] = float(profile[int(np.clip(refined_right, 0, width - 1))])
    return refined


def _wall_contrast(crop: np.ndarray, mask: np.ndarray, xs: int | np.ndarray, step: int) -> np.ndarray:
    """Signed cross-wall intensity step. A wire narrower than the sampling band cancels out."""
    rows = np.arange(crop.shape[0])
    xs = np.broadcast_to(np.rint(xs).astype(int), rows.shape)
    lo = np.clip(xs - step, 0, crop.shape[1] - 1)
    hi = np.clip(xs + step, 0, crop.shape[1] - 1)
    delta = crop[rows, hi].astype(np.int16) - crop[rows, lo]
    in_bounds = (xs >= 0) & (xs < crop.shape[1])
    in_roi = mask[rows, np.clip(xs, 0, crop.shape[1] - 1)] > 0
    return np.where(in_bounds & in_roi, delta, 0)


def _paired_wall_support(left: np.ndarray, right: np.ndarray, threshold: float = 12) -> float:
    dark_tube = (left < -threshold) & (right > threshold)
    light_tube = (left > threshold) & (right < -threshold)
    return max(np.count_nonzero(dark_tube), np.count_nonzero(light_tube)) / len(left)


def _bright_rim(crop, mask, x, step, threshold):
    """Rows with a narrow highlight brighter than both flanks of the wall."""
    rows = np.arange(len(crop))
    last = crop.shape[1] - 1
    x = np.rint(x).astype(int).clip(0, last)
    offsets = np.arange(-2, 3)
    xs = (x + offsets[:, None]).clip(0, last)
    peak = np.argmax(crop[rows, xs], axis=0)
    x = (x + offsets[peak]).clip(0, last)
    crest = crop[rows, x]
    # A much wider bright gap beside a railing also looks like a crest at
    # one offset. A narrow rim falls away on both sides at both distances.
    near = max(2, step // 2)
    flanks = np.maximum.reduce([crop[rows, (x + dx).clip(0, last)] for dx in (-step, -near, near, step)])
    return (crest.astype(np.int16) - flanks > threshold) & (mask[rows, x] > 0)


def find_edge_pair(crop: np.ndarray, mask: np.ndarray, config: DetectionConfig,
                   expected_width: float | None = None) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    height, width = crop.shape
    evidence, profile = _edge_evidence(crop, mask, config)
    peaks = _find_profile_peaks(profile, config.max_profile_peaks)

    valid_evidence = evidence[mask > 0]
    threshold = max(2.0, float(np.percentile(valid_evidence, 78))) if valid_evidence.size else 2.0
    min_sep = max(MIN_EDGE_PAIR_SEPARATION_PX, int(config.min_width_frac * width))
    max_sep = max(min_sep + 4, int(config.max_width_frac * width))

    support = {x: _boundary_support(evidence, mask, x, threshold) for _, x in peaks}
    step = max(3, min(12, int(round(width * 0.08))))
    contrast = {x: _wall_contrast(crop, mask, x, step) for _, x in peaks}
    # Estimate the noise in this crop. Each wall must show a repeatable step
    # above it, even if the other wall casts a very strong edge.
    residual = cv2.absdiff(crop, cv2.medianBlur(crop, 3))[mask > 0]
    noise_floor = max(2.0, 4.0 * float(np.median(residual))) if residual.size else 2.0
    def has_wall(delta: np.ndarray) -> bool:
        return max(np.count_nonzero(delta > noise_floor),
                   np.count_nonzero(delta < -noise_floor)) >= 0.15 * height

    # Where the sky clips, the true tube outline borders bright background on
    # both sides. A shading gradient inside the tube only satisfies one side.
    sky_left = {x: (crop[:, max(0, x-step)] >= 252) & (contrast[x] < -noise_floor) for _, x in peaks}
    sky_right = {x: (crop[:, min(width-1, x+step)] >= 252) & (contrast[x] > noise_floor) for _, x in peaks}
    near_steps = {}
    for _, x in peaks:
        # Strong edges elsewhere in the ROI can set the gradient percentile
        # above an entire washed-out wall. Require a repeatable *local* step.
        near = _wall_contrast(crop, mask, x, max(2, step // 2))
        near_steps[x] = near
        silhouette_rows = (sky_left[x] & (near < -noise_floor)) | (sky_right[x] & (near > noise_floor))
        support[x] = max(support[x], float(np.mean(silhouette_rows)))
    peaks = [(value, x) for value, x in peaks if has_wall(contrast[x]) and support[x] >= 0.15]
    rim_threshold = max(12, noise_floor)
    rims = {x: _bright_rim(crop, mask, x, step, rim_threshold) for _, x in peaks}
    wall_steps = {x: np.median([near_steps[x], _wall_contrast(crop, mask, x, max(2, round(.75 * step))),
                                contrast[x]], axis=0) for _, x in peaks}
    sustained = {x: (np.abs(contrast[x]) > rim_threshold)
                 & (np.sign(contrast[x]) * near_steps[x] > rim_threshold) for _, x in peaks}
    positive_profile = profile[profile > 0]
    background_strength = max(1.0, float(np.median(positive_profile))) if positive_profile.size else 1.0
    best: dict[str, Any] | None = None
    for i, (left_strength_raw, a) in enumerate(peaks):
        for right_strength_raw, b in peaks[i + 1 :]:
            left_x, right_x = sorted((a, b))
            tube_width = right_x - left_x
            if tube_width < min_sep or tube_width > max_sep:
                continue

            left_support = support[left_x]
            right_support = support[right_x]
            continuity = 0.5 * (left_support + right_support)

            width_frac = tube_width / max(1, width)
            if width_frac < 0.38:
                width_prior = width_frac / 0.38
            elif width_frac > 0.84:
                width_prior = max(0.55, (0.92 - width_frac) / 0.08)
            else:
                width_prior = 1.0

            score = (left_strength_raw + right_strength_raw) * (0.4 + continuity) * width_prior
            pair_support = _paired_wall_support(wall_steps[left_x], wall_steps[right_x])
            rim_support = max(np.mean(rims[left_x] & sustained[right_x]),
                              np.mean(rims[right_x] & sustained[left_x]))
            score *= 0.5 + max(pair_support, rim_support)
            silhouette = float(np.mean(sky_left[left_x] & sky_right[right_x]))
            score *= 1 + 3 * silhouette
            if expected_width:
                score *= min(tube_width / expected_width, expected_width / tube_width) ** 3

            candidate = {
                "score": float(score),
                "left_x": int(left_x),
                "right_x": int(right_x),
                "tube_width_px": int(tube_width),
                "tube_width_frac": float(width_frac),
                "left_strength": float(profile[left_x]),
                "right_strength": float(profile[right_x]),
                "continuity": float(continuity),
                "silhouette_support": silhouette,
            }
            if best is None or candidate["score"] > best["score"]:
                best = candidate

    peak_debug = [{"x": int(x), "value": round(float(value), 3)} for value, x in peaks[:10]]
    if best is None:
        return {
            "status": "failed",
            "confidence": "failed",
            "score": 0.0,
            "candidate_peaks": peak_debug,
            "failure_reason": "no valid pair of long side edges",
        }, profile, evidence

    refined = _refine_pair_with_foreground_span(best, peaks, profile, crop, mask, config)
    supported_peaks = {x for _, x in peaks}
    if refined["left_x"] in supported_peaks and refined["right_x"] in supported_peaks:
        silhouette = float(np.mean(sky_left[refined["left_x"]] & sky_right[refined["right_x"]]))
        # The dark foreground mask can connect the pipe to a crossing girder.
        # Its proposed wider pair must retain evidence for each moved wall.
        supported = all(refined[f"{side}_x"] == best[f"{side}_x"] or (
            refined[f"{side}_strength"] >= .30 * best[f"{side}_strength"]
            and support[refined[f"{side}_x"]] >= .5 * support[best[f"{side}_x"]]
        ) for side in ("left", "right"))
        width_consistent = (not expected_width or
                            abs(refined["tube_width_px"] - expected_width) <=
                            max(.08 * expected_width, abs(best["tube_width_px"] - expected_width)))
        if width_consistent and silhouette >= best["silhouette_support"] - 0.15 and (supported or silhouette >= .5):
            best = {**refined, "silhouette_support": silhouette}
    # Confidence describes the pair. A single strong peak in blank sky must
    # not compensate for a missing second boundary.
    best["continuity"] = min(support[best["left_x"]], support[best["right_x"]])
    best["edge_contrast"] = min(best["left_strength"], best["right_strength"]) / background_strength

    if (
        best["continuity"] >= 0.45
        and best["edge_contrast"] >= 2.2
        and 0.38 <= best["tube_width_frac"] <= 0.84
    ):
        confidence = "high"
    elif (best["continuity"] >= 0.25 and best["edge_contrast"] >= 1.6) or best["silhouette_support"] >= 0.40:
        confidence = "medium"
    else:
        confidence = "low"

    best["status"] = "ok"
    best["wall_noise_floor"] = noise_floor
    best["confidence"] = confidence
    best["candidate_peaks"] = peak_debug
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
    expected_x = int(np.clip(expected_x, 0, width - 1))
    x0 = max(0, expected_x - search_radius)
    x1 = min(width, expected_x + search_radius + 1)
    columns = np.arange(x0, x1, dtype=np.float32)
    weights_x = np.exp(-0.5 * ((columns - expected_x) / max(2.0, search_radius / 3.0)) ** 2)
    band = np.where(mask[:, x0:x1], evidence[:, x0:x1], 0)
    peaks = np.argmax(band * weights_x, axis=1)
    strengths = band[np.arange(height), peaks]
    ys = np.flatnonzero(strengths > min_strength)
    min_samples = max(8, int(round(height * 0.18)))
    if ys.size < min_samples:
        return None
    xs = columns[peaks[ys]]
    weights = np.sqrt(strengths[ys] + 1e-6)
    if float(np.ptp(ys)) < 4.0:
        return None

    slope, intercept = np.polyfit(ys, xs, 1, w=weights)
    predicted = slope * ys + intercept
    residual = np.abs(xs - predicted)
    # A wire crossing the tube can dominate a few rows. Refit the wall
    # using the rows consistent with its long straight edge.
    inliers = residual <= max(2.0, 3.0 * float(np.median(residual)))
    if min_samples <= np.count_nonzero(inliers) < ys.size:
        ys, xs, weights = ys[inliers], xs[inliers], weights[inliers]
        if np.ptp(ys) < 4:
            return None
        slope, intercept = np.polyfit(ys, xs, 1, w=weights)
        residual = np.abs(xs - (slope * ys + intercept))
    median_residual = float(np.median(residual))
    mid_y = (height - 1) / 2.0
    if abs(float(slope * mid_y + intercept) - float(expected_x)) > max(4.0, search_radius * 0.55):
        return None
    if abs(float(slope)) > 0.45 or median_residual > max(2.5, search_radius * 0.35):
        return None

    return BoundaryLineModel(
        slope=float(slope),
        intercept=float(intercept),
        support_rows=int(ys.size),
        median_residual=median_residual,
        support_y_min=float(ys.min()),
        support_y_max=float(ys.max()),
    )


def _with_parallel_slope(model: BoundaryLineModel, slope: float, mid_y: float) -> BoundaryLineModel:
    return BoundaryLineModel(
        slope=float(slope),
        intercept=float(model.x_at(mid_y) - slope * mid_y),
        support_rows=model.support_rows,
        median_residual=model.median_residual,
        support_y_min=model.support_y_min,
        support_y_max=model.support_y_max,
    )


def _fit_boundary_lines(
    evidence: np.ndarray,
    mask: np.ndarray,
    left_x: int,
    right_x: int,
    crop: np.ndarray,
    profile: np.ndarray,
    peaks: list[dict[str, Any]],
) -> tuple[BoundaryLineModel | None, BoundaryLineModel | None]:
    height, width = evidence.shape
    tube_width = max(1, int(right_x) - int(left_x))
    search_radius = max(6, min(int(round(tube_width * 0.45)), int(round(width * 0.24))))

    valid_evidence = evidence[mask > 0]
    positive_evidence = valid_evidence[valid_evidence > 0]
    if positive_evidence.size == 0:
        return None, None

    min_strength = float(np.percentile(positive_evidence, 70))
    # Follow the polarity of the selected step. The near edge of a parallel
    # railing across a bright gap has the opposite sign from the pipe wall.
    gx = cv2.Sobel(crop, cv2.CV_16S, 1, 0)
    def fit(x):
        step = max(2, min(5, round(tube_width * .05)))
        delta = _wall_contrast(crop, mask, x, step)
        polarity = np.sign(np.count_nonzero(delta > 2) - np.count_nonzero(delta < -2))
        signed_evidence = evidence
        if polarity:
            opposite = gx < 0 if polarity > 0 else gx > 0
            signed_evidence = np.where(opposite, 0, evidence)
        # Do not let a stronger neighboring ridge pull rows out of the basin
        # of the chosen peak (e.g. the second edge of a parallel railing).
        gaps = [abs(p["x"] - x) for p in peaks if p["x"] != x and p["value"] >= .6 * profile[x]]
        radius = min(search_radius, max(4, round(.6 * min(gaps)))) if gaps else search_radius
        return _fit_one_boundary_line(signed_evidence, mask, x, radius, min_strength)

    left_model, right_model = fit(left_x), fit(right_x)
    if left_model is None or right_model is None:
        return None, None

    if abs(left_model.slope - right_model.slope) > 0.20:
        return None, None

    # When a wire crosses one wall almost lengthwise, it can supply hundreds
    # of rows with the wrong slope. Prefer the tightly localized opposite wall.
    models = [left_model, right_model]
    weights = [model.support_rows / (.5 + model.median_residual) ** 2 for model in models]
    strong = int(weights[1] > weights[0])
    weak = 1 - strong
    mid_y = (height - 1) / 2.0
    slopes_disagree = abs(left_model.slope - right_model.slope) * height > 3
    weak_peak = (left_x, right_x)[weak]
    displaced = abs(models[weak].x_at(mid_y) - weak_peak) > 2
    if slopes_disagree or (displaced and weights[strong] > 2 * weights[weak]):
        # Vote at parallel offsets over the full length. Taking unsigned
        # gradients allows a wall to pass from bright sky to dark steelwork.
        radius = min(search_radius, max(2, round(.08 * tube_width)))
        offsets = np.arange(-radius, radius + 1)
        rows = np.arange(0, height, 2)[:, None]
        slopes = (left_model.slope, right_model.slope) if slopes_disagree else (models[strong].slope,)
        best_score = 0.0
        best_pair = None
        for slope in slopes:
            pair = []
            score = 1.0
            for x, model in zip((left_x, right_x), models, strict=True):
                cols = np.rint(x + (rows - mid_y) * slope + offsets).astype(int).clip(0, width - 1)
                band = np.where(mask[rows, cols], evidence[rows, cols], 0)
                votes = np.minimum(band, np.percentile(band.max(axis=1), 75)).mean(axis=0)
                peak = np.argmax(votes)
                score *= float(votes[peak])
                parallel = model
                if model.slope != slope:
                    parallel = _with_parallel_slope(model, slope, mid_y)
                    parallel.intercept = float(x + offsets[peak] - slope * mid_y)
                pair.append(parallel)
            if score > best_score:
                best_score, best_pair = score, pair
        if best_pair is not None:
            return best_pair[0], best_pair[1]

    total_support = left_model.support_rows + right_model.support_rows
    parallel_slope = (
        left_model.slope * left_model.support_rows + right_model.slope * right_model.support_rows
    ) / total_support
    return _with_parallel_slope(left_model, parallel_slope, mid_y), _with_parallel_slope(
        right_model,
        parallel_slope,
        mid_y,
    )


def _line_x(model: BoundaryLineModel | None, fallback_x: float, y: float, width: int) -> float:
    x = model.x_at(float(y)) if model is not None else float(fallback_x)
    return float(np.clip(x, 0.0, max(0.0, float(width - 1))))


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
    if left_model is not None and right_model is not None:
        support_min = max(float(left_model.support_y_min), float(right_model.support_y_min))
        support_max = min(float(left_model.support_y_max), float(right_model.support_y_max))
        min_support_span = max(8.0, meta.height * 0.18)
        if support_max - support_min >= min_support_span:
            y0 = max(0, int(np.floor(support_min)))
            y1 = min(meta.height - 1, int(np.ceil(support_max)))
            rows = np.arange(y0, y1 + 1, dtype=np.int32)

    def columns(y: np.ndarray, model: BoundaryLineModel | None, fallback: float) -> np.ndarray:
        values = model.slope * y + model.intercept if model else np.full(y.shape, fallback)
        return np.clip(values, 0, meta.width - 1)

    left_cols = columns(rows, left_model, left_x)
    right_cols = columns(rows, right_model, right_x)
    mask_band = cv2.dilate(mask, np.ones((1, 5), dtype=np.uint8))
    valid = np.zeros(rows.size, dtype=bool)
    for col in (left_cols, right_cols, (left_cols + right_cols) / 2):
        valid |= mask_band[rows, np.rint(col).astype(int)] > 0
    valid_rows = rows[valid]
    if not valid_rows.size:
        valid_rows = np.array([0, meta.height - 1])
    sample_rows = (np.linspace(valid_rows[0], valid_rows[-1], count)
                   if valid_rows.size > count else valid_rows)
    left_cols = columns(sample_rows, left_model, left_x)
    right_cols = columns(sample_rows, right_model, right_x)
    middle = (left_cols + right_cols) / 2 if left_model and right_model else fallback_center_x
    center_cols = np.clip(middle + center_offset_x, np.minimum(left_cols, right_cols), np.maximum(left_cols, right_cols))
    origins = meta.center + (sample_rows[:, None] + meta.long_min) * meta.long_axis

    def map_columns(col: np.ndarray) -> list[list[float]]:
        points = origins + (col[:, None] + meta.short_min) * meta.short_axis
        return np.round(points, 3).tolist()

    left, right, center = (map_columns(col) for col in (left_cols, right_cols, center_cols))
    return left, right, center


def _foreground_centerline_offset(
    crop: np.ndarray,
    mask: np.ndarray,
    left_x: int,
    right_x: int,
    config: DetectionConfig,
    left_model: BoundaryLineModel | None,
    right_model: BoundaryLineModel | None,
    paired_support: float,
) -> tuple[float, dict[str, Any] | None]:
    if not config.centerline_foreground_correction:
        return 0.0, None
    if crop.shape[0] < int(config.centerline_foreground_min_height):
        return 0.0, None

    edge_width = float(int(right_x) - int(left_x))
    if edge_width <= MIN_EDGE_PAIR_SEPARATION_PX:
        return 0.0, None

    # Keep the geometric midpoint when both walls delimit the same body over
    # most of its length. A shaded half of a round tube is not a second tube.
    # In clutter one wall may change contrast polarity against the background.
    # A shorter stretch of bilateral contrast suffices if both fitted lines
    # also have accurate, long support over the rest of the tube.
    stable_walls = all(model is not None and model.support_rows >= 0.65 * crop.shape[0]
                       and model.median_residual <= 2 for model in (left_model, right_model))
    if paired_support >= 0.50 or (stable_walls and paired_support >= 0.25):
        return 0.0, None

    foreground = _foreground_column_profile(crop, mask)
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
        if result.get("centerline_source"):
            item["centerline_source"] = result["centerline_source"]
        for key in (
            "centerline_x_offset",
            "centerline_edge_center_x",
            "centerline_foreground_center_x",
            "centerline_foreground_span",
            "centerline_foreground_span_ratio",
        ):
            if key in result:
                item[key] = result[key]
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

    expected_width = roi.get("expected_width")
    search_crop = crop
    if expected_width:
        # A thin upright in a bright gap can merge with the pipe in the
        # smoothed gradient profile. Close only narrow dark structures, and
        # only when one persists along the outer part of the search window.
        kernel = np.ones((1, 2 * max(2, round(.10 * expected_width)) + 1), np.uint8)
        closed = cv2.morphologyEx(crop, cv2.MORPH_CLOSE, kernel)
        sample = closed[::4]
        filled = (sample >= 200) & (cv2.subtract(sample, crop[::4]) >= 32) & (mask[::4] > 0)
        extent = max(1, round(crop.shape[1] * .35))
        coverage = filled.mean(axis=0)
        if max(coverage[:extent].max(), coverage[-extent:].max()) >= .30:
            search_crop = closed
    pair, profile, evidence = find_edge_pair(search_crop, mask, config, expected_width)

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
            search_crop,
            profile,
            pair["candidate_peaks"],
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
        rows = np.arange(crop.shape[0])
        step = max(3, min(12, round(pair["tube_width_px"] * 0.12)))
        left_cols = left_model.slope * rows + left_model.intercept if left_model else pair["left_x"]
        right_cols = right_model.slope * rows + right_model.intercept if right_model else pair["right_x"]
        # A narrow neighboring railing can coincide with one sampling offset.
        # Require agreement at two of three distances on each measured wall.
        def wall_step(cols):
            radii = (max(2, step // 2), max(2, round(.75 * step)), step)
            native = np.median([_wall_contrast(crop, mask, cols, radius) for radius in radii], axis=0)
            if search_crop is crop:
                return native
            cleared = np.median([_wall_contrast(search_crop, mask, cols, radius) for radius in radii], axis=0)
            # The original image must show the same step at the fitted wall.
            direct = _wall_contrast(crop, mask, cols, 2)
            return np.where(np.sign(cleared) * direct > pair["wall_noise_floor"], cleared, native)
        left_step = wall_step(left_cols)
        right_step = wall_step(right_cols)
        result["wall_pair_coverage"] = _paired_wall_support(left_step, right_step, pair["wall_noise_floor"])
        segment = slice(0, 0)
        if left_model and right_model:
            start = int(max(left_model.support_y_min, right_model.support_y_min))
            stop = 1 + int(min(left_model.support_y_max, right_model.support_y_max))
            if stop > start:
                segment = slice(start, stop)
                result["segment_wall_pair_coverage"] = _paired_wall_support(
                    left_step[segment], right_step[segment], pair["wall_noise_floor"])
        if left_model and right_model and expected_width and abs(pair["tube_width_px"] / expected_width - 1) <= .10:
            # A specular rim can be brighter than both the pipe and the
            # background. Count it only alongside a separately measured wall.
            threshold = max(12, pair["wall_noise_floor"])
            for cols, other_step in ((left_cols, right_step), (right_cols, left_step)):
                supported = _bright_rim(crop, mask, cols, step, threshold) & (np.abs(other_step) > threshold)
                result["wall_pair_coverage"] = max(result["wall_pair_coverage"], float(supported.mean()))
                if "segment_wall_pair_coverage" in result:
                    result["segment_wall_pair_coverage"] = max(result["segment_wall_pair_coverage"],
                                                               float(supported[segment].mean()))
        center_offset_x, centerline_info = _foreground_centerline_offset(
            crop,
            mask,
            int(pair["left_x"]),
            int(pair["right_x"]),
            config,
            left_model,
            right_model,
            result["wall_pair_coverage"],
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
        "evidence": evidence,
        "meta": meta,
    }
    return result, debug


def draw_roi_label(image, vertices, label, scale, color, lost, anchor=None):
    # Fit the entire badge inside the visible polygon, including diagonal and
    # partially cropped ROIs. Work only in its local bounding rectangle.
    vertices = np.asarray(vertices, dtype=np.int32).reshape(-1, 2)
    x, y = np.maximum(vertices.min(axis=0), 0)
    end_x, end_y = np.minimum(vertices.max(axis=0) + 1, image.shape[1::-1])
    if end_x <= x or end_y <= y:
        return False
    mask = np.zeros((end_y - y, end_x - x), np.uint8)
    cv2.fillPoly(mask, [vertices - [int(x), int(y)]], 255)
    layouts = [[label]]
    if " " in label:
        layouts.append(label.split())
    for lines in layouts:
        font_scale = scale
        while font_scale >= .20:
            sizes = [cv2.getTextSize(line, 0, font_scale, 1) for line in lines]
            width = max(size[0][0] for size in sizes) + 5
            (_, th), baseline = sizes[0]
            line_height = th + baseline + 2
            height = len(lines) * line_height + 2
            if width <= mask.shape[1] and height <= mask.shape[0]:
                valid = cv2.erode(mask, np.ones((height, width), np.uint8), anchor=(0, 0),
                                  borderType=cv2.BORDER_CONSTANT, borderValue=0)
                ys = np.flatnonzero(valid.any(axis=1))
                if ys.size:
                    row = (ys[len(ys) // 8] if anchor is None else
                           ys[np.argmin(np.abs(ys - (anchor[1] - y - height / 2)))])
                    xs = np.flatnonzero(valid[row])
                    col = (xs[len(xs) // 2] if anchor is None else
                           xs[np.argmin(np.abs(xs - (anchor[0] - x - width / 2)))])
                    px, py = int(x + col), int(y + row)
                    cv2.rectangle(image, (px, py), (px + width - 1, py + height - 1),
                                  (0, 0, 155) if lost else (25, 25, 25), -1)
                    for index, line in enumerate(lines):
                        origin = (px + 2, py + th + 1 + index * line_height)
                        cv2.putText(image, line, origin, 0, font_scale, color, 1, cv2.LINE_AA)
                    return True
            font_scale *= .75
    return False


def _review_polygon(vertices, previous):
    points = np.asarray(vertices, dtype=np.float32)
    if previous is None or points.shape != previous.shape:
        return points.copy()
    # The same polygon can be returned with another first corner or winding.
    if cv2.contourArea(points, oriented=True) * cv2.contourArea(previous, oriented=True) < 0:
        points = points[::-1]
    offset = (points - points.mean(axis=0)) - (previous[0] - previous.mean(axis=0))
    points = np.roll(points, -np.argmin(np.linalg.norm(offset, axis=1)), axis=0)
    basis = _basis_from_polygon(points)
    axes = np.stack((basis.short_axis, basis.long_axis))
    extent = np.ptp(points @ axes.T, axis=0)
    # Visible length varies under occlusion. Smooth more along the pipe and
    # limit lateral lag so the outline follows actual sideways motion.
    lag = .82 * ((previous - points) @ axes.T)
    limit = extent * [.07, .18]
    return (points + np.clip(lag, -limit, limit) @ axes).astype(np.float32)


def draw_overlay(gray: np.ndarray, results: list[dict[str, Any]], *, review_polygons=None) -> np.ndarray:
    overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    confidence_color = {
        "high": (0, 255, 0),
        "medium": (0, 200, 255),
        "low": (0, 0, 255),
        "failed": (0, 0, 255),
    }

    line_width = max(2, round(min(gray.shape[:2]) / 1100))
    for result in results:
        vertices = result["vertices"]
        anchor = None
        if review_polygons is not None:
            idx = result["roi_idx"]
            vertices = _review_polygon(vertices, review_polygons.get(idx))
            review_polygons[idx] = vertices
            anchor = (vertices.mean(axis=0) + vertices[0]) / 2
        vertices = np.asarray(vertices, dtype=np.int32)
        lost = result["status"] == "lost"
        color = (0, 0, 255) if lost else confidence_color.get(result["confidence"], (0, 0, 255))
        cv2.polylines(overlay, [vertices], True, (145, 145, 145), 1, cv2.LINE_AA)

        if result["status"] == "ok":
            for key, line_color in TUBE_LINES:
                pts = np.asarray(result[key], dtype=np.float32).round().astype(np.int32)
                if len(pts) >= 2:
                    cv2.polylines(overlay, [pts], False, line_color, line_width, cv2.LINE_AA)

        label = f"R{result['roi_idx']} {'LOST' if lost else result['confidence']}"
        scale = min(.85, max(.4, min(gray.shape[:2]) / 4800))
        draw_roi_label(overlay, vertices, label, scale, (255, 255, 255) if lost else color, lost, anchor)
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
            cv2.line(panel, (left_top, 0), (left_bottom, height - 1), WALL_COLOR, 2)
            cv2.line(panel, (right_top, 0), (right_bottom, height - 1), WALL_COLOR, 2)
            cv2.line(panel, (center_top, 0), (center_bottom, height - 1), CENTERLINE_COLOR, 1)
        else:
            cv2.line(panel, (left_x, 0), (left_x, height - 1), WALL_COLOR, 2)
            cv2.line(panel, (right_x, 0), (right_x, height - 1), WALL_COLOR, 2)
            cv2.line(panel, (center_x, 0), (center_x, height - 1), CENTERLINE_COLOR, 1)

    title = (
        f"ROI{result['roi_idx']} {result['confidence']} "
        f"w={result.get('tube_width_px', '-')}"
    )
    cv2.putText(panel, title, (4, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 255, 255), 1, cv2.LINE_AA)

    plot_height = 80
    plot = np.full((plot_height, width, 3), 255, dtype=np.uint8)
    max_value = max(float(profile.max()), 1.0)
    for x, value in enumerate(profile):
        y = int(plot_height - 1 - value / max_value * (plot_height - 8))
        cv2.line(plot, (x, plot_height - 1), (x, y), (0, 0, 0), 1)

    if result["status"] == "ok":
        cv2.line(plot, (int(result["left_x"]), 0), (int(result["left_x"]), plot_height - 1), WALL_COLOR, 1)
        cv2.line(plot, (int(result["right_x"]), 0), (int(result["right_x"]), plot_height - 1), WALL_COLOR, 1)

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


def detect_all(
    gray: np.ndarray, roi_payload: dict[str, Any], config: DetectionConfig, *, debug_panels_enabled: bool = True
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    results: list[dict[str, Any]] = []
    debug_panels: list[np.ndarray] = []
    for roi in roi_payload.get("rois", []):
        result, debug = detect_roi(gray, roi, config)
        results.append(result)
        if debug_panels_enabled:
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
        centerline_foreground_correction=getattr(args, "centerline_foreground_correction", False),
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
    parser.add_argument(
        "--centerline-foreground-correction",
        dest="centerline_foreground_correction",
        action="store_true",
        default=False,
        help="Shift centerlines toward the foreground/body profile when the foreground span is reliable.",
    )
    parser.add_argument(
        "--no-centerline-foreground-correction",
        dest="centerline_foreground_correction",
        action="store_false",
        help="Keep centerlines at the midpoint of detected left/right boundaries.",
    )
    args = parser.parse_args(argv)
    if args.draw_rois and args.rois:
        parser.error("--draw-rois cannot be used together with --rois.")
    if not args.draw_rois and not args.rois:
        parser.error("one of --rois or --draw-rois is required.")
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
