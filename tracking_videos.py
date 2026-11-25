"""Paired global and rectified ROI recordings of the same tracking results."""

from pathlib import Path

import cv2
import numpy as np

from detect_main_tube import CENTERLINE_COLOR, TUBE_LINES, WALL_COLOR, _basis_from_polygon


TILE_WIDTH, TILE_HEIGHT = 168, 340
HEADER = 54


def make_unique_output_path(path: Path) -> Path:
    if not path.exists():
        return path
    for index in range(2, 1000):
        candidate = path.with_name(f"{path.stem}_{index:03d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No available output filename near {path}")


def _open_video_writer(path, frame, fps):
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frame.shape[1::-1])
    if not writer.isOpened():
        raise ValueError(f"Could not open video writer: {path}")
    return writer


def _project(points, transform):
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return np.rint(xy @ transform[:, :2].T + transform[:, 2]).astype(np.int32)


class TrackingVideos:
    def __init__(self, path, fps, initial_rois, image_paths):
        self.path = Path(path)
        stem = self.path.stem
        roi_stem = (stem.replace("tracking_review", "roi_review", 1)
                    if stem.startswith("tracking_review") else f"{stem}_rois")
        self.roi_path = make_unique_output_path(self.path.with_name(roi_stem + ".mp4"))
        self.fps = fps
        self.total = len(image_paths)
        self.bases = {r["roi_idx"]: _basis_from_polygon(np.float32(r["vertices"])) for r in initial_rois}
        self.global_writer = self.roi_writer = None

    def roi_frame(self, gray, results, frame_index):
        count = max(1, len(self.bases))
        cols = min(5, count)
        rows = (count + cols - 1) // cols
        sheet = np.full((HEADER + TILE_HEIGHT * rows, TILE_WIDTH * cols, 3), 32, np.uint8)
        cv2.putText(sheet, f"Frame {frame_index + 1} / {self.total}", (6, 21), 0, .53, (255, 255, 255), 1, cv2.LINE_AA)
        for x, label, color in ((6, "Walls", WALL_COLOR), (75, "Center", CENTERLINE_COLOR)):
            cv2.putText(sheet, label, (x, 44), 0, .46, color, 1, cv2.LINE_AA)
        by_id = {r["roi_idx"]: r for r in results}
        for index, (roi_id, basis) in enumerate(self.bases.items()):
            result = by_id.get(roi_id, {})
            # Fixed orientation and magnification keep neighboring frames comparable.
            # Follow the tracked polygon even when the measured centerline is missing.
            center = basis.center
            if result.get("vertices") is not None:
                center = cv2.minAreaRect(np.float32(result["vertices"]))[0]
            axes = np.stack([basis.short_axis * TILE_WIDTH / (1.8 * basis.width),
                             basis.long_axis * TILE_HEIGHT / (1.06 * basis.height)])
            transform = np.column_stack([axes, np.array([TILE_WIDTH, TILE_HEIGHT]) / 2 - axes @ center])
            crop = cv2.warpAffine(gray, transform, (TILE_WIDTH, TILE_HEIGHT))
            tile = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
            visible = result.get("status") == "ok"
            if visible:
                for key, color in TUBE_LINES:
                    line = result[key]
                    if len(line) >= 2:
                        cv2.polylines(tile, [_project(line, transform)], False, color, 2, cv2.LINE_AA)
            label = f"R{roi_id}" if visible else f"R{roi_id} LOST"
            width = cv2.getTextSize(label, 0, .5, 1)[0][0]
            cv2.rectangle(tile, (0, 0), (width + 8, 22), (30, 30, 30) if visible else (0, 0, 155), -1)
            cv2.putText(tile, label, (4, 16), 0, .5, (255, 255, 255), 1, cv2.LINE_AA)
            y = HEADER + index // cols * TILE_HEIGHT
            x = index % cols * TILE_WIDTH
            sheet[y:y + TILE_HEIGHT, x:x + TILE_WIDTH] = tile
        return sheet

    def write(self, global_frame, gray, results, frame_index):
        local_frame = self.roi_frame(gray, results, frame_index)
        if self.global_writer is None:
            self.global_writer = _open_video_writer(self.path, global_frame, self.fps)
            self.roi_writer = _open_video_writer(self.roi_path, local_frame, self.fps)
        self.global_writer.write(global_frame)
        self.roi_writer.write(local_frame)

    def release(self):
        for writer in (self.global_writer, self.roi_writer):
            if writer is not None:
                writer.release()
