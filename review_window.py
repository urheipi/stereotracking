"""Interactive zoom and playback for the sequence tracking windows."""

from typing import Any

import cv2
import numpy as np

from detect_main_tube import RoiViewport

WINDOW_NAME = "Main Tube Sequence Review"


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
        self.viewport = RoiViewport.for_image(width, height, max_width=max_width, max_height=max_height)
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

    def _draw(self, frame: np.ndarray, paused: bool) -> None:
        canvas = self.viewport.render(frame)
        state = "Paused" if paused else "Live"
        label = f"{state}  {self.viewport.zoom:.1f}x   Wheel/+/- zoom   Right-drag pan   R fit"
        controls = "P pause   C play   N step   Q save & exit"
        for y, text in [(canvas.shape[0] - 29, label), (canvas.shape[0] - 7, controls)]:
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(canvas, text, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(WINDOW_NAME, canvas)

    def show(self, frame: np.ndarray, *, force_pause: bool) -> bool:
        pause = force_pause or not self.continuous
        while True:
            self._draw(frame, pause)
            # Pump mouse events and repaint the same frame while paused.
            key = cv2.waitKey(30 if pause else self.wait_ms) & 0xFF
            try:
                visible = cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE)
            except cv2.error:
                return False  # Some backends drop the window handle immediately.
            if visible < 1:
                return False
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
            if key in (ord("+"), ord("="), ord("-")):
                factor = 1 / 1.2 if key == ord("-") else 1.2
                self.viewport.zoom_at(self.viewport.canvas_width / 2, self.viewport.canvas_height / 2, factor)
                continue
            if not pause:
                return True

    def close(self) -> None:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass  # Already closed using the title bar.
