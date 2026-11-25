import contextlib
import io
import unittest
from unittest import mock

import cv2
import numpy as np

import detect_main_tube as dmt


class DetectMainTubeCliTest(unittest.TestCase):
    def test_parse_args_allows_drawing_rois_without_json(self):
        args = dmt.parse_args(["--image", "pic/1.bmp", "--draw-rois"])

        self.assertEqual(args.image, "pic/1.bmp")
        self.assertTrue(args.draw_rois)
        self.assertIsNone(args.rois)

    def test_parse_args_requires_roi_source_without_drawing(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                dmt.parse_args(["--image", "pic/1.bmp"])


class RoiPayloadTest(unittest.TestCase):
    def test_normalize_accepts_roi_points_from_existing_json(self):
        payload = {
            "image": "2.bmp",
            "rois": [
                {
                    "roi_idx": 7,
                    "roi_points": [[1, 2], [3.5, 4], [5, 6], [7, 8]],
                    "bbox": [1, 2, 7, 8],
                }
            ],
        }

        normalized = dmt.normalize_roi_payload(payload)

        self.assertEqual(normalized["image"], "2.bmp")
        self.assertEqual(
            normalized["rois"],
            [
                {
                    "roi_idx": 7,
                    "vertices": [[1.0, 2.0], [3.5, 4.0], [5.0, 6.0], [7.0, 8.0]],
                }
            ],
        )

    def test_normalize_assigns_roi_index_when_missing(self):
        payload = {"rois": [{"vertices": [[0, 0], [10, 0], [10, 10]]}]}

        normalized = dmt.normalize_roi_payload(payload)

        self.assertEqual(normalized["rois"][0]["roi_idx"], 1)


class RoiViewportTest(unittest.TestCase):
    def test_canvas_to_image_uses_zoom_and_offset(self):
        viewport = dmt.RoiViewport.for_image(4000, 3000, max_width=1000, max_height=750)
        viewport.zoom = 2.0
        viewport.offset_x = 1000.0
        viewport.offset_y = 500.0

        self.assertEqual(viewport.canvas_to_image(250, 125), [1500.0, 750.0])

    def test_zoom_at_keeps_cursor_on_same_image_point(self):
        viewport = dmt.RoiViewport.for_image(4000, 3000, max_width=1000, max_height=750)
        before = viewport.canvas_to_image(500, 375)

        viewport.zoom_at(500, 375, 1.5)
        after = viewport.canvas_to_image(500, 375)

        self.assertEqual(after, before)

    def test_pan_moves_view_in_image_coordinates(self):
        viewport = dmt.RoiViewport.for_image(4000, 3000, max_width=1000, max_height=750)
        viewport.zoom_at(500, 375, 2.0)

        before = viewport.canvas_to_image(300, 200)
        viewport.pan(50, -25)
        after = viewport.canvas_to_image(300, 200)

        self.assertEqual(after, [before[0] - 100.0, before[1] + 50.0])


class CenterlineSummaryTest(unittest.TestCase):
    def test_centerline_geometry_uses_endpoints(self):
        geometry = dmt.centerline_geometry([[1, 2], [4, 6], [7, 10]])

        self.assertEqual(
            geometry,
            {
                "endpoints": [1.0, 2.0, 7.0, 10.0],
                "plucker": [-4.0, 6.0, 8.0],
                "line_eq": {"A": -0.8, "B": 0.6, "C": -0.4},
            },
        )

    def test_make_centerline_summary_omits_sampling_points(self):
        payload = dmt.make_centerline_summary(
            image_path="pic/1.bmp",
            rois_source="drawn_rois.json",
            results=[
                {
                    "roi_idx": 1,
                    "status": "ok",
                    "confidence": "high",
                    "centerline": [[1, 2], [4, 6], [7, 10]],
                    "left_boundary": [[0, 0], [1, 1]],
                    "right_boundary": [[2, 2], [3, 3]],
                    "centerline_source": "foreground-body",
                    "centerline_x_offset": 8.5,
                },
                {
                    "roi_idx": 2,
                    "status": "failed",
                    "confidence": "failed",
                    "centerline": [],
                    "failure_reason": "no valid pair of long side edges",
                },
            ],
        )

        self.assertEqual(payload["image"], "pic/1.bmp")
        self.assertEqual(payload["rois"], "drawn_rois.json")
        self.assertEqual(payload["num_rois"], 2)
        self.assertEqual(payload["centerlines"][0]["centerline"]["endpoints"], [1.0, 2.0, 7.0, 10.0])
        self.assertEqual(payload["centerlines"][0]["centerline_source"], "foreground-body")
        self.assertEqual(payload["centerlines"][0]["centerline_x_offset"], 8.5)
        self.assertNotIn("left_boundary", payload["centerlines"][0])
        self.assertNotIn("right_boundary", payload["centerlines"][0])
        self.assertNotIn("centerline_points", payload["centerlines"][0])
        self.assertIsNone(payload["centerlines"][1]["centerline"])
        self.assertEqual(payload["centerlines"][1]["failure_reason"], "no valid pair of long side edges")


class RoiDetectionGeometryTest(unittest.TestCase):
    def test_one_visible_wall_and_a_quantization_step_do_not_make_a_tube(self):
        gray = np.full((190, 130), 235, dtype=np.uint8)
        gray[:, :54] = 55
        gray[:, 90:] = 234
        roi = {"roi_idx": 23, "vertices": [[30, 5], [112, 5], [112, 184], [30, 184]]}
        result, _ = dmt.detect_roi(gray, roi)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["centerline"], [])

    def test_shaded_tube_crossed_by_wires_keeps_geometric_center(self):
        gray = np.full((920, 170), 245, dtype=np.uint8)
        gray[:, 50:115] = np.linspace(178, 38, 65).astype(np.uint8)
        for y in (145, 310, 650):
            cv2.line(gray, (12, y), (155, y + 90), 22, 9)
        roi = {"roi_idx": 1, "vertices": [[38, 12], [39, 904], [128, 902], [125, 12]]}
        config = dmt.DetectionConfig(centerline_foreground_correction=True)
        result, _ = dmt.detect_roi(gray, roi, config)
        self.assertEqual(result["status"], "ok")
        center = np.asarray(result["centerline"])
        self.assertLess(np.max(np.abs(center[:, 0] - 82)), 2)
        self.assertGreater(np.ptp(center[:, 1]), 800)

    def test_sample_boundaries_uses_common_line_support_range(self):
        meta = dmt.RectifyMeta(
            center=np.asarray([0.0, 0.0], dtype=np.float32),
            long_axis=np.asarray([0.0, 1.0], dtype=np.float32),
            short_axis=np.asarray([1.0, 0.0], dtype=np.float32),
            long_min=0.0,
            short_min=0.0,
            width=100,
            height=100,
        )
        mask = np.full((100, 100), 255, dtype=np.uint8)
        left_model = dmt.BoundaryLineModel(
            slope=0.0,
            intercept=20.0,
            support_rows=30,
            median_residual=0.1,
            support_y_min=20.0,
            support_y_max=60.0,
        )
        right_model = dmt.BoundaryLineModel(
            slope=0.0,
            intercept=60.0,
            support_rows=30,
            median_residual=0.1,
            support_y_min=30.0,
            support_y_max=70.0,
        )

        _left, _right, center = dmt._sample_boundaries(
            meta,
            mask,
            left_x=20,
            right_x=60,
            count=5,
            left_model=left_model,
            right_model=right_model,
        )

        self.assertAlmostEqual(center[0][1], 30.0, places=1)
        self.assertAlmostEqual(center[-1][1], 60.0, places=1)

    def test_detect_roi_reports_oversized_roi_instead_of_crashing(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        roi = {
            "roi_idx": 9,
            "vertices": [[0, 0], [50, 0], [50, 50], [0, 50]],
        }

        with mock.patch.object(
            dmt,
            "rectify_polygon",
            side_effect=cv2.error("OpenCV remap size assertion"),
        ):
            result, debug = dmt.detect_roi(gray, roi)

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["confidence"], "failed")
        self.assertEqual(result["failure_reason"], "invalid or oversized roi")
        self.assertEqual(result["left_boundary"], [])
        self.assertEqual(result["right_boundary"], [])
        self.assertEqual(result["centerline"], [])
        self.assertEqual(debug, {})

    def test_detect_all_builds_debug_panel_for_failed_roi_without_debug_crop(self):
        gray = np.zeros((100, 100), dtype=np.uint8)
        roi_payload = {"rois": [{"roi_idx": 9, "vertices": [[0, 0], [50, 0], [50, 50], [0, 50]]}]}
        failed_result = {
            "roi_idx": 9,
            "status": "failed",
            "confidence": "failed",
            "failure_reason": "invalid or oversized roi",
            "vertices": [[0, 0], [50, 0], [50, 50], [0, 50]],
            "left_boundary": [],
            "right_boundary": [],
            "centerline": [],
        }

        with mock.patch.object(dmt, "detect_roi", return_value=(failed_result, {})):
            results, debug_panels = dmt.detect_all(gray, roi_payload, dmt.DetectionConfig())

        self.assertEqual(results[0]["failure_reason"], "invalid or oversized roi")
        self.assertEqual(len(debug_panels), 1)
        self.assertEqual(debug_panels[0].ndim, 3)
        self.assertGreater(int(debug_panels[0].sum()), 0)

    def test_draw_overlay_uses_previous_yellow_centerline_style(self):
        gray = np.zeros((70, 70), dtype=np.uint8)
        result = {
            "roi_idx": 1,
            "status": "ok",
            "confidence": "high",
            "vertices": [[5, 5], [65, 5], [65, 65], [5, 65]],
            "left_boundary": [[15, 10], [15, 60]],
            "right_boundary": [[55, 10], [55, 60]],
            "centerline": [[35, 10], [35, 60]],
        }

        overlay = dmt.draw_overlay(gray, [result])

        for x in (15, 55):
            np.testing.assert_array_equal(overlay[55, x], dmt.WALL_COLOR)
        center_pixel = overlay[55, 35]
        adjacent_pixel = overlay[55, 34]
        self.assertLess(int(center_pixel[0]), 10)
        self.assertGreater(int(center_pixel[1]), 200)
        self.assertGreater(int(center_pixel[2]), 200)
        self.assertGreater(int(adjacent_pixel[1]), 200)
        self.assertGreater(int(adjacent_pixel[2]), 200)

    def test_detect_roi_refines_centerline_direction_inside_slightly_skewed_roi(self):
        gray = np.full((170, 170), 255, dtype=np.uint8)
        cv2.rectangle(gray, (65, 20), (95, 145), 25, -1)
        roi = {
            "roi_idx": 1,
            "vertices": [[45, 18], [68, 148], [115, 148], [96, 18]],
        }

        result, _debug = dmt.detect_roi(gray, roi)

        self.assertEqual(result["status"], "ok")
        top_x, _top_y = result["centerline"][0]
        bottom_x, _bottom_y = result["centerline"][-1]
        self.assertLess(abs(top_x - bottom_x), 3.0)

    def test_detect_roi_uses_weak_outer_edge_instead_of_stronger_internal_edge(self):
        gray = np.full((180, 180), 140, dtype=np.uint8)
        cv2.rectangle(gray, (60, 20), (130, 160), 45, -1)
        cv2.rectangle(gray, (104, 20), (110, 160), 150, -1)
        cv2.rectangle(gray, (131, 20), (160, 160), 78, -1)
        roi = {
            "roi_idx": 1,
            "vertices": [[45, 18], [48, 165], [155, 165], [152, 18]],
        }

        result, _debug = dmt.detect_roi(gray, roi)

        center_x = float(np.mean([point[0] for point in result["centerline"]]))
        self.assertLess(abs(center_x - 95.0), 5.0)

    def test_find_edge_pair_accepts_narrow_valid_tube(self):
        crop = np.full((100, 28), 230, dtype=np.uint8)
        cv2.rectangle(crop, (10, 5), (18, 95), 50, -1)
        mask = np.full_like(crop, 255)

        pair, _profile, _evidence = dmt.find_edge_pair(crop, mask, dmt.DetectionConfig())

        self.assertEqual(pair["status"], "ok")
        self.assertLessEqual(pair["left_x"], 10)
        self.assertGreaterEqual(pair["right_x"], 18)


if __name__ == "__main__":
    unittest.main()
