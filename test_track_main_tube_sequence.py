import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock
from zipfile import ZipFile

import cv2
import numpy as np

import track_main_tube_sequence as tracker
import tracking_diagnostics as diagnostics


class DiagnosticsTest(unittest.TestCase):
    def test_local_bundle_preserves_rejected_evidence_and_has_no_source_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / "private-input"
            src.mkdir()
            images = []
            for i, visible in enumerate([True, False, True, True]):
                gray = np.full((160, 120), 225, np.uint8)
                gray[:, 45:75 if visible else 120] = 50
                p = src / f"private-{i}.png"
                tracker.dmt.write_image_unicode(p, gray)
                images.append(str(p))
            rois = root / "rois.json"
            rois.write_text(json.dumps({"rois": [{"roi_idx": 4, "vertices": [[30, 10], [90, 10], [90, 150], [30, 150]]}]}))
            args = tracker.parse_args(["--images", *images, "--initial-rois", str(rois), "--out-dir", tmp,
                                       "--roi-update-mode", "flow", "--no-window", "--no-video", "--diagnostics"])
            with contextlib.redirect_stdout(io.StringIO()):
                tracker.run_sequence(args)
            with ZipFile(next(root.glob("*_diagnostics.zip"))) as bundle:
                summary = bundle.read("summary.json").decode()
                self.assertNotIn("private-", summary)
                self.assertNotIn(str(root.name), summary)
                stats = json.loads(summary)["by_roi"]["4"]
                self.assertEqual((stats["lost"], stats["longest_loss"], stats["first_recovery"]), (2, [2, 3], 4))
                rows = list(csv.DictReader(io.StringIO(bundle.read("measurements.csv").decode())))
                self.assertEqual(rows[2]["reason"], "awaiting second wall measurement")
                self.assertTrue(float(rows[2]["candidate_tube_width_px"]) > 0)
                self.assertEqual(rows[2]["center_x1"], "")
                jpeg = bundle.read("samples/F000002_R4_lost.jpg")
                self.assertIsNotNone(cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR))

    def test_image_budget_is_shared_across_rois(self):
        items = [{"roi_idx": idx, "status": "ok", "wall_pair_coverage": .8} for idx in range(100)]
        frames = [{"centerlines": items}] * 5
        stats, samples = diagnostics._summarize(frames)
        self.assertEqual(len(stats), 100)
        self.assertEqual(len(samples), diagnostics.MAX_SAMPLES)
        self.assertEqual(len(set(samples)), len(samples))


class TrackingVideosTest(unittest.TestCase):
    def test_both_recordings_end_on_the_frame_where_review_stops(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            gray = np.full((160, 120), 235, np.uint8)
            gray[:, 45:75] = 55
            images = [root / f"{i}.png" for i in range(3)]
            for path in images:
                tracker.dmt.write_image_unicode(path, gray)
            rois = root / "rois.json"
            rois.write_text(json.dumps({"rois": [{"roi_idx": 7, "vertices": [[30, 8], [88, 8], [88, 152], [30, 152]]}]}))
            args = tracker.parse_args(["--images", *map(str, images), "--out-dir", str(root),
                                       "--initial-rois", str(rois)])
            with mock.patch.object(tracker, "ReviewWindow") as window:
                window.return_value.show.side_effect = [True, False]
                payload = tracker.run_sequence(args)
            self.assertEqual(payload["num_frames"], 2)
            for path in (args._resolved_out_video, args._resolved_roi_video):
                cap = cv2.VideoCapture(str(path))
                try:
                    self.assertTrue(cap.read()[0])
                    ok, last = cap.read()
                    self.assertTrue(ok)
                    self.assertFalse(cap.read()[0])
                    if path == args._resolved_roi_video:
                        blue, green, red = cv2.split(last[90:-20])
                        self.assertTrue(np.any((blue < 90) & (green > 135) & (red > 155)))
                        self.assertTrue(np.any((blue > 150) & (green > 125) & (red < 100)))
                finally:
                    cap.release()


class ReviewWindowTest(unittest.TestCase):
    def test_wheel_zoom_redraws_paused_frame_before_advancing(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        with mock.patch.multiple(cv2, namedWindow=mock.DEFAULT, resizeWindow=mock.DEFAULT,
                                 setMouseCallback=mock.DEFAULT, imshow=mock.DEFAULT), \
                mock.patch.object(cv2, "getWindowProperty", return_value=1):
            window = tracker.ReviewWindow((80, 100), max_width=100, max_height=80, review_mode="step", wait_ms=1)
            zooms = []

            def wait(delay):
                self.assertGreater(delay, 0)
                zooms.append(window.viewport.zoom)
                if len(zooms) == 1:
                    window._on_mouse(cv2.EVENT_MOUSEWHEEL, 50, 40, 120, None)
                    return -1
                return ord("n")

            with mock.patch.object(cv2, "waitKey", side_effect=wait):
                self.assertTrue(window.show(frame, force_pause=False))
            self.assertEqual(zooms, [1.0, 1.2])
            self.assertEqual(cv2.imshow.call_count, 2)


class SequenceSummaryTest(unittest.TestCase):
    def test_independent_tubes_with_reversals_and_crossing_wires(self):
        # Two tubes descend and reverse while a third travels in another
        # direction. Their ground truth is known independently of the detector.
        def box(center, axis, length, half_width):
            normal = np.array([-axis[1], axis[0]])
            return np.array([center + a * length * axis + b * half_width * normal
                             for a, b in [(-1, -1), (-1, 1), (1, 1), (1, -1)]])

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            images, rois, truth = [], [], []
            for i, shift in enumerate([0, 8, 16, 24, 32, 40, 30, 20, 10, 0]):
                gray = np.full((360, 540), 226 - i, np.uint8)
                gray[:, 360:] = 33 + i
                expected = []
                for idx, (center, axis, radius, intensity) in enumerate([
                    ((82, 130 + shift), (0.3, 1), 14, 70),
                    ((255, 132 + shift), (1, 0.3), 10, 90),
                    ((445 - shift / 3, 165 - shift / 3), (-0.6, 1), 13, 135),
                ], 1):
                    c = np.float32(center)
                    along = np.float32(axis) / np.linalg.norm(axis)
                    cv2.fillConvexPoly(gray, box(c, along, 72, radius).round().astype(np.int32), intensity)
                    if i == 0:
                        rois.append({"roi_idx": idx, "vertices": box(c, along, 78, radius + 8).tolist()})
                    expected.append([c - along * 35, c + along * 35])
                cv2.line(gray, (363, 120), (513, 189), 18, 3)
                path = root / f"{i}.png"
                tracker.dmt.write_image_unicode(path, gray)
                images.append(str(path))
                truth.append(expected)
            roi_path = root / "rois.json"
            roi_path.write_text(json.dumps({"rois": rois}))
            args = tracker.parse_args(["--images", *images, "--initial-rois", str(roi_path),
                                       "--out-dir", tmp, "--track-scale", "1", "--no-video", "--no-window"])
            frames = tracker.run_sequence(args)["frames"]
            self.assertEqual(len(frames), len(truth))
            for frame, targets in zip(frames, truth, strict=True):
                for line, target in zip(frame["centerlines"], targets, strict=True):
                    self.assertIn(line["confidence"], {"high", "medium"})
                    a, b = np.reshape(line["centerline"]["endpoints"], (2, 2))
                    v = b - a
                    for point in target:
                        nearest = a + np.clip((point - a) @ v / (v @ v), 0, 1) * v
                        self.assertLess(np.linalg.norm(point - nearest), 3)

    def test_local_lk_preserves_global_position_for_distant_roi(self):
        prev = np.zeros((1200, 1400), dtype=np.uint8)
        prev[730:810, 985:1055] = np.random.default_rng(7).integers(20, 220, (80, 70), dtype=np.uint8)
        moved = cv2.warpAffine(prev, np.float32([[1, 0, 12], [0, 1, -8]]), (1400, 1200))
        vertices = np.float32([[980, 725], [1060, 725], [1060, 815], [980, 815]])
        with mock.patch.object(cv2, "calcOpticalFlowPyrLK", wraps=cv2.calcOpticalFlowPyrLK) as lk:
            matrix, info = tracker._estimate_roi_transform(prev, moved, vertices, 40, 3, 1.5, 50)
        self.assertEqual(info["method"], "lk")
        self.assertEqual(lk.call_count, 2)
        self.assertLess(lk.call_args.args[0].size, prev.size)
        predicted = cv2.transform(vertices[None, :, :], matrix)[0]
        np.testing.assert_allclose(predicted, vertices + [12, -8], atol=0.5)

    def test_sequence_payload_contains_centerline_summaries_without_samples(self):
        frame_summary = tracker.make_frame_centerline_summary(
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
                    "centerline_points": [[1, 2], [7, 10]],
                }
            ],
        )

        payload = tracker.make_sequence_centerline_payload(
            frames=[frame_summary],
            initial_rois="drawn_rois.json",
        )

        self.assertEqual(payload["initial_rois"], "drawn_rois.json")
        self.assertEqual(payload["num_frames"], 1)
        self.assertEqual(payload["frames"][0]["image"], "pic/1.bmp")
        self.assertEqual(payload["frames"][0]["num_rois"], 1)
        centerline_item = payload["frames"][0]["centerlines"][0]
        self.assertEqual(centerline_item["roi_idx"], 1)
        self.assertEqual(centerline_item["centerline"]["endpoints"], [1.0, 2.0, 7.0, 10.0])
        self.assertNotIn("left_boundary", centerline_item)
        self.assertNotIn("right_boundary", centerline_item)
        self.assertNotIn("centerline_points", centerline_item)

    def test_frame_summary_keeps_roi_and_tracking_diagnostics(self):
        frame_summary = tracker.make_frame_centerline_summary(
            image_path="pic/2.bmp",
            rois_source="tracked_from_previous_frame",
            results=[
                {
                    "roi_idx": 1,
                    "status": "ok",
                    "confidence": "tracked",
                    "vertices": [[10, 10], [20, 10], [20, 40], [10, 40]],
                    "centerline": [[15, 12], [15, 38]],
                    "fallback_reason": "no valid pair of long side edges",
                    "left_boundary": [[12, 12], [12, 38]],
                    "right_boundary": [[18, 12], [18, 38]],
                }
            ],
            tracking=[
                {
                    "roi_idx": 1,
                    "status": "ok",
                    "method": "template",
                    "update_status": "detect-corrected",
                    "template_score": 0.92,
                }
            ],
        )

        centerline_item = frame_summary["centerlines"][0]
        self.assertEqual(centerline_item["vertices"], [[10.0, 10.0], [20.0, 10.0], [20.0, 40.0], [10.0, 40.0]])
        self.assertEqual(centerline_item["fallback_reason"], "no valid pair of long side edges")
        self.assertEqual(centerline_item["tracking"]["method"], "template")
        self.assertEqual(centerline_item["tracking"]["update_status"], "detect-corrected")
        self.assertNotIn("left_boundary", centerline_item)
        self.assertNotIn("right_boundary", centerline_item)


class RoiTrackingTest(unittest.TestCase):
    def test_detect_roi_features_crops_work_and_restores_image_coordinates(self):
        gray = np.zeros((100, 120), dtype=np.uint8)
        vertices = np.asarray([[40, 30], [60, 30], [60, 50], [40, 50]], dtype=np.float32)
        local_points = np.asarray([[[7.0, 8.0]], [[15.0, 19.0]]], dtype=np.float32)

        with mock.patch.object(cv2, "goodFeaturesToTrack", return_value=local_points) as detect:
            points = tracker._detect_roi_features(gray, vertices, max_corners=8)

        cropped_gray = detect.call_args.args[0]
        cropped_mask = detect.call_args.kwargs["mask"]
        self.assertEqual(cropped_gray.shape, (29, 29))
        self.assertEqual(cropped_mask.shape, cropped_gray.shape)
        self.assertTrue(np.array_equal(points, np.asarray([[43, 34], [51, 45]], dtype=np.float32)))

    def test_track_rois_between_frames_translates_polygon(self):
        prev_gray = np.zeros((90, 90), dtype=np.uint8)
        cv2.circle(prev_gray, (28, 28), 3, 255, -1)
        cv2.circle(prev_gray, (42, 28), 3, 220, -1)
        cv2.circle(prev_gray, (28, 45), 3, 200, -1)
        cv2.circle(prev_gray, (48, 48), 3, 180, -1)
        transform = np.float32([[1, 0, 7], [0, 1, 5]])
        next_gray = cv2.warpAffine(prev_gray, transform, (90, 90))

        roi_payload = {
            "rois": [
                {
                    "roi_idx": 3,
                    "vertices": [[20, 20], [55, 20], [55, 55], [20, 55]],
                }
            ]
        }

        tracked_payload, tracking = tracker.track_rois_between_frames(prev_gray, next_gray, roi_payload, track_scale=1.0)

        self.assertEqual(tracked_payload["num_rois"], 1)
        self.assertEqual(tracked_payload["rois"][0]["roi_idx"], 3)
        tracked_vertices = np.asarray(tracked_payload["rois"][0]["vertices"], dtype=np.float32)
        expected_vertices = np.asarray([[27, 25], [62, 25], [62, 60], [27, 60]], dtype=np.float32)
        self.assertTrue(np.allclose(tracked_vertices, expected_vertices, atol=1.5))
        self.assertEqual(tracking[0]["status"], "ok")

    def test_track_rois_between_frames_with_track_scale_keeps_original_coordinates(self):
        prev_gray = np.zeros((180, 180), dtype=np.uint8)
        cv2.circle(prev_gray, (56, 56), 5, 255, -1)
        cv2.circle(prev_gray, (86, 58), 5, 230, -1)
        cv2.circle(prev_gray, (58, 94), 5, 210, -1)
        cv2.circle(prev_gray, (98, 98), 5, 190, -1)
        transform = np.float32([[1, 0, 12], [0, 1, 8]])
        next_gray = cv2.warpAffine(prev_gray, transform, (180, 180))
        roi_payload = {
            "rois": [
                {
                    "roi_idx": 4,
                    "vertices": [[40, 40], [115, 40], [115, 115], [40, 115]],
                }
            ]
        }

        tracked_payload, tracking = tracker.track_rois_between_frames(
            prev_gray,
            next_gray,
            roi_payload,
            track_scale=0.5,
        )

        tracked_vertices = np.asarray(tracked_payload["rois"][0]["vertices"], dtype=np.float32)
        expected_vertices = np.asarray([[52, 48], [127, 48], [127, 123], [52, 123]], dtype=np.float32)
        self.assertTrue(np.allclose(tracked_vertices, expected_vertices, atol=3.0))
        self.assertEqual(tracking[0]["status"], "ok")

    def test_cluster_at_one_end_does_not_shrink_the_whole_search_window(self):
        gray = np.zeros((240, 120), np.uint8)
        vertices = np.float32([[20, 10], [80, 10], [80, 225], [20, 225]])
        points = np.float32([[31, 201], [36, 208], [33, 212], [43, 211]])
        moved = (points - points[0]) * .72 + points[0] + [3, -4]
        status = np.ones((len(points), 1), np.uint8)
        with mock.patch.object(tracker, "_detect_roi_features", return_value=points), mock.patch.object(
            cv2, "calcOpticalFlowPyrLK", side_effect=[
                (moved.astype(np.float32), status, None), (points, status, None)]
        ):
            matrix, _ = tracker._estimate_roi_transform(gray, gray, vertices, 80, 3, 1.5)
        np.testing.assert_allclose(matrix[:, :2], np.eye(2))
        np.testing.assert_allclose(matrix[:, 2], np.median(moved - points, axis=0), atol=1e-4)

    def test_track_rois_between_frames_template_fallback_handles_large_translation(self):
        prev_gray = np.zeros((130, 150), dtype=np.uint8)
        cv2.rectangle(prev_gray, (34, 34), (70, 62), 120, -1)
        cv2.circle(prev_gray, (42, 42), 4, 255, -1)
        cv2.line(prev_gray, (36, 58), (68, 38), 210, 2)
        transform = np.float32([[1, 0, 28], [0, 1, 19]])
        next_gray = cv2.warpAffine(prev_gray, transform, (150, 130))
        roi_payload = {
            "rois": [
                {
                    "roi_idx": 8,
                    "vertices": [[30, 30], [76, 30], [76, 68], [30, 68]],
                }
            ]
        }

        with mock.patch.object(
            tracker,
            "_detect_roi_features",
            return_value=np.empty((0, 2), dtype=np.float32),
        ):
            tracked_payload, tracking = tracker.track_rois_between_frames(
                prev_gray,
                next_gray,
                roi_payload,
                track_scale=1.0,
                max_roi_jump=50.0,
            )

        tracked_vertices = np.asarray(tracked_payload["rois"][0]["vertices"], dtype=np.float32)
        expected_vertices = np.asarray([[58, 49], [104, 49], [104, 87], [58, 87]], dtype=np.float32)
        self.assertTrue(np.allclose(tracked_vertices, expected_vertices, atol=1.5))
        self.assertEqual(tracking[0]["status"], "ok")
        self.assertEqual(tracking[0]["method"], "template")
        self.assertEqual(tracking[0]["lk_failure_reason"], "not enough features in roi")

    def test_track_rois_between_frames_rejects_unreasonable_roi_jump(self):
        prev_gray = np.zeros((100, 100), dtype=np.uint8)
        next_gray = np.zeros((100, 100), dtype=np.uint8)
        roi_payload = {
            "rois": [
                {
                    "roi_idx": 5,
                    "vertices": [[10, 10], [20, 10], [20, 20], [10, 20]],
                }
            ]
        }
        bad_matrix = np.asarray([[100000.0, 0.0, 0.0], [0.0, 100000.0, 0.0]], dtype=np.float32)

        with mock.patch.object(
            tracker,
            "_estimate_roi_transform",
            return_value=(bad_matrix, {"status": "ok", "num_features": 8, "num_tracked": 8, "num_inliers": 8}),
        ):
            tracked_payload, tracking = tracker.track_rois_between_frames(
                prev_gray,
                next_gray,
                roi_payload,
                track_scale=1.0,
                max_roi_jump=200.0,
            )

        self.assertEqual(tracked_payload["rois"][0], roi_payload["rois"][0])
        self.assertEqual(tracking[0]["status"], "failed")
        self.assertEqual(tracking[0]["failure_reason"], "unreasonable tracked roi")

    def test_track_rois_between_frames_rejects_group_vertical_outlier(self):
        prev_gray = np.zeros((220, 220), dtype=np.uint8)
        next_gray = np.zeros((220, 220), dtype=np.uint8)
        roi_payload = {
            "rois": [
                {"roi_idx": 1, "vertices": [[10, 10], [40, 10], [40, 50], [10, 50]]},
                {"roi_idx": 2, "vertices": [[70, 10], [100, 10], [100, 50], [70, 50]]},
                {"roi_idx": 3, "vertices": [[130, 10], [160, 10], [160, 50], [130, 50]]},
            ]
        }
        transforms = [
            (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 2.0]], dtype=np.float32), {"status": "ok"}),
            (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 3.0]], dtype=np.float32), {"status": "ok"}),
            (np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 100.0]], dtype=np.float32), {"status": "ok"}),
        ]

        with mock.patch.object(tracker, "_estimate_roi_transform", side_effect=transforms):
            tracked_payload, tracking = tracker.track_rois_between_frames(
                prev_gray,
                next_gray,
                roi_payload,
                track_scale=1.0,
                max_roi_jump=200.0,
                track_group_outlier_min_dy=55.0,
                track_group_outlier_dy_deviation=45.0,
            )

        self.assertEqual(tracking[2]["status"], "failed")
        self.assertEqual(tracking[2]["failure_reason"], "inconsistent group motion")
        self.assertEqual(tracked_payload["rois"][2], roi_payload["rois"][2])
        self.assertEqual(tracked_payload["rois"][0]["vertices"][0], [10.0, 12.0])

    def test_correct_roi_from_detection_wraps_detected_boundaries_with_padding(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[5, 5], [95, 5], [95, 75], [5, 75]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "left_boundary": [[40, 20], [40, 80]],
            "right_boundary": [[60, 20], [60, 80]],
            "centerline": [[50, 20], [50, 80]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
        )

        vertices = np.asarray(corrected["vertices"], dtype=np.float32)
        self.assertEqual(corrected["roi_idx"], 7)
        self.assertEqual(info["update_status"], "detect-corrected")
        self.assertTrue(np.allclose(vertices[:, 0].min(), 37.5, atol=0.1))
        self.assertTrue(np.allclose(vertices[:, 0].max(), 62.5, atol=0.1))
        self.assertTrue(np.allclose(vertices[:, 1].min(), 14.0, atol=0.1))
        self.assertTrue(np.allclose(vertices[:, 1].max(), 86.0, atol=0.1))

    def test_correct_roi_from_detection_rejects_large_center_jump(self):
        predicted_roi = {
            "roi_idx": 1,
            "vertices": [[0, 0], [20, 0], [20, 20], [0, 20]],
        }
        result = {
            "roi_idx": 1,
            "status": "ok",
            "left_boundary": [[200, 200], [200, 260]],
            "right_boundary": [[220, 200], [220, 260]],
            "centerline": [[210, 200], [210, 260]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=40.0,
        )

        self.assertEqual(corrected, predicted_roi)
        self.assertEqual(info["update_status"], "rejected-roi-jump")

    def test_correct_roi_from_detection_rejects_low_confidence_state_update(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[5, 5], [95, 5], [95, 75], [5, 75]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "low",
            "left_boundary": [[40, 20], [40, 80]],
            "right_boundary": [[60, 20], [60, 80]],
            "centerline": [[50, 20], [50, 80]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
            min_update_confidence="medium",
        )

        self.assertEqual(corrected, predicted_roi)
        self.assertEqual(info["update_status"], "low-confidence-detection")
        self.assertEqual(info["detection_confidence"], "low")

    def test_correct_roi_from_detection_keeps_minimum_corrected_short_span(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[5, 5], [95, 5], [95, 75], [5, 75]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "medium",
            "left_boundary": [[49, 20], [49, 80]],
            "right_boundary": [[51, 20], [51, 80]],
            "centerline": [[50, 20], [50, 80]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
            min_corrected_roi_short_span=40.0,
        )

        vertices = np.asarray(corrected["vertices"], dtype=np.float32)
        self.assertEqual(info["update_status"], "detect-corrected")
        self.assertGreaterEqual(float(vertices[:, 0].max() - vertices[:, 0].min()), 39.9)

    def test_correct_roi_from_detection_rejects_oversized_corrected_span(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[0, 0], [30, 0], [30, 40], [0, 40]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "high",
            "left_boundary": [[14, -50000], [14, 50000]],
            "right_boundary": [[16, -50000], [16, 50000]],
            "centerline": [[15, -50000], [15, 50000]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
            min_corrected_roi_short_span=40.0,
        )

        self.assertEqual(corrected, predicted_roi)
        self.assertEqual(info["update_status"], "rejected-corrected-roi")
        self.assertEqual(info["rejection_reason"], "unreasonable corrected roi")

    def test_correct_roi_from_detection_caps_growth_relative_to_predicted_roi(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[0, 0], [100, 0], [100, 20], [0, 20]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "high",
            "left_boundary": [[-20, 8], [120, 8]],
            "right_boundary": [[-20, 12], [120, 12]],
            "centerline": [[-20, 10], [120, 10]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.25,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
            min_corrected_roi_short_span=10.0,
            max_corrected_roi_growth=1.0,
        )

        vertices = np.asarray(corrected["vertices"], dtype=np.float32)
        self.assertEqual(info["update_status"], "detect-corrected")
        self.assertLessEqual(float(vertices[:, 0].max() - vertices[:, 0].min()), 100.1)
        self.assertGreaterEqual(float(vertices[:, 1].max() - vertices[:, 1].min()), 9.9)

    def test_correct_roi_from_detection_rejects_collapsed_long_span(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[0, 0], [100, 0], [100, 20], [0, 20]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "high",
            "left_boundary": [[0, 8], [50, 8]],
            "right_boundary": [[0, 12], [50, 12]],
            "centerline": [[0, 10], [50, 10]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.0,
            length_padding_frac=0.0,
            max_roi_jump=200.0,
            min_corrected_roi_long_span_ratio=0.78,
            min_corrected_roi_long_span_drop=45.0,
        )

        self.assertEqual(corrected, predicted_roi)
        self.assertEqual(info["update_status"], "rejected-corrected-roi-shrink")
        self.assertLess(info["roi_long_span_ratio"], 0.78)

    def test_correct_roi_from_detection_rejects_anchor_drift(self):
        predicted_roi = {
            "roi_idx": 7,
            "vertices": [[0, 0], [100, 0], [100, 20], [0, 20]],
        }
        anchor_roi = {
            "roi_idx": 7,
            "vertices": [[0, 0], [100, 0], [100, 20], [0, 20]],
        }
        result = {
            "roi_idx": 7,
            "status": "ok",
            "confidence": "high",
            "left_boundary": [[140, 8], [240, 8]],
            "right_boundary": [[140, 12], [240, 12]],
            "centerline": [[140, 10], [240, 10]],
        }

        corrected, info = tracker.correct_roi_from_detection(
            predicted_roi,
            result,
            padding_frac=0.0,
            length_padding_frac=0.0,
            max_roi_jump=300.0,
            anchor_roi=anchor_roi,
            max_anchor_drift=30.0,
        )

        self.assertEqual(corrected, anchor_roi)
        self.assertEqual(info["update_status"], "rejected-anchor-drift")

    def test_exposure_suspect_info_detects_bright_saturated_jump(self):
        previous_stats = tracker.measure_exposure_stats(np.full((32, 32), 90, dtype=np.uint8), sample_stride=1)
        current = np.full((32, 32), 210, dtype=np.uint8)
        current[:, :20] = 255
        current_stats = tracker.measure_exposure_stats(current, sample_stride=1)

        suspected, info = tracker.exposure_suspect_info(
            previous_stats,
            current_stats,
            mean_jump=25.0,
            saturation_jump=0.15,
            bright_mean=175.0,
            saturation_frac=0.48,
        )

        self.assertTrue(suspected)
        self.assertEqual(info["exposure_status"], "suspect")
        self.assertGreater(info["exposure_mean_delta"], 25.0)
        self.assertGreater(info["exposure_saturated_delta"], 0.15)

    def test_guard_exposure_shrunken_rois_rejects_active_exposure_shrink(self):
        predicted_payload = {
            "rois": [
                {"roi_idx": 1, "vertices": [[0, 0], [100, 0], [100, 40], [0, 40]]},
                {"roi_idx": 2, "vertices": [[120, 0], [220, 0], [220, 40], [120, 40]]},
            ]
        }
        corrected_payload = {
            "rois": [
                {"roi_idx": 1, "vertices": [[0, 5], [100, 5], [100, 30], [0, 30]]},
                {"roi_idx": 2, "vertices": [[120, 0], [220, 0], [220, 40], [120, 40]]},
            ]
        }
        updates = [
            {"roi_idx": 1, "update_status": "detect-corrected"},
            {"roi_idx": 2, "update_status": "detect-corrected"},
        ]

        guarded, guarded_updates = tracker.guard_exposure_shrunken_rois(
            predicted_payload,
            corrected_payload,
            updates,
            exposure_active=True,
            exposure_info={"exposure_status": "suspect", "exposure_mean_delta": 60.0},
            shrink_ratio_threshold=0.93,
            shrink_pixels_threshold=5.0,
            min_shrunk_rois=2,
        )

        guarded_by_idx = {roi["roi_idx"]: roi for roi in guarded["rois"]}
        updates_by_idx = {update["roi_idx"]: update for update in guarded_updates}
        self.assertEqual(guarded_by_idx[1], predicted_payload["rois"][0])
        self.assertEqual(guarded_by_idx[2], corrected_payload["rois"][1])
        self.assertEqual(updates_by_idx[1]["update_status"], "rejected-exposure-shrink")
        self.assertEqual(updates_by_idx[1]["exposure_rejected_update_status"], "detect-corrected")
        self.assertLess(updates_by_idx[1]["roi_short_span_ratio"], 0.93)

    def test_guard_exposure_shrunken_rois_uses_group_shrink_as_confirmation(self):
        predicted_payload = {
            "rois": [
                {"roi_idx": 1, "vertices": [[0, 0], [100, 0], [100, 40], [0, 40]]},
                {"roi_idx": 5, "vertices": [[120, 0], [220, 0], [220, 40], [120, 40]]},
            ]
        }
        corrected_payload = {
            "rois": [
                {"roi_idx": 1, "vertices": [[0, 4], [100, 4], [100, 30], [0, 30]]},
                {"roi_idx": 5, "vertices": [[120, 3], [220, 3], [220, 31], [120, 31]]},
            ]
        }
        updates = [
            {"roi_idx": 1, "update_status": "detect-corrected"},
            {"roi_idx": 5, "update_status": "detect-corrected"},
        ]

        guarded, guarded_updates = tracker.guard_exposure_shrunken_rois(
            predicted_payload,
            corrected_payload,
            updates,
            exposure_active=False,
            exposure_info={"exposure_status": "normal"},
            shrink_ratio_threshold=0.93,
            shrink_pixels_threshold=5.0,
            min_shrunk_rois=2,
        )

        self.assertEqual(guarded["rois"], predicted_payload["rois"])
        self.assertEqual([update["update_status"] for update in guarded_updates], ["rejected-exposure-shrink", "rejected-exposure-shrink"])

    def test_correct_rois_from_detection_keeps_predicted_roi_when_detection_failed(self):
        predicted_payload = {
            "rois": [
                {
                    "roi_idx": 2,
                    "vertices": [[10, 10], [40, 10], [40, 50], [10, 50]],
                }
            ]
        }
        results = [
            {
                "roi_idx": 2,
                "status": "failed",
                "left_boundary": [],
                "right_boundary": [],
                "centerline": [],
            }
        ]

        corrected_payload, updates = tracker.correct_rois_from_detections(
            predicted_payload,
            results,
            padding_frac=0.35,
            length_padding_frac=0.10,
            max_roi_jump=200.0,
        )

        self.assertEqual(corrected_payload["rois"][0], predicted_payload["rois"][0])
        self.assertEqual(updates[0]["update_status"], "detection-failed")

    def test_visibility_requires_two_consistent_walls_after_loss(self):
        roi = {"roi_idx": 7, "vertices": [[1, 1], [99, 1], [99, 99], [1, 99]]}
        payload = {"rois": [roi]}
        history = {}

        def measure(x=40, width=20, **changes):
            result = {**roi, "status": "ok", "confidence": "high", "wall_pair_coverage": .8,
                      "tube_width_px": width, "centerline": [[x, 10], [x, 90]],
                      "left_boundary": [[x-width/2, 10], [x-width/2, 90]],
                      "right_boundary": [[x+width/2, 10], [x+width/2, 90]], **changes}
            return tracker.assess_visibility([result], history, payload, payload, 7)[0]

        self.assertEqual(measure()["status"], "ok")
        for changes in ({"wall_pair_coverage": .05}, {}, {"confidence": "low"}, {},
                        {"x": 60}, {"x": 60, "width": 10}):
            result = measure(**changes)
            self.assertEqual(result["status"], "lost")
            for key, _ in tracker.dmt.TUBE_LINES:
                self.assertEqual(result[key], [])
        recovered = measure(x=61, width=10)
        self.assertEqual(recovered["centerline"], [[61, 10], [61, 90]])
        output = [recovered]
        tracker.reject_invalid_updates(output, [{"roi_idx": 7, "update_status": "rejected-anchor-drift"}], history)
        self.assertEqual(output[0]["status"], "lost")
        self.assertEqual(output[0]["failure_reason"], "rejected-anchor-drift")
        self.assertEqual(measure()["status"], "lost")

    def test_occluded_tube_shows_no_lines_until_two_visible_frames(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            roi = {"roi_idx": 4, "vertices": [[30, 10], [90, 10], [90, 150], [30, 150]]}
            rois = root / "rois.json"
            rois.write_text(json.dumps({"rois": [roi]}))
            images = []
            for i, visible in enumerate([True, False, True, False, True, True]):
                gray = np.full((160, 120), 225, np.uint8)
                gray[:, 45:75 if visible else 120] = 50
                path = root / f"{i}.png"
                tracker.dmt.write_image_unicode(path, gray)
                images.append(str(path))
            args = tracker.parse_args(["--images", *images, "--initial-rois", str(rois),
                                       "--roi-update-mode", "flow", "--out-dir", str(root), "--video-scale", "1", "--no-window"])
            payload = tracker.run_sequence(args)
            items = [frame["centerlines"][0] for frame in payload["frames"]]
            self.assertEqual([item["status"] for item in items], ["ok", "lost", "lost", "lost", "lost", "ok"])
            self.assertTrue(all(item["centerline"] is None for item in items[1:-1]))
            for video in [args._resolved_roi_video, args._resolved_out_video]:
                cap = cv2.VideoCapture(str(video))
                try:
                    for item in items:
                        ok, frame = cap.read()
                        self.assertTrue(ok)
                        # The middle scanline excludes the header, ROI ID and LOST badge.
                        b, g, r = frame[len(frame)//2].astype(int).T
                        for color in [(b > 145) & (g > 125) & (r < 100),
                                      (r > 145) & (g > 145) & (b < 100)]:
                            self.assertEqual(bool(color.any()), item["status"] == "ok")
                finally:
                    cap.release()

    def test_guard_long_span_collapsed_rois_rejects_skinny_history_collapse(self):
        predicted_payload = {
            "rois": [
                {"roi_idx": 7, "vertices": [[0, 0], [100, 0], [100, 80], [0, 80]]},
            ]
        }
        corrected_payload = {
            "rois": [
                {"roi_idx": 7, "vertices": [[0, 0], [50, 0], [50, 30], [0, 30]]},
            ]
        }
        updates = [{"roi_idx": 7, "update_status": "detect-corrected"}]
        history = {7: {"long_span": 120.0, "short_span": 80.0}}

        guarded_payload, guarded_updates = tracker.guard_long_span_collapsed_rois(
            predicted_payload,
            corrected_payload,
            updates,
            history,
            long_span_ratio_threshold=0.60,
            long_span_drop_threshold=40.0,
            short_span_ratio_threshold=0.85,
        )

        self.assertEqual(guarded_payload["rois"][0], predicted_payload["rois"][0])
        self.assertEqual(guarded_updates[0]["update_status"], "rejected-long-span-history-shrink")
        self.assertLess(guarded_updates[0]["roi_long_span_ratio"], 0.60)

    def test_parse_args_accepts_sequence_inputs(self):
        args = tracker.parse_args(
            [
                "--images",
                "pic/1.bmp",
                "pic/2.bmp",
                "--initial-rois",
                "output/1_draw/drawn_rois.json",
                "--out-json",
                "output/sequence_centerlines.json",
            ]
        )

        self.assertEqual(args.images, ["pic/1.bmp", "pic/2.bmp"])
        self.assertEqual(args.initial_rois, "output/1_draw/drawn_rois.json")
        self.assertEqual(args.out_json, "output/sequence_centerlines.json")
        self.assertIsNone(args.out_dir)
        self.assertEqual(args.roi_update_mode, "detect-corrected")
        self.assertEqual(args.roi_padding_frac, 0.35)
        self.assertEqual(args.roi_length_padding_frac, 0.10)
        self.assertEqual(args.max_roi_jump, 200.0)
        self.assertEqual(args.min_roi_update_confidence, "medium")
        self.assertEqual(args.min_corrected_roi_short_span, 40.0)
        self.assertEqual(args.max_corrected_roi_growth, 1.0)
        self.assertEqual(args.min_corrected_roi_long_span_ratio, 0.78)
        self.assertEqual(args.min_corrected_roi_long_span_drop, 45.0)
        self.assertEqual(args.history_long_span_ratio, 0.60)
        self.assertEqual(args.history_long_span_drop, 80.0)
        self.assertEqual(args.history_short_span_ratio, 0.85)
        self.assertEqual(args.max_anchor_drift, 0.0)
        self.assertEqual(args.exposure_guard, "shrink")
        self.assertEqual(args.exposure_mean_jump, 25.0)
        self.assertEqual(args.exposure_saturation_jump, 0.15)
        self.assertEqual(args.exposure_bright_mean, 175.0)
        self.assertEqual(args.exposure_saturation_frac, 0.48)
        self.assertEqual(args.exposure_hold_frames, 8)
        self.assertEqual(args.exposure_shrink_ratio, 0.93)
        self.assertEqual(args.exposure_shrink_pixels, 5.0)
        self.assertEqual(args.exposure_min_shrunk_rois, 2)
        self.assertEqual(args.lk_window_size, 31)
        self.assertEqual(args.lk_max_level, 4)
        self.assertEqual(args.track_group_outlier_min_dy, 0.0)
        self.assertEqual(args.track_group_outlier_dy_deviation, 45.0)
        self.assertTrue(args.centerline_foreground_correction)
        self.assertEqual(args.centerline_lateral_jump, 7.0)
        self.assertEqual(args.progress_every, 0)

    def test_parse_args_accepts_flow_roi_update_mode_for_compatibility(self):
        args = tracker.parse_args(
            [
                "--images",
                "pic/1.bmp",
                "pic/2.bmp",
                "--initial-rois",
                "output/1_draw/drawn_rois.json",
                "--roi-update-mode",
                "flow",
            ]
        )

        self.assertEqual(args.roi_update_mode, "flow")

    def test_parse_args_accepts_external_image_dir_and_current_output_dir(self):
        args = tracker.parse_args(
            [
                "--image-dir",
                "D:/external/camera_left",
                "--pattern",
                "*.bmp",
                "--initial-rois",
                "output/1_draw/drawn_rois.json",
                "--out-dir",
                "tracking_results",
            ]
        )

        self.assertEqual(args.image_dir, "D:/external/camera_left")
        self.assertEqual(args.pattern, "*.bmp")
        self.assertEqual(args.out_dir, "tracking_results")
        self.assertIsNone(args.out_json)
        self.assertEqual(args.max_frames, 0)
        self.assertTrue(args.window)
        self.assertTrue(args.video)
        self.assertEqual(args.review_mode, "step")
        self.assertEqual(args.track_scale, 0.5)

    def test_parse_args_accepts_max_frames_for_probe_runs(self):
        args = tracker.parse_args(
            [
                "--image-dir",
                "trackingbmp",
                "--initial-rois",
                "tracking_results/drawn_rois.json",
                "--max-frames",
                "520",
            ]
        )

        self.assertEqual(args.max_frames, 520)

    def test_parse_args_accepts_jupyter_friendly_mode(self):
        args = tracker.parse_args(
            [
                "--image-dir",
                "trackingbmp",
                "--initial-rois",
                "tracking_results/drawn_rois.json",
                "--jupyter-friendly",
            ]
        )

        self.assertFalse(args.window)
        self.assertFalse(args.video)
        self.assertEqual(args.review_mode, "continuous")
        self.assertEqual(args.pause_on, "none")
        self.assertEqual(args.progress_every, 1)
        self.assertEqual(args.track_scale, 0.25)

    def test_parse_args_accepts_drawing_rois_without_initial_json(self):
        args = tracker.parse_args(
            [
                "--image-dir",
                "D:/external/camera_left",
                "--pattern",
                "*.bmp",
                "--draw-rois",
                "--out-dir",
                "tracking_results",
            ]
        )

        self.assertTrue(args.draw_rois)
        self.assertIsNone(args.initial_rois)

    def test_parse_args_rejects_both_draw_rois_and_initial_json(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tracker.parse_args(
                    [
                        "--images",
                        "pic/1.bmp",
                        "pic/2.bmp",
                        "--draw-rois",
                        "--initial-rois",
                        "output/1_draw/drawn_rois.json",
                    ]
                )

    def test_parse_args_rejects_single_image(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tracker.parse_args(
                    [
                        "--images",
                        "pic/1.bmp",
                        "--initial-rois",
                        "output/1_draw/drawn_rois.json",
                        "--out-json",
                        "output/sequence_centerlines.json",
                    ]
                )

    def test_parse_args_rejects_missing_image_source(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                tracker.parse_args(
                    [
                        "--initial-rois",
                        "output/1_draw/drawn_rois.json",
                        "--out-dir",
                        "tracking_results",
                    ]
                )


class ImageSourceTest(unittest.TestCase):
    def test_collect_image_paths_sorts_by_leading_timestamp_without_changing_files(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            paths = [
                root / "200_2025-left.bmp",
                root / "100_2025-left.bmp",
                root / "150_2025-left.bmp",
            ]
            for path in paths:
                path.write_bytes(path.name.encode("ascii"))

            collected = tracker.collect_image_paths(root, "*.bmp", sort_by="timestamp")

            self.assertEqual([path.name for path in collected], ["100_2025-left.bmp", "150_2025-left.bmp", "200_2025-left.bmp"])
            for path in paths:
                self.assertEqual(path.read_bytes(), path.name.encode("ascii"))

    def test_resolve_output_json_adds_timestamp_under_current_output_dir(self):
        output = tracker.resolve_output_json(
            out_json=None,
            out_dir="tracking_results",
            run_timestamp="20260608_191400",
        )

        self.assertEqual(output, Path.cwd() / "tracking_results" / "sequence_centerlines_20260608_191400.json")

    def test_resolve_output_json_uses_tracking_results_by_default(self):
        output = tracker.resolve_output_json(
            out_json=None,
            out_dir=None,
            run_timestamp="20260608_191401",
        )

        self.assertEqual(output, Path.cwd() / "tracking_results" / "sequence_centerlines_20260608_191401.json")

    def test_resolve_output_json_respects_explicit_relative_path_without_out_dir(self):
        output = tracker.resolve_output_json(out_json="output/sequence.json", out_dir=None)

        self.assertEqual(output, Path.cwd() / "output" / "sequence.json")

    def test_resolve_output_json_avoids_existing_default_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            existing = root / "sequence_centerlines_20260608_191402.json"
            existing.write_text("old json", encoding="utf-8")

            output = tracker.resolve_output_json(
                out_json=None,
                out_dir=root,
                run_timestamp="20260608_191402",
            )

            self.assertEqual(output, root / "sequence_centerlines_20260608_191402_002.json")

    def test_resolve_output_video_adds_timestamp_under_current_output_dir(self):
        output = tracker.resolve_output_video(
            out_video=None,
            out_dir="tracking_results",
            run_timestamp="20260608_191500",
        )

        self.assertEqual(output, Path.cwd() / "tracking_results" / "tracking_review_20260608_191500.mp4")

    def test_resolve_output_video_uses_tracking_results_by_default(self):
        output = tracker.resolve_output_video(
            out_video=None,
            out_dir=None,
            run_timestamp="20260608_191501",
        )

        self.assertEqual(output, Path.cwd() / "tracking_results" / "tracking_review_20260608_191501.mp4")

    def test_resolve_output_video_keeps_explicit_relative_name_under_output_dir(self):
        output = tracker.resolve_output_video(
            out_video="review_fixed.mp4",
            out_dir="tracking_results",
            run_timestamp="20260608_191502",
        )

        self.assertEqual(output, Path.cwd() / "tracking_results" / "review_fixed.mp4")

    def test_resolve_output_video_avoids_existing_default_video_name(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            existing = root / "tracking_review_20260608_191503.mp4"
            existing.write_bytes(b"old video")

            output = tracker.resolve_output_video(
                out_video=None,
                out_dir=root,
                run_timestamp="20260608_191503",
            )

            self.assertEqual(output, root / "tracking_review_20260608_191503_002.mp4")

    def test_load_initial_roi_payload_draws_and_saves_rois_without_changing_image(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            image_path = root / "100_2025-left.bmp"
            ok, encoded = cv2.imencode(".bmp", np.zeros((20, 30, 3), dtype=np.uint8))
            self.assertTrue(ok)
            encoded.tofile(str(image_path))
            before = image_path.read_bytes()
            out_dir = root / "tracking_results"
            args = SimpleNamespace(draw_rois=True, initial_rois=None)
            drawn_payload = {
                "rois": [
                    {
                        "roi_idx": 1,
                        "vertices": [[1, 1], [10, 1], [10, 10], [1, 10]],
                    }
                ]
            }

            with mock.patch.object(tracker.dmt, "draw_rois_interactive", return_value=drawn_payload):
                roi_payload, rois_source = tracker.load_initial_roi_payload(args, image_path, out_dir)

            self.assertEqual(rois_source, out_dir / "drawn_rois.json")
            self.assertTrue(rois_source.exists())
            self.assertEqual(roi_payload["num_rois"], 1)
            self.assertEqual(image_path.read_bytes(), before)


class SequenceRoiUpdateTest(unittest.TestCase):
    def test_run_sequence_uses_corrected_initial_roi_for_next_frame_by_default(self):
        initial_payload = {
            "rois": [
                {
                    "roi_idx": 1,
                    "vertices": [[0, 0], [100, 0], [100, 100], [0, 100]],
                }
            ]
        }
        detected_results = [
            {
                "roi_idx": 1,
                "status": "ok",
                "confidence": "high",
                "tube_width_px": 20,
                "wall_pair_coverage": .9,
                "vertices": [[0, 0], [100, 0], [100, 100], [0, 100]],
                "left_boundary": [[40, 20], [40, 80]],
                "right_boundary": [[60, 20], [60, 80]],
                "centerline": [[50, 20], [50, 80]],
            }
        ]
        predicted_payload = {
            "rois": [
                {
                    "roi_idx": 1,
                    "vertices": [[10, 10], [90, 10], [90, 90], [10, 90]],
                }
            ]
        }

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = SimpleNamespace(
                image_dir=None,
                images=["frame1.bmp", "frame2.bmp"],
                pattern="*.bmp",
                sort_by="timestamp",
                out_json=str(Path(tmp_dir) / "sequence.json"),
                out_video=None,
                out_dir=tmp_dir,
                draw_rois=False,
                initial_rois="rois.json",
                min_width_frac=0.28,
                max_width_frac=0.92,
                boundary_samples=60,
                window=False,
                video=False,
                fps=10.0,
                video_scale=0.5,
                window_width=1400,
                window_height=900,
                review_mode="step",
                wait_ms=30,
                pause_on="failed",
                max_corners=80,
                min_track_points=3,
                max_forward_backward_error=1.5,
                track_scale=1.0,
                roi_update_mode="detect-corrected",
                roi_padding_frac=0.25,
                roi_length_padding_frac=0.10,
                max_roi_jump=200.0,
            )

            with (
                mock.patch.object(tracker, "load_initial_roi_payload", return_value=(initial_payload, Path("rois.json"))),
                mock.patch.object(tracker.dmt, "read_image_unicode", return_value=np.zeros((100, 100), dtype=np.uint8)),
                mock.patch.object(tracker.dmt, "detect_all", return_value=(detected_results, [])),
                mock.patch.object(tracker, "track_rois_between_frames", return_value=(predicted_payload, [])) as track_mock,
            ):
                tracker.run_sequence(args)

        tracked_input_payload = track_mock.call_args.args[2]
        self.assertEqual(track_mock.call_args.kwargs["max_roi_jump"], 200.0)
        tracked_vertices = np.asarray(tracked_input_payload["rois"][0]["vertices"], dtype=np.float32)
        self.assertTrue(np.allclose(tracked_vertices[:, 0].min(), 37.5, atol=0.1))
        self.assertTrue(np.allclose(tracked_vertices[:, 0].max(), 62.5, atol=0.1))
        self.assertTrue(np.allclose(tracked_vertices[:, 1].min(), 14.0, atol=0.1))
        self.assertTrue(np.allclose(tracked_vertices[:, 1].max(), 86.0, atol=0.1))


class ReviewOverlayTest(unittest.TestCase):
    def test_make_review_frame_draws_overlay_at_video_scale(self):
        gray = np.zeros((80, 120), dtype=np.uint8)
        results = [
            {
                "roi_idx": 1,
                "status": "ok",
                "confidence": "high",
                "vertices": [[10, 10], [80, 10], [80, 40], [10, 40]],
                "left_boundary": [[15, 12], [15, 38]],
                "right_boundary": [[75, 12], [75, 38]],
                "centerline": [[45, 12], [45, 38]],
            }
        ]

        frame = tracker.make_review_frame(
            gray,
            results,
            image_path="100_2025-left.bmp",
            frame_index=0,
            total_frames=2,
            tracking=[],
            video_scale=0.5,
        )

        self.assertEqual(frame.shape, (40, 60, 3))
        self.assertGreater(int(frame.sum()), 0)


if __name__ == "__main__":
    unittest.main()
