"""Synthetic-image and offscreen UI checks; no camera is required."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import time
import unittest
from unittest.mock import patch

from PySide6.QtGui import QImage
from PySide6.QtMultimedia import QCamera, QVideoFrame

from sift_image_features_demo import (
    QApplication, FeatureWorker, SiftDemo, SiftSettings,
    create_sift, cv2, match_descriptors, np,
)


def textured_image():
    rng = np.random.default_rng(14)
    image = np.full((360, 480, 3), 230, np.uint8)
    for _ in range(140):
        center = tuple(int(value) for value in rng.integers([15, 15], [465, 345]))
        color = tuple(int(value) for value in rng.integers(0, 210, 3))
        cv2.circle(image, center, int(rng.integers(3, 14)), color, -1)
    cv2.putText(image, "SIFT DEMO", (45, 190), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 0, 0), 3)
    return image


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cv2.setNumThreads(2)
        cls.rgb = textured_image()

    def test_rotation_scale_matching(self):
        detector = create_sift(SiftSettings())
        transform = cv2.getRotationMatrix2D((240, 180), 18, 0.88)
        current = cv2.warpAffine(self.rgb, transform, (480, 360))
        left, desc_left = detector.detectAndCompute(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY), None)
        right, desc_right = detector.detectAndCompute(cv2.cvtColor(current, cv2.COLOR_RGB2GRAY), None)
        self.assertEqual(desc_left.shape[1], 128)
        matches = match_descriptors(desc_left, desc_right, 0.75)
        self.assertGreater(len(matches), 25)
        errors = []
        for match in matches:
            expected = transform @ np.array([*left[match.queryIdx].pt, 1.0])
            errors.append(np.linalg.norm(expected - right[match.trainIdx].pt))
        self.assertLess(np.median(errors), 3.0)
        self.assertEqual(len({match.trainIdx for match in matches}), len(matches))

    def test_feature_limit_and_blank(self):
        detector = create_sift(SiftSettings(features=100))
        points, descriptors = detector.detectAndCompute(cv2.cvtColor(self.rgb, cv2.COLOR_RGB2GRAY), None)
        self.assertGreater(len(points), 0)
        self.assertEqual(descriptors.shape[1], 128)
        self.assertLess(len(points), 150)
        self.assertTrue(all(0 <= point.angle < 360 for point in points))
        points, descriptors = detector.detectAndCompute(np.zeros((240, 320), np.uint8), None)
        self.assertEqual(len(points), 0)
        self.assertIsNone(descriptors)
        self.assertEqual(match_descriptors(None, None, 0.75), [])
        self.assertEqual(match_descriptors(np.ones((2, 128), np.float32),
                                           np.ones((1, 128), np.float32), 0.75), [])
        ambiguous = np.zeros((2, 128), np.float32)
        self.assertEqual(match_descriptors(ambiguous, ambiguous, 0.75), [])

    def test_reference_cache_and_settings(self):
        worker = FeatureWorker()
        results = []
        worker.finished.connect(results.append)
        job = dict(rgb=self.rgb, reference=self.rgb.copy(), reference_id=1,
                   settings=SiftSettings(), ratio=0.75, revision=1)
        worker.process(job)
        self.assertNotIn("error", results[-1])
        original = worker.reference_descriptors
        worker.process(dict(job, ratio=0.6))
        self.assertIs(worker.reference_descriptors, original)
        worker.process(dict(job, settings=SiftSettings(contrast=0.1)))
        self.assertNotIn("error", results[-1])
        self.assertIsNot(worker.reference_descriptors, original)
        self.assertLess(len(worker.reference_descriptors), len(original))
        worker.process(dict(job, reference=np.zeros_like(self.rgb), reference_id=2))
        self.assertIsNone(worker.reference_descriptors)
        self.assertEqual(results[-1]["matches"], [])


class WindowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def wait_for_result(self, window):
        deadline = time.monotonic() + 10
        while window.busy and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.01)
        self.assertFalse(window.busy, "Worker did not finish")

    def test_capture_settings_clear_and_paint(self):
        window = SiftDemo(discover_cameras=False)
        try:
            self.assertTrue(window.sift_available)
            window.show()
            window.latest_rgb = textured_image()
            window.frame_id += 1
            window.submit_frame()
            self.wait_for_result(window)
            window.capture_reference()
            frozen = window.reference_rgb.copy()
            window.latest_rgb = cv2.warpAffine(frozen, np.float32([[1, 0, 10], [0, 1, 5]]), (480, 360))
            window.frame_id += 1
            window.submit_frame()
            self.wait_for_result(window)
            self.assertGreater(len(window.view.matches), 20)
            window.contrast.setValue(0.08)
            window.submit_frame()
            self.wait_for_result(window)
            np.testing.assert_array_equal(window.reference_rgb, frozen)
            self.assertIn("128D", window.statistics.text())
            window.resize(1100, 650)
            self.app.processEvents()
            self.assertFalse(window.grab().isNull())
            self.assertTrue(window.view.reference.isGrayscale())
            self.assertTrue(window.view.current.isGrayscale())
            window.max_lines.setValue(0)
            self.assertFalse(window.grab().isNull())
            window.clear_reference()
            window.submit_frame()
            self.wait_for_result(window)
            self.assertTrue(window.view.reference.isNull())
            self.assertEqual(window.view.matches, [])
        finally:
            window.close()
        self.assertFalse(window.worker_thread.isRunning())

    def test_camera_reset_discards_in_flight_result(self):
        window = SiftDemo(discover_cameras=False)
        try:
            window.latest_rgb = textured_image()
            window.submit_frame()
            self.assertTrue(window.busy)
            window.stop_camera()
            self.wait_for_result(window)
            self.assertTrue(window.view.current.isNull())
            self.assertIsNone(window.displayed_rgb)
            self.assertFalse(window.capture.isEnabled())
        finally:
            window.close()

    def test_video_frame_padding_and_capture_owns_pixels(self):
        window = SiftDemo(discover_cameras=False)
        try:
            window.camera = QCamera(window)
            image = QImage(161, 120, QImage.Format.Format_RGB888)
            image.fill(0x123456)
            window.receive_frame(QVideoFrame(image))
            self.assertEqual(window.latest_rgb.shape, (120, 161, 3))
            np.testing.assert_array_equal(window.latest_rgb[0, 0], [0x12, 0x34, 0x56])
            self.assertTrue(window.capture.isEnabled())
            window.capture_reference()
            window.latest_rgb[:] = 0
            np.testing.assert_array_equal(window.reference_rgb[-1, -1], [0x12, 0x34, 0x56])
            large = QImage(1600, 1200, QImage.Format.Format_RGB888)
            large.fill(0)
            window.receive_frame(QVideoFrame(large))
            self.assertEqual(window.latest_rgb.shape, (600, 800, 3))
        finally:
            window.close()

    def test_missing_sift_explains_setup(self):
        with patch("sift_image_features_demo.create_sift", side_effect=RuntimeError("SIFT unavailable: install opencv-python")):
            window = SiftDemo(discover_cameras=False)
        try:
            self.assertFalse(window.sift_available)
            self.assertFalse(window.feature_group.isEnabled())
            self.assertIn("opencv-python", window.processing_status.text())
            window.latest_rgb = textured_image()
            window.submit_frame()
            self.assertFalse(window.busy)
        finally:
            window.close()


if __name__ == "__main__":
    unittest.main()
