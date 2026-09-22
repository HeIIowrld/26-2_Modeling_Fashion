from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from quality_checker import (  # noqa: E402
    QualityChecker,
    assess_arm_pose,
    assess_front_pose,
    assess_head_framing,
)


def landmarks() -> dict[str, tuple[float, float, float]]:
    return {
        "nose": (0.5, 0.15, 0.95),
        "left_shoulder": (0.35, 0.25, 0.95),
        "right_shoulder": (0.65, 0.25, 0.95),
        "left_elbow": (0.35, 0.45, 0.95),
        "right_elbow": (0.65, 0.45, 0.95),
        "left_wrist": (0.35, 0.65, 0.95),
        "right_wrist": (0.65, 0.65, 0.95),
        "left_hip": (0.42, 0.55, 0.95),
        "right_hip": (0.58, 0.55, 0.95),
        "left_knee": (0.43, 0.75, 0.95),
        "right_knee": (0.57, 0.75, 0.95),
        "left_ankle": (0.43, 0.92, 0.95),
        "right_ankle": (0.57, 0.92, 0.95),
        "left_heel": (0.43, 0.95, 0.95),
        "right_heel": (0.57, 0.95, 0.95),
        "left_foot": (0.43, 0.98, 0.95),
        "right_foot": (0.57, 0.98, 0.95),
    }


def camera_landmarks() -> dict[str, tuple[float, float, float]]:
    """실제 MediaPipe 좌표: left_*는 사람 기준 왼쪽이라 정면 사진에서는 이미지 오른쪽에 있다.

    위 landmarks()는 좌우가 반전된 셀카와 같다. 둘 다 같은 판정이 나와야 한다.
    """
    return {name: (1.0 - x, y, v) for name, (x, y, v) in landmarks().items()}


def checked(landmark_map: dict | None, *, valid: bool = True) -> dict:
    pose = SimpleNamespace(
        valid=valid,
        warnings=[],
        full_body_score=0.95,
        landmarks=landmark_map,
    )
    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (600, 400, 3), dtype=np.uint8)
    )
    return QualityChecker(None).check_input(image, pose=pose)


def checked_pose(pose) -> dict:
    image = Image.fromarray(
        np.random.default_rng(7).integers(0, 256, (600, 400, 3), dtype=np.uint8)
    )
    return QualityChecker(None).check_input(image, pose=pose)


class FrontFullBodyValidationTests(unittest.TestCase):
    def test_invalid_pose_is_rejected(self):
        result = checked(landmarks(), valid=False)
        self.assertFalse(result["passed"])
        self.assertIn("정면 전신", result["issues"][0])

    def test_none_pose_is_rejected(self):
        result = checked_pose(None)
        self.assertFalse(result["passed"])

    def test_empty_landmarks_are_rejected(self):
        result = checked({})
        self.assertFalse(result["passed"])

    def test_missing_core_torso_landmark_is_rejected(self):
        current = landmarks()
        del current["left_hip"]
        result = checked(current)
        self.assertFalse(result["passed"])

    def test_all_nan_landmarks_are_rejected(self):
        current = {name: (np.nan, np.nan, np.nan) for name in landmarks()}
        result = checked(current)
        self.assertFalse(result["passed"])

    def test_normal_front_full_body_passes(self):
        result = checked(landmarks())
        self.assertTrue(result["passed"])
        self.assertEqual(result["head_framing"], "visible")
        self.assertEqual(result["lower_body_framing"], "visible")
        self.assertEqual(result["full_body_framing"], "visible")
        self.assertEqual(result["front_pose"]["status"], "front")
        self.assertEqual(result["arm_pose"]["status"], "arms_down")

    def test_clear_head_crop_is_rejected(self):
        current = landmarks()
        current["nose"] = (0.5, 0.02, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["head_framing"], "cropped")

    def test_clear_foot_crop_is_rejected(self):
        current = landmarks()
        current["left_foot"] = (0.43, 1.02, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["lower_body_framing"], "cropped")

    def test_uncertain_foot_does_not_become_crop(self):
        current = landmarks()
        current["left_foot"] = (0.43, 1.02, 0.1)
        result = checked(current)
        self.assertTrue(result["passed"])
        self.assertEqual(result["lower_body_framing"], "uncertain")

    def test_low_visibility_stays_uncertain(self):
        current = landmarks()
        current["nose"] = (0.5, 0.02, 0.2)
        current["left_shoulder"] = (*current["left_shoulder"][:2], 0.2)
        result = checked(current)
        self.assertTrue(result["passed"])
        self.assertEqual(result["head_framing"], "uncertain")
        self.assertEqual(result["front_pose"]["status"], "uncertain")

    def test_front_pose_passes(self):
        self.assertEqual(assess_front_pose(landmarks())["status"], "front")

    def test_clear_non_front_pose_is_rejected(self):
        current = landmarks()
        current["left_shoulder"] = (0.25, 0.10, 0.95)
        current["right_shoulder"] = (0.65, 0.30, 0.95)
        current["left_hip"] = (0.62, 0.70, 0.95)
        current["right_hip"] = (0.78, 0.70, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["front_pose"]["status"], "non_front")

    def test_arms_down_passes(self):
        self.assertEqual(assess_arm_pose(landmarks())["status"], "arms_down")

    def test_open_arms_are_rejected(self):
        current = landmarks()
        current["left_wrist"] = (0.05, 0.65, 0.95)
        current["right_wrist"] = (0.95, 0.65, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["arm_pose"]["status"], "arms_raised_or_open")

    def test_raised_arms_are_rejected(self):
        current = landmarks()
        current["left_elbow"] = (0.28, 0.15, 0.95)
        current["right_elbow"] = (0.72, 0.15, 0.95)
        current["left_wrist"] = (0.22, 0.04, 0.95)
        current["right_wrist"] = (0.78, 0.04, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["arm_pose"]["status"], "arms_raised_or_open")

    def test_crossed_arms_are_rejected(self):
        current = landmarks()
        current["left_wrist"] = (0.68, 0.58, 0.95)
        current["right_wrist"] = (0.32, 0.58, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["arm_pose"]["status"], "arms_crossed_or_occluded")

    def test_arms_down_passes_in_camera_orientation(self):
        # 2026-09-22 운영 배포에서 팔을 내린 정면 사진이 모두 '팔 교차'로 거절됐다.
        result = checked(camera_landmarks())
        self.assertEqual(result["arm_pose"]["status"], "arms_down")
        self.assertTrue(result["passed"], result["issues"])

    def test_measured_front_photo_passes(self):
        # 운영 서버에서 잰 DeepFashion 정면 사진(팔을 몸 옆에 내림)의 손목·어깨 x 좌표.
        current = camera_landmarks()
        current["left_shoulder"] = (0.71, 0.25, 0.95)
        current["right_shoulder"] = (0.40, 0.25, 0.95)
        current["left_hip"] = (0.62, 0.55, 0.95)
        current["right_hip"] = (0.47, 0.55, 0.95)
        current["left_elbow"] = (0.73, 0.45, 0.95)
        current["right_elbow"] = (0.37, 0.45, 0.95)
        current["left_wrist"] = (0.73, 0.65, 0.95)
        current["right_wrist"] = (0.36, 0.65, 0.95)
        self.assertEqual(assess_arm_pose(current)["status"], "arms_down")

    def test_relaxed_arms_slightly_outside_shoulders_pass(self):
        # 팔을 편하게 내리면 손목이 어깨보다 어깨 폭의 0.25~0.41만큼 바깥에 온다(실측 100장).
        current = camera_landmarks()
        width = current["left_shoulder"][0] - current["right_shoulder"][0]
        current["left_wrist"] = (current["left_shoulder"][0] + 0.35 * width, 0.65, 0.95)
        current["right_wrist"] = (current["right_shoulder"][0] - 0.35 * width, 0.65, 0.95)
        self.assertEqual(assess_arm_pose(current)["status"], "arms_down")

    def test_crossed_arms_are_rejected_in_camera_orientation(self):
        current = camera_landmarks()
        current["left_wrist"] = (0.32, 0.58, 0.95)
        current["right_wrist"] = (0.68, 0.58, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["arm_pose"]["status"], "arms_crossed_or_occluded")

    def test_open_arms_are_rejected_in_camera_orientation(self):
        current = camera_landmarks()
        current["left_wrist"] = (0.95, 0.65, 0.95)
        current["right_wrist"] = (0.05, 0.65, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertEqual(result["arm_pose"]["status"], "arms_raised_or_open")

    def test_multiple_failures_are_limited_to_three_messages(self):
        current = landmarks()
        current["nose"] = (0.5, 0.02, 0.95)
        current["left_foot"] = (0.43, 1.02, 0.95)
        current["left_wrist"] = (0.68, 0.58, 0.95)
        current["right_wrist"] = (0.32, 0.58, 0.95)
        result = checked(current)
        self.assertFalse(result["passed"])
        self.assertLessEqual(len(result["issues"]), 3)


if __name__ == "__main__":
    unittest.main()