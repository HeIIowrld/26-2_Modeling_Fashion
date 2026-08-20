import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from garment_attribute_analyzer import GarmentAttributeAnalyzer
from schemas import PoseAnalysis


def sample_pose() -> PoseAnalysis:
    landmarks = {
        "left_shoulder": (0.35, 0.20, 1.0),
        "right_shoulder": (0.65, 0.20, 1.0),
        "left_elbow": (0.27, 0.38, 1.0),
        "right_elbow": (0.73, 0.38, 1.0),
        "left_wrist": (0.25, 0.55, 1.0),
        "right_wrist": (0.75, 0.55, 1.0),
        "left_hip": (0.42, 0.50, 1.0),
        "right_hip": (0.58, 0.50, 1.0),
        "left_knee": (0.42, 0.70, 1.0),
        "right_knee": (0.58, 0.70, 1.0),
        "left_ankle": (0.42, 0.92, 1.0),
        "right_ankle": (0.58, 0.92, 1.0),
    }
    return PoseAnalysis(True, 1.0, "균형형", 1.0, 0.5, 0.5, "정면", landmarks=landmarks)


class GarmentAttributeTests(unittest.TestCase):
    def test_sleeve_thresholds(self):
        label = GarmentAttributeAnalyzer._sleeve_label
        self.assertEqual(label(0.1), "민소매")
        self.assertEqual(label(0.4), "반팔")
        self.assertEqual(label(0.7), "7부 소매")
        self.assertEqual(label(0.9), "긴팔")

    def test_long_pants_from_segmentation_endpoint(self):
        segmentation = np.zeros((100, 60), dtype=np.uint8)
        segmentation[20:51, 18:43] = 3  # top
        segmentation[50:93, 24:37] = 6  # pants
        result = GarmentAttributeAnalyzer().analyze(segmentation, sample_pose())
        self.assertEqual(result["upper_type"], "상의")
        self.assertEqual(result["lower_type"], "바지")
        self.assertEqual(result["bottom_length"], "긴바지")


if __name__ == "__main__":
    unittest.main()
