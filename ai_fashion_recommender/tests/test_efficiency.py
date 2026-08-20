import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fashion_model import FashionClassifier
from quality_checker import QualityChecker
from schemas import PoseAnalysis


class CountingPoseAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, _):
        self.calls += 1
        raise AssertionError("전달된 포즈가 있으면 analyze()를 다시 호출하면 안 됩니다.")


class EfficiencyTests(unittest.TestCase):
    def test_quality_checker_reuses_pose(self):
        analyzer = CountingPoseAnalyzer()
        checker = QualityChecker(analyzer)
        checkerboard = (np.indices((600, 600)).sum(axis=0) % 2 * 255).astype(np.uint8)
        image = Image.fromarray(np.repeat(checkerboard[:, :, None], 3, axis=2))
        pose = PoseAnalysis(True, 0.95, "균형형", 1.0, 0.5, 0.5, "정면")
        result = checker.check_input(image, pose=pose)
        self.assertTrue(result["passed"])
        self.assertEqual(analyzer.calls, 0)

    def test_disabled_batch_classifier_returns_fallbacks(self):
        classifier = FashionClassifier(enabled=False)
        result = classifier.best_mapped_labels(
            Image.new("RGB", (16, 16)),
            {"pattern": {"무지": "solid"}, "material": {"코튼": "cotton"}},
        )
        self.assertEqual(result["pattern"], ("분석 보류", 0.0))
        self.assertEqual(result["material"], ("분석 보류", 0.0))


if __name__ == "__main__":
    unittest.main()
