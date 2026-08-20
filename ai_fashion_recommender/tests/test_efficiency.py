import sys
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_model import FashionClassifier
from quality_checker import QualityChecker
from schemas import PoseAnalysis


class CountingPoseAnalyzer:
    def __init__(self):
        self.calls = 0

    def analyze(self, _):
        self.calls += 1
        raise AssertionError("전달된 포즈가 있으면 analyze()를 다시 호출하면 안 됩니다.")


class CountingFashionEncoder:
    def __init__(self):
        self.image_calls = 0

    def encode_image(self, batch, normalize=True):
        self.image_calls += 1
        value = torch.ones((len(batch), 8), device=batch.device)
        return torch.nn.functional.normalize(value, dim=-1) if normalize else value

    def encode_text(self, tokens, normalize=True):
        value = torch.arange(1, len(tokens) * 8 + 1, device=tokens.device, dtype=torch.float32).reshape(len(tokens), 8)
        return torch.nn.functional.normalize(value, dim=-1) if normalize else value


class RecordingAttributePredictor:
    def __init__(self):
        self.calls = 0

    def predict_features(self, features, *, tasks=None):
        self.calls += 1
        self.features = features
        return {}


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

    def test_quality_checker_accepts_320px_short_side_but_rejects_under_it(self):
        analyzer = CountingPoseAnalyzer()
        checker = QualityChecker(analyzer)
        pose = PoseAnalysis(True, 0.95, "균형형", 1.0, 0.5, 0.5, "정면")

        def checkerboard(width, height):
            values = (np.indices((height, width)).sum(axis=0) % 2 * 255).astype(np.uint8)
            return Image.fromarray(np.repeat(values[:, :, None], 3, axis=2))

        accepted = checker.check_input(checkerboard(320, 687), pose=pose)
        rejected = checker.check_input(checkerboard(319, 687), pose=pose)
        self.assertTrue(accepted["passed"])
        self.assertFalse(rejected["passed"])
        self.assertIn("320px", rejected["issues"][-1])

    def test_quality_checker_accepts_mild_blur_under_relaxed_threshold(self):
        analyzer = CountingPoseAnalyzer()
        checker = QualityChecker(analyzer)
        pose = PoseAnalysis(True, 0.70, "균형형", 1.0, 0.5, 0.5, "정면")
        y, x = np.indices((400, 400))
        coarse_checkerboard = (((x // 16 + y // 16) % 2) * 255).astype(np.uint8)
        mildly_blurred = cv2.GaussianBlur(coarse_checkerboard, (21, 21), 0)
        image = Image.fromarray(np.repeat(mildly_blurred[:, :, None], 3, axis=2))

        result = checker.check_input(image, pose=pose)

        self.assertGreaterEqual(result["sharpness"], 25.0)
        self.assertLess(result["sharpness"], 60.0)
        self.assertTrue(result["passed"])

    def test_disabled_batch_classifier_returns_fallbacks(self):
        classifier = FashionClassifier(enabled=False)
        result = classifier.best_mapped_labels(
            Image.new("RGB", (16, 16)),
            {"pattern": {"무지": "solid"}, "material": {"코튼": "cotton"}},
        )
        self.assertEqual(result["pattern"], ("분석 보류", 0.0))
        self.assertEqual(result["material"], ("분석 보류", 0.0))

    def test_crop_analysis_shares_one_image_embedding(self):
        classifier = FashionClassifier(enabled=False)
        classifier.enabled = True
        classifier._torch = torch
        classifier.device = "cpu"
        classifier.model = CountingFashionEncoder()
        classifier.preprocess = lambda _: torch.ones(3, 4, 4)
        classifier.tokenizer = lambda prompts: torch.ones((len(prompts), 2), dtype=torch.long)
        classifier.attribute_predictor = RecordingAttributePredictor()

        learned, zero_shot = classifier.analyze_crop(
            Image.new("RGB", (16, 16)),
            tasks=["category"],
            prompt_groups={"category": {"셔츠": "shirt", "니트": "knit"}},
        )

        self.assertEqual(learned, {})
        self.assertEqual(set(zero_shot), {"category"})
        self.assertEqual(classifier.model.image_calls, 1)
        self.assertEqual(classifier.attribute_predictor.calls, 1)


if __name__ == "__main__":
    unittest.main()
