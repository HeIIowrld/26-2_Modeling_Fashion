import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from garment_attribute_analyzer import GarmentAttributeAnalyzer
from clothing_parser import ClothingParser
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
    return PoseAnalysis(True, 1.0, "사각체형", 1.0, 0.5, 0.5, "정면", landmarks=landmarks)


class GarmentAttributeTests(unittest.TestCase):
    def test_outfit_does_not_replace_unobservable_length_with_confident_head(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from PIL import Image
        from fashion_attribute_model import AttributePrediction
        from outfit_analyzer import OutfitAnalyzer

        seg = np.zeros((100, 60), dtype=np.uint8)
        seg[20:50, 18:43] = 3
        seg[50:75, 24:37] = 6
        parsed = {"segmentation": seg, "upper_mask": seg == 3, "lower_mask": seg == 6,
                  "present_labels": ["top", "pants"]}
        parser = SimpleNamespace(backend="test", parse=lambda *args: parsed)

        class Classifier:
            enabled = True
            trained_attributes_enabled = True
            short = False

            def best_mapped_label(self, *args):
                return "캐주얼", 0.9

            def predict_layering(self, *args):
                return None

            def analyze_crop(self, image, *, tasks, prompt_groups, **kwargs):
                learned = {"pant_length": AttributePrediction(["풀렝스"], {"풀렝스": 0.99}, 0.99, True)} if "pant_length" in tasks else {}
                if self.short and "pant_length" in tasks:
                    learned["category"] = AttributePrediction(["쇼츠"], {"쇼츠": 0.99}, 0.99, True)
                    learned["lower_length"] = AttributePrediction(["쇼츠·미니 기장"], {"쇼츠·미니 기장": 0.99}, 0.99, True)
                return learned, {key: ("분석 보류", 0.0) for key in prompt_groups}

        pose = sample_pose()
        pose.landmarks["left_ankle"] = (0.42, 1.1, 0.03)
        with patch("outfit_analyzer._layering_zero_shot", return_value=("단일 상의", 0.9)):
            outfit, _ = OutfitAnalyzer(parser, Classifier()).analyze(Image.new("RGB", (60, 100)), pose)
        self.assertEqual(outfit.bottom_length, "분석 불가")
        self.assertEqual(outfit.pant_length, "분석 보류")
        self.assertEqual(outfit.attribute_sources["bottom_length"], "pose_unavailable")

        # 같은 발목 가림이라도 실제 밑단이 무릎 부근에 보이면 보류하지 않는다.
        seg[70:] = 0
        parsed["lower_mask"] = seg == 6
        with patch("outfit_analyzer._layering_zero_shot", return_value=("단일 상의", 0.9)):
            outfit, _ = OutfitAnalyzer(parser, Classifier()).analyze(Image.new("RGB", (60, 100)), pose)
        self.assertEqual(outfit.bottom_length, "무릎 기장 바지")
        self.assertEqual(outfit.attribute_sources["bottom_length"], "mask_knee")
        self.assertNotIn("풀렝스", outfit.to_summary_dict()["하의"])

        # 쇼츠에는 7부/앵클/풀렝스 전용 헤드를 적용하지 않는다.
        Classifier.short = True
        seg[63:] = 0
        parsed["lower_mask"] = seg == 6
        with patch("outfit_analyzer._layering_zero_shot", return_value=("단일 상의", 0.9)):
            outfit, _ = OutfitAnalyzer(parser, Classifier()).analyze(Image.new("RGB", (60, 100)), sample_pose())
        self.assertEqual(outfit.bottom_length, "반바지")
        self.assertEqual(outfit.pant_length, "해당 없음")
        self.assertNotIn("풀렝스", outfit.to_summary_dict()["하의"])

    def test_bottom_length_abstains_for_unobserved_or_invalid_ankles(self):
        mask = np.zeros((100, 60), dtype=np.uint8)
        mask[50:75, 24:37] = 1
        for point in ((0.42, 0.92, 0.03), (0.42, 1.1, 1.0),
                      (float("nan"), 0.92, 1.0), (0.42, 0.92, 0.49)):
            with self.subTest(point=point):
                pose = sample_pose()
                pose.landmarks["left_ankle"] = point
                self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, pose, "바지"), "분석 불가")
        pose = sample_pose()
        del pose.landmarks["left_ankle"]
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, pose, "바지"), "분석 불가")

    def test_visible_shorts_remain_short(self):
        mask = np.zeros((100, 60), dtype=np.uint8)
        mask[50:63, 24:37] = 1
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, sample_pose(), "바지"), "반바지")

    def test_visible_hem_uses_knees_when_ankles_are_out_of_frame(self):
        pose = sample_pose()
        for side in ("left", "right"):
            pose.landmarks[f"{side}_ankle"] = (0.5, 1.2, 0.03)
        for end, kind, expected in ((63, "바지", "반바지"), (63, "치마", "미니 기장"),
                                    (70, "치마", "무릎 기장"), (70, "바지", "무릎 기장 바지"),
                                    (90, "바지", "분석 불가"), (100, "바지", "분석 불가")):
            with self.subTest(end=end, kind=kind):
                mask = np.zeros((100, 60), dtype=np.uint8)
                mask[50:end, 24:37] = 1
                self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, pose, kind), expected)

    def test_hem_touching_frame_or_invisible_knees_still_abstains(self):
        pose = sample_pose()
        for side in ("left", "right"):
            pose.landmarks[f"{side}_ankle"] = (0.5, 1.2, 0.03)
            pose.landmarks[f"{side}_knee"] = (0.5, 0.98, 1.0)
        mask = np.zeros((100, 60), dtype=np.uint8)
        mask[50:, 24:37] = 1
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, pose, "바지"), "분석 불가")
        mask[70:] = 0
        pose.landmarks["left_knee"] = (0.5, 0.7, 0.03)
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, pose, "바지"), "분석 불가")

    def test_shoe_label_fragment_does_not_turn_shorts_into_full_length(self):
        mask = np.zeros((200, 120), dtype=np.uint8)
        mask[100:124, 40:80] = 1
        mask[179:185, 50:55] = 1  # 약 3% 면적의 신발 오라벨
        original = mask.copy()
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, sample_pose(), "바지"), "반바지")
        np.testing.assert_array_equal(mask, original)

    def test_disconnected_long_pant_legs_are_both_retained(self):
        mask = np.zeros((200, 120), dtype=np.uint8)
        mask[100:150, 40:55] = 1
        mask[100:186, 65:70] = 1  # 더 얇고 긴 다른 다리
        self.assertEqual(GarmentAttributeAnalyzer()._bottom_length(mask, sample_pose(), "바지"), "긴바지")

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
        self.assertIn("visible_sleeve_length", result["measurements"])
        self.assertEqual(
            set(result["measurements"]["sleeve_side_coverage"]),
            {"left", "right"},
        )

    def test_accessories_do_not_pollute_main_garment_masks(self):
        segmentation = np.zeros((10, 10), dtype=np.uint8)
        segmentation[1:5, 2:8] = 3
        segmentation[5:9, 2:8] = 6
        segmentation[2, 1] = 10  # scarf
        segmentation[5, 1] = 7   # belt

        class FakeParser:
            @staticmethod
            def predict(_):
                return segmentation

        parser = ClothingParser(use_fashn=False)
        parser._parser = FakeParser()
        parser.backend = "test-parser"
        parsed = parser.parse(np.zeros((10, 10, 3), dtype=np.uint8), sample_pose())
        self.assertFalse(parsed["upper_mask"][2, 1])
        self.assertFalse(parsed["lower_mask"][5, 1])
        self.assertTrue(parsed["accessory_mask"][2, 1])
        self.assertTrue(parsed["accessory_mask"][5, 1])


if __name__ == "__main__":
    unittest.main()
