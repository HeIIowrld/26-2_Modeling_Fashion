import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_attribute_model import AttributePrediction
from layering_model import LayeringHeadPrediction
from outfit_analyzer import _refine_upper_type
from wear_state_analyzer import infer_layering_state, infer_sleeve_state


class WearStateTests(unittest.TestCase):
    def test_trained_multi_roi_layering_result_has_priority(self):
        prediction = LayeringHeadPrediction(
            state="레이어드",
            confidence=0.91,
            accepted=True,
            inner_category="셔츠",
            outer_category="베스트",
            component_confidence=0.84,
        )
        result = infer_layering_state(
            "폴로 셔츠",
            "폴로 칼라",
            "라운드넥",
            "코튼",
            trained_prediction=prediction,
            zero_shot_label="단일 상의",
            zero_shot_confidence=0.95,
        )
        self.assertEqual(result.state, "레이어드")
        self.assertEqual(result.upper_items, ["셔츠", "베스트"])
        self.assertIn("멀티 ROI", result.reason)

    def test_long_sleeve_visible_as_three_quarter_is_marked_rolled(self):
        learned = AttributePrediction(
            labels=["긴팔"], scores={"긴팔": 0.92}, confidence=0.92, accepted=True
        )
        result = infer_sleeve_state(
            {
                "sleeve_coverage_ratio": 0.72,
                "sleeve_side_coverage": {"left": 0.70, "right": 0.74},
                "visible_sleeve_length": "7부 소매",
            },
            learned,
            "7부 소매",
            pose_landmarks={
                "left_wrist": (0.0, 0.0, 0.9),
                "right_wrist": (0.0, 0.0, 0.9),
            },
        )
        self.assertEqual(result.designed_length, "긴팔")
        self.assertEqual(result.visible_length, "7부 소매")
        self.assertEqual(result.state, "걷음 가능성 높음")
        self.assertFalse(result.requires_retake)

    def test_long_sleeve_rolled_to_elbow_requires_retake(self):
        learned = AttributePrediction(
            labels=["긴팔"], scores={"긴팔": 0.91}, confidence=0.91, accepted=True
        )
        result = infer_sleeve_state(
            {
                "sleeve_coverage_ratio": 0.55,
                "sleeve_side_coverage": {"left": 0.54, "right": 0.56},
                "visible_sleeve_length": "반팔",
            },
            learned,
            "반팔",
            pose_landmarks=self._visible_arm_landmarks(),
        )
        self.assertTrue(result.requires_retake)
        self.assertEqual(result.state, "재촬영 필요")
        self.assertEqual(result.error_code, "SLEEVE_ROLLUP_RETAKE_REQUIRED")

    def test_one_sleeve_at_elbow_and_the_other_long_requires_retake(self):
        learned = AttributePrediction(
            labels=["7부 소매"], scores={"7부 소매": 0.75}, confidence=0.75, accepted=True
        )
        result = infer_sleeve_state(
            {
                "sleeve_coverage_ratio": 0.72,
                "sleeve_side_coverage": {"left": 0.52, "right": 0.91},
                "visible_sleeve_length": "7부 소매",
            },
            learned,
            "7부 소매",
            pose_landmarks=self._visible_arm_landmarks(),
        )
        self.assertTrue(result.requires_retake)

    def test_symmetric_real_short_sleeves_are_not_rejected(self):
        learned = AttributePrediction(
            labels=["반팔"], scores={"반팔": 0.90}, confidence=0.90, accepted=True
        )
        result = infer_sleeve_state(
            {
                "sleeve_coverage_ratio": 0.40,
                "sleeve_side_coverage": {"left": 0.39, "right": 0.41},
                "visible_sleeve_length": "반팔",
            },
            learned,
            "반팔",
            pose_landmarks=self._visible_arm_landmarks(),
        )
        self.assertFalse(result.requires_retake)

    def test_low_visibility_does_not_trigger_a_false_retake(self):
        learned = AttributePrediction(
            labels=["긴팔"], scores={"긴팔": 0.92}, confidence=0.92, accepted=True
        )
        landmarks = self._visible_arm_landmarks()
        landmarks["left_elbow"] = (0.0, 0.0, 0.30)
        result = infer_sleeve_state(
            {
                "sleeve_coverage_ratio": 0.55,
                "sleeve_side_coverage": {"left": 0.54, "right": 0.56},
                "visible_sleeve_length": "반팔",
            },
            learned,
            "반팔",
            pose_landmarks=landmarks,
        )
        self.assertFalse(result.requires_retake)

    @staticmethod
    def _visible_arm_landmarks():
        return {
            name: (0.0, 0.0, 0.90)
            for name in (
                "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
                "left_wrist", "right_wrist",
            )
        }

    def test_knit_with_shirt_collar_is_layered(self):
        result = infer_layering_state(
            "니트", "셔츠 칼라", "V넥", "니트", zero_shot_label="단일 상의",
            zero_shot_confidence=0.55,
        )
        self.assertEqual(result.state, "레이어드")
        self.assertEqual(result.upper_items, ["셔츠", "니트"])
        self.assertEqual(result.inner_category, "셔츠")

    def test_polo_refinement_is_disabled_for_layering_or_knit_conflict(self):
        self.assertEqual(
            _refine_upper_type("티셔츠", "폴로 칼라", layering_state="레이어드"),
            "티셔츠",
        )
        self.assertEqual(
            _refine_upper_type(
                "티셔츠", "폴로 칼라", layering_state="단일 상의", material="니트"
            ),
            "티셔츠",
        )
        self.assertEqual(
            _refine_upper_type("티셔츠", "폴로 칼라", layering_state="단일 상의"),
            "폴로 셔츠",
        )


if __name__ == "__main__":
    unittest.main()
