import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fashion_attribute_model import AttributePrediction
from outfit_analyzer import _refine_upper_type
from wear_state_analyzer import infer_layering_state, infer_sleeve_state


class WearStateTests(unittest.TestCase):
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
