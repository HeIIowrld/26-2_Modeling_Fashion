"""체형 분류 경계가 실제 측정값의 분포와 맞는지 지킨다.

`shoulder_hip_ratio`는 어깨 관절과 골반 관절 사이 간격의 비율이라 1.4~2.2에 분포한다.
예전에는 신체 표면 치수(어깨너비÷엉덩이너비) 기준인 0.90/1.12를 그대로 쓰는 바람에
전신 인식된 1,145장 중 89%가 '역삼각체형'으로 쏠렸다.
"""

import json
import sys
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pose_analyzer
from pose_analyzer import LOWER_BODY_RATIO, UPPER_BODY_RATIO

REFERENCE = ROOT / "data" / "body_shape_reference.json"


class ReferenceTableTests(unittest.TestCase):
    def test_reference_file_exists(self):
        self.assertTrue(REFERENCE.is_file())

    def test_reference_declares_what_it_measures(self):
        """Size Korea 값과 재는 대상이 달라 그대로 옮겨 쓰면 안 된다는 점을 남겨 둔다."""
        data = json.loads(REFERENCE.read_text(encoding="utf-8"))
        self.assertEqual(data["metric"], "shoulder_hip_ratio")
        self.assertTrue(data["definition"])
        self.assertTrue(data["caveats"])
        self.assertGreater(data["sample"]["n"], 0)

    def test_percentiles_are_ordered(self):
        percentiles = json.loads(REFERENCE.read_text(encoding="utf-8"))["percentiles"]
        keys = sorted(percentiles, key=int)
        values = [percentiles[key] for key in keys]
        self.assertEqual(values, sorted(values))

    def test_thresholds_come_from_the_reference_table(self):
        percentiles = json.loads(REFERENCE.read_text(encoding="utf-8"))["percentiles"]
        self.assertAlmostEqual(LOWER_BODY_RATIO, float(percentiles["33"]))
        self.assertAlmostEqual(UPPER_BODY_RATIO, float(percentiles["67"]))

    def test_missing_reference_falls_back_without_crashing(self):
        with unittest.mock.patch.object(Path, "is_file", return_value=False):
            low, high = pose_analyzer._load_body_shape_reference()
        self.assertLess(low, high)


class WorldLandmarkTests(unittest.TestCase):
    """체형 비율은 3D world 좌표로 잰다.

    정규화 2D 좌표는 몸이 정면을 향해 골반이 겹쳐 보이면 간격이 0에 가까워져 비율이
    폭발한다. 같은 표본 1,409장에서 변동계수가 2D 3.80 vs world 0.18이었다.
    """

    def test_reference_records_that_world_coordinates_are_used(self):
        definition = json.loads(REFERENCE.read_text(encoding="utf-8"))["definition"]
        self.assertIn("world", definition)

    def test_distance_handles_three_dimensional_points(self):
        from pose_analyzer import _distance

        self.assertAlmostEqual(_distance((0.0, 0.0, 0.0), (3.0, 4.0, 12.0)), 13.0)
        self.assertAlmostEqual(_distance((0.0, 0.0), (3.0, 4.0)), 5.0)

    def test_rotation_around_the_vertical_axis_barely_moves_the_ratio(self):
        """2D였다면 몸이 돌아갈수록 골반 간격이 줄어 비율이 커졌다."""
        import math

        def ratio(angle: float, use_depth: bool) -> float:
            from pose_analyzer import _distance

            def rotate(x: float, z: float) -> tuple[float, float]:
                return x * math.cos(angle) - z * math.sin(angle), x * math.sin(angle) + z * math.cos(angle)

            shoulder = [rotate(sign * 0.20, 0.0) for sign in (-1, 1)]
            hip = [rotate(sign * 0.14, 0.0) for sign in (-1, 1)]
            if use_depth:
                points = [(x, 0.0, z) for x, z in shoulder], [(x, 0.0, z) for x, z in hip]
            else:
                points = [(x, 0.0) for x, _ in shoulder], [(x, 0.0) for x, _ in hip]
            return _distance(*points[0]) / max(_distance(*points[1]), 1e-6)

        world_spread = abs(ratio(0.0, True) - ratio(math.radians(40), True))
        flat_spread = abs(ratio(0.0, False) - ratio(math.radians(40), False))
        self.assertLess(world_spread, 1e-6)
        self.assertLessEqual(world_spread, flat_spread)


class SizeKoreaVocabularyTests(unittest.TestCase):
    """체형 이름은 Size Korea 『한국인의 표준체형』의 상반신 분류를 따른다."""

    def test_labels_use_size_korea_names(self):
        from schemas import BODY_SHAPES

        self.assertEqual(set(BODY_SHAPES), {"역삼각체형", "사각체형", "삼각체형"})

    def test_reference_records_the_source(self):
        classification = json.loads(REFERENCE.read_text(encoding="utf-8"))["classification"]
        self.assertIn("Size Korea", classification["source"])
        self.assertEqual(set(classification["labels"]), {"역삼각체형", "사각체형", "삼각체형"})

    def test_unimplemented_types_are_documented_with_a_reason(self):
        """둘레가 필요한 나머지 4가지를 조용히 빠뜨리지 않는다."""
        blocked = json.loads(REFERENCE.read_text(encoding="utf-8"))["classification"]["not_implemented"]
        self.assertIn("모래시계체형", blocked["labels"])
        self.assertIn("둘레", blocked["reason"])
        self.assertTrue(blocked["unlocks_if"])

    def test_waist_substitution_is_disclosed(self):
        """Size Korea는 허리둘레를 쓰지만 사진에서는 골반 폭으로 대신한다."""
        data = json.loads(REFERENCE.read_text(encoding="utf-8"))
        self.assertIn("허리둘레", data["classification"]["deviation"])

    def test_engine_rules_use_the_same_labels(self):
        """엔진과 분석기의 라벨이 어긋나면 체형 규칙이 조용히 잠든다."""
        import recommendation_engine
        from schemas import SHAPE_INVERTED_TRIANGLE, SHAPE_RECTANGLE, SHAPE_TRIANGLE

        source = Path(recommendation_engine.__file__).read_text(encoding="utf-8")
        for name in ("SHAPE_INVERTED_TRIANGLE", "SHAPE_TRIANGLE", "SHAPE_RECTANGLE"):
            with self.subTest(name=name):
                self.assertIn(name, source)
        self.assertNotIn('"상체 강조형"', source)
        self.assertEqual(
            {SHAPE_INVERTED_TRIANGLE, SHAPE_RECTANGLE, SHAPE_TRIANGLE},
            {"역삼각체형", "사각체형", "삼각체형"},
        )


class ThresholdSanityTests(unittest.TestCase):
    def test_thresholds_sit_inside_the_observed_range(self):
        """world 좌표 기준 어깨/골반 비율은 1.3~1.6에 분포한다."""
        percentiles = json.loads(REFERENCE.read_text(encoding="utf-8"))["percentiles"]
        low, high = float(percentiles["5"]), float(percentiles["95"])
        for value in (LOWER_BODY_RATIO, UPPER_BODY_RATIO):
            with self.subTest(value=value):
                self.assertGreater(value, low)
                self.assertLess(value, high)

    def test_old_surface_measurement_thresholds_are_not_used(self):
        """0.90 / 1.12는 신체 표면 치수 기준이라 이 지표에는 맞지 않는다."""
        self.assertNotAlmostEqual(LOWER_BODY_RATIO, 0.90, places=2)
        self.assertNotAlmostEqual(UPPER_BODY_RATIO, 1.12, places=2)

    def test_all_three_labels_are_reachable(self):
        """어떤 값을 넣어도 한 라벨만 나오면 분류가 의미를 잃는다."""
        labels = set()
        for ratio in (LOWER_BODY_RATIO - 0.2, (LOWER_BODY_RATIO + UPPER_BODY_RATIO) / 2,
                      UPPER_BODY_RATIO + 0.2):
            if ratio >= UPPER_BODY_RATIO:
                labels.add("역삼각체형")
            elif ratio <= LOWER_BODY_RATIO:
                labels.add("삼각체형")
            else:
                labels.add("사각체형")
        self.assertEqual(labels, {"역삼각체형", "사각체형", "삼각체형"})

    def test_reference_percentiles_split_the_population_evenly(self):
        """33/67 백분위를 쓰므로 각 구간이 대략 3분의 1이어야 한다."""
        percentiles = json.loads(REFERENCE.read_text(encoding="utf-8"))["percentiles"]
        self.assertLess(float(percentiles["25"]), LOWER_BODY_RATIO)
        self.assertGreater(float(percentiles["75"]), UPPER_BODY_RATIO)


if __name__ == "__main__":
    import unittest.mock  # noqa: F401

    unittest.main()
