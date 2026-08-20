"""둘레를 입력했을 때의 체형 판정을 지킨다.

사진에는 둘레 정보가 없어 역삼각·사각·삼각 세 가지만 구분할 수 있다.
사용자가 가슴·허리·엉덩이 둘레를 입력하면 Size Korea 문서가 "둘레 항목이 필요해
추가 분석이 필요하다"며 미룬 모래시계·마름모꼴·둥근체형까지 판정한다.
"""

import json
import sys
import unittest
import unittest.mock
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from body_shape import classify, classify_from_circumferences, thresholds
from schemas import (
    BASIS_MEASUREMENT,
    BASIS_PHOTO,
    CIRCUMFERENCE_SHAPES,
    SHAPE_DIAMOND,
    SHAPE_HOURGLASS,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_ROUND,
    SHAPE_TRIANGLE,
    PoseAnalysis,
    UserProfile,
)

REFERENCE = ROOT / "data" / "body_shape_reference.json"


def pose(shape: str = SHAPE_RECTANGLE) -> PoseAnalysis:
    return PoseAnalysis(True, 0.96, shape, 1.45, 0.66, 0.44, "정면에 가까움", 0.95)


class ShapeFromCircumferenceTests(unittest.TestCase):
    def test_hourglass_needs_a_defined_waist(self):
        shape, basis = classify_from_circumferences(chest_cm=90, waist_cm=64, hip_cm=92)
        self.assertEqual(shape, SHAPE_HOURGLASS)
        self.assertEqual(basis, BASIS_MEASUREMENT)

    def test_same_chest_and_hip_without_a_waist_is_a_rectangle(self):
        shape, _ = classify_from_circumferences(chest_cm=90, waist_cm=76, hip_cm=92)
        self.assertEqual(shape, SHAPE_RECTANGLE)

    def test_wide_hip_is_a_triangle(self):
        shape, _ = classify_from_circumferences(chest_cm=84, waist_cm=70, hip_cm=100)
        self.assertEqual(shape, SHAPE_TRIANGLE)

    def test_wide_chest_is_an_inverted_triangle(self):
        shape, _ = classify_from_circumferences(chest_cm=104, waist_cm=82, hip_cm=92)
        self.assertEqual(shape, SHAPE_INVERTED_TRIANGLE)

    def test_largest_waist_is_a_diamond(self):
        shape, _ = classify_from_circumferences(chest_cm=90, waist_cm=96, hip_cm=92)
        self.assertEqual(shape, SHAPE_DIAMOND)

    def test_barely_defined_waist_is_round(self):
        shape, _ = classify_from_circumferences(chest_cm=96, waist_cm=90, hip_cm=97)
        self.assertEqual(shape, SHAPE_ROUND)

    def test_photo_only_shapes_cannot_produce_circumference_shapes(self):
        """사진만으로는 이 세 체형이 나올 수 없어야 한다."""
        from schemas import BODY_SHAPES

        self.assertEqual(set(BODY_SHAPES) & set(CIRCUMFERENCE_SHAPES), set())

    def test_zero_or_missing_values_are_rejected(self):
        for values in ((0, 70, 92), (90, -1, 92), (90, 70, None)):
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    classify_from_circumferences(*values)


class ClassifyPrefersMeasurementsTests(unittest.TestCase):
    def test_measurements_override_the_photo_estimate(self):
        """줄자로 잰 값이 사진 추정보다 정확하므로 우선한다."""
        profile = UserProfile(chest_cm=90, waist_cm=64, hip_cm=92)
        shape, basis = classify(profile, pose(SHAPE_TRIANGLE))
        self.assertEqual(shape, SHAPE_HOURGLASS)
        self.assertEqual(basis, BASIS_MEASUREMENT)

    def test_photo_estimate_is_kept_without_measurements(self):
        shape, basis = classify(UserProfile(), pose(SHAPE_TRIANGLE))
        self.assertEqual(shape, SHAPE_TRIANGLE)
        self.assertEqual(basis, BASIS_PHOTO)

    def test_partial_measurements_fall_back_to_the_photo(self):
        profile = UserProfile(chest_cm=90, waist_cm=64)  # 엉덩이 없음
        shape, basis = classify(profile, pose(SHAPE_RECTANGLE))
        self.assertEqual(basis, BASIS_PHOTO)
        self.assertEqual(shape, SHAPE_RECTANGLE)

    def test_impossible_measurements_fall_back_instead_of_crashing(self):
        profile = UserProfile(chest_cm=0, waist_cm=0, hip_cm=0)
        shape, basis = classify(profile, pose(SHAPE_TRIANGLE))
        self.assertEqual(basis, BASIS_PHOTO)
        self.assertEqual(shape, SHAPE_TRIANGLE)


class RuleProvenanceTests(unittest.TestCase):
    def test_thresholds_come_from_the_data_file(self):
        rules = json.loads(REFERENCE.read_text(encoding="utf-8"))["circumference_rules"]
        self.assertEqual(thresholds()["balanced_tolerance"], rules["thresholds"]["balanced_tolerance"])

    def test_rules_are_marked_provisional(self):
        """Rasband 분류는 서술적 정의라 수치가 출처마다 다르다. 검수 전임을 남긴다."""
        rules = json.loads(REFERENCE.read_text(encoding="utf-8"))["circumference_rules"]
        self.assertIn("잠정", rules["status"])
        self.assertTrue(rules["why_provisional"])

    def test_documented_order_matches_the_code(self):
        rules = json.loads(REFERENCE.read_text(encoding="utf-8"))["circumference_rules"]
        documented = [entry["shape"] for entry in rules["order"]]
        self.assertEqual(
            documented,
            [SHAPE_DIAMOND, SHAPE_ROUND, SHAPE_HOURGLASS, SHAPE_INVERTED_TRIANGLE,
             SHAPE_TRIANGLE, SHAPE_RECTANGLE],
        )

    def test_unimplemented_shapes_are_explained(self):
        rules = json.loads(REFERENCE.read_text(encoding="utf-8"))["circumference_rules"]
        self.assertIn("튜브체형", rules["not_implemented"]["labels"])
        self.assertTrue(rules["not_implemented"]["reason"])



class PhotoEstimatorSeamTests(unittest.TestCase):
    """쓰리사이즈를 안 넣었을 때 사진으로 둘레를 추정하는 연결 지점."""

    def test_estimator_runs_without_extra_model_files(self):
        """MediaPipe 분할만 쓰므로 별도 다운로드나 라이선스 동의가 필요 없다."""
        from body_measure import BodyMeasurementEstimator

        self.assertTrue(BodyMeasurementEstimator().available)

    def test_status_reports_ready(self):
        from body_measure import estimator_status

        self.assertTrue(estimator_status()["available"])

    def test_disabled_estimator_says_where_to_turn_it_on(self):
        from body_measure import BodyMeasurementEstimator, MeasurementNotReady

        with self.assertRaises(MeasurementNotReady) as caught:
            BodyMeasurementEstimator(enabled=False).estimate(object(), pose())
        self.assertIn("ENABLE_BODY_MEASUREMENT", str(caught.exception))

    def test_photo_without_a_person_is_refused(self):
        from body_measure import BodyMeasurementEstimator, MeasurementNotReady

        import numpy as np

        blank = np.full((240, 160, 3), 255, dtype=np.uint8)
        with self.assertRaises(MeasurementNotReady):
            BodyMeasurementEstimator().estimate(blank, pose())

    def test_connected_estimator_is_used_and_marked_as_an_estimate(self):
        from body_measure import BodyMeasurement
        from schemas import BASIS_ESTIMATE

        class FakeEstimator:
            def estimate(self, person_image, pose_result, height_cm=None):
                return BodyMeasurement(90.0, 64.0, 92.0, "테스트")

        shape, basis = classify(
            UserProfile(height_cm=170), pose(SHAPE_TRIANGLE),
            person_image=object(), estimator=FakeEstimator(),
        )
        self.assertEqual(shape, SHAPE_HOURGLASS)
        self.assertEqual(basis, BASIS_ESTIMATE)

    def test_typed_measurements_beat_the_photo_estimate(self):
        from body_measure import BodyMeasurement

        class FakeEstimator:
            def estimate(self, *_args, **_kwargs):
                return BodyMeasurement(90.0, 64.0, 92.0, "테스트")

        profile = UserProfile(chest_cm=84, waist_cm=70, hip_cm=100, height_cm=170)
        shape, basis = classify(
            profile, pose(), person_image=object(), estimator=FakeEstimator()
        )
        self.assertEqual(shape, SHAPE_TRIANGLE)
        self.assertEqual(basis, BASIS_MEASUREMENT)

    def test_falls_back_to_photo_shape_when_estimation_fails(self):
        shape, basis = classify(UserProfile(), pose(SHAPE_INVERTED_TRIANGLE), person_image=object())
        self.assertEqual(shape, SHAPE_INVERTED_TRIANGLE)
        self.assertEqual(basis, BASIS_PHOTO)


if __name__ == "__main__":
    unittest.main()
