"""합성 후 품질 검사와 원인 기반 마스크 정책 회귀 테스트.

2026-09-15 원자료 재분석에서 확인한 실패 모드를 고정한다.
- 마스크 안인데 배를 맨살로 그린 크롭화(model_7)는 재생성 대상이다.
- 원래 크롭탑·트인 셔츠 모양 때문에 덮지 못한 피부(2-model_4·demo_2)는 경고만 한다.
- 원래 옷 윤곽을 담은 마스크는 넥라인 구멍과 짧은 밑단을 새 옷에 옮긴다.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catvton_tryon import (  # noqa: E402
    CatVTONTryOn,
    _labels_to_model,
    _points_to_model,
    agnostic_upper_mask,
    landmarks_to_pixels,
    pad_to_aspect,
    pants_flare_ratio,
    shape_lower_mask,
)
from tryon_quality import (  # noqa: E402
    DEFAULT_THRESHOLDS,
    QualityCheck,
    TryOnQualityReport,
    assess_tryon,
    body_band,
    load_thresholds,
)

H, W = 400, 300
LANDMARKS = {
    "left_shoulder": (100.0, 100.0), "right_shoulder": (200.0, 100.0),
    "left_hip": (115.0, 250.0), "right_hip": (185.0, 250.0),
    "left_knee": (120.0, 330.0), "right_knee": (180.0, 330.0),
}


def _scene(*, skin_rows=None, edit_bottom=260):
    """상의(3)를 입은 합성 사람. skin_rows 행 구간을 결과에서 몸통 피부(16)로 바꾼다."""
    before = np.full((H, W, 3), 200, np.uint8)
    seg = np.zeros((H, W), np.uint8)
    seg[90:250, 90:210] = 3
    seg[250:390, 100:200] = 6
    seg[20:80, 120:180] = 1
    before[seg == 3] = (40, 40, 160)
    before[seg == 6] = (30, 30, 30)
    before[seg == 1] = (220, 180, 150)
    edit = np.zeros((H, W), bool)
    edit[85:edit_bottom, 85:215] = True
    after = before.copy()
    after[seg == 3] = (180, 40, 40)
    after_seg = seg.copy()
    if skin_rows is not None:
        after_seg[skin_rows[0]:skin_rows[1], 90:210] = 16
    return before, after, edit, seg, after_seg


def _check(report, name):
    return next(check for check in report.checks if check.name == name)


class BodyBandTests(unittest.TestCase):
    def test_top_band_is_below_chest_and_between_hips(self):
        band = body_band(LANDMARKS, (H, W), "top")
        ys, xs = np.where(band)
        self.assertEqual(ys.min(), round(100 + 0.45 * 150))
        self.assertEqual((xs.min(), xs.max() + 1), (115, 185))

    def test_missing_joint_disables_band(self):
        partial = {k: v for k, v in LANDMARKS.items() if k != "left_hip"}
        self.assertIsNone(body_band(partial, (H, W), "top"))


class AssessTryOnTests(unittest.TestCase):
    def test_clean_result_passes(self):
        before, after, edit, seg, after_seg = _scene()
        report = assess_tryon(category="top", before=before, after=after, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                              landmarks_px=LANDMARKS)
        self.assertEqual(report.failed, [])
        self.assertFalse(report.retry_recommended)

    def test_midriff_skin_inside_mask_is_retryable(self):
        # model_7: 마스크는 허리선까지인데 명치 아래를 맨살로 그렸다.
        before, after, edit, seg, after_seg = _scene(skin_rows=(190, 250))
        report = assess_tryon(category="top", before=before, after=after, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                              landmarks_px=LANDMARKS)
        inside = _check(report, "torso_skin_inside_mask")
        self.assertFalse(inside.passed)
        self.assertTrue(report.retry_recommended)

    def test_skin_outside_mask_warns_without_retry(self):
        # 2-model_4: 원래 크롭탑 밑단에서 마스크가 끝나 그 아래 배는 새 상의가 덮을 수 없다.
        before, after, edit, seg, after_seg = _scene(skin_rows=(200, 250), edit_bottom=195)
        report = assess_tryon(category="top", before=before, after=after, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                              landmarks_px=LANDMARKS)
        self.assertFalse(_check(report, "torso_skin_outside_mask").passed)
        self.assertTrue(_check(report, "torso_skin_inside_mask").passed)
        self.assertNotIn("torso_skin_outside_mask",
                         [c.name for c in report.failed if c.retryable])

    def test_crop_product_name_is_exempt(self):
        before, after, edit, seg, after_seg = _scene(skin_rows=(190, 250))
        report = assess_tryon(category="top", before=before, after=after, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                              landmarks_px=LANDMARKS, product_name="베이직 쿨 헨리넥 크롭 반팔 니트")
        self.assertNotIn("torso_skin_inside_mask", [c.name for c in report.checks])

    def test_shifted_result_fails_preservation(self):
        before, after, edit, seg, after_seg = _scene()
        noisy = np.roll(after, 7, axis=1)
        noisy[::2] = 255 - noisy[::2]
        report = assess_tryon(category="top", before=before, after=noisy, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10))
        preservation = _check(report, "outside_preservation")
        self.assertFalse(preservation.passed)
        self.assertFalse(preservation.retryable)

    def test_result_farther_from_reference_than_original_is_retryable(self):
        # 2026-09-15 재생: 원래 옷과 상품이 같은 검정 계열이면 차이가 -0.02 안팎이라
        # 0 기준은 오탐이었다. 원래 옷보다 확실히(-0.03 이하) 멀어진 결과만 실패로 본다.
        before, after, edit, seg, after_seg = _scene()
        reference = np.zeros((120, 80, 3), np.uint8)
        reference[:] = (180, 40, 40)

        def embed(image):
            # 파랑(원래 옷)과 빨강(상품)을 구분하는 가짜 임베딩
            rgb = np.asarray(image, np.float32).reshape(-1, 3).mean(axis=0)
            vector = np.array([rgb[0], rgb[2]])
            return vector / np.linalg.norm(vector)

        wrong = before.copy()
        wrong[seg == 3] = (20, 200, 220)  # 상품과 더 먼 다른 옷을 그린 경우
        report = assess_tryon(category="top", before=before, after=wrong, edit_mask=edit,
                              after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                              reference_rgb=reference, embed=embed)
        fidelity = _check(report, "reference_fidelity")
        self.assertFalse(fidelity.passed)
        self.assertTrue(fidelity.retryable)
        good = assess_tryon(category="top", before=before, after=after, edit_mask=edit,
                            after_seg=after_seg, before_seg=seg, target_labels=(3, 4, 10),
                            reference_rgb=reference, embed=embed)
        self.assertTrue(_check(good, "reference_fidelity").passed)

    def test_long_pants_rendered_as_shorts_is_structural_failure(self):
        # 2026-09-15 재생: 긴바지 상품이 원래 반바지·치마 모양대로 짧게 그려진 사례.
        seg = np.zeros((H, W), np.uint8)
        seg[250:290, 100:200] = 6        # 합성 결과의 하의가 허벅지에서 끝남
        seg[290:395, 110:190] = 14       # 그 아래는 맨다리
        legs = {**LANDMARKS, "left_ankle": (120.0, 390.0), "right_ankle": (180.0, 390.0)}
        image = np.full((H, W, 3), 128, np.uint8)
        edit = seg > 0
        report = assess_tryon(category="bottom", before=image, after=image, edit_mask=edit,
                              after_seg=seg, target_labels=(4, 5, 6, 7), landmarks_px=legs,
                              reference_length="롱·긴바지 기장")
        check = _check(report, "bottom_length_fidelity")
        self.assertFalse(check.passed)
        self.assertFalse(check.retryable)
        same = assess_tryon(category="bottom", before=image, after=image, edit_mask=edit,
                            after_seg=seg, target_labels=(4, 5, 6, 7), landmarks_px=legs,
                            reference_length="쇼츠·미니 기장")
        self.assertTrue(_check(same, "bottom_length_fidelity").passed)

    def test_resolution_mismatch_is_skipped_not_passed(self):
        before, after, edit, seg, after_seg = _scene()
        report = assess_tryon(category="top", before=before, after=after[:-1], edit_mask=edit,
                              after_seg=after_seg, target_labels=(3,))
        self.assertEqual(report.checks, [])
        self.assertTrue(report.skipped)

    def test_thresholds_file_is_marked_provisional(self):
        import json
        from tryon_quality import THRESHOLDS_PATH

        stored = json.loads(THRESHOLDS_PATH.read_text(encoding="utf-8"))
        self.assertIn("잠정", stored["status"])
        self.assertEqual(set(stored["thresholds"]), set(DEFAULT_THRESHOLDS))
        self.assertEqual(set(load_thresholds()), set(DEFAULT_THRESHOLDS))


class _FakePipeline:
    def __init__(self):
        self.calls = 0

    def __call__(self, **kwargs):
        self.calls += 1
        return [kwargs["image"]]


class RetryPolicyTests(unittest.TestCase):
    def _adapter(self, reports):
        adapter = CatVTONTryOn(max_retries=1)
        adapter.device = "cpu"
        pipeline = _FakePipeline()
        adapter._load_pipeline = lambda: pipeline
        adapter._apply_scheduler = lambda _pipeline: None
        adapter._garment_sharpness = lambda image, mask: 100.0
        queue = list(reports)
        adapter._assess_attempt = lambda *args: queue.pop(0)
        return adapter, pipeline

    def _run(self, adapter):
        person = Image.new("RGB", (60, 80), "gray")
        mask = Image.new("L", (60, 80), 255)
        return adapter._tryon_once(person, person, mask, quality={"category": "top"})

    def test_retryable_failure_triggers_one_more_generation(self):
        bad = TryOnQualityReport("top", [QualityCheck("torso_skin_inside_mask", 0.2, 0.08, False, True, "배 노출")])
        good = TryOnQualityReport("top", [QualityCheck("torso_skin_inside_mask", 0.0, 0.08, True, True)])
        adapter, pipeline = self._adapter([bad, good])
        self._run(adapter)
        self.assertEqual(pipeline.calls, 2)
        self.assertEqual(adapter.last_warnings, [])
        self.assertEqual(adapter.last_quality_reports[0]["attempts"], 2)

    def test_structural_failure_does_not_waste_a_generation(self):
        structural = TryOnQualityReport(
            "top", [QualityCheck("torso_skin_outside_mask", 0.3, 0.08, False, False, "원래 옷 모양")])
        adapter, pipeline = self._adapter([structural])
        self._run(adapter)
        self.assertEqual(pipeline.calls, 1)
        self.assertEqual(len(adapter.last_warnings), 1)
        self.assertIn("원래 옷 모양", adapter.last_warnings[0])

    def test_persistent_failure_is_disclosed_after_retry(self):
        bad = TryOnQualityReport("top", [QualityCheck("garment_coverage", 0.2, 0.5, False, True, "덜 덮음")])
        adapter, pipeline = self._adapter([bad, bad])
        self._run(adapter)
        self.assertEqual(pipeline.calls, 2)
        self.assertIn("다시 생성 1회 후", adapter.last_warnings[0])

    def test_gate_can_be_disabled(self):
        adapter, pipeline = self._adapter([])
        adapter.post_quality_gate = False
        self._run(adapter)
        self.assertEqual(pipeline.calls, 1)
        self.assertEqual(adapter.last_quality_reports, [])


class AgnosticUpperMaskTests(unittest.TestCase):
    def _v_neck_crop_top(self):
        seg = np.zeros((H, W), np.uint8)
        seg[90:200, 90:210] = 3          # 크롭탑
        seg[90:160, 140:160] = 16        # V넥 구멍
        seg[200:250, 100:200] = 16       # 드러난 배
        seg[250:390, 100:200] = 6        # 바지
        style = np.isin(seg, (3, 12))
        return seg, style

    def test_fills_neckline_hole_and_extends_hem_over_person_pixels(self):
        seg, style = self._v_neck_crop_top()
        mask = agnostic_upper_mask(style, seg, LANDMARKS)
        self.assertTrue(mask[120, 150])       # V 구멍
        self.assertTrue(mask[230, 150])       # 배
        self.assertTrue(mask[260, 150])       # 골반선 여유만큼 바지 윗부분
        self.assertFalse(mask[260, 20])       # 배경은 넣지 않는다
        self.assertFalse(mask[300, 150])      # 골반 아래 바지는 그대로

    def test_crop_products_keep_their_hem(self):
        seg, style = self._v_neck_crop_top()
        mask = agnostic_upper_mask(style, seg, LANDMARKS, extend_hem=False)
        self.assertTrue(mask[230, 150])       # 구멍 메우기는 유지
        self.assertFalse(mask[260, 150])

    def test_lower_overlap_mode_stops_shortly_below_lower_garment_top(self):
        # 중간 점검: 골반까지 넓히면 치마 윗부분에 갈색 띠가 새로 생겼다.
        seg, style = self._v_neck_crop_top()
        mask = agnostic_upper_mask(style, seg, LANDMARKS, hem_mode="lower", lower_overlap=0.04)
        self.assertTrue(mask[230, 150])       # 배는 메운다
        self.assertTrue(mask[252, 150])       # 하의 윗단(250)에서 조금 겹친다
        self.assertFalse(mask[260, 150])      # 골반 기준(265)보다 먼저 멈춘다

    def test_requires_shoulders_and_hips(self):
        seg, style = self._v_neck_crop_top()
        self.assertIsNone(agnostic_upper_mask(style, seg, {"left_shoulder": (1, 1)}))

    def test_invisible_or_offscreen_landmarks_are_dropped(self):
        points = landmarks_to_pixels({"a": (0.5, 0.5, 0.9), "b": (0.5, 1.2, 0.9), "c": (0.5, 0.5, 0.1)}, W, H)
        self.assertEqual(set(points), {"a"})


def _pants(hem_width, height=300, waist=80):
    mask = np.zeros((height + 20, 200), bool)
    for y in range(10, 10 + height):
        progress = (y - 10) / height
        width = int(round(waist + progress * (hem_width - waist)))
        mask[y, 100 - width // 2:100 + width // 2] = True
    return mask


class ReferenceShapedLowerMaskTests(unittest.TestCase):
    def test_flare_ratio_separates_slim_from_wide(self):
        self.assertLess(pants_flare_ratio(_pants(40)), 0.7)
        self.assertGreater(pants_flare_ratio(_pants(110)), 1.1)

    def test_widens_only_when_reference_is_clearly_wider(self):
        person = _pants(50)
        points = {"left_hip": (80.0, 15.0), "right_hip": (120.0, 15.0)}
        current = pants_flare_ratio(person)
        self.assertIsNone(shape_lower_mask(person, points, current + 0.05, current))
        shaped = shape_lower_mask(person, points, current + 0.6, current)
        self.assertGreater(shaped[300].sum(), person[300].sum())
        self.assertTrue((shaped | person).sum() == shaped.sum())  # 좁히지 않는다
        self.assertEqual(shaped[20].sum(), person[20].sum())       # 허리는 그대로

    def test_unknown_current_shape_is_not_widened(self):
        person = _pants(50)
        self.assertIsNone(shape_lower_mask(person, {"left_hip": (80, 15), "right_hip": (120, 15)}, 1.5, None))


class ModelCoordinateTests(unittest.TestCase):
    def test_labels_and_points_follow_the_person_padding(self):
        labels = np.zeros((200, 100), np.uint8)
        labels[50, 50] = 7
        target = (150, 200)
        padded, box = pad_to_aspect(Image.fromarray(labels), target, 0)
        mapped = _labels_to_model(labels, target)
        point = _points_to_model({"p": (50.0, 50.0)}, box, padded.size, target)["p"]
        ys, xs = np.where(mapped == 7)
        self.assertLessEqual(abs(xs.mean() - point[0]), 1.5)
        self.assertLessEqual(abs(ys.mean() - point[1]), 1.5)


class DefaultPolicyTests(unittest.TestCase):
    def test_validated_mask_policies_are_the_defaults(self):
        # 2026-09-15 사전 기준: 상의 agnostic 통과·채택, 하의 reference-shape 실패·기각.
        adapter = CatVTONTryOn()
        self.assertEqual(adapter.upper_mask_policy, "agnostic")
        self.assertEqual(adapter.lower_mask_policy, "native")
        self.assertEqual(CatVTONTryOn.fast().upper_mask_policy, "agnostic")

    def test_unknown_policy_is_rejected(self):
        with self.assertRaises(ValueError):
            CatVTONTryOn(upper_mask_policy="convex")


if __name__ == "__main__":
    unittest.main()
