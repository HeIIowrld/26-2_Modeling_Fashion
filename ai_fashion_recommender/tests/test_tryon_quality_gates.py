"""VTON 합성 신뢰도 게이트 회귀 테스트.

female_012 통제 실험(2026-08-20)에서 확인한 두 실패 모드를 고정한다:
- 레퍼런스 coverage 0.15 → 시스루 렌더링
- 하의 기장 gap=3 → 다리 전체가 옷 텍스처로 채워짐 (gap=1은 정상)
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from catvton_tryon import (
    CatVTONTryOn,
    UNRELIABLE_LENGTH_GAP,
    _restore_original_regions,
    bottom_length_gap,
    evaluate_garment_reference,
    outerwear_level,
    pad_to_aspect,
    sleeve_length_gap,
    split_outerwear_mask,
    unpad_result,
)


class TunedDefaultTests(unittest.TestCase):
    """2026-08-21 A/B로 정한 값을 고정한다.

    근거는 reports/vton_quality/param_tuning_2026-08-21.md에 있다. 바꿀 거라면
    같은 사진·같은 시드로 다시 재고 리포트를 갱신할 것.
    """

    def test_repaint_band_stays_narrow(self):
        # 블렌딩 반경은 height // divisor다. 밴드가 넓으면 원래 옷 색이 번진다.
        # 150에서 300으로 좁혀 halo를 줄였다.
        self.assertEqual(CatVTONTryOn().repaint_blur_divisor, 300)

    def test_fast_preset_uses_dpm_solver(self):
        # 2026-08-22 A/B: DPM++ 2M Karras 25스텝이 DDIM 50스텝보다 의류 텍스처
        # 지표가 높고 색 충실도도 나았다(3장 전부). 시간은 절반(장당 22초).
        preset = CatVTONTryOn.fast()
        self.assertEqual(preset.num_inference_steps, 25)
        self.assertEqual(preset.scheduler, "dpmpp_2m_karras")
        # 나머지는 기본값이어야 비교가 성립한다.
        self.assertEqual(preset.guidance_scale, CatVTONTryOn().guidance_scale)
        self.assertEqual(preset.repaint_blur_divisor, CatVTONTryOn().repaint_blur_divisor)

    def test_fast_preset_accepts_overrides(self):
        self.assertEqual(CatVTONTryOn.fast(num_inference_steps=10).num_inference_steps, 10)

    def test_tuned_defaults_2026_08_22(self):
        # 2026-08-22 같은 시드 A/B로 승격한 기본값. 바꿀 거라면 그날 리포트와 같은
        # 프로토콜(같은 사진·같은 시드, tune_vton.py)로 재고 근거를 남길 것.
        tryon = CatVTONTryOn()
        self.assertEqual(tryon.scheduler, "ddim")  # 기본 경로는 보수적으로 유지
        self.assertEqual(tryon.eta, 1.0)  # CatVTON 공식 기본(확률 샘플링)
        self.assertEqual(tryon.outerwear_policy, "reassign")  # 롱코트 4장 전부 개선
        self.assertTrue(tryon.protect_restore)  # 가방 끈·버클 복원, 부작용 없음
        self.assertFalse(tryon.pipeline_recarve)  # 미검증이라 꺼짐
        self.assertEqual(tryon.skirt_guidance_scale, 1.5)  # 시스루·치마 슬릿 완화


class SchedulerSwapTests(unittest.TestCase):
    """스케줄러 드롭인 교체가 설정→적용→복원 왕복에서 안전한지 고정한다."""

    def _dummy_pipeline(self):
        from diffusers import DDIMScheduler

        return SimpleNamespace(noise_scheduler=DDIMScheduler())

    def test_dpm_swap_and_restore(self):
        from diffusers import DPMSolverMultistepScheduler

        tryon = CatVTONTryOn(scheduler="dpmpp_2m_karras")
        pipeline = self._dummy_pipeline()
        original = pipeline.noise_scheduler
        tryon._apply_scheduler(pipeline)
        self.assertIsInstance(pipeline.noise_scheduler, DPMSolverMultistepScheduler)
        self.assertEqual(pipeline.noise_scheduler.config.algorithm_type, "dpmsolver++")
        self.assertTrue(pipeline.noise_scheduler.config.use_karras_sigmas)
        # ddim으로 되돌리면 원본 인스턴스가 그대로 돌아와야 한다.
        tryon.scheduler = "ddim"
        tryon._apply_scheduler(pipeline)
        self.assertIs(pipeline.noise_scheduler, original)

    def test_unipc_swap(self):
        from diffusers import UniPCMultistepScheduler

        tryon = CatVTONTryOn(scheduler="unipc")
        pipeline = self._dummy_pipeline()
        tryon._apply_scheduler(pipeline)
        self.assertIsInstance(pipeline.noise_scheduler, UniPCMultistepScheduler)

    def test_unknown_scheduler_raises(self):
        tryon = CatVTONTryOn(scheduler="euler_a")
        with self.assertRaises(ValueError):
            tryon._apply_scheduler(self._dummy_pipeline())


class OuterwearGateTests(unittest.TestCase):
    """아우터 감지: 30장 배치 재판독에서 하드셋 7/8 적중·오탐 0인 기준을 고정한다."""

    def test_hard_types(self):
        for upper in ("코트", "재킷", "블레이저", "블랙 코트", "재킷 추정"):
            self.assertEqual(outerwear_level(upper), "hard", upper)

    def test_soft_types_warn_only(self):
        # 가디건은 성공작(IMG_5424)에도 있어 정책 발동 대상이 아니다.
        for upper in ("가디건", "베스트", "점퍼"):
            self.assertEqual(outerwear_level(upper), "soft", upper)

    def test_non_outerwear(self):
        for upper in ("티셔츠", "셔츠", "니트", "탑", "", None):
            self.assertIsNone(outerwear_level(upper), upper)

    def test_skirt_reference_falls_back_to_product_name(self):
        tryon = CatVTONTryOn(skirt_guidance_scale=2.0)
        garment = Image.new("RGB", (10, 10), "white")
        skirt = SimpleNamespace(name="플리츠 미디 스커트", category="bottom")
        # 카탈로그에 영문명이 섞여 있다 — 실제 미발동 사례(MS7076570)를 고정한다.
        skirt_en = SimpleNamespace(name="Cotton Veil Skirt pale blue", category="bottom")
        pants = SimpleNamespace(name="와이드 데님 팬츠", category="bottom")
        self.assertTrue(tryon._is_skirt_reference(garment, skirt, {}))
        self.assertTrue(tryon._is_skirt_reference(garment, skirt_en, {}))
        self.assertFalse(tryon._is_skirt_reference(garment, pants, {}))


class OuterwearMaskSurgeryTests(unittest.TestCase):
    """힙 클립+오버행 재배정의 기하를 합성 마스크로 고정한다."""

    HEIGHT, WIDTH = 200, 100

    def _landmarks(self):
        # 어깨 y=0.2(40px), 힙 y=0.5(100px). 팔은 몸통 밖(x 0.05/0.95)으로 내린다.
        return {
            "left_shoulder": (0.30, 0.20, 1.0),
            "right_shoulder": (0.70, 0.20, 1.0),
            "left_hip": (0.40, 0.50, 1.0),
            "right_hip": (0.60, 0.50, 1.0),
            "left_elbow": (0.05, 0.40, 1.0),
            "right_elbow": (0.95, 0.40, 1.0),
            "left_wrist": (0.05, 0.60, 1.0),
            "right_wrist": (0.95, 0.60, 1.0),
        }

    def _masks(self):
        upper = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        upper[30:170, 25:75] = True  # 롱코트: 힙(100px) 한참 아래까지 내려온다
        lower = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        lower[95:190, 30:70] = True
        return upper, lower

    def test_overhang_moves_from_upper_to_lower(self):
        upper, lower = self._masks()
        split = split_outerwear_mask(upper, lower, self._landmarks())
        self.assertIsNotNone(split)
        new_upper, new_lower = split
        # clip_y = 100 + 0.3*(100-40) = 118. 그 아래 몸통 중앙(팔 회랑 밖)은
        # 상의 마스크에서 빠지고 하의 마스크로 넘어가야 한다.
        self.assertFalse(new_upper[130:170, 45:55].any())
        self.assertTrue(new_lower[130:170, 45:55].all())
        # 클립 위는 그대로다.
        self.assertTrue(new_upper[30:110, 30:70].all())
        # 마스크 합집합은 보존된다(영역을 잃지 않고 소유권만 이동).
        np.testing.assert_array_equal(new_upper | new_lower, upper | lower)

    def test_missing_landmarks_return_none(self):
        upper, lower = self._masks()
        landmarks = self._landmarks()
        del landmarks["left_hip"]
        self.assertIsNone(split_outerwear_mask(upper, lower, landmarks))

    def test_no_overhang_returns_none(self):
        upper = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        upper[30:90, 25:75] = True  # 힙 위에서 끝나는 짧은 상의
        lower = np.zeros((self.HEIGHT, self.WIDTH), dtype=bool)
        lower[95:190, 30:70] = True
        self.assertIsNone(split_outerwear_mask(upper, lower, self._landmarks()))


class ProtectRestoreTests(unittest.TestCase):
    def test_restore_forces_original_pixels(self):
        person = Image.new("RGB", (60, 60), (255, 0, 0))
        result = Image.new("RGB", (60, 60), (0, 0, 255))
        restore = np.zeros((60, 60), dtype=bool)
        restore[20:40, 20:40] = True
        blended = _restore_original_regions(person, result, restore)
        # 보호 영역 중심은 원본(빨강), 바깥은 합성 결과(파랑)여야 한다.
        self.assertEqual(blended.getpixel((30, 30)), (255, 0, 0))
        self.assertEqual(blended.getpixel((5, 5)), (0, 0, 255))


class LengthGapTests(unittest.TestCase):
    def test_long_pants_to_shorts_is_flagged(self):
        # 실측: 긴바지 → 쇼츠·미니에서 다리 전체가 니트 텍스처로 덮였다.
        gap = bottom_length_gap("긴바지", "쇼츠·미니 기장")
        self.assertEqual(gap, 3)
        self.assertGreaterEqual(gap, UNRELIABLE_LENGTH_GAP)

    def test_long_pants_to_midi_is_allowed(self):
        # 실측: 긴바지 → 미디·7부(페인터 팬츠)는 정상 합성됐다.
        gap = bottom_length_gap("긴바지", "미디·7부 기장")
        self.assertEqual(gap, 1)
        self.assertLess(gap, UNRELIABLE_LENGTH_GAP)

    def test_estimate_suffix_is_ignored(self):
        self.assertEqual(bottom_length_gap("긴바지", "무릎 기장 추정"), 2)

    def test_unknown_labels_return_none(self):
        self.assertIsNone(bottom_length_gap("분석 불가", "쇼츠·미니 기장"))
        self.assertIsNone(bottom_length_gap("긴바지", ""))

    def test_sleeve_gap_uses_same_scale(self):
        self.assertEqual(sleeve_length_gap("긴팔", "민소매"), 3)
        self.assertEqual(sleeve_length_gap("긴팔", "7부 소매"), 1)
        self.assertEqual(sleeve_length_gap("반팔", "긴팔"), -2)


class NecklineGapTests(unittest.TestCase):
    def test_high_neck_to_round_neck_is_disclosed(self):
        tryon = CatVTONTryOn()
        tryon._check_neckline_gap(
            SimpleNamespace(neckline="라운드넥"),
            {"outfit": SimpleNamespace(neckline="터틀넥")},
        )
        self.assertEqual(len(tryon.last_warnings), 1)
        self.assertIn("기존 칼라", tryon.last_warnings[0])

    def test_similar_coverage_does_not_warn(self):
        tryon = CatVTONTryOn()
        tryon._check_neckline_gap(
            SimpleNamespace(neckline="스탠드 칼라"),
            {"outfit": SimpleNamespace(neckline="터틀넥")},
        )
        self.assertEqual(tryon.last_warnings, [])


class ReferenceQualityTests(unittest.TestCase):
    @staticmethod
    def _reference(mask_pixels: int, brightness: int, size: int = 100):
        rgb = np.full((size, size, 3), brightness, dtype=np.uint8)
        mask = np.zeros((size, size), dtype=bool)
        rows = mask_pixels // size
        mask[:rows, :] = True
        return rgb, mask

    def test_coverage_scales_with_target_resolution(self):
        rgb, mask = self._reference(5000, 120)
        report = evaluate_garment_reference(rgb, mask, target_pixels=10000)
        self.assertAlmostEqual(report["coverage"], 0.5, places=3)
        self.assertAlmostEqual(report["garment_pixels"], 5000.0, places=3)

    def test_white_garment_has_low_contrast(self):
        rgb, mask = self._reference(5000, 250)
        report = evaluate_garment_reference(rgb, mask, target_pixels=10000)
        self.assertLess(report["contrast"], 10)

    def test_empty_mask_is_safe(self):
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        report = evaluate_garment_reference(rgb, np.zeros((10, 10), bool), target_pixels=100)
        self.assertEqual(report["coverage"], 0.0)
        self.assertEqual(report["garment_pixels"], 0.0)


class AspectPaddingTests(unittest.TestCase):
    """세로로 긴 인스타 수집본(중앙값 약 1:2)이 잘리지 않는지 고정한다."""

    TARGET = (768, 1024)

    def test_tall_image_is_padded_not_cropped(self):
        image = Image.new("RGB", (465, 1433), "red")
        padded, box = pad_to_aspect(image, self.TARGET, (0, 0, 0))
        # 목표 비율 0.75에 맞춰 좌우로만 넓어지고 세로는 그대로여야 한다.
        self.assertEqual(padded.height, 1433)
        self.assertAlmostEqual(padded.width / padded.height, 0.75, places=2)
        self.assertGreater(padded.width, image.width)
        self.assertEqual(box[2] - box[0], 465)
        self.assertEqual(box[3] - box[1], 1433)

    def test_matching_aspect_is_untouched(self):
        image = Image.new("RGB", (768, 1024), "red")
        padded, box = pad_to_aspect(image, self.TARGET, (0, 0, 0))
        self.assertIs(padded, image)
        self.assertEqual(box, (0, 0, 768, 1024))

    def test_content_is_centered_and_recoverable(self):
        image = Image.new("RGB", (400, 1200), "red")
        padded, box = pad_to_aspect(image, self.TARGET, (0, 0, 0))
        # 원본 영역만 빨강, 여백은 검정이어야 한다.
        self.assertEqual(padded.getpixel(((box[0] + box[2]) // 2, 600)), (255, 0, 0))
        self.assertEqual(padded.getpixel((1, 600)), (0, 0, 0))
        # 렌더 크기로 줄인 결과에서 여백을 걷어내면 원본 크기로 돌아와야 한다.
        rendered = padded.resize(self.TARGET, Image.LANCZOS)
        restored = unpad_result(rendered, box, padded.size, image.size)
        self.assertEqual(restored.size, image.size)

    def test_unpad_is_noop_without_padding(self):
        rendered = Image.new("RGB", self.TARGET, "blue")
        restored = unpad_result(rendered, (0, 0, 768, 1024), self.TARGET, self.TARGET)
        self.assertIs(restored, rendered)


if __name__ == "__main__":
    unittest.main()
