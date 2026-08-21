"""VTON 합성 신뢰도 게이트 회귀 테스트.

female_012 통제 실험(2026-08-20)에서 확인한 두 실패 모드를 고정한다:
- 레퍼런스 coverage 0.15 → 시스루 렌더링
- 하의 기장 gap=3 → 다리 전체가 옷 텍스처로 채워짐 (gap=1은 정상)
"""

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from PIL import Image

from catvton_tryon import (
    CatVTONTryOn,
    UNRELIABLE_LENGTH_GAP,
    bottom_length_gap,
    evaluate_garment_reference,
    pad_to_aspect,
    sleeve_length_gap,
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

    def test_fast_preset_halves_the_steps(self):
        preset = CatVTONTryOn.fast()
        self.assertEqual(preset.num_inference_steps, 25)
        # 프리셋은 스텝만 건드린다. 나머지는 기본값이어야 비교가 성립한다.
        self.assertEqual(preset.guidance_scale, CatVTONTryOn().guidance_scale)
        self.assertEqual(preset.repaint_blur_divisor, CatVTONTryOn().repaint_blur_divisor)

    def test_fast_preset_accepts_overrides(self):
        self.assertEqual(CatVTONTryOn.fast(num_inference_steps=10).num_inference_steps, 10)


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
