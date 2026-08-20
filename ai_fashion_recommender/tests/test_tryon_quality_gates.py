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
sys.path.insert(0, str(ROOT))

from catvton_tryon import (
    UNRELIABLE_LENGTH_GAP,
    bottom_length_gap,
    evaluate_garment_reference,
    sleeve_length_gap,
)


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


if __name__ == "__main__":
    unittest.main()
