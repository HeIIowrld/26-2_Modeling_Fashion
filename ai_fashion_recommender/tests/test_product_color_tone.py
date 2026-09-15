from __future__ import annotations

import sys
import unittest
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_color_tone import COOL_TONE, WARM_TONE, classify_product_tone


class ProductColorToneTests(unittest.TestCase):
    def test_warm_and_cool_general_colors(self):
        warm = classify_product_tone(Image.new("RGB", (120, 120), (190, 112, 48)), "오렌지 니트")
        cool = classify_product_tone(Image.new("RGB", (120, 120), (48, 88, 174)), "코발트 셔츠")
        self.assertEqual(warm.tone, WARM_TONE)
        self.assertEqual(cool.tone, COOL_TONE)

    def test_warm_denim_uses_washing_and_stitch_rules(self):
        result = classify_product_tone(
            Image.new("RGB", (120, 120), (102, 119, 121)),
            "빈티지 브라운 워싱 오렌지 스티치 데님 팬츠",
            ["데님"],
        )
        self.assertTrue(result.is_denim)
        self.assertEqual(result.tone, WARM_TONE)
        self.assertTrue(any("데님 전용" in value for value in result.evidence))

    def test_cool_denim_uses_indigo_and_silver_rules(self):
        result = classify_product_tone(
            Image.new("RGB", (120, 120), (49, 72, 112)),
            "딥 인디고 실버 리벳 데님",
            ["데님"],
        )
        self.assertTrue(result.is_denim)
        self.assertEqual(result.tone, COOL_TONE)

    def test_nearly_achromatic_image_is_not_forced(self):
        result = classify_product_tone(Image.new("RGB", (120, 120), (128, 128, 128)), "그레이 티셔츠")
        self.assertEqual(result.tone, "판단 불가")


if __name__ == "__main__":
    unittest.main()
