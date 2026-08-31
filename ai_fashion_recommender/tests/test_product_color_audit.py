import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from audit_product_colors import analyze_image_color


class ProductColorAuditTests(unittest.TestCase):
    def _image(self, colors: list[str]) -> Path:
        directory = Path(tempfile.mkdtemp())
        path = directory / "product.jpg"
        image = Image.new("RGB", (360, 420), "white")
        draw = ImageDraw.Draw(image)
        width = 240 // len(colors)
        for index, color in enumerate(colors):
            draw.rectangle(
                (60 + index * width, 70, 60 + (index + 1) * width, 350),
                fill=color,
            )
        image.save(path)
        return path

    def test_single_color_product_can_correct_catalog_metadata(self):
        result = analyze_image_color(self._image(["#d21f30"]), "그린")
        self.assertEqual(result["image_color"], "레드")
        self.assertEqual(result["override"], "true")
        self.assertGreaterEqual(float(result["confidence"]), 0.90)

    def test_matching_catalog_color_is_not_marked_as_an_override(self):
        result = analyze_image_color(self._image(["#d21f30"]), "레드")
        self.assertEqual(result["override"], "false")
        self.assertEqual(result["reason"], "catalog_agrees")

    def test_multicolor_graphic_is_not_automatically_relabelled(self):
        result = analyze_image_color(self._image(["#d21f30", "#2769c7"]), "그린")
        self.assertEqual(result["override"], "false")

    def test_neutral_mismatch_is_left_for_manual_review(self):
        result = analyze_image_color(self._image(["#202020"]), "그레이")
        self.assertEqual(result["image_color"], "블랙")
        self.assertEqual(result["override"], "false")
        self.assertTrue(result["reason"].startswith("manual_review_"))

    def test_similar_chromatic_color_is_left_for_manual_review(self):
        result = analyze_image_color(self._image(["#8b263e"]), "레드")
        self.assertEqual(result["image_color"], "버건디")
        self.assertEqual(result["override"], "false")


if __name__ == "__main__":
    unittest.main()
