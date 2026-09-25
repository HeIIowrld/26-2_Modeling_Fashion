"""Small synthetic checks for research-only layered-model source preservation."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from composite_layered_vton import composite_layered_result, unpad_generated  # noqa: E402


class LayeredCompositeTests(unittest.TestCase):
    def test_unpad_wide_photo(self) -> None:
        source = Image.new("RGB", (750, 1101), "red")
        from eval_layering_vton_outerwear import pad_image

        padded = pad_image(source)
        restored = unpad_generated(padded, source.size)
        self.assertEqual(restored.size, source.size)
        self.assertTrue(np.all(np.asarray(restored) == (255, 0, 0)))

    def test_unpad_narrow_photo(self) -> None:
        source = Image.new("RGB", (400, 1100), "green")
        from eval_layering_vton_outerwear import pad_image

        restored = unpad_generated(pad_image(source), source.size)
        self.assertEqual(restored.size, source.size)
        self.assertTrue(np.all(np.asarray(restored) == (0, 128, 0)))

    def test_only_editable_pixels_change(self) -> None:
        source = Image.new("RGB", (64, 96), "blue")
        generated = Image.new("RGB", (512, 896), "red")
        erase = np.zeros((96, 64), dtype=bool)
        erase[10:50, 10:50] = True
        protected = np.zeros_like(erase)
        protected[20:30, 20:30] = True
        result, metrics = composite_layered_result(source, generated, erase, protected)
        pixels = np.asarray(result)
        self.assertTrue(np.all(pixels[~erase] == (0, 0, 255)))
        self.assertTrue(np.all(pixels[protected] == (0, 0, 255)))
        self.assertTrue(np.any(np.all(pixels[40:45, 40:45] == (255, 0, 0), axis=2)))
        self.assertTrue(metrics["composite_outside_exact"])
        self.assertTrue(metrics["composite_protected_exact"])
        self.assertEqual(metrics["raw_outside_changed_fraction"], 1.0)

    def test_rejects_mask_size_mismatch(self) -> None:
        source = Image.new("RGB", (64, 96), "blue")
        generated = Image.new("RGB", (512, 896), "red")
        erase = np.ones((96, 64), dtype=bool)
        wrong = np.zeros((96, 63), dtype=bool)
        with self.assertRaises(ValueError):
            composite_layered_result(source, generated, erase, wrong)

    def test_accepts_binary_255_masks(self) -> None:
        source = Image.new("RGB", (64, 96), "blue")
        generated = Image.new("RGB", (512, 896), "red")
        erase = np.zeros((96, 64), dtype=np.uint8)
        erase[10:50, 10:50] = 255
        protected = np.zeros_like(erase)
        protected[20:30, 20:30] = 255
        result, metrics = composite_layered_result(source, generated, erase, protected)
        self.assertTrue(np.all(np.asarray(result)[protected != 0] == (0, 0, 255)))
        self.assertTrue(metrics["composite_protected_exact"])

    def test_same_size_generator_is_also_supported(self) -> None:
        source = Image.new("RGB", (64, 96), "blue")
        generated = Image.new("RGB", source.size, "red")
        erase = np.zeros((96, 64), dtype=bool)
        erase[10:50, 10:50] = True
        result, metrics = composite_layered_result(
            source, generated, erase, np.zeros_like(erase)
        )
        self.assertTrue(np.all(np.asarray(result)[~erase] == (0, 0, 255)))
        self.assertTrue(metrics["composite_outside_exact"])


if __name__ == "__main__":
    unittest.main()
