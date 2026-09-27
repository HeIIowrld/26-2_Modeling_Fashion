"""Small invariants for the experimental VTON v1.5 guarded composite."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from composite_vton15_outerwear import guarded_composite  # noqa: E402


class GuardedCompositeTests(unittest.TestCase):
    def test_source_is_exactly_preserved_outside_edit_mask(self):
        source = Image.new("RGB", (20, 20), (10, 20, 30))
        generated = Image.new("RGB", (10, 10), (200, 210, 220))
        mask = np.zeros((20, 20), dtype=np.uint8)
        mask[4:16, 4:16] = 255
        result, metrics = guarded_composite(
            source, generated, Image.fromarray(mask), feather_px=2
        )
        pixels = np.asarray(result)
        self.assertTrue(metrics["outside_mask_preserved"])
        self.assertTrue(np.all(pixels[~mask.astype(bool)] == [10, 20, 30]))
        self.assertEqual(tuple(pixels[10, 10]), (200, 210, 220))

    def test_rejects_wrong_mask_size_and_empty_mask(self):
        source = Image.new("RGB", (20, 20))
        generated = Image.new("RGB", (10, 10))
        with self.assertRaisesRegex(ValueError, "original image size"):
            guarded_composite(source, generated, Image.new("L", (10, 10), 255))
        with self.assertRaisesRegex(ValueError, "empty"):
            guarded_composite(source, generated, Image.new("L", (20, 20)))


if __name__ == "__main__":
    unittest.main()
