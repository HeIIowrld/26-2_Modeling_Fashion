"""Regression checks for narrow studio seam restoration."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from restore_accessory_background import harmonize_accessory_background  # noqa: E402


class AccessoryBackgroundTests(unittest.TestCase):
    def test_dark_strap_is_replaced_with_studio_color_only_inside_edit(self):
        source = np.full((300, 220, 3), 250, np.uint8)
        source[45:95, 80:87] = (25, 25, 25)
        source[40:100, 140:170] = (55, 90, 140)  # lower clothing
        edited = source.copy()
        edited[40:100, 70:100] = (240, 240, 242)
        allowed = np.zeros((300, 220), bool)
        allowed[40:100, 70:100] = True
        accessory = np.zeros((300, 220), bool)
        accessory[45:95, 80:87] = True
        result, metrics = harmonize_accessory_background(source, edited, allowed, accessory)
        self.assertTrue(metrics["applied"])
        self.assertTrue(metrics["outside_edit_exact_to_input"])
        np.testing.assert_array_equal(result[60, 82], [250, 250, 250])
        np.testing.assert_array_equal(result[60, 150], source[60, 150])
        np.testing.assert_array_equal(result[~allowed], edited[~allowed])

    def test_empty_accessory_fails_closed(self):
        source = np.full((40, 40, 3), 245, np.uint8)
        allowed = np.ones((40, 40), bool)
        result, metrics = harmonize_accessory_background(
            source, source.copy(), allowed, np.zeros((40, 40), bool)
        )
        self.assertFalse(metrics["applied"])
        np.testing.assert_array_equal(result, source)


if __name__ == "__main__":
    unittest.main()
