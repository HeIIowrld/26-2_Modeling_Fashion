"""Small exact-label checks for the oracle mask morphology diagnostic."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_upper_mask_morphology import evaluate, kernel_size, morph_mask  # noqa: E402
from eval_catvton_upper_morphology import build_custom_target_mask  # noqa: E402
from catvton_tryon import _dilate_mask, _solidify_mask  # noqa: E402


class UpperMaskMorphologyTests(unittest.TestCase):
    def test_uses_catvton_odd_kernel_rule(self):
        self.assertEqual(kernel_size((1101, 750), 0.05), 37)
        self.assertEqual(kernel_size((1101, 750), 0.03), 23)

    def test_standard_variant_matches_catvton_helpers(self):
        mask = np.zeros((1101, 750), dtype=np.uint8)
        mask[200:620, 210:500] = 1
        mask[290:345, 330:380] = 0
        expected = _dilate_mask(_solidify_mask(mask)) > 0
        np.testing.assert_array_equal(morph_mask(mask, 0.05, 0.03), expected)

    def test_expansion_counts_only_pixels_outside_oracle_top_outer(self):
        seg = np.zeros((40, 40), dtype=np.uint8)
        seg[10:30, 10:30] = 2
        seg[15:25, 15:25] = 1
        seg[29:35, 14:26] = 5
        rows = {row["variant"]: row for row in evaluate(seg, "fixture.png")}
        self.assertEqual(rows["raw"]["added_pixels"], 0)
        self.assertEqual(rows["raw"]["raw_pixels"], 388)
        self.assertEqual(rows["close_5_dilate_3"]["removed_raw_pixels"], 0)
        self.assertGreater(rows["close_5_dilate_3"]["added_background"], 0)
        self.assertGreater(rows["close_5_dilate_3"]["added_pants_leggings"], 0)

    def test_morphology_rejects_non_mask(self):
        with self.assertRaisesRegex(ValueError, "two-dimensional"):
            morph_mask(np.zeros((4, 4, 3)), 0.05, 0.03)

    def test_custom_target_unions_old_inner_without_touching_protected(self):
        pose = np.zeros((40, 40), dtype=bool)
        pose[10:20, 10:20] = True
        inner = np.zeros_like(pose)
        inner[18:28, 15:25] = True
        protected = np.zeros_like(pose)
        protected[12:16, 12:16] = True
        custom = build_custom_target_mask(pose, inner, protected)
        self.assertTrue(custom[25, 20])
        self.assertFalse(custom[13, 13])
        self.assertFalse(custom[0, 0])

    def test_custom_target_rejects_broad_or_mismatched_mask(self):
        broad = np.ones((40, 40), dtype=bool)
        empty = np.zeros_like(broad)
        with self.assertRaisesRegex(ValueError, "too large"):
            build_custom_target_mask(broad, empty, empty)
        with self.assertRaisesRegex(ValueError, "equal-sized"):
            build_custom_target_mask(empty, empty[:, :-1], empty)


if __name__ == "__main__":
    unittest.main()
