"""Safety checks for experimental VTON hair/coat boundary release."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from eval_fashn_vton15_outerwear import (  # noqa: E402
    hair_coat_release_mask,
    studio_background_condition,
)


class HairCoatReleaseTests(unittest.TestCase):
    def test_only_nearby_hair_pixels_with_coat_proposal_are_released(self):
        seg = np.zeros((40, 40), dtype=np.uint8)
        seg[7:12, 19:22] = 2
        seg[30:35, 30:35] = 2
        base = np.zeros_like(seg, dtype=bool)
        base[12:25, 15:25] = True
        coat = np.zeros_like(base)
        coat[10, 20] = True
        coat[32, 32] = True
        coat[13, 20] = True  # Already editable.
        coat[10, 19] = True
        release = hair_coat_release_mask(seg, base, coat, near_ratio=0.05)
        self.assertTrue(release[10, 20])
        self.assertTrue(release[10, 19])
        self.assertFalse(release[32, 32])
        self.assertFalse(release[13, 20])
        self.assertEqual(int(release.sum()), 2)

    def test_rejects_oversize_or_mismatched_proposals(self):
        seg = np.full((20, 20), 2, dtype=np.uint8)
        base = np.zeros((20, 20), dtype=bool)
        base[8:12, 8:12] = True
        with self.assertRaisesRegex(ValueError, "image-area cap"):
            hair_coat_release_mask(seg, base, np.ones_like(base), near_ratio=0.1, max_fraction=0.01)
        with self.assertRaisesRegex(ValueError, "identical dimensions"):
            hair_coat_release_mask(seg, base, np.zeros((19, 20), dtype=bool))

    def test_studio_condition_only_changes_outer_edit_ring(self):
        original = np.full((20, 20, 3), 220, dtype=np.uint8)
        agnostic = original.copy()
        edit = np.zeros((20, 20), dtype=bool)
        edit[4:16, 4:16] = True
        agnostic[edit] = 127
        target = np.zeros_like(edit)
        target[7:13, 7:13] = True
        conditioned, ring = studio_background_condition(agnostic, original, edit, target)
        self.assertEqual(int(ring.sum()), int(edit.sum() - target.sum()))
        self.assertTrue(np.all(conditioned[ring] == 220))
        self.assertTrue(np.all(conditioned[target] == 127))
        np.testing.assert_array_equal(conditioned[~edit], original[~edit])


if __name__ == "__main__":
    unittest.main()
