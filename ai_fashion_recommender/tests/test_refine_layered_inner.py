"""Safety checks for the research-only old-inner carryover proposal."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from refine_layered_inner import coarse_product_contrast, residual_inner_mask  # noqa: E402


class ResidualInnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = np.full((200, 160, 3), 200, dtype=np.uint8)
        self.candidate = np.zeros_like(self.source)
        self.inner = np.zeros((200, 160), dtype=bool)
        self.inner[40:120, 35:125] = True
        self.forbidden = np.zeros_like(self.inner)

    def test_detects_only_large_unchanged_component(self) -> None:
        self.candidate[90:120, 45:115] = 200
        self.candidate[45:47, 45:47] = 200
        edit, selected, metrics = residual_inner_mask(
            self.source, self.candidate, self.inner, self.forbidden
        )
        self.assertEqual(metrics["selected_components"], 1)
        self.assertTrue(selected[100, 80])
        self.assertFalse(selected[45, 45])
        self.assertTrue(edit[88, 80])
        self.assertFalse(edit[20, 80])

    def test_protected_region_is_never_edited(self) -> None:
        self.candidate[90:120, 45:115] = 200
        self.forbidden[105:115, 70:90] = True
        edit, _, _ = residual_inner_mask(
            self.source, self.candidate, self.inner, self.forbidden
        )
        self.assertFalse(edit[self.forbidden].any())

    def test_component_box_covers_deoccluded_gap(self) -> None:
        # The old top can occupy the whole new gap even though the source
        # label only observed a U-shaped piece beneath the jacket.
        self.candidate[90:120, 45:115] = 200
        self.candidate[92:115, 65:95] = 0
        exact, _, _ = residual_inner_mask(
            self.source, self.candidate, self.inner, self.forbidden
        )
        boxed, _, metrics = residual_inner_mask(
            self.source, self.candidate, self.inner, self.forbidden,
            fill_component_boxes=True,
        )
        self.assertFalse(exact[100, 80])
        self.assertTrue(boxed[100, 80])
        self.assertTrue(metrics["fill_component_boxes"])

    def test_rejects_missing_residual(self) -> None:
        with self.assertRaisesRegex(ValueError, "No sufficiently large"):
            residual_inner_mask(self.source, self.candidate, self.inner, self.forbidden)

    def test_rejects_shape_mismatch(self) -> None:
        with self.assertRaisesRegex(ValueError, "Masks must match"):
            residual_inner_mask(
                self.source, self.candidate, self.inner[:, :-1], self.forbidden
            )

    def test_rejects_broad_edit(self) -> None:
        self.candidate[self.inner] = 200
        with self.assertRaisesRegex(ValueError, "too large"):
            residual_inner_mask(self.source, self.candidate, self.inner, self.forbidden)

    def test_product_contrast_distinguishes_black_from_white(self) -> None:
        dark_product = np.full((20, 20, 3), 20, dtype=np.uint8)
        white_product = np.full((20, 20, 3), 210, dtype=np.uint8)
        self.assertGreater(
            coarse_product_contrast(self.source, self.inner, dark_product), 45
        )
        self.assertLess(
            coarse_product_contrast(self.source, self.inner, white_product), 45
        )


if __name__ == "__main__":
    unittest.main()
