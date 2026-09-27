"""Safety checks for the isolated fuzzy-coat-boundary proposal."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from eval_coat_boundary_grabcut import expand_coat_boundary, propose_dark_straps, score  # noqa: E402


class CoatBoundaryTests(unittest.TestCase):
    def test_refinement_never_erases_baseline_pixels(self):
        image = np.full((120, 160, 3), 250, np.uint8)
        image[25:95, 40:120] = (30, 45, 75)
        baseline = np.zeros((120, 160), bool)
        baseline[30:90, 45:115] = True
        refined, metrics = expand_coat_boundary(
            image, baseline, halo_ratio=0.10, max_added_fraction=0.1, iterations=2
        )
        self.assertTrue(np.all(refined[baseline]))
        self.assertTrue(metrics["accepted"])

    def test_invalid_or_empty_seed_fails_closed(self):
        image = np.zeros((100, 100, 3), np.uint8)
        empty = np.zeros((100, 100), bool)
        with self.assertRaises(ValueError):
            expand_coat_boundary(image, empty, halo_ratio=0.05)
        with self.assertRaises(ValueError):
            expand_coat_boundary(image, np.ones((100, 100), bool), halo_ratio=0.50)

    def test_score_penalizes_inner_overreach(self):
        official = np.ones((10, 10), np.uint8)
        official[:, :5] = 2
        pred = np.zeros((10, 10), bool)
        pred[:, :6] = True
        result = score(official, pred)
        self.assertEqual(result["outer_recall"], 1.0)
        self.assertEqual(result["outer_iou"], round(50 / 60, 5))
        self.assertEqual(result["inner_false_coat"], 0.2)

    def test_dark_studio_straps_are_separate_from_coat_seed(self):
        image = np.full((300, 220, 3), 245, np.uint8)
        coat = np.zeros((300, 220), bool)
        coat[30:180, 35:185] = True
        image[coat] = (25, 25, 35)
        image[180:225, 40:45] = (25, 25, 35)
        image[180:230, 175:180] = (25, 25, 35)
        proposal, diagnostics = propose_dark_straps(image, coat)
        self.assertTrue(diagnostics["accepted"])
        self.assertTrue(proposal[200, 42])
        self.assertTrue(proposal[200, 177])
        self.assertFalse(proposal[50, 70])

    def test_dark_strap_detector_abstains_on_nonstudio_background(self):
        image = np.full((200, 160, 3), 80, np.uint8)
        coat = np.zeros((200, 160), bool)
        coat[30:110, 35:125] = True
        proposal, diagnostics = propose_dark_straps(image, coat)
        self.assertFalse(proposal.any())
        self.assertEqual(diagnostics["reason"], "complex_background")


if __name__ == "__main__":
    unittest.main()
