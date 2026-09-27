import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from refine_studio_vton_background import refine_background  # noqa: E402


class RefineStudioVtonBackgroundTests(unittest.TestCase):
    def setUp(self):
        self.source = np.full((120, 100, 3), 248, dtype=np.uint8)
        self.source_labels = np.zeros((120, 100), dtype=np.uint8)
        self.source[25:105, 25:75] = (150, 120, 100)
        self.source_labels[25:105, 25:75] = 3
        self.candidate = self.source.copy()
        self.candidate_labels = self.source_labels.copy()
        self.candidate[25:105, 25:75] = (225, 224, 222)
        self.candidate_labels[25:105, 25:75] = 3
        self.candidate[35:95, 15:25] = (228, 227, 225)

    def test_removes_connected_halo_without_touching_garment(self):
        result, metrics = refine_background(
            self.source, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertTrue(metrics["studio_reliable"])
        self.assertGreater(metrics["changed_pixels"], 0)
        self.assertGreater(result[55, 17, 0], self.candidate[55, 17, 0])
        self.assertTrue(np.array_equal(result[45, 45], self.candidate[45, 45]))

    def test_does_not_erase_dark_or_isolated_parser_holes(self):
        self.candidate[40:45, 5:10] = (40, 40, 40)
        self.candidate_labels[55:65, 45:55] = 0
        self.candidate[55:65, 45:55] = (230, 230, 230)
        result, _ = refine_background(
            self.source, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertEqual(tuple(result[42, 7]), (40, 40, 40))
        self.assertEqual(tuple(result[60, 50]), (230, 230, 230))

    def test_keeps_pixels_already_exactly_from_source(self):
        self.source[60, 17] = (235, 234, 233)
        self.candidate[60, 17] = self.source[60, 17]
        result, _ = refine_background(
            self.source, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertTrue(np.array_equal(result[60, 17], self.source[60, 17]))

    def test_non_studio_fails_closed(self):
        self.source[:, :12] = (40, 130, 70)
        self.source[:, -12:] = (40, 130, 70)
        result, metrics = refine_background(
            self.source, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertFalse(metrics["studio_reliable"])
        self.assertTrue(np.array_equal(result, self.candidate))

    def test_gray_studio_fails_closed_and_source_identity_is_exact(self):
        gray = self.source.copy()
        gray[:] = (215, 215, 217)
        output, metrics = refine_background(
            gray, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertFalse(metrics["studio_reliable"])
        self.assertTrue(np.array_equal(output, self.candidate))

        self.source_labels[50:60, 15:20] = 13
        self.candidate[50:60, 15:20] = (230, 220, 210)
        output, _ = refine_background(
            self.source, self.candidate, self.source_labels, self.candidate_labels
        )
        self.assertTrue(np.array_equal(output[50:60, 15:20], self.candidate[50:60, 15:20]))

    def test_mismatched_labels_raise(self):
        with self.assertRaises(ValueError):
            refine_background(self.source, self.candidate, self.source_labels[:10], self.candidate_labels)


if __name__ == "__main__":
    unittest.main()
