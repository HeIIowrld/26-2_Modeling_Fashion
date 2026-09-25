"""Invariants for the isolated three-zone outerwear composite."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from composite_layered_source_zones import (  # noqa: E402
    build_source_zones, composite_zones, unpad_generated_labels,
)


class SourceZoneTests(unittest.TestCase):
    def sample(self):
        original = np.full((100, 80, 3), 245, np.uint8)
        labels = np.zeros((100, 80), np.uint8)
        labels[10:25, 30:50] = 1  # face
        labels[25:65, 20:60] = 3  # coat/top
        labels[65:95, 28:52] = 6  # visible pants
        original[labels == 1] = (210, 170, 145)
        original[labels == 3] = (20, 30, 70)
        original[labels == 6] = (70, 90, 120)
        coat = labels == 3
        return original, labels, coat

    def test_identity_pants_and_distant_studio_background_are_exact(self):
        original, labels, coat = self.sample()
        generated = np.full_like(original, 125)
        output, metrics, zones = composite_zones(
            Image.fromarray(original), Image.fromarray(generated), labels, coat
        )
        pixels = np.asarray(output)
        self.assertTrue(metrics["identity_exact"])
        self.assertTrue(metrics["observed_pants_exact"])
        self.assertTrue(metrics["distant_background_exact"])
        self.assertTrue(zones["distant_background"][0, 0])
        np.testing.assert_array_equal(pixels[labels == 6], original[labels == 6])
        self.assertNotEqual(tuple(pixels[40, 40]), tuple(original[40, 40]))

    def test_complex_background_does_not_claim_exact_distant_backdrop(self):
        original, labels, coat = self.sample()
        original[:, :10] = (45, 70, 90)
        original[:, -10:] = (170, 130, 80)
        _, zones, metrics = build_source_zones(original, labels, coat)
        self.assertFalse(metrics["studio_background_reliable"])
        self.assertFalse(zones["distant_background"].any())

    def test_invalid_coat_seed_fails_closed(self):
        original, labels, coat = self.sample()
        with self.assertRaises(ValueError):
            build_source_zones(original, labels, np.zeros_like(coat))

    def test_new_top_can_occlude_original_pants_without_blue_patch(self):
        original, labels, coat = self.sample()
        generated = np.full_like(original, 125)
        generated_labels = np.zeros(labels.shape, np.uint8)
        generated_labels[60:78, 25:55] = 3
        output, metrics, zones = composite_zones(
            Image.fromarray(original), Image.fromarray(generated), labels, coat,
            generated_labels=generated_labels,
        )
        pixels = np.asarray(output)
        self.assertGreater(metrics["occluded_pants_fraction"], 0)
        self.assertTrue(zones["occluded_pants"][70, 40])
        self.assertFalse(zones["observed_pants"][70, 40])
        self.assertNotEqual(tuple(pixels[70, 40]), tuple(original[70, 40]))
        np.testing.assert_array_equal(pixels[88, 40], original[88, 40])

    def test_unpad_generated_labels_keeps_integer_classes(self):
        labels = np.zeros((896, 512), np.uint8)
        labels[200:400, 100:300] = 3
        projected = unpad_generated_labels(Image.fromarray(labels), (750, 1101))
        self.assertEqual(projected.shape, (1101, 750))
        self.assertLessEqual(set(np.unique(projected)), {0, 3})


if __name__ == "__main__":
    unittest.main()
