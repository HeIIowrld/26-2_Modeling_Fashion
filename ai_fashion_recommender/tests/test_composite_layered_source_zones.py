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
    build_source_zones, composite_zones, displaced_pocket_hand_release,
    unpad_generated_labels, validate_carried_pocket_hand_mask,
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

    def pocket_hand_sample(self):
        original = np.full((100, 80, 3), 245, np.uint8)
        labels = np.zeros((100, 80), np.uint8)
        coat = np.zeros((100, 80), bool)
        coat[20:70, 20:70] = True
        labels[coat] = 3
        labels[65:95, 20:70] = 6
        # The source hand is a sparse thumb/wrist island at the coat pocket.
        labels[52:56, 44:64] = 13
        labels[56:72, 60:64] = 13
        labels[74:94, 5:13] = 13  # A separate, normally hanging hand.
        original[labels == 3] = (55, 45, 35)
        original[labels == 6] = (50, 60, 75)
        original[labels == 13] = (200, 155, 125)
        generated = np.full_like(original, 120)
        generated_labels = np.zeros(labels.shape, np.uint8)
        generated_labels[20:65, 20:70] = 3
        generated_labels[54:72, 56:69] = 13
        generated_labels[74:94, 5:13] = 13
        return original, generated, labels, coat, generated_labels

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

    def test_displaced_coat_pocket_hand_is_replaced_but_other_hand_is_locked(self):
        original, generated, labels, coat, generated_labels = self.pocket_hand_sample()
        output, metrics, zones = composite_zones(
            Image.fromarray(original), Image.fromarray(generated), labels, coat,
            generated_labels=generated_labels, release_displaced_pocket_hand=True,
        )
        self.assertEqual(metrics["released_pocket_hand_regions"], 1)
        self.assertTrue(zones["released_pocket_hand"][54, 50])
        self.assertFalse(zones["identity"][54, 50])
        self.assertTrue(zones["identity"][80, 8])
        self.assertTrue(metrics["identity_exact"])
        pixels = np.asarray(output)
        self.assertNotEqual(tuple(pixels[54, 50]), tuple(original[54, 50]))
        np.testing.assert_array_equal(pixels[74:94, 5:13], original[74:94, 5:13])

    def test_pocket_release_fails_closed_without_displaced_generated_hand(self):
        original, _, labels, coat, generated_labels = self.pocket_hand_sample()
        generated_labels[54:72, 56:69] = 0
        generated_labels[52:56, 44:64] = 13
        generated_labels[56:72, 60:64] = 13
        released, count = displaced_pocket_hand_release(labels, generated_labels, coat)
        self.assertEqual(count, 0)
        self.assertFalse(released.any())
        with self.assertRaises(ValueError):
            build_source_zones(original, labels, coat,
                               release_displaced_pocket_hand=True)

    def test_pocket_release_rejects_noncoat_hand_and_extra_generated_hand(self):
        _, _, labels, coat, generated_labels = self.pocket_hand_sample()
        no_coat_at_hand = coat.copy()
        no_coat_at_hand[50:75, 40:70] = False
        released, count = displaced_pocket_hand_release(
            labels, generated_labels, no_coat_at_hand
        )
        self.assertEqual(count, 0)
        self.assertFalse(released.any())
        generated_labels[78:90, 65:79] = 13
        released, count = displaced_pocket_hand_release(labels, generated_labels, coat)
        self.assertEqual(count, 0)
        self.assertFalse(released.any())

    def test_pocket_release_rejects_two_ambiguous_displaced_hands(self):
        _, _, labels, coat, generated_labels = self.pocket_hand_sample()
        labels[74:94, 5:13] = 0
        generated_labels[74:94, 5:13] = 0
        labels[52:56, 18:38] = 13
        labels[56:72, 18:22] = 13
        generated_labels[54:72, 10:23] = 13
        released, count = displaced_pocket_hand_release(labels, generated_labels, coat)
        self.assertEqual(count, 0)
        self.assertFalse(released.any())

    def test_first_stage_hand_decision_survives_final_source_lock(self):
        original, generated, labels, coat, generated_labels = self.pocket_hand_sample()
        release, count = displaced_pocket_hand_release(labels, generated_labels, coat)
        self.assertEqual(count, 1)
        # A later try-on may hide the pocket hand entirely. Recomputing from
        # that final parser map would lock the old hand back into the image.
        final_labels = generated_labels.copy()
        final_labels[54:72, 56:69] = 0
        output, metrics, zones = composite_zones(
            Image.fromarray(original), Image.fromarray(generated), labels, coat,
            generated_labels=final_labels, carried_pocket_hand_mask=release,
        )
        self.assertTrue(metrics["carried_pocket_hand_release"])
        self.assertEqual(metrics["released_pocket_hand_regions"], 1)
        self.assertTrue(zones["released_pocket_hand"][54, 50])
        self.assertNotEqual(tuple(np.asarray(output)[54, 50]), tuple(original[54, 50]))
        np.testing.assert_array_equal(np.asarray(output)[74:94, 5:13], original[74:94, 5:13])

    def test_carried_hand_mask_rejects_face_or_broad_release(self):
        _, _, labels, coat, generated_labels = self.pocket_hand_sample()
        release, count = displaced_pocket_hand_release(labels, generated_labels, coat)
        self.assertEqual(count, 1)
        unsafe = release.copy()
        labels[10:20, 30:40] = 1
        unsafe[12, 35] = True
        with self.assertRaises(ValueError):
            validate_carried_pocket_hand_mask(labels, coat, unsafe)
        with self.assertRaises(ValueError):
            validate_carried_pocket_hand_mask(labels, coat, np.ones_like(release))


if __name__ == "__main__":
    unittest.main()
