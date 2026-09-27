"""Small invariants for the item-held-out, research-only layer correction."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
from eval_layer_corrector import (  # noqa: E402
    feature_maps, grouped_folds, item_key, predict, save_unlabeled_review,
    score_one, separate_targets, summarize,
)
from outerwear_layer_parser import OuterwearLayerParser  # noqa: E402


class _Session:
    def run(self, outputs, inputs):
        logits = np.zeros((1, 20, 473, 473), dtype=np.float32)
        logits[:, 7, :, :235] = 3.0
        logits[:, 5, :, 235:] = 3.0
        return [logits]


class LayerCorrectorTests(unittest.TestCase):
    def test_item_groups_cannot_cross_folds(self):
        first = "MEN-Jackets_Vests-id_00005047-01_4_full.jpg"
        second = "MEN-Jackets_Vests-id_00005047-02_4_full.jpg"
        other = "WOMEN-Jackets_Coats-id_00001582-03_4_full.jpg"
        self.assertEqual(item_key(first), item_key(second))
        records = [{"image": name, "outer_pixels": 1200} for name in (first, second, other)]
        folds = grouped_folds(records, 2)
        self.assertEqual(len(folds), 2)
        self.assertEqual(folds[item_key(first)], folds[item_key(second)])

    def test_all_views_of_target_item_are_excluded_from_development(self):
        first = "MEN-Jackets_Vests-id_00005047-01_4_full.jpg"
        second = "MEN-Jackets_Vests-id_00005047-02_4_full.jpg"
        unrelated = "MEN-Shirts_Polos-id_00000193-07_4_full.jpg"
        development, targets = separate_targets(
            [{"image": name} for name in (first, second, unrelated)], (first,)
        )
        self.assertEqual([row["image"] for row in development], [unrelated])
        self.assertEqual([row["image"] for row in targets], [first, second])

    def test_features_keep_inner_and_coat_as_candidates(self):
        parser = OuterwearLayerParser("unused.onnx", session=_Session())
        features, candidate, baseline = feature_maps(Image.new("RGB", (64, 96), "white"), parser, 32)
        self.assertEqual(features.shape[:2], (48, 32))
        self.assertTrue(candidate.all())
        self.assertTrue(baseline[:, :14].all())
        self.assertFalse(baseline[:, 18:].any())
        self.assertTrue(np.isfinite(features).all())

    def test_scoring_penalizes_old_inner_as_coat(self):
        official = np.ones((60, 60), dtype=np.uint8)
        official[:, :30] = 2
        prediction = np.zeros((60, 60), dtype=bool)
        prediction[:, :45] = True
        row = score_one(official, prediction, "case.jpg", "our_corrector", "fold_0")
        self.assertEqual(row["outer_true_positive"], 1800)
        self.assertEqual(row["inner_false_coat"], 900)
        self.assertEqual(summarize([row])["outer_iou_macro"], round(1800 / 2700, 5))

    def test_empty_candidate_is_an_empty_coat_prediction(self):
        class NeverCalled:
            def predict_proba(self, features):
                raise AssertionError("No candidate pixels should reach the model")

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "empty.npz"
            np.savez(cache, features=np.zeros((3, 4, 20), np.float16),
                     candidate=np.zeros((3, 4), bool), baseline=np.zeros((3, 4), bool))
            baseline, corrected, vetoed = predict(cache, NeverCalled())
        self.assertFalse(baseline.any())
        self.assertFalse(corrected.any())
        self.assertFalse(vetoed.any())

    def test_conservative_veto_only_removes_very_unlikely_coat_pixels(self):
        class Probabilities:
            def predict_proba(self, features):
                p = np.asarray((0.05, 0.20, 0.80), dtype=np.float32)
                return np.column_stack((1 - p, p))

        with tempfile.TemporaryDirectory() as directory:
            cache = Path(directory) / "three.npz"
            np.savez(cache, features=np.zeros((1, 3, 20), np.float16),
                     candidate=np.ones((1, 3), bool), baseline=np.asarray([[1, 1, 0]], bool))
            baseline, corrected, vetoed = predict(cache, Probabilities())
        np.testing.assert_array_equal(baseline, [[1, 1, 0]])
        np.testing.assert_array_equal(corrected, [[0, 0, 1]])
        np.testing.assert_array_equal(vetoed, [[0, 1, 0]])

    def test_unlabeled_comparison_keeps_original_as_first_panel(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.jpg"
            output = Path(directory) / "review.jpg"
            Image.new("RGB", (80, 50), (120, 130, 140)).save(source)
            masks = (np.zeros((25, 40), bool),) * 3
            save_unlabeled_review(source, masks, output)
            with Image.open(source) as opened:
                original = np.asarray(opened.convert("RGB"))
            with Image.open(output) as opened:
                review = np.asarray(opened.convert("RGB"))
        self.assertEqual(review.shape[:2], (50, 320))
        self.assertLess(np.abs(review[:, :80].astype(int) - original.astype(int)).mean(), 3)


if __name__ == "__main__":
    unittest.main()
