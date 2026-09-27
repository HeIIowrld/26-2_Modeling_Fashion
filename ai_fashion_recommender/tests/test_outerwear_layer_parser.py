"""Regression tests for the optional coat/inner parsing experiment."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_outerwear_layer_parser import evaluate_one  # noqa: E402
from calibrate_outerwear_layer_parser import evaluate_thresholds, grouped_folds  # noqa: E402
from prepare_outerwear_coat_proposal import coat_proposal  # noqa: E402
from prepare_outerwear_second_mask import generated_top_proposal  # noqa: E402
from outerwear_layer_parser import OuterwearLayerParser, preprocess_schp  # noqa: E402


class _Session:
    def __init__(self):
        self.received = None

    def run(self, outputs, inputs):
        self.received = (outputs, inputs)
        logits = np.zeros((1, 20, 473, 473), dtype=np.float32)
        logits[:, 7, :, :236] = 2.0
        logits[:, 5, :, 236:] = 2.0
        return [logits]


class _BadSession:
    def run(self, outputs, inputs):
        return [np.zeros((1, 19, 473, 473), dtype=np.float32)]


class OuterwearLayerParserTests(unittest.TestCase):
    def test_preprocessing_uses_published_rgb_normalization(self):
        pixels = preprocess_schp(Image.new("RGB", (12, 20), (0, 255, 128)))
        self.assertEqual(pixels.shape, (1, 3, 473, 473))
        self.assertEqual(pixels.dtype, np.float32)
        self.assertAlmostEqual(float(pixels[0, 0, 0, 0]), -0.406 / 0.225, places=5)
        self.assertAlmostEqual(float(pixels[0, 1, 0, 0]), (1 - 0.456) / 0.224, places=5)

    def test_coat_and_upper_labels_remain_separate_at_source_resolution(self):
        session = _Session()
        parser = OuterwearLayerParser("unused.onnx", session=session)
        predicted = parser.predict(Image.new("RGB", (80, 120), "white"))
        logits = parser.predict_logits(Image.new("RGB", (80, 120), "white"))
        self.assertEqual(predicted.shape, (120, 80))
        self.assertEqual(logits.shape, (20, 120, 80))
        self.assertGreater(float(logits[7, 60, 10]), float(logits[5, 60, 10]))
        self.assertEqual(predicted[60, 10], 7)
        self.assertEqual(predicted[60, 70], 5)
        self.assertEqual(session.received[0], ["logits"])
        self.assertEqual(session.received[1]["pixel_values"].shape, (1, 3, 473, 473))

    def test_evaluation_detects_outer_to_inner_confusion(self):
        official = np.zeros((4, 4), dtype=np.uint8)
        official[:, :2] = 2
        official[:, 2:] = 1
        predicted = np.full((4, 4), 5, dtype=np.uint8)
        predicted[:2, :2] = 7
        report = evaluate_one(official, predicted)
        self.assertEqual(report["outer_as_coat_recall"], 0.5)
        self.assertEqual(report["outer_as_upper_fraction"], 0.5)
        self.assertEqual(report["inner_as_coat_fraction"], 0.0)
        self.assertEqual(report["outer_coat_precision"], 1.0)

    def test_unexpected_model_taxonomy_fails_closed(self):
        parser = OuterwearLayerParser("unused.onnx", session=_BadSession())
        with self.assertRaises(RuntimeError):
            parser.predict(Image.new("RGB", (80, 120), "white"))

    def test_generated_top_proposal_only_expands_upper_clothes(self):
        labels = np.zeros((9, 9), dtype=np.uint8)
        labels[4, 4] = 5
        labels[0, 0] = 7
        proposal = generated_top_proposal(labels)
        self.assertTrue(proposal[4, 4])
        self.assertTrue(proposal[4, 6])
        self.assertFalse(proposal[0, 0])
        with self.assertRaises(ValueError):
            generated_top_proposal(labels, kernel_size=4)
        with self.assertRaises(ValueError):
            generated_top_proposal(np.zeros((9, 9), dtype=np.uint8))

    def test_coat_proposal_keeps_only_coat_label(self):
        labels = np.zeros((8, 8), dtype=np.uint8)
        labels[2, 2] = 7
        labels[3, 3] = 5
        mask = coat_proposal(labels)
        self.assertTrue(mask[2, 2])
        self.assertFalse(mask[3, 3])
        self.assertTrue(coat_proposal(labels, kernel_size=3)[2, 3])
        with self.assertRaises(ValueError):
            coat_proposal(labels, kernel_size=4)
        with self.assertRaises(ValueError):
            coat_proposal(np.zeros((8, 8), dtype=np.uint8))

    def test_threshold_sweep_matches_argmax_at_zero(self):
        official = np.zeros((4, 4), dtype=np.uint8)
        official[:, :2] = 2
        official[:, 2:] = 1
        logits = np.zeros((20, 4, 4), dtype=np.float32)
        logits[7, :, :2] = 2
        logits[5, :, 2:] = 2
        rows = evaluate_thresholds(official, logits, "WOMEN-Jackets_Coats-id_00000001-01_4_full.jpg")
        baseline = next(row for row in rows if row["threshold"] == 0.0)
        lenient = next(row for row in rows if row["threshold"] == -4.0)
        self.assertEqual(baseline["outer_true_positive"], 8)
        self.assertEqual(baseline["inner_false_coat"], 0)
        self.assertEqual(lenient["inner_false_coat"], 8)

    def test_grouped_folds_keep_same_item_together(self):
        names = [
            "WOMEN-Jackets_Coats-id_00007431-03_4_full.jpg",
            "WOMEN-Jackets_Coats-id_00007431-04_4_full.jpg",
        ]
        rows = [{"image": name, "official_outer_pixels": 1000} for name in names]
        folds = grouped_folds(rows)
        self.assertEqual(len(folds), 1)


if __name__ == "__main__":
    unittest.main()
