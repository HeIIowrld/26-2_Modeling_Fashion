from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import torch
from shoe_model import (SHOE_LABELS, ShoePredictor, build_shoe_heads, shoe_classification_loss,
                        save_shoe_checkpoint, shoe_crop)


class ShoeHeadTests(unittest.TestCase):
    def test_no_checkpoint_does_not_predict(self):
        from unittest.mock import patch
        from fashion_model import FashionClassifier
        import os
        with patch.dict(os.environ, {}, clear=True):
            result = FashionClassifier(enabled=False).predict_shoes(None, None)
        self.assertEqual(result["status"], "not_trained")
        self.assertFalse(result["accepted"])

    def test_apparel_checkpoint_schema_unchanged(self):
        from fashion_attribute_model import build_attribute_heads, save_attribute_checkpoint, load_attribute_heads
        from fashion_attribute_schema import ATTRIBUTE_TASKS
        heads = build_attribute_heads(16, hidden_dim=8, dropout=0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "apparel.pt"
            save_attribute_checkpoint(path, heads, backbone_model_id="test")
            loaded, metadata = load_attribute_heads(path)
            self.assertEqual(set(metadata["tasks"]), set(ATTRIBUTE_TASKS))
            self.assertNotIn("shoes", loaded.heads)
            for name, value in heads.state_dict().items():
                torch.testing.assert_close(value, loaded.state_dict()[name])

    def test_trainable_and_checkpoint_roundtrip(self):
        heads = build_shoe_heads(16, hidden_dim=8, dropout=0)
        features = torch.randn(3, 16)
        logits = heads(features)
        self.assertEqual(tuple(logits.shape), (3, len(SHOE_LABELS)))
        loss = shoe_classification_loss(logits, torch.tensor([0, -1, 2]))
        loss.backward()
        self.assertIsNotNone(heads[-1].weight.grad)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shoes.pt"
            save_shoe_checkpoint(path, heads, backbone_model_id="test", training_examples=2, threshold=.8)
            predictor = ShoePredictor(path, model_id="test")
            torch.testing.assert_close(heads(features), predictor.heads(features))
            with self.assertRaises(ValueError):
                ShoePredictor(path, model_id="different-backbone")
            with self.assertRaises(ValueError):
                save_shoe_checkpoint(path, heads, backbone_model_id="test", training_examples=0, threshold=.8)

    def test_crop_requires_visible_feet(self):
        rgb = np.zeros((50, 50, 3), dtype=np.uint8)
        segmentation = np.zeros((50, 50), dtype=np.uint8)
        self.assertIsNone(shoe_crop(rgb, segmentation))
        segmentation[30:45, 10:30] = 15
        self.assertEqual(shoe_crop(rgb, segmentation).size, (20, 15))

    def test_barefoot_is_not_accepted(self):
        heads = build_shoe_heads(16, hidden_dim=8)
        with torch.no_grad():
            heads[-1].weight.zero_()
            heads[-1].bias.zero_()
            heads[-1].bias[SHOE_LABELS.index("맨발")] = 100
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "shoes.pt"
            save_shoe_checkpoint(path, heads, backbone_model_id="test", training_examples=1, threshold=.8)
            result = ShoePredictor(path, model_id="test").predict(torch.zeros(1, 16))
            self.assertFalse(result["accepted"])
            self.assertEqual(result["item_type"], "분석 보류")


if __name__ == "__main__":
    unittest.main()
