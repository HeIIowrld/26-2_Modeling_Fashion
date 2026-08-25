import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layering_model import (
    LAYERING_ROI_NAMES,
    build_layering_heads,
    layering_roi_crops,
    load_layering_heads,
    save_layering_checkpoint,
)
from layering_training import (
    LayeringTrainingConfig,
    load_layering_csv,
    train_layering_heads,
)


class LayeringHeadTests(unittest.TestCase):
    def test_roi_extractor_exposes_all_planned_visual_cues(self):
        image = Image.new("RGB", (200, 300), "gray")
        rois = layering_roi_crops(image)
        self.assertEqual(tuple(rois), LAYERING_ROI_NAMES)
        self.assertEqual(rois["global"].size, image.size)
        self.assertLess(rois["neck_collar"].height, image.height)
        self.assertLess(rois["placket"].width, image.width)

    def test_head_outputs_layer_state_and_two_component_categories(self):
        heads = build_layering_heads(8, hidden_dim=8, dropout=0.0)
        output = heads(torch.randn(3, len(LAYERING_ROI_NAMES), 8))
        self.assertEqual(output["layering"].shape, (3, 2))
        self.assertEqual(output["inner_category"].shape[0], 3)
        self.assertEqual(output["outer_category"].shape, output["inner_category"].shape)
        self.assertTrue(torch.allclose(output["roi_attention"].sum(dim=1), torch.ones(3)))

    def test_checkpoint_round_trip_preserves_multi_roi_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layering.pt"
            heads = build_layering_heads(8, hidden_dim=8, dropout=0.0)
            save_layering_checkpoint(path, heads, backbone_model_id="test-backbone")
            loaded, metadata = load_layering_heads(path)
            self.assertEqual(metadata["backbone_model_id"], "test-backbone")
            self.assertEqual(tuple(loaded.roi_names), LAYERING_ROI_NAMES)

    def test_csv_accepts_binary_state_and_optional_components(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 30), "white").save(root / "single.jpg")
            Image.new("RGB", (20, 30), "white").save(root / "layered.jpg")
            csv_path = root / "labels.csv"
            csv_path.write_text(
                "image_path,split,is_layered,inner_category,outer_category\n"
                "single.jpg,train,0,,\n"
                "layered.jpg,val,겹쳐입음,셔츠,니트\n",
                encoding="utf-8",
            )
            train = load_layering_csv(csv_path, root, split="train")
            val = load_layering_csv(csv_path, root, split="val")
            self.assertEqual(train[0].is_layered, 0)
            self.assertEqual(val[0].is_layered, 1)
            self.assertEqual((val[0].inner_category, val[0].outer_category), ("셔츠", "니트"))

    def test_head_only_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_cache(path, count):
                torch.save(
                    {
                        "version": 1,
                        "backbone_model_id": "test-backbone",
                        "roi_names": list(LAYERING_ROI_NAMES),
                        "features": torch.randn(count, len(LAYERING_ROI_NAMES), 8),
                        "layering_targets": torch.arange(count) % 2,
                        "inner_targets": torch.where(torch.arange(count) % 2 == 1, torch.zeros(count, dtype=torch.long), -1),
                        "outer_targets": torch.where(torch.arange(count) % 2 == 1, torch.ones(count, dtype=torch.long), -1),
                        "image_paths": [f"{index}.jpg" for index in range(count)],
                    },
                    path,
                )

            train, val, output = root / "train.pt", root / "val.pt", root / "heads.pt"
            write_cache(train, 12)
            write_cache(val, 6)
            summary = train_layering_heads(
                train,
                val,
                output,
                config=LayeringTrainingConfig(
                    epochs=2, batch_size=4, hidden_dim=8, dropout=0.0, patience=2
                ),
                device="cpu",
            )
            self.assertTrue(output.is_file())
            self.assertIn("layering_accuracy", summary["metrics"])


if __name__ == "__main__":
    unittest.main()
