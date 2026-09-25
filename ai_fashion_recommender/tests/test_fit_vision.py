from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fit_vision_dataset import fit_csv_header, load_fit_vision_csv
from fit_labeling import FitAnnotationStore, infer_group_id, stable_group_split
from fit_vision_model import FIT_GEOMETRY_DIM, apply_fit_feature_mode, fit_geometry_vector
from fit_vision_schema import FIT_VISION_TASKS, tasks_for_category
from fit_vision_training import FitVisionTrainingConfig, train_fit_vision_heads


class FitVisionSchemaTests(unittest.TestCase):
    def test_fit_and_length_are_separate_tasks(self):
        self.assertEqual(tasks_for_category("top"), ("upper_fit", "upper_length", "quality"))
        self.assertIn("세미와이드", FIT_VISION_TASKS["bottom_silhouette"].labels)
        self.assertNotIn("크롭", FIT_VISION_TASKS["upper_fit"].labels)

    def test_geometry_is_scale_normalized_and_tracks_width_profile(self):
        narrow = np.zeros((100, 60), dtype=bool)
        narrow[5:95, 22:38] = True
        wide = np.zeros((200, 120), dtype=bool)
        wide[10:190, 28:92] = True
        narrow_vector = fit_geometry_vector(narrow)
        wide_vector = fit_geometry_vector(wide)
        self.assertEqual(len(narrow_vector), FIT_GEOMETRY_DIM)
        self.assertGreater(wide_vector[0], narrow_vector[0])

    def test_group_identity_cannot_cross_splits(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "labels.csv"
            rows = [
                {"image_path": "a.jpg", "split": "train", "category": "bottom",
                 "source_domain": "shop", "group_id": "P1", "bottom_silhouette": "wide"},
                {"image_path": "b.jpg", "split": "val", "category": "bottom",
                 "source_domain": "shop", "group_id": "P1", "bottom_silhouette": "straight"},
            ]
            # CSV contract uses the Korean runtime labels even when the external
            # labeling guide documents their English equivalents.
            rows[0]["bottom_silhouette"] = "와이드"
            rows[1]["bottom_silhouette"] = "스트레이트"
            with csv_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fit_csv_header())
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "여러 split"):
                load_fit_vision_csv(csv_path, require_images=False)

    def test_ablation_modes_zero_only_the_disabled_branches(self):
        import torch

        context = torch.ones((1, 2))
        garment = torch.full((1, 2), 2.0)
        geometry = torch.full((1, FIT_GEOMETRY_DIM), 3.0)
        rgb, masked, shape = apply_fit_feature_mode(
            context, garment, geometry, "rgb_geometry"
        )
        self.assertEqual(float(rgb.sum()), 2.0)
        self.assertEqual(float(masked.sum()), 0.0)
        self.assertEqual(float(shape.sum()), 30.0)

    def test_training_report_separates_user_and_shop_domains(self):
        import torch

        def cache(path: Path):
            count = 6
            targets = {
                name: torch.full((count,), -1, dtype=torch.long)
                for name in FIT_VISION_TASKS
            }
            targets["bottom_silhouette"] = torch.tensor([2, 3, 4, 2, 3, 4])
            valid = {name: value >= 0 for name, value in targets.items()}
            torch.save({
                "version": 1,
                "backbone_model_id": "test-backbone",
                "context_features": torch.randn(count, 8),
                "garment_features": torch.randn(count, 8),
                "geometry": torch.randn(count, FIT_GEOMETRY_DIM),
                "targets": targets,
                "valid": valid,
                "image_paths": [f"{index}.jpg" for index in range(count)],
                "source_domains": ["user", "user", "user", "shop", "shop", "shop"],
                "group_ids": [str(index) for index in range(count)],
            }, path)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_path, val_path = root / "train.pt", root / "val.pt"
            cache(train_path)
            cache(val_path)
            summary = train_fit_vision_heads(
                train_path,
                val_path,
                root / "fit.pt",
                config=FitVisionTrainingConfig(
                    epochs=1, batch_size=3, hidden_dim=8, patience=1,
                    feature_mode="rgb_mask_geometry",
                ),
                device="cpu",
            )
        metrics = summary["metrics"]["bottom_silhouette"]
        self.assertEqual(set(metrics["by_domain"]), {"user", "shop"})
        self.assertIn("세미와이드", metrics["per_class_f1"])

    def test_cpu_label_store_writes_portable_csv_and_stable_group_split(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "images" / "MS123456_model_01.jpg"
            image.parent.mkdir()
            image.write_bytes(b"not-decoded-by-store")
            store = FitAnnotationStore(root / "annotations.csv", root)
            row = store.save(
                image,
                category="bottom",
                source_domain="shop",
                group_id=infer_group_id(image, "shop"),
                labels={
                    "bottom_silhouette": "세미와이드",
                    "bottom_length": "풀렝스",
                    "quality": "판정 가능",
                },
            )
            self.assertEqual(row["image_path"], "images/MS123456_model_01.jpg")
            self.assertEqual(row["group_id"], "product_123456")
            self.assertEqual(row["split"], stable_group_split("shop:product_123456"))
            records = load_fit_vision_csv(root / "annotations.csv", root)
            self.assertEqual(records[0].labels["bottom_silhouette"], "세미와이드")


if __name__ == "__main__":
    unittest.main()
