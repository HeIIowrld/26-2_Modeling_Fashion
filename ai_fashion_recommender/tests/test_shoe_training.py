import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


def module(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


prep = module("prepare_shoe_dataset")
train = module("train_shoe_heads")
weak = module("build_weak_shoe_manifest")


class ShoeTrainingTests(unittest.TestCase):
    def row(self, **kwargs):
        return {"shoe_label": "로퍼", "split": "train", "domain": "catalog", "product_id": "p1", "image_path": "a.jpg", **kwargs}

    def test_product_leakage(self):
        with self.assertRaisesRegex(ValueError, "leakage"):
            prep.validate_manifest([self.row(), self.row(split="val", image_path="b.jpg")])

    def test_outfit_requires_subject(self):
        with self.assertRaisesRegex(ValueError, "subject"):
            prep.validate_manifest([self.row(domain="outfit")])

    def test_unknown_label(self):
        with self.assertRaisesRegex(ValueError, "shoe_label"):
            prep.validate_manifest([self.row(shoe_label="Running")])

    def test_negative_classes_never_accepted(self):
        probabilities = np.eye(12)[[10, 11, 2]]
        metrics = train.metrics(np.array([10, 11, 2]), probabilities, .5)
        self.assertEqual(metrics["accepted_count"], 1)
        self.assertEqual(metrics["barefoot_false_accept_rate"], 0)

    def test_no_supported_threshold(self):
        probabilities = np.eye(12)[[2, 2]] * .9 + .1 / 12
        threshold, _, met = train.select_threshold(np.array([11, 11]), probabilities, .85, 1)
        self.assertFalse(met)
        self.assertEqual(threshold, 1.0)

    def test_cache_checksum_leakage(self):
        a = {"backbone_model_id": "m", "features": torch.zeros(1, 3), "paths": ["a"],
             "subject_ids": [""], "product_ids": ["p"], "session_ids": [""], "checksums": ["same"]}
        b = {**a, "paths": ["b"], "product_ids": ["q"]}
        with self.assertRaisesRegex(ValueError, "checksums"):
            train.validate_pair(a, b)

    def test_group_split_is_deterministic(self):
        self.assertEqual(weak.stable_split("same-person-and-product"),
                         weak.stable_split("same-person-and-product"))
        self.assertIn(weak.stable_split("same-person-and-product"), {"train", "val", "test"})

    def test_cache_requires_label_provenance(self):
        cache = {"features": torch.zeros(1, 3), "labels": torch.tensor([2]),
                 "label_names": tuple(weak.SHOE_LABELS), "preprocessing": "squash",
                 "backbone_model_id": "m", "paths": ["a"], "subject_ids": ["s"],
                 "product_ids": ["p"], "session_ids": ["x"], "checksums": ["h"]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pt"
            torch.save(cache, path)
            with self.assertRaisesRegex(KeyError, "label_sources"):
                train.load_cache(path)


if __name__ == "__main__":
    unittest.main()
