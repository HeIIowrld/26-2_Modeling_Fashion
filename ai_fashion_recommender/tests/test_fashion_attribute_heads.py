import json
import sys
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))  # 전처리 스크립트의 헬퍼도 검증한다

from fashion_attribute_dataset import (
    convert_fashionpedia_instances,
    encode_record_targets,
    load_attribute_csv,
)
from fashion_attribute_model import (
    GEOMETRY_DIM,
    AttributePrediction,
    FashionAttributePredictor,
    build_attribute_heads,
    fuse_measured_and_learned,
    save_attribute_checkpoint,
)
from fashion_attribute_schema import ATTRIBUTE_TASKS, UPPER_CATEGORIES
from fashion_attribute_training import (
    TrainingConfig,
    build_embedding_cache,
    filter_embedding_cache,
    load_embedding_cache,
    merge_embedding_caches,
    train_attribute_heads,
)
from prepare_fashion200k_bottoms import _labels, _lower_details, _lower_subtype, _pant_leg_shape, _pant_length
from fashion_prompts import UPPER_TYPE_PROMPTS
from outfit_analyzer import _refine_upper_type


class FakeEncoder:
    def __init__(self):
        self.calls = 0

    def encode_image(self, batch, normalize=True):
        self.calls += 1
        features = torch.ones((len(batch), 8), device=batch.device)
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features


class FakeBatchEncoder:
    model_id = "test-backbone"
    preprocessing = "squash"

    def __init__(self):
        self.encoded_images = 0

    def encode_pil_batch(self, images):
        self.encoded_images += len(images)
        return torch.full((len(images), 4), 3.0)


class FashionAttributeHeadTests(unittest.TestCase):
    def test_zero_shot_upper_fallback_covers_trained_categories(self):
        self.assertEqual(set(UPPER_TYPE_PROMPTS), UPPER_CATEGORIES)

    def test_csv_supports_single_and_multi_label_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new("RGB", (20, 20), "white").save(root / "top.jpg")
            csv_path = root / "labels.csv"
            csv_path.write_text(
                "image_path,split,bbox_x,bbox_y,bbox_w,bbox_h,category,pattern,material,detail\n"
                "top.jpg,train,1,2,10,12,셔츠,스트라이프|그래픽,코튼|레이스,단추|포켓\n",
                encoding="utf-8",
            )
            records = load_attribute_csv(csv_path, root, split="train")
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0].labels["category"], ["셔츠"])
            self.assertEqual(records[0].labels["pattern"], ["스트라이프", "그래픽"])
            targets, valid = encode_record_targets(records[0])
            self.assertTrue(valid["category"])
            self.assertEqual(sum(targets["pattern"]), 2.0)
            self.assertFalse(valid["neckline"])

    def test_fashionpedia_converter_maps_instance_crop_and_attributes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "instances.json"
            source.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "image.jpg"}],
                        "categories": [
                            {"id": 10, "name": "shirt, blouse"},
                            {"id": 11, "name": "sleeve"},
                            {"id": 12, "name": "pocket"},
                        ],
                        "attributes": [
                            {"id": 100, "name": "wrist", "supercategory": "sleeve length"},
                            {"id": 101, "name": "striped", "supercategory": "textile pattern"},
                        ],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 10, "attribute_ids": [101], "bbox": [2, 3, 10, 12]},
                            {"id": 2, "image_id": 1, "category_id": 11, "attribute_ids": [100], "bbox": [3, 4, 3, 8]},
                            {"id": 3, "image_id": 1, "category_id": 12, "attribute_ids": [], "bbox": [5, 6, 2, 2]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "converted.csv"
            result = convert_fashionpedia_instances(source, output, split="train", image_prefix="train")
            self.assertEqual(result["written"], 1)
            text = output.read_text(encoding="utf-8-sig")
            self.assertIn("셔츠", text)
            self.assertIn("긴팔", text)
            self.assertIn("스트라이프", text)
            self.assertIn("포켓", text)
            self.assertIn("train/image.jpg", text)

    def test_polo_collar_refines_tshirt_category(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "instances.json"
            source.write_text(
                json.dumps(
                    {
                        "images": [{"id": 1, "file_name": "polo.jpg"}],
                        "categories": [
                            {"id": 10, "name": "top, t-shirt, sweatshirt"},
                            {"id": 11, "name": "collar"},
                        ],
                        "attributes": [{"id": 100, "name": "polo", "supercategory": "collar"}],
                        "annotations": [
                            {"id": 1, "image_id": 1, "category_id": 10, "attribute_ids": [], "bbox": [0, 0, 20, 20]},
                            {"id": 2, "image_id": 1, "category_id": 11, "attribute_ids": [100], "bbox": [5, 2, 10, 5]},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            output = root / "converted.csv"
            convert_fashionpedia_instances(source, output, split="train")
            text = output.read_text(encoding="utf-8-sig")
            self.assertIn("폴로 셔츠", text)
            self.assertIn("폴로 칼라", text)
        self.assertEqual(_refine_upper_type("티셔츠", "폴로 칼라"), "폴로 셔츠")
        self.assertEqual(_refine_upper_type("니트", "폴로 칼라"), "니트")

    def test_fusion_prefers_agreement_and_strong_learned_result(self):
        agreed = AttributePrediction(["긴팔"], {"긴팔": 0.8}, 0.8, True)
        label, confidence, source = fuse_measured_and_learned("긴팔", 0.7, agreed)
        self.assertEqual(label, "긴팔")
        self.assertEqual(source, "fused_agreement")
        self.assertGreater(confidence, 0.7)

        strong = AttributePrediction(["오버핏"], {"오버핏": 0.95}, 0.95, True)
        label, _, source = fuse_measured_and_learned("슬림핏 추정", 0.5, strong)
        self.assertEqual(label, "오버핏")
        self.assertEqual(source, "trained_head")

    def test_checkpoint_predictor_uses_existing_image_encoder(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0)
            save_attribute_checkpoint(checkpoint, heads, backbone_model_id="test-backbone")
            predictor = FashionAttributePredictor(
                checkpoint,
                image_encoder=FakeEncoder(),
                preprocess=lambda _: torch.ones(3, 4, 4),
                model_id="test-backbone",
                device="cpu",
            )
            result = predictor.predict(Image.new("RGB", (8, 8)), tasks=["category", "pattern"])
            self.assertEqual(set(result), {"category", "pattern"})
            self.assertEqual(len(result["category"].scores), len(ATTRIBUTE_TASKS["category"].labels))

    def test_precomputed_features_skip_a_second_backbone_call(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0)
            save_attribute_checkpoint(checkpoint, heads, backbone_model_id="test-backbone")
            encoder = FakeEncoder()
            predictor = FashionAttributePredictor(
                checkpoint,
                image_encoder=encoder,
                preprocess=lambda _: torch.ones(3, 4, 4),
                model_id="test-backbone",
                device="cpu",
            )
            features = encoder.encode_image(torch.ones(1, 3, 4, 4), normalize=True)
            result = predictor.predict_features(features, tasks=["category", "pattern"])
            self.assertEqual(encoder.calls, 1)
            self.assertEqual(set(result), {"category", "pattern"})

    def test_checkpoint_masks_labels_without_enough_training_examples(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0)
            support = {
                "category": {
                    label: (10 if label == "티셔츠" else 0)
                    for label in ATTRIBUTE_TASKS["category"].labels
                }
            }
            save_attribute_checkpoint(
                checkpoint,
                heads,
                backbone_model_id="test-backbone",
                label_support=support,
                minimum_label_examples=5,
            )
            predictor = FashionAttributePredictor(
                checkpoint,
                image_encoder=FakeEncoder(),
                preprocess=lambda _: torch.ones(3, 4, 4),
                model_id="test-backbone",
                device="cpu",
            )
            result = predictor.predict(Image.new("RGB", (8, 8)), tasks=["category"])["category"]
            self.assertFalse(result.accepted)
            self.assertEqual(result.scores["니트"], 0.0)

    def test_single_label_predictor_uses_calibrated_checkpoint_threshold(self):
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0)
            save_attribute_checkpoint(
                checkpoint,
                heads,
                backbone_model_id="test-backbone",
                thresholds={"pant_leg_shape": 0.99},
            )
            predictor = FashionAttributePredictor(
                checkpoint,
                image_encoder=FakeEncoder(),
                preprocess=lambda _: torch.ones(3, 4, 4),
                model_id="test-backbone",
                device="cpu",
            )
            result = predictor.predict(
                Image.new("RGB", (8, 8)), tasks=["pant_leg_shape"]
            )["pant_leg_shape"]
            self.assertFalse(result.accepted)

    def test_head_only_training_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(7)

            def write_cache(path, count):
                targets = {}
                valid = {}
                for name, task in ATTRIBUTE_TASKS.items():
                    if task.multi_label:
                        value = torch.zeros((count, len(task.labels)), dtype=torch.float32)
                        value[torch.arange(count), torch.arange(count) % len(task.labels)] = 1.0
                    else:
                        value = torch.arange(count) % len(task.labels)
                    targets[name] = value
                    valid[name] = torch.ones(count, dtype=torch.bool)
                torch.save(
                    {
                        "version": 2,
                        "backbone_model_id": "test-backbone",
                        "features": torch.randn((count, 8), generator=generator),
                        "geometry": torch.randn((count, GEOMETRY_DIM), generator=generator),
                        "targets": targets,
                        "valid": valid,
                        "image_paths": [f"{index}.jpg" for index in range(count)],
                    },
                    path,
                )

            train_cache = root / "train.pt"
            val_cache = root / "val.pt"
            output = root / "heads.pt"
            write_cache(train_cache, 20)
            write_cache(val_cache, 8)
            summary = train_attribute_heads(
                train_cache,
                val_cache,
                output,
                config=TrainingConfig(epochs=2, batch_size=8, hidden_dim=8, dropout=0.0, patience=2),
                device="cpu",
            )
            self.assertTrue(output.is_file())
            self.assertTrue(output.with_suffix(".metrics.json").is_file())
            self.assertEqual(summary["train_samples"], 20)
            self.assertIn("category", summary["metrics"])

    def test_merge_embedding_caches_preserves_all_rows(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)

            def write_cache(path, count, offset):
                targets = {}
                valid = {}
                for name, task in ATTRIBUTE_TASKS.items():
                    if task.multi_label:
                        targets[name] = torch.zeros((count, len(task.labels)), dtype=torch.float32)
                    else:
                        targets[name] = torch.zeros(count, dtype=torch.long)
                    valid[name] = torch.ones(count, dtype=torch.bool)
                torch.save(
                    {
                        "version": 1,
                        "backbone_model_id": "same-backbone",
                        "features": torch.full((count, 4), float(offset)),
                        "targets": targets,
                        "valid": valid,
                        "image_paths": [f"{offset}-{index}.jpg" for index in range(count)],
                    },
                    path,
                )

            first, second, output = root / "a.pt", root / "b.pt", root / "merged.pt"
            write_cache(first, 2, 1)
            write_cache(second, 3, 2)
            merge_embedding_caches([first, second], output)
            merged = load_embedding_cache(output)
            self.assertEqual(len(merged["features"]), 5)
            self.assertEqual(merged["image_paths"], ["1-0.jpg", "1-1.jpg", "2-0.jpg", "2-1.jpg", "2-2.jpg"])

    def test_fashion200k_bottom_subtype_uses_explicit_product_name(self):
        base = {"category1": "pants", "category2": "straight-leg pants"}
        self.assertEqual(_lower_subtype({**base, "category3": "black slim tailored trousers"}), "슬랙스")
        self.assertEqual(_lower_subtype({**base, "category3": "beige chino pants"}), "치노 팬츠")
        self.assertEqual(_lower_subtype({**base, "category3": "green cargo jogger pants"}), "카고 팬츠")
        self.assertIsNone(_lower_subtype({**base, "category3": "black straight-leg pants"}))

    def test_fashion200k_bottom_axes_are_independent(self):
        row = {
            "category1": "pants",
            "category2": "cropped pants",
            "category3": "wide-leg cargo pants with drawstring and side stripe",
        }
        labels = _labels(row)
        self.assertEqual(_lower_subtype(row), "카고 팬츠")
        self.assertEqual(_pant_leg_shape(row), "와이드")
        self.assertEqual(_pant_length(row), "크롭·앵클")
        self.assertEqual(_lower_details(row, "카고 팬츠"), ["카고 포켓", "드로스트링", "사이드 스트라이프"])
        self.assertEqual(labels["lower_subtype"], ["카고 팬츠"])
        self.assertEqual(labels["pant_leg_shape"], ["와이드"])
        self.assertEqual(labels["pant_length"], ["크롭·앵클"])
        self.assertEqual(labels["lower_fit"], ["와이드핏"])

    def test_fashion200k_bottom_shape_separates_bootcut_from_flare(self):
        base = {"category1": "pants", "category2": "full length pants"}
        self.assertEqual(_pant_leg_shape({**base, "category3": "dark denim bootcut pants"}), "부츠컷")
        self.assertEqual(_pant_leg_shape({**base, "category3": "black flared trousers"}), "플레어")
        self.assertEqual(_pant_length({**base, "category3": "black capri trousers"}), "카프리·7부")

    def test_old_embedding_cache_adds_new_tasks_as_unannotated(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.pt"
            targets = {}
            valid = {}
            for name, task in ATTRIBUTE_TASKS.items():
                if name == "lower_subtype":
                    continue
                targets[name] = (
                    torch.zeros((2, len(task.labels)), dtype=torch.float32)
                    if task.multi_label else torch.zeros(2, dtype=torch.long)
                )
                valid[name] = torch.ones(2, dtype=torch.bool)
            torch.save(
                {
                    "version": 1,
                    "backbone_model_id": "test-backbone",
                    "features": torch.zeros((2, 4)),
                    "targets": targets,
                    "valid": valid,
                    "image_paths": ["a.jpg", "b.jpg"],
                },
                path,
            )
            upgraded = load_embedding_cache(path)
            self.assertIn("lower_subtype", upgraded["targets"])
            self.assertFalse(bool(upgraded["valid"]["lower_subtype"].any()))

    def test_filter_embedding_cache_excludes_only_matching_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.pt", root / "filtered.pt"
            targets = {}
            valid = {}
            for name, task in ATTRIBUTE_TASKS.items():
                targets[name] = (
                    torch.zeros((3, len(task.labels)), dtype=torch.float32)
                    if task.multi_label else torch.zeros(3, dtype=torch.long)
                )
                valid[name] = torch.ones(3, dtype=torch.bool)
            torch.save(
                {
                    "version": 1,
                    "backbone_model_id": "test-backbone",
                    "features": torch.arange(12, dtype=torch.float32).reshape(3, 4),
                    "targets": targets,
                    "valid": valid,
                    "image_paths": ["base/a.jpg", "fashion200k_bottoms/b.jpg", "base/c.jpg"],
                },
                source,
            )
            filter_embedding_cache(
                source,
                output,
                exclude_path_fragments=["fashion200k_bottoms"],
            )
            filtered = load_embedding_cache(output)
            self.assertEqual(filtered["image_paths"], ["base/a.jpg", "base/c.jpg"])
            self.assertEqual(filtered["features"].shape, (2, 4))

    def test_build_embedding_cache_reuses_matching_features(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image_paths = [root / f"{index}.jpg" for index in range(3)]
            for index, path in enumerate(image_paths):
                Image.new("RGB", (8, 8), (index, index, index)).save(path)
            records = [
                load_attribute_csv(
                    self._write_csv(root, path),
                    root,
                    split="train",
                )[0]
                for path in image_paths
            ]
            reuse = root / "reuse.pt"
            targets = {}
            valid = {}
            for name, task in ATTRIBUTE_TASKS.items():
                targets[name] = (
                    torch.zeros((1, len(task.labels)), dtype=torch.float32)
                    if task.multi_label else torch.zeros(1, dtype=torch.long)
                )
                valid[name] = torch.zeros(1, dtype=torch.bool)
            torch.save(
                {
                    "version": 1,
                    "backbone_model_id": "test-backbone",
                    "features": torch.full((1, 4), 9.0),
                    "targets": targets,
                    "valid": valid,
                    "image_paths": [str(image_paths[1].resolve())],
                },
                reuse,
            )
            encoder = FakeBatchEncoder()
            output = root / "output.pt"
            build_embedding_cache(records, encoder, output, batch_size=2, reuse_cache_paths=[reuse])
            cache = load_embedding_cache(output)
            self.assertEqual(encoder.encoded_images, 2)
            self.assertTrue(torch.equal(cache["features"][1], torch.full((4,), 9.0)))

    @staticmethod
    def _write_csv(root: Path, image_path: Path) -> Path:
        path = root / f"{image_path.stem}.csv"
        path.write_text(
            "image_path,split,category\n" + f"{image_path.name},train,팬츠\n",
            encoding="utf-8-sig",
        )
        return path


if __name__ == "__main__":
    unittest.main()
