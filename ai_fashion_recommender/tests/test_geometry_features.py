"""기하 특징과 crop 통일을 지킨다.

FashionSigLIP 전처리는 crop을 224x224로 눌러 종횡비를 없앤다. 그래서 핏·다리 모양처럼
비율이 핵심인 속성은 임베딩만으로 맞힐 수 없고, crop에서 잰 기하 특징을 따로 넣어야 한다.
학습과 추론이 그 값을 다르게 계산하면 효과가 사라지므로 두 경로가 일치하는지 검사한다.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다

from fashion_attribute_dataset import AttributeRecord
from fashion_attribute_model import (
    GEOMETRY_DIM,
    build_attribute_heads,
    build_legacy_attribute_heads,
    geometry_vector,
    load_attribute_heads,
    save_attribute_checkpoint,
)
from fashion_attribute_training import _record_geometry
from outfit_analyzer import _crop_geometry, _garment_crop, _masked_crop


class GeometryVectorTests(unittest.TestCase):
    def test_wide_and_tall_crops_get_opposite_signs(self):
        wide = geometry_vector(300, 100, tight_crop=True)
        tall = geometry_vector(100, 300, tight_crop=True)
        self.assertGreater(wide[0], 0)
        self.assertLess(tall[0], 0)

    def test_square_crop_is_neutral(self):
        self.assertAlmostEqual(geometry_vector(200, 200, tight_crop=True)[0], 0.0, places=6)

    def test_values_stay_bounded_for_extreme_shapes(self):
        for width, height in ((1, 4000), (4000, 1), (1, 1)):
            with self.subTest(size=(width, height)):
                value = geometry_vector(width, height, tight_crop=True)[0]
                self.assertGreaterEqual(value, -1.0)
                self.assertLessEqual(value, 1.0)

    def test_tight_flag_is_reported(self):
        self.assertEqual(geometry_vector(10, 20, tight_crop=True)[1], 1.0)
        self.assertEqual(geometry_vector(10, 20, tight_crop=False)[1], 0.0)

    def test_vector_length_matches_declared_dimension(self):
        self.assertEqual(len(geometry_vector(10, 20, tight_crop=True)), GEOMETRY_DIM)

    def test_zero_sized_crop_does_not_crash(self):
        self.assertEqual(len(geometry_vector(0, 0, tight_crop=True)), GEOMETRY_DIM)


class TrainingAndInferenceAgreeTests(unittest.TestCase):
    """같은 옷 영역이면 학습 경로와 추론 경로가 같은 기하 특징을 내야 한다."""

    def test_same_box_produces_the_same_vector(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "person.jpg"
            Image.new("RGB", (400, 600), "white").save(image_path)

            # 학습 경로: bbox 주석
            record = AttributeRecord(image_path, "train", (100.0, 200.0, 120.0, 260.0), {})
            training_vector = _record_geometry(record)

            # 추론 경로: 같은 영역을 덮는 마스크
            mask = np.zeros((600, 400), dtype=bool)
            mask[200:460, 100:220] = True
            inference_vector = _crop_geometry(mask)

            self.assertEqual(training_vector, inference_vector)

    def test_whole_image_records_are_marked_as_not_tight(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "product.jpg"
            Image.new("RGB", (300, 450), "white").save(image_path)
            record = AttributeRecord(image_path, "train", None, {})
            self.assertEqual(_record_geometry(record)[1], 0.0)

    def test_empty_mask_is_marked_as_not_tight(self):
        self.assertEqual(_crop_geometry(np.zeros((10, 10), dtype=bool))[1], 0.0)


class GarmentCropTests(unittest.TestCase):
    def test_analysis_crop_keeps_the_background_like_training_data(self):
        """학습 표본은 배경이 남은 bbox crop이라 추론도 같아야 한다."""
        rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        rgb[40:60, 40:60] = (10, 20, 30)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        rgb[45:55, 10:20] = (255, 0, 0)  # 마스크 밖 배경

        crop = np.asarray(_garment_crop(rgb, mask))
        self.assertEqual(crop.shape[:2], (20, 20))
        self.assertFalse((crop == 255).all(axis=2).any())

    def test_masked_crop_still_removes_background_for_colour_analysis(self):
        rgb = np.full((100, 100, 3), 200, dtype=np.uint8)
        mask = np.zeros((100, 100), dtype=bool)
        mask[40:60, 40:60] = True
        crop = np.asarray(_masked_crop(rgb, mask))
        self.assertTrue((crop == 255).all(axis=2).any())

    def test_crop_matches_the_geometry_it_reports(self):
        rgb = np.zeros((200, 120, 3), dtype=np.uint8)
        mask = np.zeros((200, 120), dtype=bool)
        mask[20:180, 30:70] = True
        crop = _garment_crop(rgb, mask)
        self.assertEqual(_crop_geometry(mask), geometry_vector(crop.width, crop.height, True))


class HeadWiringTests(unittest.TestCase):
    def test_geometry_changes_the_output(self):
        """기하 특징이 실제로 쓰이는지 확인한다. 무시되면 개선이 있을 수 없다."""
        torch.manual_seed(0)
        heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0, geometry_dim=GEOMETRY_DIM)
        features = torch.randn(1, 8)
        wide = heads(features, torch.tensor([[0.9, 1.0]]))["category"]
        tall = heads(features, torch.tensor([[-0.9, 1.0]]))["category"]
        self.assertFalse(torch.allclose(wide, tall))

    def test_missing_geometry_fails_loudly(self):
        heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0, geometry_dim=GEOMETRY_DIM)
        with self.assertRaises(ValueError):
            heads(torch.randn(1, 8))

    def test_heads_without_geometry_ignore_it(self):
        heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0, geometry_dim=0)
        features = torch.randn(1, 8)
        self.assertTrue(
            torch.allclose(heads(features)["category"], heads(features, torch.tensor([[1.0, 1.0]]))["category"])
        )


class LetterboxTests(unittest.TestCase):
    """crop을 눌러 정사각형으로 만들면 비율 정보가 사라진다. 레터박스는 그것을 보존한다."""

    def test_letterbox_makes_a_square_without_stretching(self):
        from fashion_attribute_model import letterbox_image

        boxed = letterbox_image(Image.new("RGB", (120, 300), "black"))
        self.assertEqual(boxed.size, (300, 300))

    def test_original_pixels_survive_at_the_centre(self):
        from fashion_attribute_model import letterbox_image

        source = Image.new("RGB", (100, 200), (10, 20, 30))
        boxed = letterbox_image(source)
        self.assertEqual(boxed.getpixel((100, 100)), (10, 20, 30))
        self.assertEqual(boxed.getpixel((5, 100)), (255, 255, 255))

    def test_square_input_is_unchanged(self):
        from fashion_attribute_model import letterbox_image

        source = Image.new("RGB", (64, 64), "black")
        self.assertIs(letterbox_image(source), source)

    def test_different_shapes_stay_different_after_letterbox(self):
        """찌그러뜨리면 두 모양이 같은 입력이 되지만 레터박스는 다르게 남긴다."""
        from fashion_attribute_model import letterbox_image

        def bar(width, height):
            image = Image.new("RGB", (width, height), "white")
            image.paste(Image.new("RGB", (width, height), "black"), (0, 0))
            return image

        narrow, wide = bar(60, 300), bar(240, 300)
        squashed = [np.asarray(image.resize((224, 224))) for image in (narrow, wide)]
        boxed = [np.asarray(letterbox_image(image).resize((224, 224))) for image in (narrow, wide)]
        self.assertTrue(np.array_equal(*squashed))       # 눌러버리면 구분 불가
        self.assertFalse(np.array_equal(*boxed))         # 레터박스는 구분됨

    def test_unknown_mode_is_rejected(self):
        from fashion_attribute_model import apply_preprocess_mode

        with self.assertRaises(ValueError):
            apply_preprocess_mode(Image.new("RGB", (8, 8)), "stretch")

    def test_checkpoint_records_the_mode_it_was_trained_with(self):
        from fashion_attribute_model import PREPROCESS_LETTERBOX, PREPROCESS_SQUASH

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0)
            save_attribute_checkpoint(
                path, heads, backbone_model_id="test-backbone", preprocessing=PREPROCESS_LETTERBOX
            )
            _, payload = load_attribute_heads(path, "cpu")
            self.assertEqual(payload["preprocessing"], PREPROCESS_LETTERBOX)

        # 예전 체크포인트에는 항목이 없다. 기본 전처리로 학습한 것으로 본다.
        self.assertEqual(
            {"version": 1}.get("preprocessing", PREPROCESS_SQUASH), PREPROCESS_SQUASH
        )


class TaskRoutingTests(unittest.TestCase):
    """속성마다 다른 crop 처리를 쓰는 구조를 지킨다."""

    def test_every_task_has_a_preprocessing_assignment(self):
        from fashion_attribute_model import PREPROCESS_MODES, TASK_PREPROCESSING
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        self.assertEqual(set(TASK_PREPROCESSING), set(ATTRIBUTE_TASKS))
        for task, mode in TASK_PREPROCESSING.items():
            with self.subTest(task=task):
                self.assertIn(mode, PREPROCESS_MODES)

    def test_proportion_attributes_use_letterbox(self):
        """비율이 답을 정하는 속성은 종횡비를 보존해야 한다."""
        from fashion_attribute_model import PREPROCESS_LETTERBOX, TASK_PREPROCESSING

        for task in ("upper_fit", "lower_fit", "upper_length", "lower_length", "pant_length"):
            with self.subTest(task=task):
                self.assertEqual(TASK_PREPROCESSING[task], PREPROCESS_LETTERBOX)

    def test_local_detail_attributes_use_squash(self):
        """레터박스는 여백만큼 옷이 작아져 국소 디테일 해상도가 떨어진다."""
        from fashion_attribute_model import PREPROCESS_SQUASH, TASK_PREPROCESSING

        for task in ("neckline", "collar", "pattern", "material", "detail"):
            with self.subTest(task=task):
                self.assertEqual(TASK_PREPROCESSING[task], PREPROCESS_SQUASH)

    def test_string_preprocessing_expands_to_every_task(self):
        from fashion_attribute_model import PREPROCESS_SQUASH, resolve_task_preprocessing
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        resolved = resolve_task_preprocessing(PREPROCESS_SQUASH)
        self.assertEqual(set(resolved), set(ATTRIBUTE_TASKS))
        self.assertEqual(set(resolved.values()), {PREPROCESS_SQUASH})

    def test_incomplete_routing_is_rejected(self):
        from fashion_attribute_model import PREPROCESS_SQUASH, resolve_task_preprocessing

        with self.assertRaises(ValueError):
            resolve_task_preprocessing({"category": PREPROCESS_SQUASH})

    def test_each_head_reads_only_its_assigned_embedding(self):
        from fashion_attribute_model import (
            PREPROCESS_LETTERBOX,
            PREPROCESS_SQUASH,
            TASK_PREPROCESSING,
        )

        torch.manual_seed(0)
        heads = build_attribute_heads(
            8, hidden_dim=4, dropout=0.0, task_preprocessing=dict(TASK_PREPROCESSING)
        )
        squash = torch.randn(1, 8)
        letterbox = torch.randn(1, 8)
        base = heads({PREPROCESS_SQUASH: squash, PREPROCESS_LETTERBOX: letterbox})
        # letterbox 임베딩만 바꾸면 letterbox 배정 속성만 달라져야 한다.
        changed = heads({PREPROCESS_SQUASH: squash, PREPROCESS_LETTERBOX: torch.randn(1, 8)})
        for task, mode in TASK_PREPROCESSING.items():
            with self.subTest(task=task, mode=mode):
                same = torch.allclose(base[task], changed[task])
                self.assertEqual(same, mode == PREPROCESS_SQUASH)

    def test_routed_checkpoint_requires_every_embedding(self):
        from fashion_attribute_model import TASK_PREPROCESSING

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hybrid.pt"
            heads = build_attribute_heads(
                8, hidden_dim=4, dropout=0.0, task_preprocessing=dict(TASK_PREPROCESSING)
            )
            save_attribute_checkpoint(
                path, heads, backbone_model_id="test-backbone", preprocessing=dict(TASK_PREPROCESSING)
            )
            loaded, payload = load_attribute_heads(path, "cpu")
            self.assertIsInstance(payload["preprocessing"], dict)
            self.assertEqual(loaded.task_preprocessing, dict(TASK_PREPROCESSING))


class TrainingSwitchTests(unittest.TestCase):
    def test_geometry_is_off_by_default(self):
        """5,984 crop A/B에서 이득이 없었으므로 기본값은 꺼둔다."""
        from fashion_attribute_training import TrainingConfig

        self.assertFalse(TrainingConfig().use_geometry)

    def test_turning_geometry_on_produces_a_geometry_checkpoint(self):
        from fashion_attribute_schema import ATTRIBUTE_TASKS
        from fashion_attribute_training import TrainingConfig, train_attribute_heads

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            generator = torch.Generator().manual_seed(0)

            def write_cache(path: Path, count: int) -> None:
                targets, valid = {}, {}
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

            write_cache(root / "train.pt", 16)
            write_cache(root / "val.pt", 8)
            output = root / "heads.pt"
            train_attribute_heads(
                root / "train.pt",
                root / "val.pt",
                output,
                config=TrainingConfig(
                    epochs=1, batch_size=8, hidden_dim=4, dropout=0.0, patience=1, use_geometry=True
                ),
                device="cpu",
            )
            _, payload = load_attribute_heads(output, "cpu")
            self.assertEqual(payload["geometry_dim"], GEOMETRY_DIM)

    def test_old_cache_without_geometry_is_refused_when_enabled(self):
        from fashion_attribute_schema import ATTRIBUTE_TASKS
        from fashion_attribute_training import TrainingConfig, train_attribute_heads

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            targets = {
                name: (
                    torch.zeros((4, len(task.labels)), dtype=torch.float32)
                    if task.multi_label
                    else torch.zeros(4, dtype=torch.long)
                )
                for name, task in ATTRIBUTE_TASKS.items()
            }
            valid = {name: torch.ones(4, dtype=torch.bool) for name in ATTRIBUTE_TASKS}
            for path in ("train.pt", "val.pt"):
                torch.save(
                    {
                        "version": 1,
                        "backbone_model_id": "test-backbone",
                        "features": torch.randn(4, 8),
                        "targets": targets,
                        "valid": valid,
                        "image_paths": ["a.jpg"] * 4,
                    },
                    root / path,
                )
            with self.assertRaisesRegex(ValueError, "기하 특징이 없는"):
                train_attribute_heads(
                    root / "train.pt",
                    root / "val.pt",
                    root / "heads.pt",
                    config=TrainingConfig(epochs=1, use_geometry=True),
                    device="cpu",
                )


class CheckpointCompatibilityTests(unittest.TestCase):
    def test_geometry_checkpoint_round_trips(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "heads.pt"
            heads = build_attribute_heads(8, hidden_dim=4, dropout=0.0, geometry_dim=GEOMETRY_DIM)
            save_attribute_checkpoint(path, heads, backbone_model_id="test-backbone")
            loaded, payload = load_attribute_heads(path, "cpu")
            self.assertEqual(payload["geometry_dim"], GEOMETRY_DIM)
            self.assertEqual(loaded.geometry_dim, GEOMETRY_DIM)

    def test_version_1_checkpoints_still_load(self):
        """배포 중인 체크포인트를 못 읽게 되면 서비스가 멈춘다."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy.pt"
            legacy = build_legacy_attribute_heads(8, hidden_dim=4, dropout=0.0)
            save_attribute_checkpoint(path, legacy, backbone_model_id="test-backbone")
            payload = torch.load(path, map_location="cpu", weights_only=False)
            payload["version"] = 1
            torch.save(payload, path)

            loaded, meta = load_attribute_heads(path, "cpu")
            self.assertEqual(meta["version"], 1)
            self.assertEqual(loaded.geometry_dim, 0)
            self.assertIn("category", loaded(torch.randn(1, 8)))

    def test_deployed_checkpoint_loads(self):
        checkpoint = ROOT / "models" / "fashion_attribute_heads.pt"
        if not checkpoint.is_file():
            self.skipTest("배포 체크포인트가 없습니다.")
        heads, payload = load_attribute_heads(checkpoint, "cpu")
        self.assertIn(payload["version"], (1, 2))
        self.assertIn("category", heads(torch.randn(1, payload["input_dim"])
                                        if payload.get("geometry_dim", 0) == 0
                                        else torch.randn(1, payload["input_dim"]),
                                        None if payload.get("geometry_dim", 0) == 0
                                        else torch.zeros(1, payload["geometry_dim"])))


if __name__ == "__main__":
    unittest.main()
