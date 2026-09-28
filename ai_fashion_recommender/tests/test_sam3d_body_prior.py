import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sam3d_body_prior import SAM3DBodyProjection, SAM3DBodyProvider  # noqa: E402
from catvton_tryon import (  # noqa: E402
    mesh_target_mask,
    observed_garment_mask,
    residual_garment_ring,
)
from catvton_tryon import CatVTONTryOn  # noqa: E402
from schemas import Product, Recommendation  # noqa: E402


class FakeEstimator:
    def __init__(self):
        self.faces = np.asarray([[0, 1, 2]], np.int32)
        self.calls = []

    def process_one_image(self, image, **kwargs):
        self.calls.append(kwargs)
        return [{
            "focal_length": 500.0,
            "pred_vertices": np.zeros((3, 3), np.float32),
            "pred_cam_t": np.asarray([0, 0, 3], np.float32),
        }]


class FakeRenderer:
    def __init__(self, **_kwargs):
        pass

    def __call__(self, _vertices, _cam, canvas, **_kwargs):
        rgba = np.zeros((*canvas.shape[:2], 4), np.float32)
        rgba[18:82, 38:62, 3] = 1
        return rgba


class SAM3DBodyProviderTests(unittest.TestCase):
    def test_existing_person_mask_is_reused_as_prompt_and_result_is_cached(self):
        estimator = FakeEstimator()
        provider = SAM3DBodyProvider(
            "/missing", "/missing/model", "/missing/mhr",
            estimator=estimator, renderer_factory=FakeRenderer,
        )
        image = np.zeros((100, 100, 3), np.uint8)
        visible = np.zeros((100, 100), bool)
        visible[10:90, 25:75] = True

        first = provider.estimate(image, visible)
        second = provider.estimate(image, visible)

        self.assertIs(first, second)
        self.assertEqual(len(estimator.calls), 1)
        self.assertEqual(estimator.calls[0]["inference_type"], "body")
        self.assertEqual(estimator.calls[0]["masks"].shape, (1, 100, 100))
        self.assertEqual(first.diagnostics["backend"], "sam-3d-body")
        self.assertTrue(first.mask[50, 50])
        self.assertGreater(first.confidence, 0.8)

    def test_bad_mesh_overlap_fails_closed(self):
        class OutsideRenderer(FakeRenderer):
            def __call__(self, _vertices, _cam, canvas, **_kwargs):
                rgba = np.zeros((*canvas.shape[:2], 4), np.float32)
                rgba[:20, :20, 3] = 1
                return rgba

        visible = np.zeros((100, 100), bool)
        visible[50:95, 50:95] = True
        class TrackingProvider(SAM3DBodyProvider):
            def _release_estimator(self):
                self.release_calls = getattr(self, "release_calls", 0) + 1

        provider = TrackingProvider(
            "/missing", "/missing/model", "/missing/mhr",
            estimator=FakeEstimator(), renderer_factory=OutsideRenderer,
        )
        self.assertIsNone(provider.estimate(np.zeros((100, 100, 3), np.uint8), visible))
        self.assertEqual(provider.release_calls, 1)


class TwoStageMaskTests(unittest.TestCase):
    def test_background_source_excludes_limbs_in_style_mask(self):
        style = np.ones((20, 20), bool)
        segmentation = np.full((20, 20), 14, np.uint8)  # visible legs
        segmentation[3:14, 5:15] = 6  # trousers

        garment = observed_garment_mask(style, segmentation, "bottom")

        self.assertTrue(garment[8, 10])
        self.assertFalse(garment[18, 10])

    def test_loose_outer_ring_becomes_background_and_mesh_becomes_target(self):
        central = np.zeros((120, 100), bool)
        central[20:110, 42:58] = True
        prior = type("Prior", (), {
            "central_mask": central,
            "geometry_source": "sam-3d-body",
        })()
        target = mesh_target_mask(prior, "bottom", "스트레이트핏")
        source = np.zeros_like(central)
        source[15:112, 20:80] = True
        ring = residual_garment_ring(source, target)

        self.assertIsNotNone(target)
        self.assertIsNotNone(ring)
        self.assertTrue(ring[60, 25])
        self.assertFalse(ring[60, 50])
        self.assertTrue(target[60, 50])

    def test_mesh_route_is_only_used_for_narrow_targets(self):
        central = np.ones((40, 30), bool)
        prior = type("Prior", (), {
            "central_mask": central,
            "geometry_source": "sam-3d-body",
        })()
        self.assertIsNone(mesh_target_mask(prior, "bottom", "와이드핏"))


class TwoStageTryOnIntegrationTests(unittest.TestCase):
    def test_warm_generation_models_are_released_before_mesh_inference(self):
        class Editor:
            def __init__(self):
                self._pipeline = object()
                self.calls = 0

            def release_pipeline(self):
                self.calls += 1
                self._pipeline = None

        editor = Editor()
        adapter = CatVTONTryOn(transition_editor=editor)
        adapter._pipeline = object()
        adapter._original_scheduler = object()

        adapter._release_generation_models_for_body_mesh()

        self.assertIsNone(adapter._pipeline)
        self.assertIsNone(adapter._original_scheduler)
        self.assertEqual(editor.calls, 1)
        self.assertTrue(any("released-generation-models" in note for note in adapter.last_mask_notes))

    def test_sam_mesh_restores_outer_ring_then_uses_catvton_on_body(self):
        height, width = 320, 180
        seg = np.zeros((height, width), np.uint8)
        seg[145:305, 25:155] = 6
        lower = seg == 6
        points = {
            "left_hip": (68.0, 155.0), "right_hip": (112.0, 155.0),
            "left_knee": (70.0, 230.0), "right_knee": (110.0, 230.0),
            "left_ankle": (72.0, 300.0), "right_ankle": (108.0, 300.0),
        }
        pose = SimpleNamespace(
            landmarks={name: (x / width, y / height, 0.99) for name, (x, y) in points.items()}
        )
        outfit = SimpleNamespace(
            lower_fit="와이드핏", pant_leg_shape="와이드", layering_state="단일 상의",
            attribute_sources={"pant_leg_shape": "shared_fit_head"},
        )
        mesh = np.zeros_like(lower)
        mesh[145:305, 60:84] = True
        mesh[145:305, 96:120] = True

        class Provider:
            available = True

            def estimate(self, _image, _person):
                return SAM3DBodyProjection(
                    mesh, .88, {"backend": "sam-3d-body", "metric_body_measurement": False}
                )

        class Editor:
            available = True

            def __init__(self):
                self.masks = []

            def restore_background(self, person, mask, **_kwargs):
                self.masks.append(mask.copy())
                return person.copy()

        editor = Editor()
        adapter = CatVTONTryOn(
            width=90, height=160, transition_editor=editor,
            body_mesh_provider=Provider(), post_quality_gate=False,
        )
        captured = {}

        def fake_tryon(current, _garment, mask, **kwargs):
            captured["mask"] = np.asarray(mask)
            captured["quality"] = kwargs["quality"]
            return current.copy()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person = root / "person.png"
            garment = root / "garment.png"
            Image.new("RGB", (width, height), (150, 150, 150)).save(person)
            Image.new("RGB", (80, 120), "black").save(garment)
            product = Product(
                product_id="P1", name="스트레이트 데님", category="bottom", color="블랙",
                style="", purposes=[], body_shapes=[], price=0, season="", stock=True,
                fit="스트레이트핏", image_path=str(garment),
            )
            recommendation = Recommendation(
                rank=1, products=[product], total_score=0, score_breakdown={}, reasons=[]
            )
            fake_utils = SimpleNamespace(resize_and_crop=lambda image, size: image.resize(size))
            with (
                mock.patch.dict(sys.modules, {"utils": fake_utils}),
                mock.patch.object(adapter, "_load_pipeline"),
                mock.patch.object(
                    adapter, "_prepare_garment_reference",
                    return_value=Image.open(garment).convert("RGB"),
                ),
                mock.patch.object(adapter, "_check_length_gap", return_value="롱·긴바지 기장"),
                mock.patch.object(adapter, "_tryon_once", side_effect=fake_tryon),
            ):
                adapter.generate(
                    person, recommendation, root / "out.png",
                    context={
                        "lower_style_mask": lower,
                        "segmentation": seg,
                        "pose": pose,
                        "outfit": outfit,
                    },
                )

        self.assertEqual(len(editor.masks), 1)
        self.assertGreater(editor.masks[0].sum(), 0)
        self.assertEqual(captured["quality"]["transition_prompt"], "")
        self.assertTrue(any("sam3d-two-stage" in note for note in adapter.last_mask_notes))
        self.assertEqual(adapter.last_body_mesh["status"], "accepted")


if __name__ == "__main__":
    unittest.main()
