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

from body_shape_prior import (  # noqa: E402
    current_fit_label,
    estimate_body_shape_prior,
    fit_level,
    fit_transition_needed,
    target_fit_label,
)
from tryon_transition import transition_envelope, transition_prompt  # noqa: E402
from catvton_tryon import CatVTONTryOn, agnostic_upper_mask  # noqa: E402
from schemas import Product, Recommendation  # noqa: E402


H, W = 320, 180
POINTS = {
    "left_shoulder": (55.0, 55.0), "right_shoulder": (125.0, 55.0),
    "left_elbow": (42.0, 105.0), "right_elbow": (138.0, 105.0),
    "left_wrist": (38.0, 155.0), "right_wrist": (142.0, 155.0),
    "left_hip": (68.0, 155.0), "right_hip": (112.0, 155.0),
    "left_knee": (70.0, 230.0), "right_knee": (110.0, 230.0),
    "left_ankle": (72.0, 300.0), "right_ankle": (108.0, 300.0),
}


def scene():
    seg = np.zeros((H, W), np.uint8)
    seg[45:160, 28:152] = 3
    seg[150:260, 22:158] = 6
    seg[260:305, 55:82] = 6
    seg[260:305, 98:125] = 6
    return seg


class FitVocabularyTests(unittest.TestCase):
    def test_project_labels_map_from_narrow_to_wide(self):
        self.assertEqual(fit_level("top", "슬림핏"), 0)
        self.assertEqual(fit_level("top", "오버핏"), 3)
        self.assertEqual(fit_level("bottom", "세미와이드"), 2)
        self.assertEqual(fit_level("bottom", "와이드핏"), 3)

    def test_only_meaningful_narrowing_routes_to_transition_editor(self):
        self.assertTrue(fit_transition_needed("top", "오버핏", "레귤러핏"))
        self.assertTrue(fit_transition_needed("bottom", "와이드", "스트레이트"))
        self.assertFalse(fit_transition_needed("bottom", "와이드", "세미와이드"))
        self.assertFalse(fit_transition_needed("top", "레귤러핏", "오버핏"))

    def test_outfit_and_product_vocabulary_are_read_without_mutation(self):
        outfit = SimpleNamespace(
            fit="오버핏", lower_fit="와이드핏", pant_leg_shape="세미와이드",
            attribute_sources={"fit": "shared_fit_head", "pant_leg_shape": "shared_fit_head"},
        )
        self.assertEqual(current_fit_label(outfit, "top"), ("오버핏", "shared_fit_head"))
        self.assertEqual(current_fit_label(outfit, "bottom"), ("세미와이드", "shared_fit_head"))
        product = SimpleNamespace(fit="스트레이트핏", name="베이직 팬츠")
        self.assertEqual(target_fit_label(product, "bottom"), "스트레이트핏")

    def test_explicit_product_fit_overrides_a_conflicting_photo_prediction(self):
        skinny = SimpleNamespace(fit="와이드핏", name="하이웨이스트 스키니 진")
        semiwide = SimpleNamespace(fit="슬림핏", name="세미와이드 데님")
        oversized = SimpleNamespace(fit="슬림핏", name="오버핏 셔츠")
        self.assertEqual(fit_level("bottom", target_fit_label(skinny, "bottom")), 0)
        self.assertEqual(fit_level("bottom", target_fit_label(semiwide, "bottom")), 2)
        self.assertEqual(fit_level("top", target_fit_label(oversized, "top")), 3)

    def test_conflicting_name_fit_terms_use_structured_fit_or_abstain(self):
        ambiguous = SimpleNamespace(fit="레귤러핏", name="슬림 와이드 데님")
        ungrounded = SimpleNamespace(fit="", name="슬림 와이드 데님")
        self.assertEqual(target_fit_label(ambiguous, "bottom"), "레귤러핏")
        self.assertEqual(target_fit_label(ungrounded, "bottom"), "")

    def test_seller_fit_is_kept_when_marketing_name_differs(self):
        product = SimpleNamespace(
            seller_fit="여유핏", fit="레귤러핏", name="오버핏 니트"
        )
        self.assertEqual(target_fit_label(product, "top"), "여유핏")

    def test_transition_prompt_explains_fit_change_without_claiming_measurements(self):
        prompt = transition_prompt(
            "bottom", "긴바지", False,
            source_fit="와이드", target_fit="스트레이트", body_prior_confidence=0.7,
        )
        self.assertIn("와이드", prompt)
        self.assertIn("스트레이트", prompt)
        self.assertIn("Do not infer exact measurements", prompt)


class BodyShapePriorTests(unittest.TestCase):
    def test_wide_fit_increases_uncertainty_not_the_central_body(self):
        seg = scene()
        observed = seg != 0
        slim = estimate_body_shape_prior(
            seg, POINTS, "bottom", "슬림", fit_source="shared_fit_head", observed_mask=observed
        )
        wide = estimate_body_shape_prior(
            seg, POINTS, "bottom", "와이드", fit_source="shared_fit_head", observed_mask=observed
        )
        self.assertIsNotNone(slim)
        self.assertIsNotNone(wide)
        self.assertGreater(wide.plausible_mask.sum(), slim.plausible_mask.sum())
        self.assertLess(wide.central_mask.sum(), observed.sum())
        # The anatomical core keeps the two lower legs separate rather than
        # treating the whole wide trouser rectangle as a single thick body.
        self.assertFalse(wide.central_mask[280, 90])
        self.assertTrue(wide.central_mask[280, 72])

    def test_layered_upper_prior_has_lower_confidence(self):
        seg = scene()
        single = estimate_body_shape_prior(
            seg, POINTS, "top", "오버핏", fit_source="shared_fit_head", layering_state="단일 상의"
        )
        layered = estimate_body_shape_prior(
            seg, POINTS, "top", "오버핏", fit_source="shared_fit_head", layering_state="레이어드"
        )
        self.assertIsNotNone(single)
        self.assertIsNotNone(layered)
        self.assertLess(layered.confidence, single.confidence)
        self.assertFalse(layered.to_diagnostic()["calibrated"])

    def test_missing_required_joints_fails_closed(self):
        self.assertIsNone(estimate_body_shape_prior(scene(), {}, "bottom", "와이드"))

    def test_masks_and_confidence_keep_input_shape(self):
        prior = estimate_body_shape_prior(scene(), POINTS, "top", "레귤러핏")
        self.assertEqual(prior.central_mask.shape, (H, W))
        self.assertEqual(prior.plausible_mask.shape, (H, W))
        self.assertEqual(prior.confidence_map.shape, (H, W))
        self.assertTrue(np.all(prior.central_mask <= prior.plausible_mask))
        self.assertGreater(prior.confidence_map[90, 90], 0)
        self.assertEqual(float(prior.confidence_map[0, 0]), 0.0)


class BodyPriorTryOnIntegrationTests(unittest.TestCase):
    def test_wide_to_straight_uses_existing_transition_editor(self):
        seg = scene()
        lower = np.isin(seg, (4, 5, 6, 7))
        pose = SimpleNamespace(
            landmarks={name: (x / W, y / H, 0.99) for name, (x, y) in POINTS.items()}
        )
        outfit = SimpleNamespace(
            fit="레귤러핏",
            lower_fit="와이드핏",
            pant_leg_shape="와이드",
            sleeve_length="긴팔",
            layering_state="단일 상의",
            attribute_sources={"pant_leg_shape": "shared_fit_head", "lower_fit": "shared_fit_head"},
        )
        editor = SimpleNamespace(available=True)
        adapter = CatVTONTryOn(
            width=90, height=160, transition_editor=editor, post_quality_gate=False
        )
        captured = {}

        def fake_tryon(current, _garment, mask, **kwargs):
            captured["mask"] = np.asarray(mask)
            captured["quality"] = kwargs["quality"]
            return current.copy()

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            person = directory / "person.png"
            garment = directory / "garment.png"
            Image.new("RGB", (W, H), (128, 128, 128)).save(person)
            Image.new("RGB", (80, 120), "black").save(garment)
            product = Product(
                product_id="P1", name="베이직 데님", category="bottom", color="블랙",
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
                mock.patch.object(adapter, "_prepare_garment_reference", return_value=Image.open(garment).convert("RGB")),
                mock.patch.object(adapter, "_check_length_gap", return_value="긴바지"),
                mock.patch.object(adapter, "_tryon_once", side_effect=fake_tryon),
            ):
                adapter.generate(
                    person,
                    recommendation,
                    directory / "out.png",
                    context={
                        "lower_style_mask": lower,
                        "segmentation": seg,
                        "pose": pose,
                        "outfit": outfit,
                    },
                )

        self.assertIn("bottom", adapter.last_body_priors)
        self.assertIn("bottom", adapter.last_body_prior_maps)
        self.assertTrue(any("fit-transition(와이드->스트레이트핏)" in note for note in adapter.last_mask_notes))
        self.assertIn("와이드", captured["quality"]["transition_prompt"])
        self.assertIn("스트레이트핏", captured["quality"]["transition_prompt"])
        self.assertGreater(captured["mask"].sum(), 0)
        self.assertFalse(adapter.last_raw_masks["bottom"][seg == 3].any())

    def test_upper_fit_transition_does_not_edit_the_kept_pants(self):
        seg = scene()
        upper = seg == 3
        pose = SimpleNamespace(
            landmarks={name: (x / W, y / H, 0.99) for name, (x, y) in POINTS.items()}
        )
        outfit = SimpleNamespace(
            fit="오버핏", sleeve_length="긴팔", layering_state="단일 상의",
            attribute_sources={"fit": "shared_fit_head"},
        )
        adapter = CatVTONTryOn(
            width=90, height=160, transition_editor=SimpleNamespace(available=True),
            post_quality_gate=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            person = directory / "person.png"
            garment = directory / "garment.png"
            Image.new("RGB", (W, H), (128, 128, 128)).save(person)
            Image.new("RGB", (80, 120), "black").save(garment)
            product = Product(
                product_id="T1", name="베이직 셔츠", category="top", color="블랙",
                style="", purposes=[], body_shapes=[], price=0, season="", stock=True,
                fit="레귤러핏", image_path=str(garment),
            )
            recommendation = Recommendation(
                rank=1, products=[product], total_score=0, score_breakdown={}, reasons=[]
            )
            fake_utils = SimpleNamespace(resize_and_crop=lambda image, size: image.resize(size))
            with (
                mock.patch.dict(sys.modules, {"utils": fake_utils}),
                mock.patch.object(adapter, "_load_pipeline"),
                mock.patch.object(adapter, "_prepare_garment_reference", return_value=Image.open(garment).convert("RGB")),
                mock.patch.object(adapter, "_tryon_once", side_effect=lambda current, *_args, **_kwargs: current.copy()),
            ):
                adapter.generate(
                    person, recommendation, directory / "out.png",
                    context={"upper_style_mask": upper, "segmentation": seg, "pose": pose, "outfit": outfit},
                )

        self.assertTrue(any("fit-transition" in note for note in adapter.last_mask_notes))
        policy_mask = agnostic_upper_mask(upper, seg, POINTS)
        base_envelope = transition_envelope(policy_mask, seg, POINTS, "top")
        prior_extension = adapter.last_raw_masks["top"] & ~base_envelope
        self.assertFalse(prior_extension[seg == 6].any())


if __name__ == "__main__":
    unittest.main()
