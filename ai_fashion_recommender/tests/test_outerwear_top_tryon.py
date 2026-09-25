"""Outerwear-removal mask geometry and experimental editor regression tests."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_outerwear_parsing import evaluate_one  # noqa: E402
from rasterize_outerwear_accessories import rasterize_annotation  # noqa: E402
from outerwear_top_tryon import (  # noqa: E402
    OUTER_ONLY_REMOVAL_PROMPT,
    OUTERWEAR_REMOVAL_PROMPT,
    OuterwearTopMasks,
    OuterwearTopTryOn,
    build_official_layer_masks,
    build_outerwear_top_masks,
    composite_masked,
    hand_region_count,
    include_annotated_outerwear_accessories,
    is_outerwear_input,
    outer_ring_needs_review,
    release_pocket_hand_masks,
    release_hair_labeled_coat_masks,
    resolve_second_stage_mask,
    restore_studio_background,
    target_width_scale,
    visible_inner_reference,
)
from virtual_tryon import TryOnNotReady  # noqa: E402


HEIGHT, WIDTH = 200, 120


def pose():
    return SimpleNamespace(landmarks={
        "left_shoulder": (0.35, 0.22, 1.0),
        "right_shoulder": (0.65, 0.22, 1.0),
        "left_hip": (0.42, 0.53, 1.0),
        "right_hip": (0.58, 0.53, 1.0),
        "left_elbow": (0.20, 0.40, 1.0),
        "right_elbow": (0.80, 0.40, 1.0),
        "left_wrist": (0.18, 0.59, 1.0),
        "right_wrist": (0.82, 0.59, 1.0),
    })


def coat_masks():
    segmentation = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    segmentation[30:175, 8:112] = 3
    # Hands must remain protected even when the upper-style mask covers them.
    segmentation[112:128, 15:27] = 13
    segmentation[112:128, 93:105] = 13
    upper_style = np.isin(segmentation, (3, 12, 13))
    return segmentation, upper_style


class OuterwearMaskTests(unittest.TestCase):
    def test_accessory_polyline_annotation_is_bounded(self):
        annotation = {
            "image_size": [40, 40],
            "polylines": [{"points": [[20, 25], [20, 34]], "width": 1}],
        }
        mask = rasterize_annotation(annotation)
        self.assertTrue(mask[30, 20])
        self.assertFalse(mask[30, 5])
        annotation["polylines"][0]["points"] = [[20, 25], [50, 34]]
        with self.assertRaises(ValueError):
            rasterize_annotation(annotation)

    def test_annotated_cord_mask_adds_only_unprotected_pixels(self):
        erase = np.zeros((40, 40), dtype=bool)
        erase[10:30, 10:30] = True
        protected = np.zeros_like(erase)
        protected[34, 22] = True
        masks = OuterwearTopMasks(erase, np.zeros_like(erase), erase.copy(), protected)
        proposal = np.zeros_like(erase)
        proposal[31, 20] = True
        proposal[34, 22] = True
        revised, added = include_annotated_outerwear_accessories(masks, proposal)
        self.assertEqual(int(added.sum()), 1)
        self.assertTrue(revised.erase_mask[31, 20])
        self.assertFalse(revised.erase_mask[34, 22])
        self.assertTrue(revised.protected_mask[34, 22])
        with self.assertRaises(TryOnNotReady):
            include_annotated_outerwear_accessories(masks, np.zeros((2, 2)))

    def test_coat_proposal_releases_only_adjacent_hair_labeled_pixels(self):
        seg = np.zeros((40, 40), dtype=np.uint8)
        erase = np.zeros((40, 40), dtype=bool)
        erase[10:30, 10:30] = True
        seg[15, 30] = 2
        seg[2, 2] = 2
        seg[20, 30] = 13
        protected = np.isin(seg, (2, 13))
        masks = OuterwearTopMasks(erase, np.zeros_like(erase), erase.copy(), protected)
        proposal = np.zeros_like(erase)
        proposal[15, 30] = True
        proposal[2, 2] = True
        proposal[20, 30] = True
        revised, released = release_hair_labeled_coat_masks(masks, seg, proposal)
        self.assertEqual(int(released.sum()), 1)
        self.assertTrue(revised.erase_mask[15, 30])
        self.assertFalse(revised.protected_mask[15, 30])
        self.assertTrue(revised.protected_mask[2, 2])
        self.assertTrue(revised.protected_mask[20, 30])
        self.assertFalse(masks.erase_mask[15, 30])
        with self.assertRaises(TryOnNotReady):
            release_hair_labeled_coat_masks(masks, seg, np.zeros((4, 4)))

    def test_official_outer_and_inner_labels_reveal_fashn_class_collapse(self):
        segmentation = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        segmentation[30:110, 18:102] = 3
        official = np.zeros_like(segmentation)
        official[30:110, 18:102] = 2
        official[40:100, 48:72] = 1
        report = evaluate_one(official, segmentation, segmentation == 3, pose())

        self.assertEqual(report["outer_labeled_fashn_top"], 1.0)
        self.assertEqual(report["inner_labeled_fashn_top"], 1.0)
        self.assertEqual(report["outer_erase_recall"], 1.0)
        self.assertEqual(report["inner_erase_recall"], 1.0)

    def test_oracle_outer_only_does_not_force_edit_the_visible_inner_top(self):
        official = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        official[30:175, 8:112] = 2
        official[45:105, 48:72] = 1
        both = build_official_layer_masks(official, pose())
        outer_only = build_official_layer_masks(official, pose(), erase_inner=False)
        self.assertTrue(both.erase_mask[70, 60])
        self.assertFalse(outer_only.erase_mask[70, 60])
        self.assertTrue(outer_only.erase_mask[70, 20])
        self.assertEqual(int((outer_only.erase_mask & (official == 1)).sum()), 0)
        self.assertTrue(np.all(outer_only.protected_mask[official == 1]))

    def test_visible_inner_reference_excludes_the_coat_pixels(self):
        source = np.full((HEIGHT, WIDTH, 3), (220, 20, 20), dtype=np.uint8)
        official = np.full((HEIGHT, WIDTH), 2, dtype=np.uint8)
        official[40:100, 45:75] = 1
        source[official == 1] = (20, 220, 20)
        reference = np.asarray(visible_inner_reference(Image.fromarray(source), official))
        self.assertEqual(tuple(reference[384, 384]), (20, 220, 20))
        self.assertFalse(np.any(np.all(reference == (220, 20, 20), axis=2)))
        with self.assertRaises(TryOnNotReady):
            visible_inner_reference(Image.fromarray(source), np.zeros_like(official))

    def test_second_stage_override_keeps_pose_and_protected_regions(self):
        segmentation, style = coat_masks()
        masks = build_outerwear_top_masks(segmentation, style, pose())
        proposed = masks.outer_ring | masks.protected_mask
        resolved = resolve_second_stage_mask(masks, proposed)
        self.assertTrue(np.all(resolved[masks.target_body_mask]))
        self.assertTrue(np.all(resolved[masks.outer_ring]))
        self.assertFalse(np.any(resolved[masks.protected_mask]))
        with self.assertRaises(TryOnNotReady):
            resolve_second_stage_mask(masks, np.zeros((2, 2), dtype=np.uint8))

    def test_official_layer_mask_covers_coat_tail_and_preserves_visible_skin(self):
        official = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
        official[30:180, 8:112] = 2
        official[45:105, 48:72] = 1
        official[112:128, 15:27] = 15
        official[112:128, 93:105] = 15
        official[30:50, 48:72] = 13
        masks = build_official_layer_masks(official, pose())

        self.assertTrue(masks.erase_mask[170, 10])
        self.assertTrue(masks.erase_mask[70, 60])
        self.assertFalse(masks.erase_mask[118, 20])
        self.assertTrue(masks.protected_mask[118, 20])
        self.assertFalse(masks.erase_mask[38, 60])

    def test_pants_protection_removes_visible_trousers_from_erase_mask(self):
        segmentation, _ = coat_masks()
        segmentation[94:180, 43:77] = 6
        style = np.isin(segmentation, (3, 4, 10, 12))
        baseline = build_outerwear_top_masks(segmentation, style, pose())
        protected = build_outerwear_top_masks(
            segmentation, style, pose(), preserve_observed_pants=True
        )
        self.assertTrue(baseline.erase_mask[100, 60])
        self.assertFalse(protected.erase_mask[100, 60])
        self.assertTrue(protected.protected_mask[100, 60])
        self.assertTrue(protected.erase_mask[80, 60])

    def test_erase_mask_keeps_coat_but_target_prior_does_not_inherit_its_width(self):
        segmentation, upper_style = coat_masks()
        masks = build_outerwear_top_masks(segmentation, upper_style, pose())

        self.assertTrue(masks.erase_mask[80, 10])
        self.assertFalse(masks.target_body_mask[80, 10])
        self.assertGreater(masks.outer_ring.sum(), 100)
        self.assertLess(masks.target_body_mask[70:100].sum(), masks.erase_mask[70:100].sum())
        self.assertFalse(masks.erase_mask[118, 20])
        self.assertTrue(masks.protected_mask[118, 20])

    def test_target_prior_is_not_labeled_as_a_hidden_body_measurement(self):
        segmentation, upper_style = coat_masks()
        masks = build_outerwear_top_masks(segmentation, upper_style, pose())
        # It provides editing room at the open center even if the parser missed it.
        upper_style[55:95, 52:68] = False
        rebuilt = build_outerwear_top_masks(segmentation, upper_style, pose())
        self.assertTrue(rebuilt.target_body_mask[70, 60])
        self.assertTrue(masks.target_body_mask[70, 60])

    def test_missing_arm_landmark_fails_closed(self):
        segmentation, upper_style = coat_masks()
        broken = pose()
        del broken.landmarks["left_elbow"]
        with self.assertRaises(TryOnNotReady):
            build_outerwear_top_masks(segmentation, upper_style, broken)

    def test_outerwear_evidence_uses_type_material_or_real_layering(self):
        self.assertTrue(is_outerwear_input(SimpleNamespace(
            upper_type="니트", outer_category="해당 없음", material="퍼·플리스",
            layering_state="단일 상의",
        )))
        self.assertTrue(is_outerwear_input(SimpleNamespace(
            upper_type="셔츠", outer_category="가디건", material="면",
            layering_state="레이어드",
        )))
        self.assertFalse(is_outerwear_input(SimpleNamespace(
            upper_type="티셔츠", outer_category="해당 없음", material="면",
            layering_state="단일 상의",
        )))

    def test_target_fit_changes_prior_width_without_using_source_coat_width(self):
        self.assertEqual(target_width_scale(SimpleNamespace(fit="슬림핏", name="", item_type="")), 1.08)
        self.assertEqual(target_width_scale(SimpleNamespace(fit="레귤러핏", name="", item_type="")), 1.18)
        self.assertEqual(target_width_scale(SimpleNamespace(fit="오버핏", name="", item_type="")), 1.42)

    def test_outer_ring_warning_requires_actual_width_overflow(self):
        long_sleeve_report = {
            "outer_ring_top_fraction": 0.33,
            "after_top_width_per_shoulder": 1.42,
            "target_width_per_shoulder": 1.46,
        }
        oversized_report = {
            "outer_ring_top_fraction": 0.33,
            "after_top_width_per_shoulder": 1.70,
            "target_width_per_shoulder": 1.46,
        }
        self.assertFalse(outer_ring_needs_review(long_sleeve_report))
        self.assertTrue(outer_ring_needs_review(oversized_report))

    def test_hand_region_alert_ignores_tiny_islands_but_counts_extra_hand(self):
        segmentation = np.zeros((200, 120), dtype=np.uint8)
        segmentation[120:145, 10:25] = 13
        segmentation[120:145, 90:105] = 13
        segmentation[60:65, 60:65] = 13
        self.assertEqual(hand_region_count(segmentation), 2)
        segmentation[80:100, 60:80] = 13
        self.assertEqual(hand_region_count(segmentation), 3)

    def test_pocket_hand_can_be_edited_without_releasing_hanging_hand(self):
        segmentation, _ = coat_masks()
        segmentation[90:106, 54:66] = 13  # In front of the hip and torso.
        segmentation[135:151, 15:27] = 13  # Hanging beside the coat hem.
        style = np.isin(segmentation, (3, 4, 10, 12))
        masks = build_outerwear_top_masks(segmentation, style, pose())
        updated, selected, regions = release_pocket_hand_masks(masks, segmentation, pose())

        self.assertEqual(regions, 1)
        self.assertTrue(selected[98, 60])
        self.assertTrue(updated.erase_mask[98, 60])
        self.assertFalse(updated.protected_mask[98, 60])
        self.assertFalse(selected[143, 20])
        self.assertTrue(updated.protected_mask[143, 20])
        self.assertFalse(updated.erase_mask[143, 20])


class CompositeTests(unittest.TestCase):
    def test_pixels_outside_erase_mask_are_exactly_preserved(self):
        original = Image.new("RGB", (60, 80), "white")
        generated = Image.new("RGB", (30, 40), "red")
        mask = np.zeros((80, 60), dtype=bool)
        mask[20:60, 15:45] = True
        result = np.asarray(composite_masked(original, generated, mask, (15, 20, 45, 60)))
        source = np.asarray(original)
        np.testing.assert_array_equal(result[~mask], source[~mask])
        self.assertTrue(np.any(result[mask] != source[mask]))

    def test_studio_cleanup_removes_only_background_ghost_inside_erase(self):
        height, width = 40, 60
        original = Image.new("RGB", (width, height), (246, 247, 248))
        edited = np.asarray(original).copy()
        edited[8:32, 12:48] = (240, 240, 240)
        edited[12:29, 25:35] = (190, 80, 40)
        before_seg = np.zeros((height, width), dtype=np.uint8)
        after_seg = np.zeros_like(before_seg)
        after_seg[10:30, 22:38] = 3
        after_seg[12:29, 25:35] = 3
        after_seg[13:20, 23:25] = 0  # A background-classified gap inside target.
        erase = np.zeros_like(before_seg, dtype=bool)
        erase[8:32, 12:48] = True
        target = np.zeros_like(erase)
        target[10:30, 22:38] = True
        protected = np.zeros_like(erase)
        protected[8:11, 12:16] = True
        ring = erase & ~target & ~protected
        masks = SimpleNamespace(
            erase_mask=erase, target_body_mask=target,
            outer_ring=ring, protected_mask=protected,
        )
        result, report = restore_studio_background(
            original, Image.fromarray(edited), before_seg, after_seg, masks,
        )
        actual = np.asarray(result)
        self.assertTrue(report["studio_background_applied"])
        self.assertEqual(tuple(actual[20, 17]), (246, 247, 248))
        self.assertEqual(tuple(actual[20, 30]), (190, 80, 40))
        self.assertEqual(tuple(actual[16, 23]), (246, 247, 248))
        self.assertEqual(tuple(actual[9, 13]), (240, 240, 240))
        np.testing.assert_array_equal(actual[~erase], np.asarray(original)[~erase])

    def test_studio_cleanup_fails_closed_on_two_tone_background(self):
        height, width = 40, 60
        source = np.empty((height, width, 3), dtype=np.uint8)
        source[:, :30] = (240, 240, 240)
        source[:, 30:] = (150, 150, 150)
        edited = source.copy()
        edited[8:32, 12:48] = (220, 220, 220)
        erase = np.zeros((height, width), dtype=bool)
        erase[8:32, 12:48] = True
        target = np.zeros_like(erase)
        target[10:30, 22:38] = True
        masks = SimpleNamespace(
            erase_mask=erase, target_body_mask=target,
            outer_ring=erase & ~target, protected_mask=np.zeros_like(erase),
        )
        segmentation = np.zeros((height, width), dtype=np.uint8)
        after_segmentation = segmentation.copy()
        after_segmentation[10:30, 22:38] = 3
        result, report = restore_studio_background(
            Image.fromarray(source), Image.fromarray(edited),
            segmentation, after_segmentation, masks,
        )
        self.assertFalse(report["studio_background_applied"])
        np.testing.assert_array_equal(np.asarray(result), edited)

    def test_studio_cleanup_interpolates_horizontal_background_gradient(self):
        height, width = 40, 60
        levels = 240 - np.rint(np.arange(width) * 0.2).astype(np.uint8)
        source = np.broadcast_to(levels[None, :, None], (height, width, 3)).copy()
        edited = source.copy()
        edited[8:32, 12:48] = 250
        erase = np.zeros((height, width), dtype=bool)
        erase[8:32, 12:48] = True
        target = np.zeros_like(erase)
        target[10:30, 22:38] = True
        after_seg = np.zeros_like(erase, dtype=np.uint8)
        after_seg[target] = 3
        masks = SimpleNamespace(
            erase_mask=erase, target_body_mask=target,
            outer_ring=erase & ~target, protected_mask=np.zeros_like(erase),
        )
        result, report = restore_studio_background(
            Image.fromarray(source), Image.fromarray(edited),
            np.zeros_like(after_seg), after_seg, masks,
        )
        actual = np.asarray(result)
        self.assertTrue(report["studio_background_applied"])
        for x in (15, 45):
            self.assertLessEqual(abs(int(actual[20, x, 0]) - int(source[20, x, 0])), 1)
        np.testing.assert_array_equal(actual[~erase], source[~erase])

    def test_studio_cleanup_skips_white_top_with_weak_parser_support(self):
        height, width = 40, 60
        source = Image.new("RGB", (width, height), (245, 245, 245))
        edited = Image.new("RGB", (width, height), (250, 250, 250))
        erase = np.zeros((height, width), dtype=bool)
        erase[8:32, 12:48] = True
        target = np.zeros_like(erase)
        target[10:30, 22:38] = True
        masks = SimpleNamespace(
            erase_mask=erase, target_body_mask=target,
            outer_ring=erase & ~target, protected_mask=np.zeros_like(erase),
        )
        segmentation = np.zeros((height, width), dtype=np.uint8)
        after_segmentation = segmentation.copy()
        after_segmentation[12:15, 27:32] = 3
        result, report = restore_studio_background(
            source, edited, segmentation, after_segmentation, masks,
        )
        self.assertFalse(report["studio_background_applied"])
        np.testing.assert_array_equal(np.asarray(result), np.asarray(edited))


class _Generator:
    def manual_seed(self, seed):
        self.seed = seed
        return self


class OuterwearEditorTests(unittest.TestCase):
    def test_reference_reaches_editor_and_debug_masks_are_saved(self):
        segmentation, upper_style = coat_masks()
        after_segmentation = np.zeros_like(segmentation)
        after_segmentation[38:112, 38:82] = 3
        after_segmentation[112:128, 15:27] = 13
        after_segmentation[112:128, 93:105] = 13
        after_segmentation[132:148, 55:67] = 13
        captured = {}

        def pipeline(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "red")])

        editor = SimpleNamespace(
            available=True,
            seed=123,
            _load_pipeline=lambda: pipeline,
        )
        clothing = SimpleNamespace(
            reference_bottom_lengths={},
            _prepare_garment_reference=lambda path, category: Image.new("RGB", (50, 70), "blue"),
        )
        parser = SimpleNamespace(
            backend="fashn-human-parser",
            parse=lambda image, current_pose: {"segmentation": after_segmentation},
        )

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person_path = root / "person.png"
            reference_path = root / "top.png"
            output_path = root / "result.png"
            debug_dir = root / "debug"
            Image.new("RGB", (WIDTH, HEIGHT), "white").save(person_path)
            Image.new("RGB", (50, 70), "blue").save(reference_path)
            product = SimpleNamespace(
                category="top", image_path=str(reference_path), name="테스트 상의",
                fit="레귤러핏", item_type="티셔츠",
            )
            recommendation = SimpleNamespace(products=[product])
            outfit = SimpleNamespace(
                upper_type="퍼 코트", outer_category="코트", material="퍼·플리스",
                layering_state="레이어드",
            )
            adapter = OuterwearTopTryOn(
                clothing, editor, parser, debug_dir=debug_dir, fit_guidance=True
            )
            fake_torch = SimpleNamespace(Generator=lambda **kwargs: _Generator())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                actual = adapter.generate(person_path, recommendation, output_path, {
                    "segmentation": segmentation,
                    "upper_style_mask": upper_style,
                    "pose": pose(),
                    "outfit": outfit,
                })

            self.assertEqual(actual, output_path)
            self.assertTrue(output_path.is_file())
            self.assertEqual(captured["image_reference"].getpixel((384, 384)), (0, 0, 255))
            self.assertEqual(captured["mask_image"].getextrema(), (0, 255))
            self.assertEqual(captured["generator"].seed, 123)
            self.assertIn("regular straight silhouette", captured["prompt"])
            self.assertTrue(adapter.last_quality_reports[0]["outside_mask_preserved"])
            self.assertFalse(adapter.last_quality_reports[0]["calibrated"])
            self.assertTrue(adapter.last_quality_reports[0]["extra_hand_region_needs_review"])
            self.assertTrue(any("중복" in warning for warning in adapter.last_warnings))
            self.assertTrue((debug_dir / "result_erase.png").is_file())
            self.assertTrue((debug_dir / "result_target.png").is_file())
            self.assertTrue((debug_dir / "result_outer_ring.png").is_file())
            self.assertTrue((debug_dir / "result_overlay.jpg").is_file())
            self.assertTrue((debug_dir / "result_flux_stage.png").is_file())

    def test_second_pass_flux_uses_narrow_mask_and_preserves_first_stage_ring(self):
        segmentation, upper_style = coat_masks()
        after_segmentation = np.zeros_like(segmentation)
        after_segmentation[38:112, 38:82] = 3
        calls = []

        def pipeline(**kwargs):
            calls.append(kwargs)
            color = "red" if len(calls) == 1 else "green"
            return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, color)])

        editor = SimpleNamespace(available=True, seed=123, _load_pipeline=lambda: pipeline)
        clothing = SimpleNamespace(
            reference_bottom_lengths={},
            _prepare_garment_reference=lambda path, category: Image.new("RGB", (50, 70), "blue"),
        )
        parser = SimpleNamespace(
            backend="fashn-human-parser",
            parse=lambda image, current_pose: {"segmentation": after_segmentation},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person_path, reference_path = root / "person.png", root / "top.png"
            Image.new("RGB", (WIDTH, HEIGHT), "white").save(person_path)
            Image.new("RGB", (50, 70), "blue").save(reference_path)
            product = SimpleNamespace(
                category="top", image_path=str(reference_path), name="슬림 상의",
                fit="슬림핏", item_type="티셔츠",
            )
            adapter = OuterwearTopTryOn(
                clothing, editor, parser, debug_dir=root / "debug",
                second_pass_flux=True, fit_guidance=True,
            )
            fake_torch = SimpleNamespace(Generator=lambda **kwargs: _Generator())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                adapter.generate(
                    person_path, SimpleNamespace(products=[product]),
                    root / "result.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "pose": pose(),
                        "outfit": SimpleNamespace(
                            upper_type="퍼 코트", outer_category="코트",
                            material="퍼·플리스", layering_state="레이어드",
                        ),
                    },
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["generator"].seed, 123)
            self.assertEqual(calls[1]["generator"].seed, 124)
            self.assertEqual(calls[0]["image_reference"].getpixel((384, 384)), (127, 127, 127))
            self.assertEqual(calls[1]["image_reference"].getpixel((384, 384)), (0, 0, 255))
            self.assertIn("neutral", calls[0]["prompt"])
            self.assertIn("slim silhouette", calls[1]["prompt"])
            self.assertEqual(adapter.last_quality_reports[0]["stage"], "flux_outerwear_removal")
            self.assertEqual(adapter.last_quality_reports[1]["stage"], "flux_target_refinement")
            self.assertTrue(adapter.last_quality_reports[1]["second_stage_outer_ring_preserved"])
            self.assertTrue(any("중립 상의" in warning for warning in adapter.last_warnings))
            self.assertTrue((root / "debug/result_flux_target_stage.png").is_file())
            stage = np.asarray(Image.open(root / "debug/result_flux_stage.png").convert("RGB"))
            final = np.asarray(Image.open(root / "result.png").convert("RGB"))
            target = adapter.last_debug_masks.target_body_mask
            np.testing.assert_array_equal(final[~target], stage[~target])
            self.assertTrue(np.any(final[target] != stage[target]))

            calls.clear()
            with patch.dict(sys.modules, {"torch": fake_torch}):
                adapter.generate(
                    person_path, SimpleNamespace(products=[product]),
                    root / "result_shape.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "pose": pose(),
                        "outfit": SimpleNamespace(
                            upper_type="퍼 코트", outer_category="코트",
                            material="퍼·플리스", layering_state="레이어드",
                        ),
                        "coatless_shape_reference": Image.new("RGB", (176, 256), "magenta"),
                    },
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(calls[0]["image_reference"].getpixel((384, 384)), (255, 0, 255))
            self.assertEqual(calls[1]["image_reference"].getpixel((384, 384)), (0, 0, 255))
            self.assertTrue(adapter.last_quality_reports[0]["coatless_shape_reference_used"])

            calls.clear()
            extra = adapter.last_debug_masks.outer_ring
            measured_adapter = OuterwearTopTryOn(
                clothing, editor, parser, debug_dir=root / "debug",
                second_pass_flux=True, fit_guidance=True,
                bilateral_sleeves=True,
            )
            with patch.dict(sys.modules, {"torch": fake_torch}):
                measured_adapter.generate(
                    person_path, SimpleNamespace(products=[product]),
                    root / "result_measured.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "pose": pose(),
                        "second_stage_mask": extra,
                        "outfit": SimpleNamespace(
                            upper_type="퍼 코트", outer_category="코트",
                            material="퍼·플리스", layering_state="레이어드",
                        ),
                    },
                )
            self.assertEqual(len(calls), 2)
            self.assertIn("Both sleeves must have the same length", calls[1]["prompt"])
            self.assertEqual(
                measured_adapter.last_quality_reports[-1]["stage"],
                "flux_measured_garment_refinement",
            )
            self.assertFalse(measured_adapter.last_quality_reports[-1]["second_stage_outer_ring_preserved"])
            self.assertGreater(measured_adapter.last_quality_reports[-1]["second_stage_added_fraction"], 0)
            self.assertTrue(measured_adapter.last_quality_reports[-1]["outside_combined_edit_preserved"])
            self.assertTrue(measured_adapter.last_quality_reports[-1]["remaining_protected_pixels_preserved"])
            self.assertTrue(measured_adapter.last_quality_reports[-1]["bilateral_sleeve_guidance_used"])
            self.assertTrue((root / "debug/result_measured_second_stage_mask.png").is_file())
            measured = np.asarray(Image.open(root / "result_measured.png").convert("RGB"))
            self.assertTrue(np.any(measured[extra] != stage[extra]))

    def test_second_pass_backends_are_mutually_exclusive(self):
        with self.assertRaises(ValueError):
            OuterwearTopTryOn(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
                second_pass_catvton=True, second_pass_flux=True,
            )

    def test_oracle_outer_only_preserves_original_inner_and_uses_neutral_reference(self):
        segmentation, upper_style = coat_masks()
        official = np.zeros_like(segmentation)
        official[30:175, 8:112] = 2
        official[45:105, 48:72] = 1
        after_segmentation = np.zeros_like(segmentation)
        after_segmentation[38:112, 38:82] = 3
        captured = {}

        def pipeline(**kwargs):
            captured.update(kwargs)
            return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "red")])

        editor = SimpleNamespace(available=True, seed=123, _load_pipeline=lambda: pipeline)
        clothing = SimpleNamespace(
            reference_bottom_lengths={},
            _prepare_garment_reference=lambda path, category: Image.new("RGB", (50, 70), "blue"),
        )
        parser = SimpleNamespace(
            parse=lambda image, current_pose: {"segmentation": after_segmentation},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person_path, reference_path = root / "person.png", root / "top.png"
            Image.new("RGB", (WIDTH, HEIGHT), "white").save(person_path)
            Image.new("RGB", (50, 70), "blue").save(reference_path)
            adapter = OuterwearTopTryOn(
                clothing, editor, parser, oracle_outer_only=True,
            )
            product = SimpleNamespace(
                category="top", image_path=str(reference_path), name="상의",
                fit="레귤러핏", item_type="티셔츠",
            )
            fake_torch = SimpleNamespace(Generator=lambda **kwargs: _Generator())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                adapter.generate(
                    person_path, SimpleNamespace(products=[product]), root / "result.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "official_segmentation": official,
                        "pose": pose(),
                        "outfit": SimpleNamespace(upper_type="코트", outer_category="코트"),
                    },
                )
            result = Image.open(root / "result.png").convert("RGB")
            self.assertEqual(result.getpixel((60, 70)), (255, 255, 255))
            self.assertNotEqual(result.getpixel((20, 70)), (255, 255, 255))
            self.assertEqual(captured["prompt"], OUTER_ONLY_REMOVAL_PROMPT)
            self.assertEqual(captured["image_reference"].getpixel((384, 384)), (127, 127, 127))
            self.assertEqual(adapter.last_quality_reports[0]["stage"], "flux_oracle_outer_only")
            self.assertTrue(any("상품 상의 착장" in warning for warning in adapter.last_warnings))

            inner_adapter = OuterwearTopTryOn(
                clothing, editor, parser, oracle_outer_only=True,
                oracle_inner_reference=True,
            )
            with patch.dict(sys.modules, {"torch": fake_torch}):
                inner_adapter.generate(
                    person_path, SimpleNamespace(products=[product]), root / "inner_result.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "official_segmentation": official,
                        "pose": pose(),
                        "outfit": SimpleNamespace(upper_type="코트", outer_category="코트"),
                    },
                )
            self.assertEqual(captured["image_reference"].getpixel((384, 384)), (255, 255, 255))
            self.assertEqual(captured["image_reference"].getpixel((0, 0)), (240, 240, 240))
            self.assertIn("reference image shows the visible part", captured["prompt"])
            self.assertTrue(inner_adapter.last_quality_reports[0]["oracle_inner_reference_used"])

    def test_inner_reference_requires_coat_only_oracle(self):
        with self.assertRaises(ValueError):
            OuterwearTopTryOn(
                SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
                oracle_inner_reference=True,
            )

    def test_second_pass_catvton_receives_only_the_narrow_target_mask(self):
        segmentation, upper_style = coat_masks()
        after_segmentation = np.zeros_like(segmentation)
        after_segmentation[38:112, 38:82] = 3
        captured = {}

        def pipeline(**kwargs):
            captured["flux_prompt"] = kwargs["prompt"]
            captured["flux_reference"] = kwargs["image_reference"]
            return SimpleNamespace(images=[Image.new("RGB", kwargs["image"].size, "red")])

        class Clothing:
            available = True
            reference_bottom_lengths = {}
            last_quality_reports = [{"backend": "fake-catvton"}]
            last_warnings = ["catvton checked"]

            @staticmethod
            def _prepare_garment_reference(path, category):
                return Image.new("RGB", (50, 70), "blue")

            @staticmethod
            def generate(source, recommendation, destination, context):
                captured.update(context)
                with Image.open(source) as opened:
                    Image.new("RGB", opened.size, "green").save(destination)
                return destination

        editor = SimpleNamespace(available=True, seed=123, _load_pipeline=lambda: pipeline)
        parser = SimpleNamespace(
            backend="fashn-human-parser",
            parse=lambda image, current_pose: {"segmentation": after_segmentation},
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            person_path, reference_path = root / "person.png", root / "top.png"
            Image.new("RGB", (WIDTH, HEIGHT), "white").save(person_path)
            Image.new("RGB", (50, 70), "blue").save(reference_path)
            product = SimpleNamespace(
                category="top", image_path=str(reference_path), name="슬림 상의",
                fit="슬림핏", item_type="티셔츠",
            )
            adapter = OuterwearTopTryOn(
                Clothing(), editor, parser, debug_dir=root / "debug",
                second_pass_catvton=True,
            )
            fake_torch = SimpleNamespace(Generator=lambda **kwargs: _Generator())
            with patch.dict(sys.modules, {"torch": fake_torch}):
                adapter.generate(
                    person_path,
                    SimpleNamespace(products=[product]),
                    root / "result.png",
                    {
                        "segmentation": segmentation,
                        "upper_style_mask": upper_style,
                        "pose": pose(),
                        "outfit": SimpleNamespace(
                            upper_type="퍼 코트", outer_category="코트",
                            material="퍼·플리스", layering_state="레이어드",
                        ),
                    },
                )

            np.testing.assert_array_equal(captured["upper_mask"], adapter.last_debug_masks.target_body_mask)
            np.testing.assert_array_equal(
                captured["upper_style_mask"], adapter.last_debug_masks.target_body_mask
            )
            self.assertNotIn("outfit", captured)
            self.assertEqual(captured["flux_prompt"], OUTERWEAR_REMOVAL_PROMPT)
            self.assertEqual(captured["flux_reference"].getpixel((384, 384)), (127, 127, 127))
            self.assertEqual(adapter.last_quality_reports[0]["stage"], "flux_outerwear_removal")
            self.assertEqual(adapter.last_quality_reports[-1]["stage"], "catvton_target_refinement")
            self.assertIn("catvton checked", adapter.last_warnings)
            final = np.asarray(Image.open(root / "result.png").convert("RGB"))
            flux = np.asarray(Image.open(root / "debug/result_flux_stage.png").convert("RGB"))
            target = adapter.last_debug_masks.target_body_mask
            np.testing.assert_array_equal(final[~target], flux[~target])
            self.assertTrue(np.any(final[target] != flux[target]))

    def test_non_outerwear_input_is_not_silently_edited(self):
        adapter = OuterwearTopTryOn(
            SimpleNamespace(), SimpleNamespace(available=True), SimpleNamespace()
        )
        recommendation = SimpleNamespace(products=[SimpleNamespace(category="top")])
        outfit = SimpleNamespace(
            upper_type="티셔츠", outer_category="해당 없음", material="면",
            layering_state="단일 상의",
        )
        with self.assertRaises(TryOnNotReady):
            adapter.generate("missing.png", recommendation, "out.png", {"outfit": outfit})


if __name__ == "__main__":
    unittest.main()
