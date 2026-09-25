"""Regression checks for colour transfer and editable clothing boundaries."""
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from tryon_transition import (
    FITTED_NAME, exposed_limb_coverage, harmonize_exposed_skin, transition_envelope, transition_prompt,
)


@pytest.mark.parametrize("category,limb,clothes", [("bottom", 14, 6), ("top", 12, 3)])
@pytest.mark.parametrize("tone", [(195, 143, 112), (91, 60, 46), (235, 195, 176)])
def test_new_skin_moves_towards_source_without_flattening_shading(category, limb, clothes, tone):
    source = np.full((160, 120, 3), 245, np.uint8)
    before = np.zeros((160, 120), np.uint8)
    before[10:35, 35:85] = 1
    source[before == 1] = tone
    before[50:145, 35:85] = clothes
    after = before.copy()
    after[75:140, 40:80] = limb
    result = source.copy()
    for y in range(75, 140):
        result[y, 40:80] = np.clip(np.array(tone) + (22, -15, 30) + (y - 105) // 3, 0, 255)
    edit = before == clothes
    corrected, diagnostic = harmonize_exposed_skin(source, result, before, after, edit, category)
    exposed = after == limb
    assert diagnostic["status"] == "corrected"
    assert np.array_equal(corrected[~exposed], result[~exposed])
    def error(rgb):
        lab = cv2.cvtColor(rgb.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        ref = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
        return np.linalg.norm(np.median(lab[exposed], axis=0) - np.median(ref[before == 1], axis=0))
    assert error(corrected) < error(result)
    generated_l = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[85:130, 50:70, 0]
    corrected_l = cv2.cvtColor(corrected.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)[85:130, 50:70, 0]
    assert np.std(corrected_l) >= 0.9 * np.std(generated_l)


def test_visible_skin_and_skin_coloured_clothes_are_untouched():
    before = np.full((100, 100), 6, np.uint8)
    before[:40] = 14
    rgb = np.full((100, 100, 3), (180, 130, 100), np.uint8)
    output, stats = harmonize_exposed_skin(rgb, rgb.copy(), before, before, np.ones((100, 100), bool), "bottom")
    assert np.array_equal(output, rgb)
    assert stats["status"] == "no_new_skin"


def test_missing_reference_does_not_invent_default_skin():
    before = np.full((100, 100), 6, np.uint8)
    after = np.full_like(before, 14)
    rgb = np.full((100, 100, 3), 128, np.uint8)
    output, stats = harmonize_exposed_skin(rgb, rgb.copy(), before, after, np.ones_like(before, bool), "bottom")
    assert np.array_equal(output, rgb)
    assert stats["status"] == "no_reliable_source_skin"


def _lower_scene():
    seg = np.zeros((240, 160), np.uint8)
    seg[80:210, 40:120] = 6
    seg[30:80, 40:120] = 3
    seg[150:180, 30:40] = 13
    seg[200:225, 50:70] = 15
    points = {f"{side}_{joint}": (x, y) for side, x in (("left", 55), ("right", 105))
              for joint, y in (("hip", 85), ("knee", 150), ("ankle", 210))}
    return seg, points


def test_envelope_erases_all_old_clothes_and_allows_background_reconstruction():
    seg, points = _lower_scene()
    old = seg == 6
    changed = transition_envelope(old, seg, points, "bottom")
    assert changed[old].all()  # shrinking the mask would paste the old wide jeans back
    assert changed[130, 35]  # old side seam is now inside, not at the mask boundary
    assert not changed[np.isin(seg, (3, 13, 15))].any()


def test_missing_pose_holds_shape_correction():
    seg, _ = _lower_scene()
    assert transition_envelope(seg == 6, seg, {}, "bottom") is None


@pytest.mark.parametrize("name,expected", [("SKINNY jeans", True), ("슬림핏 티셔츠", True),
                                           ("relaxed jeans", False), ("slimming cream", False)])
def test_explicit_product_fit_trigger(name, expected):
    assert bool(FITTED_NAME.search(name)) == expected


def test_shortened_legs_filled_with_clothes_fail_the_skin_check():
    seg, points = _lower_scene()
    edit = seg == 6
    assert exposed_limb_coverage(seg, seg, edit, points, "bottom", "반바지") == 0
    after = seg.copy()
    after[(np.indices(seg.shape)[0] > 145) & edit] = 14
    assert exposed_limb_coverage(seg, after, edit, points, "bottom", "반바지") == 1
    assert exposed_limb_coverage(seg, seg, edit, points, "bottom", "긴바지") is None
    assert exposed_limb_coverage(seg, seg, edit, {}, "bottom", "반바지") is None


def test_short_to_short_detects_fabric_over_previously_bare_legs():
    seg, points = _lower_scene()
    seg[150:210, 40:120] = 14
    after = seg.copy()
    after[150:210, 40:120] = 6
    assert exposed_limb_coverage(seg, after, np.ones_like(seg, bool), points, "bottom", "반바지") == 0


def test_one_failed_leg_is_not_hidden_by_the_other_leg():
    seg, points = _lower_scene()
    after = seg.copy()
    after[150:210, 40:120] = 14
    after[180:210, 90:120] = 6
    coverage = exposed_limb_coverage(seg, after, seg == 6, points, "bottom", "반바지")
    assert coverage < 0.6


def test_reference_editor_receives_prompt_mask_and_retry_seed():
    from catvton_tryon import CatVTONTryOn
    from tryon_quality import QualityCheck, TryOnQualityReport
    person = Image.new("RGB", (120, 160), "gray")
    mask = Image.new("L", person.size, 255)
    editor = mock.Mock(available=True)
    editor.generate.return_value = person
    adapter = CatVTONTryOn(transition_editor=editor, max_retries=1)
    adapter.device = "cpu"
    rejected = TryOnQualityReport("bottom", [QualityCheck("exposed_limb_skin", 0, .6, False, True)])
    accepted = TryOnQualityReport("bottom", [QualityCheck("exposed_limb_skin", 1, .6, True, True)])
    prompt = transition_prompt("bottom", "반바지", True)
    with (mock.patch.object(adapter, "_load_pipeline") as catvton,
          mock.patch.object(adapter, "_apply_scheduler"),
          mock.patch.object(adapter, "_assess_attempt", side_effect=[rejected, accepted]),
          mock.patch.object(adapter, "_garment_sharpness", return_value=100)):
        adapter._tryon_once(person, person, mask, quality={"category": "bottom", "transition_prompt": prompt})
    assert editor.generate.call_count == 2
    assert editor.generate.call_args_list[0].kwargs["seed"] == 42
    assert editor.generate.call_args_list[1].kwargs["seed"] == 43
    assert editor.generate.call_args.kwargs["full_context"] is True
    assert editor.generate.call_args.kwargs["prompt"] == prompt
    catvton.return_value.assert_not_called()  # envelope must never reach CatVTON
    assert adapter.last_quality_reports[0]["attempts"] == 2


def test_skin_colour_reference_is_original_input_not_previous_clothing_pass():
    from catvton_tryon import CatVTONTryOn
    original = Image.new("RGB", (120, 160), (90, 60, 45))
    intermediate = Image.new("RGB", original.size, (220, 190, 170))
    labels = np.full((160, 120), 6, np.uint8)
    parser = SimpleNamespace(backend="fashn-human-parser", parse=lambda *a, **k: {"segmentation": labels})
    editor = mock.Mock(available=True)
    editor.generate.return_value = intermediate
    adapter = CatVTONTryOn(transition_editor=editor, post_quality_gate=False, max_retries=0)
    adapter.device = "cpu"
    with (mock.patch.object(adapter, "_load_pipeline"), mock.patch.object(adapter, "_apply_scheduler"),
          mock.patch("catvton_tryon.harmonize_exposed_skin", return_value=(np.asarray(intermediate), {"status": "no_new_skin"})) as transfer):
        adapter._tryon_once(intermediate, intermediate, Image.new("L", original.size, 255), quality={
            "category": "bottom", "transition_prompt": "edit", "source_person": original,
            "segmentation": labels, "parser": parser,
        })
    assert np.array_equal(transfer.call_args.args[0], np.asarray(original))


@pytest.mark.parametrize("editor_available", [False, True])
@pytest.mark.parametrize("name,length", [("스키니 진", "긴바지"), ("바이커 쇼츠", "반바지")])
def test_generate_routes_expanded_envelope_only_to_reference_editor(tmp_path, editor_available, name, length):
    from catvton_tryon import CatVTONTryOn
    from schemas import Product, Recommendation
    seg, points = _lower_scene()
    if length == "반바지":
        seg[150:210, 40:120] = 14
    source = tmp_path / "person.png"
    reference = tmp_path / "product.png"
    Image.new("RGB", (160, 240), "gray").save(source)
    Image.new("RGB", (160, 240), "black").save(reference)
    product = Product("test", name, "bottom", "", "", [], [], 0, "", True, image_path=str(reference))
    rec = Recommendation(1, [product], 0, {}, [])
    adapter = CatVTONTryOn(width=160, height=240, post_quality_gate=False,
                          transition_editor=mock.Mock(available=editor_available))
    pose = SimpleNamespace(landmarks={key: (x / 160, y / 240, 1.0) for key, (x, y) in points.items()})
    with (mock.patch.object(adapter, "_load_pipeline"),
          mock.patch.object(adapter, "_prepare_garment_reference", return_value=Image.open(reference).convert("RGB")),
          mock.patch.object(adapter, "_check_length_gap", return_value=length),
          mock.patch.object(adapter, "_tryon_once", side_effect=lambda person, *a, **k: person) as generate,
          mock.patch.dict(sys.modules, {"utils": SimpleNamespace(resize_and_crop=lambda im, size: im.resize(size))})):
        adapter.generate(source, rec, tmp_path / "result.png", {
            "lower_style_mask": seg == 6, "segmentation": seg, "pose": pose,
        })
    assert bool(generate.call_args.kwargs["quality"]["transition_prompt"]) == editor_available
    assert bool(adapter.last_raw_masks["bottom"][130, 35]) == editor_available
    assert adapter.last_raw_masks["bottom"][seg == 6].all()


@pytest.mark.parametrize("category,name,expected", [("top", "반팔 티셔츠", "반팔"),
                                                     ("bottom", "바이커 쇼츠", "쇼츠·미니 기장")])
def test_target_length_is_available_without_source_outfit(category, name, expected):
    from catvton_tryon import CatVTONTryOn
    adapter = CatVTONTryOn()
    target = adapter._check_length_gap(Image.new("RGB", (100, 100)), category, {},
                                       product=SimpleNamespace(name=name, product_id="test"))
    assert target == expected


@pytest.mark.parametrize("name,expected", [("머슬핏 반팔 티", "반팔"), ("롱슬리브", "긴팔"),
                                           ("sleeveless top", "민소매"), ("반팔 긴팔 세트", "")])
def test_explicit_sleeve_name_works_without_attribute_model(name, expected):
    from catvton_tryon import classify_reference_sleeve_length
    assert classify_reference_sleeve_length(None, None, product=SimpleNamespace(name=name)) == expected


def test_short_skirt_is_not_prompted_as_shorts():
    prompt = transition_prompt("bottom", "쇼츠·미니 기장", False, skirt=True)
    assert "short skirt" in prompt
    assert "target is shorts" not in prompt
