import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from schemas import Product, Recommendation
from shoe_tryon import OutfitTryOn, ShoeTryOn, composite_feet, edit_crop, foot_edit_mask
from virtual_tryon import TryOnNotReady


def context():
    seg = np.zeros((200, 120), np.uint8)
    seg[155:176, 25:45] = 6
    seg[155:176, 75:95] = 6
    seg[176:189, 25:45] = 15
    seg[176:189, 75:95] = 15
    pose = SimpleNamespace(landmarks={
        'left_ankle': (.29, .86, .99), 'left_foot': (.29, .92, .99),
        'right_ankle': (.71, .86, .99), 'right_foot': (.71, .92, .99),
    })
    return {'segmentation': seg, 'pose': pose}


def product(category, image):
    return Product(category, category, category, 'black', '', [], [], 0, '', True, image_path=str(image))


def test_mask_excludes_trousers_and_preserves_every_pixel_elsewhere():
    ctx = context()
    mask = foot_edit_mask(**ctx)
    assert mask[182, 35] and mask[182, 85]
    assert not mask[ctx['segmentation'] == 6].any()
    original = Image.fromarray(np.random.default_rng(42).integers(0, 256, (200, 120, 3), dtype=np.uint8))
    result = np.asarray(composite_feet(original, Image.new('RGB', (30, 20), 'red'), mask, edit_crop(mask)))
    assert np.array_equal(result[~mask], np.asarray(original)[~mask])
    assert not np.array_equal(result[mask], np.asarray(original)[mask])


@pytest.mark.parametrize('point', [(0.29, 1.01, .99), (.29, .92, .1), (float('nan'), .92, .99)])
def test_invisible_or_offscreen_foot_is_rejected(point):
    ctx = context()
    ctx['pose'].landmarks['left_foot'] = point
    with pytest.raises(TryOnNotReady, match='양쪽'):
        foot_edit_mask(**ctx)


def test_missing_foot_segmentation_is_not_replaced_with_a_leg_mask():
    ctx = context()
    ctx['segmentation'][ctx['segmentation'] == 15] = 14
    with pytest.raises(TryOnNotReady, match='가려져'):
        foot_edit_mask(**ctx)


def test_one_visible_shoe_cannot_supply_both_feet():
    ctx = context()
    ctx['segmentation'][176:189, 75:95] = 0
    ctx['pose'].landmarks['right_foot'] = (.45, .92, .99)
    with pytest.raises(TryOnNotReady, match='양쪽 발'):
        foot_edit_mask(**ctx)


def test_touching_shoes_are_allowed_when_both_toes_have_pixel_support():
    ctx = context()
    ctx['segmentation'][180:185, 45:75] = 15
    mask = foot_edit_mask(**ctx)
    assert mask[184, 35] and mask[184, 85]


@pytest.mark.parametrize('index', ['[]', '{"weight_map": []}', '{"weight_map": null}'])
def test_malformed_checkpoint_index_is_unavailable(tmp_path, index):
    files = [
        'model_index.json', 'transformer/diffusion_pytorch_model.safetensors',
        'transformer/config.json', 'vae/diffusion_pytorch_model.safetensors',
        'vae/config.json', 'text_encoder/config.json', 'text_encoder/model.safetensors.index.json',
        'scheduler/scheduler_config.json', 'tokenizer/tokenizer_config.json', 'tokenizer/tokenizer.json',
    ]
    for name in files:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{}')
    (tmp_path / 'text_encoder/model.safetensors.index.json').write_text(index)
    assert not ShoeTryOn(tmp_path).available


def test_failed_offload_setup_is_not_cached_and_next_request_can_retry(tmp_path, monkeypatch):
    import torch
    failed, ready = Mock(), Mock()
    failed.enable_model_cpu_offload.side_effect = RuntimeError('offload setup failed')
    factory = Mock(side_effect=[failed, ready])
    monkeypatch.setitem(sys.modules, 'diffusers', SimpleNamespace(
        Flux2KleinInpaintPipeline=SimpleNamespace(from_pretrained=factory)))
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(ShoeTryOn, 'available', property(lambda self: True))
    model = ShoeTryOn(tmp_path)
    with pytest.raises(RuntimeError, match='offload setup failed'):
        model._load_pipeline()
    assert model._pipeline is None
    assert model._load_pipeline() is ready
    assert factory.call_count == 2
    assert model._load_pipeline() is ready
    assert factory.call_count == 2


@pytest.mark.parametrize('fail_shoes', [False, True])
def test_three_item_request_uses_actual_shoe_reference_and_commits_only_complete_output(tmp_path, fail_shoes):
    ctx = context()
    person = tmp_path / 'person.png'
    reference = tmp_path / 'musinsa_shoe.png'
    Image.new('RGB', (120, 200), 'white').save(person)
    Image.new('RGB', (70, 40), 'blue').save(reference)
    clothing = Mock(available=True, last_warnings=[], last_quality_reports=[], last_render_kind='tryon')

    def clothes_generate(_person, recommendation, output, _context):
        assert [p.category for p in recommendation.products] == ['top', 'bottom']
        Image.new('RGB', (120, 200), 'green').save(output)
        return output

    clothing.generate.side_effect = clothes_generate
    shoes = Mock(available=True, last_report={'reference_fidelity': 'not_calibrated'})

    def shoe_generate(image, actual_reference, mask):
        assert image.getpixel((0, 0)) == (0, 128, 0)
        assert actual_reference.getpixel((0, 0)) == (0, 0, 255)
        assert mask[182, 35]
        if fail_shoes:
            raise RuntimeError('shoe stage failed')
        return composite_feet(image, Image.new('RGB', (30, 20), 'blue'), mask, edit_crop(mask))

    shoes.generate.side_effect = shoe_generate
    parser = Mock(backend='fashn-human-parser')
    parser.parse.return_value = ctx
    adapter = OutfitTryOn(clothing, shoes, parser)
    reco = Recommendation(1, [product(c, reference) for c in ['top', 'bottom', 'shoes']], 0, {}, [])
    output = tmp_path / 'final.png'
    if fail_shoes:
        with pytest.raises(RuntimeError, match='shoe stage'):
            adapter.generate(person, reco, output, ctx)
        assert not output.exists()
    else:
        assert adapter.generate(person, reco, output, ctx) == output
        assert Image.open(output).getpixel((0, 0)) == (0, 128, 0)
        assert adapter.last_quality_reports[-1]['reference_fidelity'] == 'not_calibrated'
    assert not list(tmp_path.glob('shoe_stage_*'))


def test_hidden_feet_fail_before_any_clothing_generation(tmp_path):
    clothing = Mock(available=True)
    adapter = OutfitTryOn(clothing, Mock(available=True))
    reco = Recommendation(1, [product('top', 'x'), product('shoes', 'y')], 0, {}, [])
    with pytest.raises(TryOnNotReady):
        adapter.generate('person.png', reco, tmp_path / 'final.png', {})
    clothing.generate.assert_not_called()


def test_shoes_disabled_without_complete_checkpoint(tmp_path):
    (tmp_path / 'model_index.json').write_text('{}')
    assert not ShoeTryOn(tmp_path).available
    assert 'shoes' not in OutfitTryOn(Mock(available=True), ShoeTryOn(tmp_path)).supported_categories


def test_reference_image_reaches_inpainting_and_output_cannot_modify_upper_body(tmp_path, monkeypatch):
    import torch
    ctx = context()
    mask = foot_edit_mask(**ctx)
    model = ShoeTryOn(tmp_path)
    captured = {}

    def pipeline(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(images=[Image.new('RGB', kwargs['image'].size, 'red')])

    monkeypatch.setattr(model, '_load_pipeline', lambda: pipeline)
    generator = SimpleNamespace(manual_seed=lambda seed: seed)
    monkeypatch.setattr(torch, 'Generator', lambda **kwargs: generator)
    original = Image.new('RGB', (120, 200), 'white')
    actual_product = Image.new('RGB', (100, 60), 'blue')
    result = np.asarray(model.generate(original, actual_product, mask))
    assert captured['image_reference'].getpixel((256, 256)) == (0, 0, 255)
    assert captured['mask_image'].getextrema() == (0, 255)
    assert np.array_equal(result[~mask], np.asarray(original)[~mask])
    assert model.last_report['outside_mask_preserved']
    assert model.last_report['reference_fidelity'] == 'not_calibrated'


def test_shoes_only_skips_clothing_generator(tmp_path):
    ctx = context()
    person, reference = tmp_path / 'person.png', tmp_path / 'shoe.png'
    Image.new('RGB', (120, 200), 'white').save(person)
    Image.new('RGB', (100, 60), 'blue').save(reference)
    clothing = Mock(available=True)
    shoes = Mock(available=True, last_report={})
    shoes.generate.return_value = Image.open(person).convert('RGB')
    parser = Mock(backend='fashn-human-parser')
    parser.parse.return_value = ctx
    adapter = OutfitTryOn(clothing, shoes, parser)
    adapter.generate(person, Recommendation(1, [product('shoes', reference)], 0, {}, []), tmp_path / 'out.png', ctx)
    clothing.generate.assert_not_called()


def test_clothing_only_preserves_previous_adapter_behavior(tmp_path):
    clothing = Mock(available=True, last_warnings=['existing warning'], last_quality_reports=[{'passed': True}],
                    last_render_kind='tryon')
    adapter = OutfitTryOn(clothing, Mock(available=False))
    reco = Recommendation(1, [product('top', 'top.png')], 0, {}, [])
    adapter.generate('person.png', reco, tmp_path / 'out.png', {})
    clothing.generate.assert_called_once_with('person.png', reco, tmp_path / 'out.png', {})
    assert adapter.last_warnings == ['existing warning']
    assert adapter.last_quality_reports == [{'passed': True}]
