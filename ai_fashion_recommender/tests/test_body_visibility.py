from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from body_visibility import assess_body_visibility, with_body_visibility


def sample():
    outfit = SimpleNamespace(fit="레귤러핏 추정", lower_fit="스트레이트핏 추정", attribute_sources={})
    return outfit, {"backend": "fashn-human-parser", "segmentation": np.full((100, 100), 6)}


@pytest.mark.parametrize('label', [4, 5])
def test_skirt_or_dress_blocks_silhouette_measurement(label):
    outfit, parsed = sample()
    parsed['segmentation'][50:] = label
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'occluded'
    assert not result['passed']
    assert '치마' in result['issues'][0]


def test_isolated_parser_noise_does_not_reject_regular_clothes():
    outfit, parsed = sample()
    parsed['segmentation'][0, 0] = 5
    assert assess_body_visibility(outfit, parsed)['passed']


@pytest.mark.parametrize('field,label', [('fit', '오버핏'), ('fit', '여유핏'), ('lower_fit', '와이드핏'), ('lower_fit', '배기핏')])
def test_loose_clothes_are_blocked_using_existing_attribute_evidence(field, label):
    outfit, parsed = sample()
    setattr(outfit, field, label)
    outfit.attribute_sources[field] = 'trained_head'
    assert assess_body_visibility(outfit, parsed)['status'] == 'occluded'


def test_mask_width_alone_is_uncertainty_not_a_body_size_judgement():
    outfit, parsed = sample()
    outfit.lower_fit = '와이드핏 추정'
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'uncertain'
    assert not result['passed']
    assert not result['calibrated']


def test_missing_parser_or_unknown_fit_cannot_silently_pass():
    outfit, parsed = sample()
    parsed['backend'] = 'pose-guided-fallback'
    assert not assess_body_visibility(outfit, parsed)['passed']
    outfit, parsed = sample()
    outfit.fit = '분석 보류'
    assert not assess_body_visibility(outfit, parsed)['passed']


def test_padded_outerwear_is_rejected_even_with_regular_fit_label():
    outfit, parsed = sample()
    outfit.outer_category = '패딩'
    assert assess_body_visibility(outfit, parsed)['status'] == 'occluded'


def test_visibility_does_not_override_other_quality_failures():
    outfit, parsed = sample()
    result = with_body_visibility({'passed': False, 'issues': ['blur']}, outfit, parsed)
    assert not result['passed']
    assert result['issues'] == ['blur']


def test_no_body_width_gender_or_weight_filter_for_regular_clothing():
    outfit, parsed = sample()
    for width in (40, 200):
        parsed['segmentation'] = np.full((100, width), 6)
        assert assess_body_visibility(outfit, parsed)['passed']
