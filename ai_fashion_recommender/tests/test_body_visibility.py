from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from body_visibility import (
    assess_body_visibility,
    body_shape_analysis_allowed,
    with_body_visibility,
)


def sample():
    outfit = SimpleNamespace(fit="레귤러핏 추정", lower_fit="스트레이트핏 추정", attribute_sources={})
    return outfit, {"backend": "fashn-human-parser", "segmentation": np.full((100, 100), 6)}


@pytest.mark.parametrize('label', [4, 5])
def test_skirt_or_dress_blocks_input_before_body_shape_analysis(label):
    outfit, parsed = sample()
    parsed['segmentation'][50:] = label
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'occluded'
    assert not result['passed']
    assert any('치마' in issue for issue in result['issues'])
    assert result['issues'][0].count('\n') == 1
    assert not result['body_shape_reliable']


def test_isolated_parser_noise_does_not_reject_regular_clothes():
    outfit, parsed = sample()
    parsed['segmentation'][0, 0] = 5
    assert assess_body_visibility(outfit, parsed)['passed']


@pytest.mark.parametrize('field,label', [('fit', '오버핏'), ('fit', '여유핏'), ('lower_fit', '와이드핏'), ('lower_fit', '배기핏')])
def test_loose_clothes_warn_but_continue_using_existing_attribute_evidence(field, label):
    outfit, parsed = sample()
    setattr(outfit, field, label)
    outfit.attribute_sources[field] = 'trained_head'
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'uncertain'
    assert result['passed']
    assert not result['issues']
    assert any('덜 정확' in message for message in result['warnings'])


def test_thick_outerwear_and_wide_pants_warn_without_blocking():
    outfit, parsed = sample()
    outfit.outer_category = '패딩'
    outer = assess_body_visibility(outfit, parsed)
    assert outer['passed']
    assert not outer['issues']
    assert outer['warnings']

    outfit.outer_category = ''
    outfit.lower_fit = '와이드핏'
    outfit.attribute_sources['lower_fit'] = 'trained_head'
    allowed = assess_body_visibility(outfit, parsed)
    assert allowed['passed']
    assert not allowed['issues']


def test_mask_width_alone_warns_instead_of_rejecting_the_photo():
    # 마스크 폭만 근거인 '넉넉함'으로 막으면 몸선이 보이는 사진의 92%가 거절된다
    # (reports/body_visibility_2026-09-23.md, 사람 핏 라벨 464장).
    outfit, parsed = sample()
    outfit.lower_fit = '와이드핏 추정'
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'uncertain'
    assert result['passed']
    assert not result['issues']
    assert result['warnings'][0].count('\n') == 2
    assert not result['calibrated']


def test_missing_parser_or_unknown_fit_warns_without_blocking():
    for setup in ('parser', 'fit'):
        outfit, parsed = sample()
        if setup == 'parser':
            parsed['backend'] = 'pose-guided-fallback'
        else:
            outfit.fit = '분석 보류'
        result = assess_body_visibility(outfit, parsed)
        assert result['status'] == 'uncertain'
        assert result['passed'] and result['warnings'] and not result['issues']


def test_multiple_loose_fit_signals_remain_warnings_without_retake():
    outfit, parsed = sample()
    outfit.fit = '오버핏'
    outfit.attribute_sources['fit'] = 'trained_head'
    outfit.lower_fit = '와이드핏 추정'
    result = assess_body_visibility(outfit, parsed)
    assert result['passed'] and not result['issues'] and result['warnings']
    assert result['status'] == 'uncertain'


def test_warnings_reach_the_caller_alongside_existing_quality_warnings():
    outfit, parsed = sample()
    outfit.fit = '분석 보류'
    result = with_body_visibility({'passed': True, 'issues': [], 'warnings': ['측면 기울기']}, outfit, parsed)
    assert result['passed']
    assert result['warnings'][0] == '측면 기울기'
    assert len(result['warnings']) > 1


def test_occluded_photo_is_rejected_before_photo_body_shape_rules():
    outfit, parsed = sample()
    parsed['segmentation'][50:] = 5
    result = with_body_visibility({'passed': True, 'issues': [], 'warnings': []}, outfit, parsed)
    assert not result['passed']
    assert not body_shape_analysis_allowed(result)
    assert body_shape_analysis_allowed(result, has_circumferences=True)


def test_padded_outerwear_warns_without_rejecting_the_photo():
    outfit, parsed = sample()
    outfit.outer_category = '패딩'
    result = assess_body_visibility(outfit, parsed)
    assert result['status'] == 'uncertain'
    assert result['passed'] and not result['issues']
    assert result['warnings']


def test_visibility_does_not_override_other_quality_failures():
    outfit, parsed = sample()
    result = with_body_visibility({'passed': False, 'issues': ['blur']}, outfit, parsed)
    assert not result['passed']
    assert result['issues'] == ['blur']


def test_non_front_and_skirt_issues_are_both_preserved():
    outfit, parsed = sample()
    parsed['segmentation'][50:] = 5
    result = with_body_visibility({
        'passed': False,
        'issues': ['몸을 정면으로 향한 전신사진을 올려주세요.'],
        'warnings': [],
    }, outfit, parsed)
    assert len(result['issues']) == 2
    assert any('정면' in issue for issue in result['issues'])
    assert any('치마·원피스' in issue for issue in result['issues'])


def test_no_body_width_gender_or_weight_filter_for_regular_clothing():
    outfit, parsed = sample()
    for width in (40, 200):
        parsed['segmentation'] = np.full((100, width), 6)
        assert assess_body_visibility(outfit, parsed)['passed']
