"""평가 누락과 재생성 성공률의 과대 집계 방지."""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compare_vton_variants import compare_seeds, flag_summary, load_quality, paired


def write_rows(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def quality(pair, *, retry=False, structural=False):
    return {"pair": pair, "category": "top", "person": "person1", "checks": [
        {"name": "coverage", "value": 0.2, "passed": not retry, "retryable": True},
        {"name": "preservation", "value": 20, "passed": not structural, "retryable": False},
    ], "penalty": float(retry) + 0.25 * structural}


def test_skipped_is_separate_from_pass(tmp_path):
    path = write_rows(tmp_path / "q.jsonl", [{"pair": "a", "category": "top", "skipped": "parser", "checks": []}])
    with pytest.raises(ValueError, match="미검사"):
        load_quality(path)
    result = flag_summary(path)["top"]
    assert result == {"pairs": 1, "unassessed": 1}


def test_cleared_retry_flag_does_not_mean_all_checks_passed(tmp_path):
    first = write_rows(tmp_path / "a.jsonl", [quality("a", retry=True), quality("b", structural=True)])
    second = write_rows(tmp_path / "b.jsonl", [quality("a", structural=True), quality("b")])
    result = compare_seeds(first, [second])
    assert result["retry_flags_cleared_in_second_attempt"] == 1
    assert result["selected_all_checks_passed_after_retry"] == 0
    # 구조적 실패에는 재생성을 요청하지 않으므로 두 번째 시드가 좋아도 교체하지 않는다.
    assert result["selected_any_failure_rate"] == 1.0


def test_missing_seed_result_is_not_silently_dropped(tmp_path):
    first = write_rows(tmp_path / "a.jsonl", [quality("a"), quality("b")])
    second = write_rows(tmp_path / "b.jsonl", [quality("a"), quality("b")])
    third = write_rows(tmp_path / "c.jsonl", [quality("a")])
    with pytest.raises(ValueError, match="누락"):
        compare_seeds(first, [second, third])


def test_failed_generation_prevents_adoption(tmp_path):
    base = write_rows(tmp_path / "base.jsonl", [{"pair": "a", "category": "top"}])
    var = write_rows(tmp_path / "var.jsonl", [{"pair": "a", "category": "top", "error": "OOM"}])
    q = write_rows(tmp_path / "q.jsonl", [quality("a")])
    with pytest.raises(ValueError, match="생성 오류"):
        paired(base, q, var, q, "top")


def test_missing_quality_result_prevents_adoption(tmp_path):
    base = write_rows(tmp_path / "base.jsonl", [{"pair": "a", "category": "top"}])
    q = write_rows(tmp_path / "q.jsonl", [quality("b")])
    with pytest.raises(ValueError, match="누락"):
        paired(base, q, base, q, "top")


def test_lower_width_does_not_fallback_to_different_body_part(monkeypatch):
    import compare_vton_variants as comparison
    monkeypatch.setattr(comparison, "cluster_ci", lambda *args, **kwargs: (0.0, 0.0, 0.0))
    rows = []
    for pid in ("wide", "slim"):
        rows.append({"pair": pid, "person": "p", "product_id": pid, "seller_fit": "레귤러핏",
                     "base": {"width_orig": {"shin": 0.2}, "width_res": {"shin": 0.2, "thigh": 0.5}},
                     "var": {"width_res": {"shin": None, "thigh": 0.5}},
                     "bq": {}, "vq": {}})
    result = comparison.compare_lower_shape(rows, {"wide": {"name": "와이드 팬츠"}, "slim": {"name": "슬림 팬츠"}})
    assert result["pairs"] == 0
    assert not result["criteria"][0]["passed"]
    assert not result["adopt"]
