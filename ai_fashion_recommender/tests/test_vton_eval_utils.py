import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from vton_eval_utils import cluster_ci, is_short_bottom, is_wide_bottom


def test_resume_rejects_changed_inputs_and_unversioned_results(tmp_path):
    import pytest
    from vton_eval_utils import verify_run_manifest
    result = tmp_path / "results.jsonl"
    config = {"seed": 42, "products": ["a", "b"]}
    verify_run_manifest(result, config)
    result.write_text('{"pair":"a"}\n', encoding="utf-8")
    verify_run_manifest(result, config)
    with pytest.raises(ValueError, match="변경"):
        verify_run_manifest(result, {**config, "seed": 43})
    legacy = tmp_path / "legacy.jsonl"
    legacy.write_text('{"pair":"b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="manifest"):
        verify_run_manifest(legacy, config)


def test_resampled_people_keep_multiplicity_after_regrouping():
    class FixedDraw:
        def choice(self, *args, **kwargs):
            return np.array([0, 0, 1])

    def person_mean(items):
        groups = defaultdict(list)
        for row in items:
            groups[row["person"]].append(row["value"])
        return np.mean([np.mean(g) for g in groups.values()])

    rows = [{"person": name, "value": value} for name, value in (("a", 0), ("b", 9), ("c", 30))]
    assert cluster_ci(rows, person_mean, n=1, rng=FixedDraw()) == (13, 3, 3)
    assert [row["person"] for row in rows] == ["a", "b", "c"]


def test_empty_cluster_sample_and_nonfinite_statistics():
    assert all(np.isnan(v) for v in cluster_ci([], lambda s: 0))
    result = cluster_ci([{"person": "a"}], lambda s: float("inf"), n=2)
    assert np.isnan(result[1]) and np.isnan(result[2])


def test_seller_text_filters_short_bottoms_without_model_labels():
    for row in ({"name": "Hot Pants"}, {"name": "Half Tights"},
                {"name": "Wide Shorts"}, {"detail_category": "바지 > 숏팬츠"}):
        assert is_short_bottom(row)
        assert not is_wide_bottom(row)
    assert is_wide_bottom({"name": "Curved Wide Pants"})
    assert is_wide_bottom({"name": "벌룬 팬츠", "catalog_length": "반바지"})
    assert not is_short_bottom({"name": "팬츠", "catalog_length": "반바지"})
