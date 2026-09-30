"""실측 → 선택 옵션 → 실제 최종 순위까지의 회귀 검사."""
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from product_measurements import normalize_size_table
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine
from musinsa_live_search import MusinsaLiveSearch, ShoppingProduct
from schemas import UserProfile, PoseAnalysis, OutfitAnalysis
from size_fit import compare_sizes, size_score, blend_size_score


def record(width=54, length=70):
    return normalize_size_table("MS1", {"data": {"sizes": [{"name": "M", "items": [
        {"name": "가슴단면", "value": width}, {"name": "총장", "value": length}]}]}})


REFERENCE = {"chest_width_cm": 54, "length_cm": 70}


def test_size_uses_weakest_dimension_and_missing_is_distinct():
    good = compare_sizes(record(), "top", REFERENCE)
    small = compare_sizes(record(45), "top", REFERENCE)
    missing = compare_sizes({}, "top", REFERENCE)
    assert size_score(good) == 100
    assert size_score(small) == 0
    assert size_score(missing) is None
    assert blend_size_score(80, [good]) == (83, 100, 1)
    assert blend_size_score(80, [small])[0] < blend_size_score(80, [missing])[0]
    assert blend_size_score(80, [good, missing]) == (79.25, 75, .5)
    assert "차이가 커요" in small["summary"]


def test_sold_out_without_reference_and_unmeasured_variant():
    table = record()
    table["sizes"][0]["available"] = False
    assert compare_sizes(table, "top")["status"] == "no_available_sizes"
    table["variants"] = [{"size_label": "XL", "available": True}]
    assert compare_sizes(table, "top", REFERENCE)["status"] == "insufficient_measurements"


def test_color_specific_stock_is_not_promised():
    table = record()
    table["sizes"][0]["available"] = True
    table["color_options"] = ["블랙", "화이트"]
    assert "색상·사이즈 옵션" in compare_sizes(table, "top", REFERENCE)["summary"]


def test_inventory_expires_even_in_a_long_running_engine():
    table = record()
    table["fetched_at"] = "2020-01-01T00:00:00+00:00"
    table["sizes"][0]["available"] = False
    result = compare_sizes(table, "top", REFERENCE)
    assert result["status"] == "compared"
    assert result["availability"] is None
    assert table["sizes"][0]["available"] is False


@pytest.mark.parametrize("shape", ["모래시계체형", "마름모꼴체형", "둥근체형"])
def test_circumference_shapes_keep_balanced_catalog_guidance(shape):
    catalog = ProductCatalog(ROOT / "data" / "products.csv")
    engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", catalog)
    from schemas import GOAL_BALANCE
    profile = UserProfile(silhouette_goal=GOAL_BALANCE, chest_cm=100, waist_cm=90, hip_cm=100)
    pose = PoseAnalysis(True, .9, shape, 1, .5, .55, "정면", .9)
    outfit = OutfitAnalysis("test", "화이트", "블랙", "보통 조합", [], "캐주얼")
    top = engine._garment(next(p for p in catalog.products if p.category == "top"), "top", outfit)
    bottom = engine._garment(None, "bottom", outfit)
    top["body_shapes"] = ["균형형"]
    _, _, rules = engine._silhouette_score(top, bottom, profile, pose)
    assert "R-BOD-04" in rules


def test_body_circumference_and_height_never_substitute_reference_clothing():
    profile = UserProfile(height_cm=180, chest_cm=108, waist_cm=80, hip_cm=100)
    assert compare_sizes(record(), "top", profile.reference_measurements.get("top"))["status"] == "needs_reference"


def test_final_candidate_score_is_not_overwritten_by_diagnostics():
    catalog = ProductCatalog(ROOT / "data" / "products.csv")
    engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", catalog)
    template = next(p for p in catalog.products if p.category == "top")
    profile = UserProfile(reference_measurements={"top": REFERENCE})
    pose = PoseAnalysis(True, .9, "사각체형", 1, .5, .55, "정면")
    outfit = OutfitAnalysis("test", "화이트", "블랙", "보통 조합", [], "캐주얼")
    scores = [engine._score_candidate(replace(template, measurement_record=r), None,
                                     profile, pose, outfit) for r in (record(), {}, record(45))]
    assert scores[0][0] > scores[1][0] > scores[2][0]
    assert scores[0][0] - scores[2][0] == pytest.approx(15, abs=.01)
    assert scores[0][1]["size_fit_coverage"] == 100
    assert scores[1][1]["size_fit_coverage"] == 0


def test_live_ranking_and_stock_filter_with_and_without_reference():
    class Client:
        def get(self, product_id):
            result = record(54 if product_id == "good" else 45)
            if product_id == "sold":
                result["sizes"][0]["available"] = False
            return result
    search = MusinsaLiveSearch(measurements=Client())
    def product(name):
        return ShoppingProduct(name, name, "", 1, "", "", "top", retrieval_score=20)
    try:
        grouped = {"top": [product(n) for n in ("bad", "good", "sold")]}
        search._compare_shortlist(grouped, UserProfile(reference_measurements={"top": REFERENCE}))
        assert [p.product_id for p in grouped["top"]] == ["good", "bad"]
        grouped = {"top": [product("sold")]}
        search._compare_shortlist(grouped, UserProfile())
        assert grouped["top"] == []
    finally:
        search._executor.shutdown(wait=True)


def test_offline_cache_reuses_measurements_but_not_stale_stock(tmp_path):
    import shutil
    csv_path = tmp_path / "products.csv"
    shutil.copyfile(ROOT / "data" / "products.csv", csv_path)
    table = record()
    table["sizes"][0]["available"] = False
    path = tmp_path / "cache" / "product_measurements" / "MS1.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"record": table, "cached_at": time.time() - 7200}), encoding="utf-8")
    catalog = ProductCatalog(csv_path)
    cached = catalog._measurement_record("MS1")
    assert cached["sizes"][0]["available"] is None
    assert size_score(compare_sizes(cached, "top", REFERENCE)) == 100
    assert json.loads(path.read_text(encoding="utf-8"))["record"]["sizes"][0]["available"] is False
