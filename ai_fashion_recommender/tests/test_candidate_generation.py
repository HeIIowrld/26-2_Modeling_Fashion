from __future__ import annotations

import sys
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from musinsa_live_search import MusinsaLiveSearch
from product_measurements import normalize_size_table
from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile
if __package__:
    from .test_musinsa_live_search import item
else:
    from test_musinsa_live_search import item


class CandidateGenerationTests(unittest.TestCase):
    def setUp(self):
        self.targets = TargetKeywordResult(mode="mixed", targets={
            "top": {"fit": ["여유핏", "레귤러"], "material": ["니트"], "style": ["캐주얼"]},
        })
        self.profile = UserProfile()
        self.search = MusinsaLiveSearch()
        self.addCleanup(self.search.close)

    def test_later_query_and_new_sort_can_win_after_first_twelve_matches(self):
        first_query = self.search._queries("top", self.targets.targets["top"])[0]
        calls = []
        def fetch(category, query, **kwargs):
            calls.append((query, kwargs["sort_code"]))
            if query != first_query and kwargs["sort_code"] == "NEW":
                return [item(900, "오버핏 니트 캐주얼", reviews=1)]
            return [item(i, "인기 기본 니트", reviews=10000) for i in range(1, 41)]
        with patch.object(self.search, "_fetch", side_effect=fetch):
            result = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual(result[0].product_id, "MS900")
        self.assertEqual(len(calls), 8)
        self.assertEqual(self.search.last_search_stats["unique_candidates"]["top"], 41)

    def test_result_beyond_position_forty_is_considered(self):
        rows = [item(i, "베이직 상의") for i in range(1, 61)] + [item(61, "오버핏 니트 캐주얼")]
        with patch.object(self.search, "_fetch", return_value=rows):
            result = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual(result[0].product_id, "MS61")

    def test_request_cache_separates_sort_page_size_and_subcategory(self):
        with patch("musinsa_live_search.fetch_json", return_value={"data": {"list": []}}) as fetch:
            self.search._fetch("top", "니트")
            self.search._fetch("top", "니트")
            self.search._fetch("top", "니트", sort_code="NEW")
            self.search._fetch("top", "니트", page=2)
            self.search._fetch("top", "니트", size=20)
            self.search._fetch("top", "니트", category_code="001006")
        self.assertEqual(fetch.call_count, 5)

    def test_search_rank_and_review_count_do_not_change_attribute_score(self):
        attributes = self.targets.targets["top"]
        high, _ = self.search._score(item(1, "니트", reviews=100000), attributes, 0)
        low, _ = self.search._score(item(2, "니트", reviews=0), attributes, 99)
        self.assertEqual(high, low)

    def test_one_malformed_item_does_not_discard_good_results(self):
        malformed = item(1, "니트")
        malformed["finalPrice"] = "bad price"
        with patch.object(self.search, "_fetch", return_value=[None, malformed, item(2, "니트")]):
            result = self.search.search(self.targets, self.profile)
        self.assertEqual([p.product_id for p in result], ["MS2"])

    def test_deadline_returns_completed_results_without_waiting_for_slow_requests(self):
        self.search.search_budget = 0.03
        first_query = self.search._queries("top", self.targets.targets["top"])[0]
        def fetch(category, query, **kwargs):
            if query == first_query and kwargs["sort_code"] == "POPULAR":
                return [item(1, "니트")]
            time.sleep(0.15)
            return [item(2, "니트")]
        with patch.object(self.search, "_fetch", side_effect=fetch):
            start = time.monotonic()
            result = self.search.search(self.targets, self.profile)
            elapsed = time.monotonic() - start
        self.assertLess(elapsed, 0.12)
        self.assertEqual([p.product_id for p in result], ["MS1"])

    def test_measurements_rerank_before_the_final_limit(self):
        self.profile.reference_measurements = {"top": {"chest_width_cm": 54, "length_cm": 70}}
        client = Mock()
        def measurements(product_id):
            chest = 54 if product_id == "MS2" else 68
            return normalize_size_table(product_id, {"data": {"sizes": [{"name": "M", "items": [
                {"name": "가슴단면", "value": chest}, {"name": "총장", "value": 70}]}]}})
        client.get.side_effect = measurements
        self.search.measurements = client
        with patch.object(self.search, "_fetch", return_value=[item(1, "니트", reviews=1000), item(2, "니트", reviews=1)]):
            results = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual(results[0].product_id, "MS2")
        self.assertEqual(results[0].size_fit["closest_size"], "M")
        self.assertEqual(client.get.call_count, 2)
        self.assertNotIn("ranking_bonus", results[0].public_dict()["size_fit"])

    def test_measurement_failure_keeps_shopping_results(self):
        self.search.measurements = Mock()
        self.search.measurements.get.side_effect = OSError("unavailable")
        with patch.object(self.search, "_fetch", return_value=[item(1, "니트")]):
            result = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual(result[0].size_fit["status"], "unavailable")

    def test_shoes_share_search_without_using_clothing_size_measurements(self):
        self.targets.targets["shoes"] = {"item_type": ["로퍼"]}
        self.search.measurements = Mock()
        self.search.measurements.get.return_value = {"status": "unavailable"}
        with patch.object(self.search, "_fetch", side_effect=lambda c, *a, **kw:
                          [item(1, "니트")] if c == "top" else [item(2, "로퍼")]):
            result = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual([p.category for p in result], ["top", "shoes"])
        self.search.measurements.get.assert_called_once_with("MS1")
        self.assertEqual(result[1].size_fit, {})

    def test_partial_category_failure_can_use_explicit_fallback(self):
        self.targets.targets["bottom"] = {"material": ["데님"]}
        fallback = Product("MS2", "데님 팬츠", "bottom", "블루", "캐주얼", [], [], 50000, "사계절", True,
                           url="https://www.musinsa.com/products/2", image_url="https://image.msscdn.net/2.jpg")
        with patch.object(self.search, "_fetch", side_effect=lambda c, *a, **kw: [item(1, "니트")] if c == "top" else []):
            result = self.search.search(self.targets, self.profile, fallback_products=[fallback])
        self.assertEqual({p.category for p in result}, {"top", "bottom"})

    def test_same_brand_is_limited_when_equally_relevant_alternatives_exist(self):
        rows = [item(i, "니트") for i in (1, 2, 3, 4)]
        rows[-1]["brandName"] = "다른 브랜드"
        with patch.object(self.search, "_fetch", return_value=rows):
            result = self.search.search(self.targets, self.profile, limit=3)
        self.assertEqual([p.product_id for p in result], ["MS1", "MS2", "MS4"])


if __name__ == "__main__":
    unittest.main()
