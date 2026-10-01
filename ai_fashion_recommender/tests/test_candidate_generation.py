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

    def test_each_selected_clothing_material_gets_a_query_but_shoes_get_none(self):
        attributes = {"material": ["데님", "니트"], "style": ["캐주얼"]}
        top_queries = self.search._queries("top", attributes)
        shoe_queries = self.search._queries(
            "shoes", {"item_type": ["스니커즈"], "material": ["데님", "니트"]}
        )

        self.assertTrue(any("데님" in query for query in top_queries))
        self.assertTrue(any("니트" in query for query in top_queries))
        self.assertFalse(any("데님" in query or "니트" in query for query in shoe_queries))

    def test_multiple_context_item_types_receive_separate_queries(self):
        queries = self.search._queries("top", {
            "item_type": ["티셔츠", "맨투맨", "니트"],
            "style": ["캐주얼"], "fit": ["여유핏"],
        })

        self.assertIn("캐주얼 티셔츠", queries)
        self.assertIn("캐주얼 맨투맨", queries)
        self.assertIn("캐주얼 니트", queries)
        self.assertIn("베이직 상의", queries)

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

    def test_measurements_are_attached_to_the_shortlist(self):
        """기준 옷 비교를 없앤 뒤(2026-10-01)에도 상품 실측표는 계속 붙어야 한다.

        예전에는 이 실측으로 순위를 다시 매겼지만 지금은 화면 표시용이다.
        그래도 조회는 상위 후보에만 하고, 결과가 카드에 실려야 한다.
        """
        client = Mock()
        def measurements(product_id):
            chest = 54 if product_id == "MS2" else 68
            return normalize_size_table(product_id, {"data": {"sizes": [{"name": "M", "items": [
                {"name": "가슴단면", "value": chest}, {"name": "총장", "value": 70}]}]}})
        client.get.side_effect = measurements
        self.search.measurements = client
        with patch.object(self.search, "_fetch", return_value=[item(1, "니트", reviews=1000), item(2, "니트", reviews=1)]):
            results = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual(client.get.call_count, 2)
        fit = results[0].public_dict()["size_fit"]
        self.assertEqual([row["size"] for row in fit["size_options"]], ["M"])
        self.assertEqual(fit["columns"], {"chest_width_cm": "가슴단면", "length_cm": "총장"})
        # 비교가 사라졌으므로 순위 보정값도 없어야 한다.
        self.assertNotIn("ranking_bonus", fit)
        self.assertNotIn("closest_size", fit)

    def test_every_size_sold_out_drops_the_product(self):
        """확인된 사이즈가 전부 품절이면 후보에서 뺀다.

        2026-10-01 사이즈 비교를 없애면서 이 필터가 조용히 꺼져 있었다
        (`status != "no_available_sizes"` 로 걸렀는데 그 status 를 더는 만들지 않았다).
        살 수 없는 상품을 추천하지 않기 위한 장치라 다시 켜고 여기서 지킨다.
        """
        def table(product_id, available):
            record = normalize_size_table(product_id, {"data": {"sizes": [{"name": "M", "items": [
                {"name": "가슴단면", "value": 54}, {"name": "총장", "value": 70}]}]}})
            record["sizes"][0]["available"] = available
            return record
        client = Mock()
        client.get.side_effect = lambda pid: table(pid, pid != "MS1")
        self.search.measurements = client
        with patch.object(self.search, "_fetch", return_value=[item(1, "니트", reviews=1000),
                                                               item(2, "니트", reviews=1)]):
            results = self.search.search(self.targets, self.profile, limit=2)
        self.assertEqual([p.product_id for p in results], ["MS2"])

    def test_unknown_stock_is_not_treated_as_sold_out(self):
        """조회에 실패해 사이즈를 모르는 상품은 거르지 않는다.

        모르는 것과 품절은 다르다. 실측 조회가 느리거나 실패했다고 해서
        멀쩡한 상품이 추천에서 사라지면 안 된다.
        """
        client = Mock()
        client.get.side_effect = OSError("unavailable")
        self.search.measurements = client
        with patch.object(self.search, "_fetch", return_value=[item(1, "니트")]):
            results = self.search.search(self.targets, self.profile, limit=1)
        self.assertEqual([p.product_id for p in results], ["MS1"])

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
        # 서로 다른 모델이되 같은 브랜드인 후보로 브랜드 다양성만 검증한다.
        # 동일 상품의 색상 옵션은 별도 정책에서 하나의 모델로 묶는다.
        rows = [item(i, f"니트 모델 {i}") for i in (1, 2, 3, 4)]
        rows[-1]["brandName"] = "다른 브랜드"
        with patch.object(self.search, "_fetch", return_value=rows):
            result = self.search.search(self.targets, self.profile, limit=3)
        self.assertEqual([p.product_id for p in result], ["MS1", "MS2", "MS4"])


if __name__ == "__main__":
    unittest.main()
