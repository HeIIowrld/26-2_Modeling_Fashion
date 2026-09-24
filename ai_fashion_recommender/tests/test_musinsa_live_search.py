from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musinsa_live_search import MusinsaLiveSearch, ShoppingProduct
from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile


def item(
    goods_no: int,
    name: str,
    *,
    price: int = 59_000,
    gender: str = "공용",
    reviews: int = 100,
) -> dict:
    return {
        "goodsNo": goods_no,
        "goodsName": name,
        "goodsLinkUrl": f"https://www.musinsa.com/products/{goods_no}",
        "thumbnail": f"https://image.msscdn.net/{goods_no}_500.jpg",
        "displayGenderText": gender,
        "isSoldOut": False,
        "finalPrice": price,
        "brand": "brand",
        "brandName": "테스트 브랜드",
        "reviewCount": reviews,
        "reviewScore": 96,
    }


class StubSearch(MusinsaLiveSearch):
    def __init__(self, by_category: dict[str, list[dict]] | None = None, fail: bool = False):
        super().__init__(timeout=0.01)
        self.by_category = by_category or {}
        self.fail = fail
        self.calls: list[tuple[str, str]] = []

    def _fetch(self, category: str, query: str, size: int = 100, **kwargs) -> list[dict]:
        self.calls.append((category, query))
        if self.fail:
            raise OSError("network unavailable")
        return self.by_category.get(category, [])


class StubMeasurements:
    """사이즈표 응답에 색 옵션이 같이 오는 상황을 그대로 흉내 낸다(추가 요청 없음)."""

    def __init__(self, colors_by_id: dict[str, list[str]]):
        self.colors_by_id = colors_by_id
        self.calls: list[str] = []

    def get(self, product_id: str) -> dict:
        self.calls.append(product_id)
        return {"status": "unavailable", "sizes": [], "issues": [],
                "color_options": list(self.colors_by_id.get(product_id, []))}


class ColorOptionTests(unittest.TestCase):
    """색 옵션은 회피 색 검사에만 쓴다.

    카드 사진과 가상 피팅은 대표 사진 한 장으로 돈다. 그런데 첫 컬러칩과 대표 사진 색이
    같은 경우가 345개 중 46%뿐이었다(2026-09-25). 그래서 "파는 색 중에 원하는 색이
    있으니 추천"은 하지 않는다 — 다른 색으로 합성하면 색을 맞춘 의미가 없다.
    반대로 파는 색이 전부 회피 색이면 어느 사진이 뜨든 회피 색이므로 뺄 수 있다.
    """

    def targets(self, color="블랙"):
        return TargetKeywordResult(mode="user_input",
                                   targets={"top": {"category": ["상의"], "color": [color]}})

    def search_with(self, colors_by_id, profile=None, color="블랙"):
        search = StubSearch({"top": [item(1, "베이직 반팔 티셔츠"), item(2, "무지 반팔 티셔츠")]})
        search.measurements = StubMeasurements(colors_by_id)
        profile = profile or UserProfile(budget=120_000, change_scope="전체 변경",
                                         provided_fields=["budget", "change_scope"])
        return search, search.search(self.targets(color), profile, limit=2)

    def test_a_sold_color_does_not_promote_a_product(self):
        search, results = self.search_with({"MS2": ["아이보리", "(19)BLACK"]})
        # 블랙을 팔지만 대표 사진이 블랙이라는 보장이 없으므로 순위를 올리지 않는다.
        self.assertEqual([product.product_id for product in results], ["MS1", "MS2"])
        self.assertEqual([product.retrieval_score for product in results], [0.0, 0.0])
        self.assertNotIn("블랙", results[1].matched_keywords)

    def test_products_whose_every_color_is_avoided_are_dropped(self):
        profile = UserProfile(budget=120_000, change_scope="전체 변경",
                              provided_fields=["budget", "change_scope"], avoided_colors=["블랙"])
        _, results = self.search_with({"MS1": ["블랙", "차콜"], "MS2": ["아이보리"]}, profile)
        self.assertEqual([product.product_id for product in results], ["MS2"])

    def test_one_wearable_color_keeps_the_product(self):
        profile = UserProfile(budget=120_000, change_scope="전체 변경",
                              provided_fields=["budget", "change_scope"], avoided_colors=["블랙"])
        _, results = self.search_with({"MS1": ["블랙", "아이보리"]}, profile)
        self.assertIn("MS1", [product.product_id for product in results])

    def test_unknown_colors_change_nothing(self):
        search, results = self.search_with({})
        self.assertEqual(len(results), 2)
        self.assertEqual(search.last_search_stats["color_options"]["colors_known"], 0)

    def test_colors_reach_the_web_payload(self):
        _, results = self.search_with({"MS2": ["아이보리", "(19)BLACK"]})
        payload = next(p for p in results if p.product_id == "MS2").public_dict()
        self.assertEqual(payload["color_options"], ["아이보리", "(19)BLACK"])


class MusinsaLiveSearchTests(unittest.TestCase):
    def setUp(self):
        self.targets = TargetKeywordResult(
            mode="mixed",
            targets={
                "top": {
                    "category": ["상의"], "fit": ["여유핏"],
                    "material": ["니트"], "style": ["캐주얼"],
                },
                "bottom": {
                    "category": ["하의"], "fit": ["세미와이드"],
                    "length": ["풀렝스"], "material": ["데님"],
                },
            },
        )
        self.profile = UserProfile(
            budget=120_000, change_scope="전체 변경", provided_fields=["budget", "change_scope"]
        )

    def test_keyword_match_reranks_live_results(self):
        search = StubSearch({
            "bottom": [
                item(1, "인기 베이직 팬츠", reviews=10_000),
                item(2, "세미 와이드 데님 팬츠", reviews=20),
            ]
        })
        bottom_only = TargetKeywordResult(mode="mixed", targets={"bottom": self.targets.targets["bottom"]})

        results = search.search(bottom_only, self.profile, limit=2)

        self.assertEqual(results[0].product_id, "MS2")
        self.assertIn("세미와이드", results[0].matched_keywords)
        self.assertIn("데님", results[0].matched_keywords)
        self.assertEqual(results[0].search_keywords, ["세미와이드", "데님", "풀렝스"])

    def test_limit_applies_to_each_requested_category(self):
        search = StubSearch({
            "top": [item(10, "오버핏 니트"), item(11, "루즈 니트")],
            "bottom": [item(20, "세미 와이드 데님"), item(21, "와이드 데님")],
        })

        results = search.search(self.targets, self.profile, limit=3)

        self.assertEqual(len(results), 4)  # 부족한 후보를 복제하거나 다른 카테고리로 채우지 않음
        self.assertEqual([result.category for result in results], ["top", "top", "bottom", "bottom"])

    def test_three_each_for_all_categories(self):
        targets = TargetKeywordResult(mode="user_input", targets={
            "top": {}, "bottom": {}, "shoes": {"item_type": ["로퍼"], "color": ["블랙"]},
        })
        search = StubSearch({category: [item(offset + i, "블랙 로퍼") for i in range(5)]
                             for category, offset in (("top", 10), ("bottom", 20), ("shoes", 30))})
        results = search.search(targets, self.profile)
        self.assertEqual(len(results), 9)
        for category in targets.targets:
            self.assertEqual(sum(p.category == category for p in results), 3)
        self.assertIn(("shoes", "블랙 로퍼"), search.calls)
        self.assertIn("로퍼", results[-1].search_keywords)

    def test_budget_gender_and_exclusions_are_hard_filters(self):
        profile = UserProfile(
            gender="여성", min_budget=30_000, max_budget=80_000,
            avoided_colors=["베이지"], provided_fields=["min_budget", "max_budget"],
        )
        search = StubSearch({
            "bottom": [
                item(1, "베이지 세미 와이드 데님", price=50_000, gender="여성"),
                item(2, "블루 세미 와이드 데님", price=90_000, gender="여성"),
                item(3, "블루 세미 와이드 데님", price=50_000, gender="남성"),
                item(4, "블루 세미 와이드 데님", price=50_000, gender="공용"),
            ]
        })
        bottom_only = TargetKeywordResult(mode="mixed", targets={"bottom": self.targets.targets["bottom"]})

        results = search.search(bottom_only, profile, limit=3)

        self.assertEqual([result.product_id for result in results], ["MS4"])

    def test_network_failure_uses_enriched_catalog_fallback(self):
        search = StubSearch(fail=True)
        fallback = Product(
            "MSLOCAL", "세미와이드 데님 팬츠", "bottom", "블루", "캐주얼",
            ["데일리"], [], 69_000, "사계절", True,
            url="https://www.musinsa.com/products/999",
            image_url="https://image.msscdn.net/999_500.jpg",
            fit="세미와이드", material="데님",
        )
        bottom_only = TargetKeywordResult(mode="mixed", targets={"bottom": self.targets.targets["bottom"]})

        results = search.search(bottom_only, self.profile, limit=3, fallback_products=[fallback])

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].source, "musinsa_catalog_fallback")

    def test_public_payload_hides_internal_keywords_and_score(self):
        product = ShoppingProduct(
            "MS1", "상품", "브랜드", 10_000, "https://image", "https://product", "top",
            matched_keywords=["오버핏"], retrieval_score=9.5,
        )

        payload = product.public_dict()

        self.assertNotIn("matched_keywords", payload)
        self.assertNotIn("retrieval_score", payload)
        self.assertEqual(payload["url"], "https://product")
        self.assertIn("recommendation_reason", payload)

    def test_public_payload_exposes_only_three_representative_search_keywords(self):
        product = ShoppingProduct(
            "MS1", "상품", "브랜드", 10_000, "https://image", "https://product", "top",
            search_keywords=["여유핏", "니트", "캐주얼"],
            matched_keywords=["여유핏", "니트"], retrieval_score=9.5,
        )

        payload = product.public_dict()

        self.assertEqual(payload["search_keywords"], ["여유핏", "니트", "캐주얼"])
        self.assertNotIn("matched_keywords", payload)
        self.assertNotIn("retrieval_score", payload)

    def test_live_result_uses_canonical_musinsa_product_url(self):
        unsafe_item = item(77, "세미 와이드 데님 팬츠")
        unsafe_item["goodsLinkUrl"] = "javascript:alert(1)"
        search = StubSearch({"bottom": [unsafe_item]})
        bottom_only = TargetKeywordResult(mode="mixed", targets={"bottom": self.targets.targets["bottom"]})

        results = search.search(bottom_only, self.profile, limit=1)

        self.assertEqual(results[0].url, "https://www.musinsa.com/products/77")


if __name__ == "__main__":
    unittest.main()
