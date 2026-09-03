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

    def _fetch(self, category: str, query: str, size: int = 40) -> list[dict]:
        self.calls.append((category, query))
        if self.fail:
            raise OSError("network unavailable")
        return self.by_category.get(category, [])


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

    def test_three_results_are_balanced_across_requested_categories(self):
        search = StubSearch({
            "top": [item(10, "오버핏 니트"), item(11, "루즈 니트")],
            "bottom": [item(20, "세미 와이드 데님"), item(21, "와이드 데님")],
        })

        results = search.search(self.targets, self.profile, limit=3)

        self.assertEqual(len(results), 3)
        self.assertEqual([result.category for result in results], ["top", "bottom", "top"])

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
