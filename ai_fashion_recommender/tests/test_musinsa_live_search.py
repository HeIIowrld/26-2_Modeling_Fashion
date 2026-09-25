from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musinsa_live_search import MusinsaLiveSearch, ShoppingProduct
from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile
from fashion_ranking_policy import BOTTOM_FIT_TREND_WEIGHTS


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
        self.assertEqual(payload["ranking_evidence_source"], "title")

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

    def test_wide_bottom_gets_small_configurable_trend_bonus(self):
        wide = ShoppingProduct("W", "와이드 팬츠", "", 1, "", "", "bottom", retrieval_score=5)
        straight = ShoppingProduct("S", "스트레이트 팬츠", "", 1, "", "", "bottom", retrieval_score=5)
        self.assertEqual(BOTTOM_FIT_TREND_WEIGHTS["와이드핏"], 3.0)
        self.search = MusinsaLiveSearch()
        self.addCleanup(self.search.close)
        self.search._apply_fashion_policy_adjustments(
            {"bottom": [straight, wide]}, UserProfile(purpose="데일리", desired_style="캐주얼")
        )
        self.assertEqual(wide.retrieval_score, 8)
        self.assertEqual(straight.retrieval_score, 3)
        self.assertEqual(wide.ranking_adjustments["bottom_fit_trend"]["rule_id"], "R-TREND-01")

    def test_semiwide_receives_seventy_percent_of_straight_deduction(self):
        semiwide = ShoppingProduct("SW", "세미와이드 팬츠", "", 1, "", "", "bottom", retrieval_score=5)
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        search._apply_fashion_policy_adjustments(
            {"bottom": [semiwide]}, UserProfile(purpose="데일리", desired_style="캐주얼")
        )
        self.assertEqual(BOTTOM_FIT_TREND_WEIGHTS["세미와이드"], -1.4)
        self.assertEqual(semiwide.retrieval_score, 3.6)

    def test_visual_straight_overrides_a_semiwide_product_title(self):
        product = ShoppingProduct(
            "SW", "세미와이드 팬츠", "", 1, "", "", "bottom", retrieval_score=5,
            photo_attributes={"fit": {
                "label": "스트레이트핏", "confidence": .96,
                "source": "product_photo", "policy": "worn-fit-length-v1",
            }},
        )
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        search._apply_fashion_policy_adjustments(
            {"bottom": [product]}, UserProfile(purpose="데일리", desired_style="캐주얼")
        )
        self.assertEqual(product.retrieval_score, 3)
        self.assertEqual(product.ranking_adjustments["bottom_fit_trend"]["fit"], "스트레이트핏")

    def test_current_policy_can_outrank_legacy_casual_fit_keywords(self):
        search = StubSearch({"bottom": [
            item(1, "스트레이트 데님 팬츠"),
            item(2, "세미와이드 데님 팬츠"),
            item(3, "와이드 데님 팬츠"),
        ]})
        target = TargetKeywordResult("mixed", {"bottom": {
            "fit": ["스트레이트", "세미와이드"], "material": ["데님"],
        }})
        results = search.search(target, self.profile, limit=3)
        self.assertEqual([product.product_id for product in results], ["MS3", "MS2", "MS1"])
        self.assertFalse(any(query.startswith("와이드 ") for category, query in search.calls
                             if category == "bottom"))

    def test_trend_bonus_is_disabled_for_formal_work_interview_and_classic(self):
        profiles = (
            UserProfile(purpose="출근", desired_style="캐주얼"),
            UserProfile(purpose="면접", desired_style="캐주얼"),
            UserProfile(purpose="데일리", desired_style="포멀"),
            UserProfile(purpose="데일리", desired_style="클래식"),
            UserProfile(purpose="데일리", desired_style="캐주얼", dress_code="포멀"),
        )
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        for profile in profiles:
            product = ShoppingProduct("W", "와이드 팬츠", "", 1, "", "", "bottom", retrieval_score=5)
            search._apply_fashion_policy_adjustments({"bottom": [product]}, profile)
            self.assertEqual(product.retrieval_score, 5)

    def test_plain_and_logo_only_tees_are_penalised_but_artwork_and_ringer_are_not(self):
        plain = ShoppingProduct("P", "브랜드 기본 반팔 티셔츠", "", 1, "", "", "top", retrieval_score=5,
                                photo_attributes={"design": {"item_type": "티셔츠", "plain_basic": True}})
        graphic = ShoppingProduct("G", "아트워크 반팔 티셔츠", "", 1, "", "", "top", retrieval_score=5,
                                  photo_attributes={"design": {"item_type": "티셔츠", "plain_basic": False}})
        ringer = ShoppingProduct("R", "링거 반팔 티셔츠", "", 1, "", "", "top", retrieval_score=5,
                                 photo_attributes={"design": {"item_type": "티셔츠", "plain_basic": True}})
        large_logo = ShoppingProduct("L", "아치 로고 반팔 티셔츠", "", 1, "", "", "top", retrieval_score=5,
                                     photo_attributes={"design": {"item_type": "티셔츠", "plain_basic": False}})
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        search._apply_fashion_policy_adjustments(
            {"top": [plain, graphic, ringer, large_logo]}, UserProfile()
        )
        self.assertEqual(plain.retrieval_score, 3.75)
        self.assertEqual(graphic.retrieval_score, 5)
        self.assertEqual(ringer.retrieval_score, 5)
        self.assertEqual(large_logo.retrieval_score, 3.75)
        self.assertNotIn("ranking_adjustments", plain.public_dict())

    def test_formal_context_guard_reranks_existing_candidates_only(self):
        search = StubSearch({"top": [
            item(1, "오버핏 후드티"),
            item(2, "코튼 셔츠"),
            item(3, "테일러드 블레이저"),
        ]})
        target = TargetKeywordResult("user_input", {"top": {}})
        profile = UserProfile(purpose="출근", desired_style="클래식")
        results = search.search(target, profile, limit=3)
        self.assertEqual([product.product_id for product in results], ["MS3", "MS2", "MS1"])
        self.assertTrue(all("와이드" not in query for _, query in search.calls))
        self.assertEqual(results[0].ranking_adjustments["formal_context"]["rule_id"], "R-CTX-01")

    def test_formal_query_uses_context_item_type_and_keeps_broad_fallback(self):
        search = StubSearch({"top": [item(1, "옥스포드 셔츠")]})
        target = TargetKeywordResult("user_input", {"top": {
            "item_type": ["셔츠", "블레이저"], "style": ["포멀"],
            "fit": ["레귤러"],
        }})
        search.search(target, UserProfile(purpose="출근", desired_style="포멀"), limit=1)

        queries = [query for category, query in search.calls if category == "top"]
        self.assertTrue(any("셔츠" in query for query in queries))
        self.assertIn("베이직 상의", queries)

    def test_oversized_leather_shacket_does_not_beat_regular_work_shirt(self):
        shacket = ShoppingProduct(
            "SH", "레더 카라 오버핏 체크 셔켓", "", 1, "", "", "top",
            retrieval_score=6,
        )
        shirt = ShoppingProduct(
            "DR", "레귤러 코튼 드레스 셔츠", "", 1, "", "", "top",
            retrieval_score=5,
        )
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        grouped = {"top": [shacket, shirt]}
        search._apply_fashion_policy_adjustments(
            grouped, UserProfile(purpose="출근", desired_style="포멀")
        )
        grouped["top"].sort(key=search._sort_key)

        self.assertEqual(grouped["top"][0].product_id, "DR")
        self.assertLess(shacket.ranking_adjustments["formal_context"]["value"], 0)

    def test_interview_guard_is_stricter_than_work_guard(self):
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        work = ShoppingProduct("W", "그래픽 티셔츠", "", 1, "", "", "top", retrieval_score=5)
        interview = ShoppingProduct("I", "그래픽 티셔츠", "", 1, "", "", "top", retrieval_score=5)
        search._apply_fashion_policy_adjustments(
            {"top": [work]}, UserProfile(purpose="출근", desired_style="캐주얼")
        )
        search._apply_fashion_policy_adjustments(
            {"top": [interview]}, UserProfile(purpose="면접", desired_style="캐주얼")
        )
        self.assertLess(interview.retrieval_score, work.retrieval_score)

    def test_daily_casual_context_is_not_formality_reranked(self):
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        hoodie = ShoppingProduct("H", "후드티", "", 1, "", "", "top", retrieval_score=5)
        search._apply_fashion_policy_adjustments(
            {"top": [hoodie]}, UserProfile(purpose="데일리", desired_style="캐주얼")
        )
        self.assertEqual(hoodie.retrieval_score, 5)
        self.assertNotIn("formal_context", hoodie.ranking_adjustments)

    def test_sporty_context_penalises_generic_denim_but_exempts_sports_brand_denim(self):
        generic = ShoppingProduct(
            "G", "레귤러 데님 팬츠", "패션브랜드", 1, "", "", "bottom",
            retrieval_score=5,
        )
        branded = ShoppingProduct(
            "N", "레귤러 데님 팬츠", "나이키", 1, "", "", "bottom",
            retrieval_score=5,
        )
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        search._apply_fashion_policy_adjustments(
            {"bottom": [generic, branded]},
            UserProfile(purpose="데일리", desired_style="스포티"),
        )

        self.assertEqual(generic.retrieval_score, 2.5)
        self.assertEqual(branded.retrieval_score, 5)
        self.assertTrue(branded.ranking_adjustments["sporty_context"]["brand_exempt"])
        self.assertEqual(branded.ranking_adjustments["sporty_context"]["tier"], "compatible")

    def test_sporty_context_rewards_track_items_in_more_than_shoes(self):
        shirt = ShoppingProduct("SH", "체크 오버핏 셔츠", "", 1, "", "", "top", retrieval_score=5)
        track = ShoppingProduct("TR", "사이드라인 트랙 재킷", "", 1, "", "", "top", retrieval_score=5)
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        search._apply_fashion_policy_adjustments(
            {"top": [shirt, track]}, UserProfile(purpose="데일리", desired_style="스포티")
        )
        self.assertGreater(track.retrieval_score, shirt.retrieval_score)
        self.assertEqual(track.ranking_adjustments["sporty_context"]["tier"], "strong")

    def test_formal_guard_covers_bottoms_and_shoes(self):
        search = MusinsaLiveSearch()
        self.addCleanup(search.close)
        slacks = ShoppingProduct("SL", "테일러드 슬랙스", "", 1, "", "", "bottom", retrieval_score=5)
        jogger = ShoppingProduct("JG", "스웨트 조거 팬츠", "", 1, "", "", "bottom", retrieval_score=5)
        loafer = ShoppingProduct("LF", "페니 로퍼", "", 1, "", "", "shoes", retrieval_score=5)
        sneaker = ShoppingProduct("SN", "러닝 스니커즈", "", 1, "", "", "shoes", retrieval_score=5)
        grouped = {"bottom": [slacks, jogger], "shoes": [loafer, sneaker]}
        search._apply_fashion_policy_adjustments(
            grouped, UserProfile(purpose="면접", desired_style="포멀")
        )
        self.assertGreater(slacks.retrieval_score, jogger.retrieval_score)
        self.assertGreater(loafer.retrieval_score, sneaker.retrieval_score)


if __name__ == "__main__":
    unittest.main()
