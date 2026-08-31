import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outfit_analyzer import _dominant_palette, _dominant_rgb, _nearest_color, color_harmony
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine, NoBudgetMatch, MinGreaterThanMax
from schemas import OutfitAnalysis, PoseAnalysis, Product, UserProfile, WardrobeItem


class StubCatalog:
    def __init__(self, products):
        self.products = products

    def available(self, category=None):
        return [
            product for product in self.products
            if product.stock and (category is None or product.category == category)
        ]


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ProductCatalog(ROOT / "data" / "products.csv")
        self.engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", self.catalog)
        self.pose = PoseAnalysis(True, 0.9, "사각체형", 1.0, 0.5, 0.55, "정면에 가까움")
        self.outfit = OutfitAnalysis("test", "화이트", "블랙", "안정적인 무채색 조합", ["top", "pants"], "캐주얼")

    def test_catalog_loads(self):
        self.assertTrue(self.catalog.products)
        categories = {product.category for product in self.catalog.products}
        self.assertEqual(categories, {"top", "bottom"})
        self.assertTrue(all(product.price > 0 for product in self.catalog.products))

    def test_full_change_returns_ranked_pairs(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼", budget=180_000, change_scope="전체 변경")
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=3)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(len(item.products) == 2 for item in recommendations))
        self.assertGreaterEqual(recommendations[0].total_score, recommendations[1].total_score)
        self.assertEqual(recommendations[0].score_coverage, 80.0)
        self.assertIn("R-CTX-01", recommendations[0].applied_rules)

    def test_tied_candidates_are_split_between_people(self):
        """최고점 동점이 수백~수천 개라 동점 처리가 곧 추천 결과를 정한다.

        카탈로그 행 순서로 자르면 모든 사람이 같은 조합을 받는다(2026-08-21 실측).
        사진마다 다른 씨앗으로 갈라야 한다.
        """
        template = next(
            product for product in self.catalog.products
            if product.product_id == "TOP012"
        )
        tied_products = [
            replace(template, product_id=f"TIE{index:02d}")
            for index in range(8)
        ]
        engine = RecommendationEngine(
            ROOT / "FASHION_RULES_MASTER.md", StubCatalog(tied_products)
        )
        profile = UserProfile(purpose="데일리", budget=180_000, change_scope="상의만 변경")
        chosen = set()
        for index in range(6):
            pose = PoseAnalysis(
                True, 0.9, "사각체형", 1.0 + index * 0.01, 0.5, 0.55, "정면에 가까움"
            )
            recommendation = engine.recommend(profile, pose, self.outfit, top_k=1)[0]
            chosen.add(tuple(product.product_id for product in recommendation.products))
        self.assertGreater(len(chosen), 1)

    def test_same_pose_always_gets_the_same_recommendation(self):
        """같은 사람은 매번 같은 결과를 받아야 한다. 실행마다 흔들리면 안 된다."""
        profile = UserProfile(purpose="데일리", budget=180_000, change_scope="전체 변경")
        first, second = (
            [
                tuple(product.product_id for product in recommendation.products)
                for recommendation in self.engine.recommend(
                    profile, self.pose, self.outfit, top_k=3
                )
            ]
            for _ in range(2)
        )
        self.assertEqual(first, second)

    def test_tie_break_does_not_lower_the_score(self):
        """동점끼리만 섞는다. 최고점 자체는 그대로여야 한다."""
        profile = UserProfile(purpose="데일리", budget=180_000, change_scope="전체 변경")
        scores = [
            self.engine.recommend(
                profile,
                PoseAnalysis(True, 0.9, "사각체형", 1.0 + index * 0.01, 0.5, 0.55, "정면에 가까움"),
                self.outfit,
                top_k=1,
            )[0].total_score
            for index in range(4)
        ]
        self.assertEqual(len(set(scores)), 1)

    def test_equal_primary_scores_are_reported_as_a_shared_rank(self):
        template = next(
            product for product in self.catalog.products
            if product.product_id == "TOP012"
        )
        products = [replace(template, product_id=f"SHARED{index}") for index in range(4)]
        engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", StubCatalog(products))
        recommendations = engine.recommend(
            UserProfile(purpose="데일리", change_scope="상의만 변경"),
            self.pose,
            self.outfit,
            top_k=3,
        )
        self.assertTrue(all(item.ranking_tied for item in recommendations))
        self.assertEqual([item.display_rank for item in recommendations], [1, 1, 1])
        self.assertTrue(all("공동 순위" in item.ranking_reason for item in recommendations))

    def test_scope_changes_only_top(self):
        profile = UserProfile(change_scope="상의만 변경")
        recommendation = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)[0]
        self.assertEqual([product.category for product in recommendation.products], ["top"])

    def test_neutral_color_harmony(self):
        self.assertEqual(color_harmony("화이트", "네이비"), "안정적인 무채색 조합")

    def test_dominant_color_returns_rgb_triplet(self):
        image = np.full((20, 20, 3), (200, 80, 120), dtype=np.uint8)
        mask = np.ones((20, 20), dtype=bool)
        color = _dominant_rgb(image, mask)
        self.assertEqual(len(color), 3)
        self.assertTrue(all(isinstance(value, (int, np.integer)) for value in color))

    def test_palette_reports_color_area_proportions(self):
        image = np.zeros((10, 10, 3), dtype=np.uint8)
        image[:7] = (230, 230, 230)
        image[7:] = (40, 60, 110)
        palette = _dominant_palette(image, np.ones((10, 10), dtype=bool), max_colors=2)
        self.assertEqual(len(palette), 2)
        self.assertAlmostEqual(sum(item["proportion"] for item in palette), 1.0, places=2)
        self.assertGreater(palette[0]["proportion"], palette[1]["proportion"])

    def test_dark_navy_is_not_collapsed_into_black(self):
        self.assertEqual(_nearest_color((23, 23, 43)), "네이비")
        self.assertEqual(_nearest_color((43, 40, 61)), "네이비")

    def test_shaded_beige_is_not_collapsed_into_gray(self):
        self.assertEqual(_nearest_color((167, 150, 122)), "베이지")
        self.assertEqual(_nearest_color((225, 207, 179)), "베이지")

    def test_olive_khaki_is_not_collapsed_into_brown(self):
        self.assertEqual(_nearest_color((79, 75, 54)), "카키")
        self.assertEqual(_nearest_color((125, 119, 91)), "카키")

    def test_shaded_clusters_of_one_colour_are_merged(self):
        image = np.empty((20, 20, 3), dtype=np.uint8)
        image[:12] = (23, 23, 43)
        image[12:] = (43, 40, 61)
        palette = _dominant_palette(image, np.ones((20, 20), dtype=bool), max_colors=3)
        self.assertEqual(palette[0]["name"], "네이비")
        self.assertAlmostEqual(palette[0]["proportion"], 1.0, places=2)

    def test_rules_markdown_is_loaded(self):
        self.assertEqual(self.engine.rules_source.name, "FASHION_RULES_MASTER.md")
        self.assertEqual(len(self.engine.documented_rule_ids), 55)
        self.assertEqual(len(self.engine.active_rule_ids), 48)
        self.assertEqual(len(self.engine.scoring_rule_ids), 36)
        self.assertIn("R-SIL-01", self.engine.active_rule_ids)
        self.assertEqual(
            set(self.engine.unsupported_rule_ids),
            {"R-SIL-02", "R-SIL-04", "R-COL-06", "R-COL-07", "R-COL-12", "R-ACC-03", "R-TREND-01"},
        )

    def _cheapest_daily_pair(self) -> int:
        """카탈로그가 바뀌어도 유효한 '가장 싼 데일리 조합' 가격."""
        profile = UserProfile(purpose="데일리", change_scope="전체 변경")
        tops = self.engine._available_for_profile("top", profile)
        bottoms = self.engine._available_for_profile("bottom", profile)
        return min(top.price for top in tops) + min(bottom.price for bottom in bottoms)

    def test_budget_is_a_hard_filter(self):
        probe = UserProfile(purpose="데일리", change_scope="전체 변경")
        floor = min(
            min(product.price for product in self.engine._available_for_profile("top", probe)),
            min(product.price for product in self.engine._available_for_profile("bottom", probe)),
        )
        profile = UserProfile(purpose="데일리", budget=floor - 1, change_scope="전체 변경")
        with self.assertRaisesRegex(ValueError, "조건을 모두 만족"):
            self.engine.recommend(profile, self.pose, self.outfit)

    def test_budget_just_above_the_floor_returns_a_pair(self):
        floor = self._cheapest_daily_pair()
        profile = UserProfile(purpose="데일리", budget=floor, change_scope="전체 변경")
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)
        self.assertEqual(len(recommendations), 1)
        self.assertLessEqual(
            sum(product.price for product in recommendations[0].products), floor
        )

    def test_budget_range_includes_min_boundary(self):
        """최소값 경계가 포함되는지 확인한다: 합계 == min_budget 이면 반환되어야 한다."""
        floor = self._cheapest_daily_pair()
        profile = UserProfile(purpose="데일리", change_scope="전체 변경", min_budget=floor, max_budget=floor)
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)
        self.assertEqual(len(recommendations), 1)
        total = sum(product.price for product in recommendations[0].products)
        self.assertEqual(total, floor)

    def test_budget_range_includes_max_boundary(self):
        """최대값 경계가 포함되는지 확인한다: 합계 == max_budget 이면 반환되어야 한다."""
        floor = self._cheapest_daily_pair()
        profile = UserProfile(purpose="데일리", change_scope="전체 변경", min_budget=0, max_budget=floor)
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)
        self.assertEqual(len(recommendations), 1)
        total = sum(product.price for product in recommendations[0].products)
        self.assertLessEqual(total, floor)

    def test_no_candidates_in_budget_range(self):
        """범위 내에 추천 조합이 없을 때는 명확한 예외를 던진다."""
        floor = self._cheapest_daily_pair()
        profile = UserProfile(purpose="데일리", change_scope="전체 변경", min_budget=floor + 1, max_budget=floor + 10)
        with self.assertRaisesRegex(NoBudgetMatch, "예산 범위"):
            self.engine.recommend(profile, self.pose, self.outfit)

    def test_min_greater_than_max_is_rejected(self):
        """최소값이 최대값보다 큰 입력은 명확한 에러를 발생시킨다."""
        profile = UserProfile(purpose="데일리", change_scope="전체 변경", min_budget=100_000, max_budget=10_000)
        with self.assertRaisesRegex(MinGreaterThanMax, "최소 예산이 최대 예산보다 큽니다"):
            self.engine.recommend(profile, self.pose, self.outfit)

    def test_user_exclusions_are_hard_filters(self):
        profile = UserProfile(
            purpose="데일리",
            budget=100_000,
            change_scope="상의만 변경",
            avoided_colors=["화이트", "그레이"],
        )
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=6)
        self.assertTrue(all(product.color not in profile.avoided_colors for item in recommendations for product in item.products))

    def test_hot_humid_weather_prioritizes_breathability(self):
        products = [
            Product(
                "HOT", "통기성 상의", "top", "베이지", "캐주얼", ["데일리"], [], 50_000,
                "여름", True, item_type="티셔츠", fit="레귤러핏", material="코튼",
                breathability=5, warmth=1, formality=2,
            ),
            Product(
                "WARM", "두꺼운 상의", "top", "베이지", "캐주얼", ["데일리"], [], 50_000,
                "여름", True, item_type="니트", fit="레귤러핏", material="니트",
                breathability=1, warmth=5, formality=2,
            ),
        ]
        engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", StubCatalog(products))
        profile = UserProfile(
            purpose="데일리", desired_style="캐주얼", budget=100_000,
            change_scope="상의만 변경", season="여름", temperature_c=31, humidity=80,
        )
        recommendations = engine.recommend(profile, self.pose, self.outfit, top_k=2)
        self.assertEqual(recommendations[0].products[0].product_id, "HOT")
        self.assertIn("R-WEA-01", recommendations[0].applied_rules)
        self.assertIn("R-WEA-02", recommendations[0].applied_rules)

    def test_formality_changes_workwear_ranking(self):
        products = [
            Product(
                "FORMAL", "포멀 상의", "top", "네이비", "포멀", ["출근"], [], 50_000,
                "사계절", True, item_type="재킷", fit="레귤러핏", material="우븐", formality=5,
            ),
            Product(
                "CASUAL", "캐주얼 상의", "top", "네이비", "포멀", ["출근"], [], 50_000,
                "사계절", True, item_type="티셔츠", fit="레귤러핏", material="코튼", formality=1,
            ),
        ]
        engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", StubCatalog(products))
        profile = UserProfile(
            purpose="출근", desired_style="포멀", budget=100_000,
            change_scope="상의만 변경", dress_code="포멀",
        )
        recommendations = engine.recommend(profile, self.pose, self.outfit, top_k=2)
        self.assertEqual(recommendations[0].products[0].product_id, "FORMAL")
        self.assertIn("R-CTX-02", recommendations[0].applied_rules)

    def test_owned_items_enable_wardrobe_score(self):
        profile = UserProfile(
            purpose="데일리",
            desired_style="캐주얼",
            budget=100_000,
            change_scope="상의만 변경",
            owned_items=[WardrobeItem("OWN-BOT", "bottom", "네이비", style="캐주얼")],
        )
        recommendation = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)[0]
        self.assertEqual(recommendation.score_coverage, 85.0)
        self.assertIn("R-OWN-01", recommendation.applied_rules)

    def test_accessory_rules_return_guidance_not_fake_products(self):
        profile = UserProfile(purpose="여행", budget=100_000, change_scope="상의만 변경")
        recommendation = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)[0]
        self.assertTrue(recommendation.styling_tips)
        self.assertIn("R-CAT-01", recommendation.applied_rules)
        self.assertIn("R-ACC-06", recommendation.applied_rules)
        self.assertTrue(all(product.category in {"top", "bottom"} for product in recommendation.products))

    def test_current_outfit_score_is_separate_and_can_skip_recommendation(self):
        outfit = OutfitAnalysis(
            "test", "네이비", "베이지", "안정적인 무채색 조합", ["top", "pants"], "캐주얼",
            upper_type="니트", lower_type="팬츠", sleeve_length="긴팔",
            upper_length="기본 기장", bottom_length="긴바지", fit="레귤러핏",
            lower_fit="와이드핏", pattern="무지", material="니트",
            lower_pattern="무지", lower_material="코튼", attribute_confidence=0.90,
        )
        profile = UserProfile(purpose="데일리", desired_style="캐주얼", change_scope="전체 변경")
        evaluation = self.engine.evaluate_current_outfit(
            profile, self._scorable_pose(), outfit, keep_threshold=85
        )
        self.assertTrue(evaluation.reliable)
        self.assertTrue(evaluation.should_keep)
        self.assertIn("top_situation_fit", evaluation.score_breakdown)
        self.assertTrue(all(all(row.values()) for row in evaluation.pass_matrix.values()))
        recommendation = self.engine.recommend(
            profile, self._scorable_pose(), outfit, current_outfit_keep_threshold=85
        )[0]
        self.assertEqual(recommendation.products, [])
        self.assertIn("좋은 코디", recommendation.reasons[0])

    def _scorable_outfit(self, **changes):
        values = dict(
            parser_backend="test", upper_color="네이비", lower_color="베이지",
            color_harmony="안정적인 무채색 조합", detected_items=["top", "pants"],
            style="캐주얼", upper_type="티셔츠", lower_type="팬츠",
            upper_length="기본 기장", bottom_length="긴바지",
            fit="레귤러핏", lower_fit="스트레이트핏", neckline="라운드넥",
            pattern="무지", material="코튼", lower_pattern="무지",
            lower_material="코튼", attribute_confidence=0.95,
        )
        values.update(changes)
        return OutfitAnalysis(**values)

    @staticmethod
    def _scorable_pose(shape="사각체형", leg_ratio=0.65):
        return PoseAnalysis(
            True, 0.96, shape, 1.0, 0.55, leg_ratio,
            "정면에 가까움", body_shape_confidence=0.95,
        )

    def test_date_situation_scores_hoodie_lower_than_blouse(self):
        profile = UserProfile(purpose="데이트", desired_style="캐주얼")
        hoodie = self.engine.evaluate_current_outfit(
            profile, self._scorable_pose(), self._scorable_outfit(upper_type="후드티")
        )
        blouse = self.engine.evaluate_current_outfit(
            profile, self._scorable_pose(), self._scorable_outfit(upper_type="블라우스")
        )
        hoodie_score = hoodie.diagnostic_matrix["top"]["situation_fit"]
        blouse_score = blouse.diagnostic_matrix["top"]["situation_fit"]
        self.assertLess(hoodie_score, 50.0)
        self.assertGreaterEqual(blouse_score - hoodie_score, 40.0)
        self.assertIn("R-CTX-03", hoodie.applied_rules)

    def test_each_diagnostic_cell_passes_at_85(self):
        evaluation = self.engine.evaluate_current_outfit(
            UserProfile(purpose="데일리", desired_style="캐주얼"),
            self._scorable_pose(),
            self._scorable_outfit(),
        )
        for section, values in evaluation.diagnostic_matrix.items():
            for name, value in values.items():
                self.assertEqual(evaluation.pass_matrix[section][name], value >= 85.0)

    def test_long_oversized_top_and_long_wide_bottom_is_trendy_harmony(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼")
        trend = self.engine.evaluate_current_outfit(
            profile,
            self._scorable_pose(),
            self._scorable_outfit(
                fit="오버핏", upper_length="롱 기장",
                lower_fit="와이드핏", bottom_length="롱·긴바지 기장",
            ),
        )
        safe = self.engine.evaluate_current_outfit(
            profile,
            self._scorable_pose(),
            self._scorable_outfit(
                fit="오버핏", upper_length="기본 기장",
                lower_fit="스트레이트핏", bottom_length="긴바지",
            ),
        )
        self.assertGreaterEqual(trend.harmony_breakdown["silhouette"], 90.0)
        self.assertGreaterEqual(safe.harmony_breakdown["silhouette"], 65.0)
        self.assertLess(safe.harmony_breakdown["silhouette"], 80.0)
        self.assertGreaterEqual(
            trend.harmony_breakdown["silhouette"] - safe.harmony_breakdown["silhouette"],
            15.0,
        )

    def test_harmony_uses_the_same_85_point_pass_threshold(self):
        evaluation = self.engine.evaluate_current_outfit(
            UserProfile(purpose="데일리", desired_style="캐주얼"),
            self._scorable_pose(),
            self._scorable_outfit(
                fit="오버핏", upper_length="롱 기장",
                lower_fit="와이드핏", bottom_length="롱·긴바지 기장",
            ),
        )
        self.assertEqual(evaluation.harmony_passed, evaluation.harmony_score >= 85.0)
        self.assertIn("R-CMP-03", evaluation.applied_rules)

    def test_auto_scope_uses_delta_minus_change_cost(self):
        profile = UserProfile(
            purpose="데이트", desired_style="캐주얼", budget=180_000,
            change_scope="전체 변경",
        )
        outfit = self._scorable_outfit(upper_type="후드티")
        evaluation = self.engine.evaluate_current_outfit(
            profile, self._scorable_pose(), outfit
        )
        recommendation = self.engine.recommend(
            profile, self._scorable_pose(), outfit, top_k=1
        )[0]
        self.assertEqual(evaluation.change_target, "top")
        self.assertEqual(recommendation.change_target, "top")
        self.assertGreater(recommendation.delta_score, 4.0)
        self.assertAlmostEqual(
            recommendation.utility_score,
            recommendation.delta_score - recommendation.change_cost,
        )

    def test_items_to_keep_prevents_that_item_from_being_replaced(self):
        profile = UserProfile(
            purpose="데이트", desired_style="캐주얼", budget=180_000,
            change_scope="전체 변경", items_to_keep=["top"],
        )
        recommendations = self.engine.recommend(
            profile, self._scorable_pose(), self._scorable_outfit(upper_type="후드티"), top_k=3
        )
        self.assertTrue(all(
            not recommendation.products
            or all(product.category == "bottom" for product in recommendation.products)
            for recommendation in recommendations
        ))

    def test_short_leg_ratio_penalizes_short_tight_lower_combination(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼")
        pose = self._scorable_pose(leg_ratio=0.48)
        tight = self.engine.evaluate_current_outfit(
            profile,
            pose,
            self._scorable_outfit(
                upper_length="롱 기장", bottom_length="쇼츠·미니 기장",
                fit="슬림핏", lower_fit="슬림핏",
            ),
        )
        balanced = self.engine.evaluate_current_outfit(
            profile,
            pose,
            self._scorable_outfit(
                upper_length="크롭 기장", bottom_length="롱·긴바지 기장",
                lower_fit="와이드핏", material="니트", lower_material="데님",
            ),
        )
        self.assertGreaterEqual(balanced.total_score - tight.total_score, 8.0)
        self.assertIn("R-BOD-05", tight.applied_rules)

    def test_narrow_shoulders_penalize_unstructured_tight_top(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼")
        pose = self._scorable_pose(shape="삼각체형")
        tight = self.engine.evaluate_current_outfit(
            profile, pose, self._scorable_outfit(fit="슬림핏")
        )
        structured = self.engine.evaluate_current_outfit(
            profile,
            pose,
            self._scorable_outfit(upper_type="재킷", neckline="라펠 칼라"),
        )
        self.assertGreaterEqual(structured.total_score - tight.total_score, 4.0)
        self.assertIn("R-BOD-06", tight.applied_rules)

    def test_current_outfit_does_not_skip_when_analysis_is_unreliable(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼", change_scope="전체 변경")
        evaluation = self.engine.evaluate_current_outfit(
            profile, self.pose, self.outfit, keep_threshold=0
        )
        self.assertFalse(evaluation.reliable)
        self.assertFalse(evaluation.should_keep)


if __name__ == "__main__":
    unittest.main()
