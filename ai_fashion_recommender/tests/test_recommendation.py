import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from outfit_analyzer import _dominant_palette, _dominant_rgb, color_harmony
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine
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

    def test_rules_markdown_is_loaded(self):
        self.assertEqual(self.engine.rules_source.name, "FASHION_RULES_MASTER.md")
        self.assertEqual(len(self.engine.documented_rule_ids), 50)
        self.assertEqual(len(self.engine.active_rule_ids), 43)
        self.assertEqual(len(self.engine.scoring_rule_ids), 31)
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
        floor = self._cheapest_daily_pair()
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
        self.assertIn("wardrobe", recommendation.score_breakdown)
        self.assertIn("R-OWN-01", recommendation.applied_rules)

    def test_accessory_rules_return_guidance_not_fake_products(self):
        profile = UserProfile(purpose="여행", budget=100_000, change_scope="상의만 변경")
        recommendation = self.engine.recommend(profile, self.pose, self.outfit, top_k=1)[0]
        self.assertTrue(recommendation.styling_tips)
        self.assertIn("R-CAT-01", recommendation.applied_rules)
        self.assertIn("R-ACC-06", recommendation.applied_rules)
        self.assertTrue(all(product.category in {"top", "bottom"} for product in recommendation.products))


if __name__ == "__main__":
    unittest.main()
