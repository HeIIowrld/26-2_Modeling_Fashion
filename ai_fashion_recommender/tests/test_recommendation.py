import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from outfit_analyzer import _dominant_rgb, color_harmony
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine
from schemas import OutfitAnalysis, PoseAnalysis, UserProfile


class RecommendationTests(unittest.TestCase):
    def setUp(self):
        self.catalog = ProductCatalog(ROOT / "data" / "products.csv")
        self.engine = RecommendationEngine(ROOT / "data" / "fashion_rules.json", self.catalog)
        self.pose = PoseAnalysis(True, 0.9, "균형형", 1.0, 0.5, 0.55, "정면에 가까움")
        self.outfit = OutfitAnalysis("test", "화이트", "블랙", "안정적인 무채색 조합", ["top", "pants"], "캐주얼")

    def test_catalog_loads(self):
        self.assertEqual(len(self.catalog.products), 12)

    def test_full_change_returns_ranked_pairs(self):
        profile = UserProfile(purpose="데일리", desired_style="캐주얼", budget=180_000, change_scope="전체 변경")
        recommendations = self.engine.recommend(profile, self.pose, self.outfit, top_k=3)
        self.assertEqual(len(recommendations), 3)
        self.assertTrue(all(len(item.products) == 2 for item in recommendations))
        self.assertGreaterEqual(recommendations[0].total_score, recommendations[1].total_score)

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


if __name__ == "__main__":
    unittest.main()
