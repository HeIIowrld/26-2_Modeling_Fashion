import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musinsa_live_search import ShoppingProduct
from recommendation_explanations import add_product_recommendation_reasons
from recommendation_keywords import RecommendationKeywordGenerator
from schemas import OutfitAnalysis, PoseAnalysis, UserProfile


class RecommendationEvidenceTests(unittest.TestCase):
    def test_rule_mapping_and_product_payload_fields(self):
        profile = UserProfile(
            purpose="데일리",
            desired_style="캐주얼",
            change_scope="하의만 변경",
            silhouette_goal="다리가 길어 보이게",
            provided_fields={"purpose", "desired_style", "change_scope", "silhouette_goal"},
        )
        pose = PoseAnalysis(True, 0.9, "삼각체형", 0.9, 0.48, 0.46, "정면", 0.82)
        outfit = OutfitAnalysis("test", "블랙", "블루", "보통 조합", [], "캐주얼")
        targets = RecommendationKeywordGenerator().generate(profile, pose, outfit)
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님 팬츠", "브랜드", 59000,
            "https://image", "https://product", "bottom",
            matched_keywords=["세미와이드"],
        )

        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            add_product_recommendation_reasons([product], profile, pose, targets)

        self.assertIn("R-BOD-05", targets.keyword_rules["bottom"]["세미와이드"])
        self.assertTrue(product.fit_evidence)
        self.assertIn("세미와이드", product.fit_evidence[0])
        self.assertIn("R-BOD-05", product.reason_rule_ids)
        self.assertIn("fit_evidence", product.public_dict())
        self.assertIn("reason_rule_ids", product.public_dict())


if __name__ == "__main__":
    unittest.main()