from __future__ import annotations

import os
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musinsa_live_search import ShoppingProduct
from recommendation_explanations import (
    add_product_recommendation_reasons,
    build_outfit_summary_points,
)
from recommendation_keywords import TargetKeywordResult
from schemas import CurrentOutfitEvaluation, OutfitAnalysis, PoseAnalysis, UserProfile


class RecommendationExplanationTests(unittest.TestCase):
    def setUp(self):
        self.profile = UserProfile(
            purpose="데이트",
            desired_style="미니멀",
            min_budget=30_000,
            max_budget=90_000,
            silhouette_goal="다리가 길어 보이게",
        )
        self.pose = PoseAnalysis(True, 0.9, "삼각체형", 0.9, 0.48, 0.46, "정면", 0.82)

    def test_rule_fallback_is_always_attached_without_llm(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님", "풀렝스"],
        )
        targets = TargetKeywordResult("mixed", {"bottom": {"fit": ["세미와이드"]}})

        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)

        self.assertIn("데이트·미니멀", product.recommendation_reason)
        self.assertIn("다리가 길어 보이게", product.recommendation_reason)
        self.assertEqual(product.recommendation_reason_source, "rules")

    def test_gemini_reason_replaces_fallback_when_explicitly_enabled(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님", "풀렝스"],
        )
        targets = TargetKeywordResult("mixed", {"bottom": {"fit": ["세미와이드"]}})
        generated = json.dumps({
            "items": [{"product_id": "MS1", "reason": "데이트와 미니멀 취향, 다리 보완 목표에 맞는 상품이에요."}]
        }, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                return False

            def read(self):
                payload = {"candidates": [{"content": {"parts": [{"text": generated}]}}]}
                return json.dumps(payload, ensure_ascii=False).encode("utf-8")

        settings = {
            "FASHION_LLM_REASONS": "1",
            "FASHION_LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, settings, clear=False), patch(
            "recommendation_explanations.urllib.request.urlopen", return_value=FakeResponse()
        ):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)

        self.assertEqual(product.recommendation_reason_source, "llm")
        self.assertIn("다리 보완 목표", product.recommendation_reason)

    def test_current_outfit_summary_has_exactly_three_points(self):
        outfit = OutfitAnalysis(
            "test", "네이비", "그레이", "톤온톤", ["셔츠", "슬랙스"], "미니멀",
            upper_type="셔츠", lower_type="슬랙스",
        )
        matrix = {
            "top": {"body_fit": 88.0, "situation_fit": 91.0, "style_fit": 89.0},
            "bottom": {"body_fit": 84.0, "situation_fit": 90.0, "style_fit": 87.0},
        }
        evaluation = CurrentOutfitEvaluation(
            87.0, {}, [], [], 100.0, 0.8, True, 85.0, False,
            "추천 코디로 보완할 수 있어요", diagnostic_matrix=matrix, harmony_score=88.0,
        )

        points = build_outfit_summary_points(outfit, evaluation)

        self.assertEqual(len(points), 3)
        self.assertIn("체형 적합도 88점", points[0])
        self.assertIn("상·하의 조화는 88점", points[2])


if __name__ == "__main__":
    unittest.main()
