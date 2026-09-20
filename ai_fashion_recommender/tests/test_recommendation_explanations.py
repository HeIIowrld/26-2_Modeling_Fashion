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
            matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"},
            applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )

        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)

        self.assertEqual(
            product.recommendation_reason,
            "추천 규칙에서 도출된 '세미와이드' 핏이 상품명과 일치합니다.",
        )
        self.assertNotIn("예산", product.recommendation_reason)
        self.assertNotIn("데이트", product.recommendation_reason)
        self.assertNotIn("체형", product.recommendation_reason)
        self.assertEqual(product.recommendation_reason_source, "rules")

    def test_shoe_reason_excludes_body_correction(self):
        product = ShoppingProduct("MS2", "블랙 로퍼", "브랜드", 59000,
                                  "https://image", "https://product", "shoes",
                                  search_keywords=["로퍼", "미니멀", "블랙"],
                                  matched_keywords=["로퍼"])
        targets = TargetKeywordResult(
            "user_input", {"shoes": {"item_type": ["로퍼"]}},
            sources={"shoes.item_type": "user_style_rule"},
            applied_rules=["R-CTX-01", "R-ACC-06"],
            keyword_rules={"shoes": {"로퍼": ["R-CTX-01", "R-ACC-06"]}},
        )
        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)
        self.assertEqual(product.recommendation_reason, "추천 조건에서 도출된 '로퍼' 종류가 상품명과 일치합니다.")
        self.assertNotIn("다리", product.recommendation_reason)
        self.assertNotIn("체형", product.recommendation_reason)

    def test_gemini_reason_replaces_fallback_when_explicitly_enabled(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님", "풀렝스"],
            matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )
        generated = json.dumps({
            "items": [{
                "product_id": "MS1",
                "summary": "추천 규칙에서 확인된 세미와이드 핏이 상품명과 일치해 추천했어요.",
                "evidence_ids": ["MS1-E1"],
            }]
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
        self.assertEqual(product.recommendation_reason, "추천 규칙에서 확인된 세미와이드 핏이 상품명과 일치해 추천했어요.")

    def test_llm_without_matched_keyword_falls_back(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님"], matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )
        generated = json.dumps({
            "items": [{"product_id": "MS1", "summary": "그냥 멋진 상품이라 추천해요.", "evidence_ids": ["MS1-E1"]}]
        }, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        settings = {"FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "test-key"}
        with patch.dict(os.environ, settings, clear=False), patch(
            "recommendation_explanations.urllib.request.urlopen", return_value=FakeResponse()
        ):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)
        self.assertEqual(product.recommendation_reason_source, "rules")
        self.assertIn("세미와이드", product.recommendation_reason)

    def test_llm_copy_that_mentions_budget_falls_back(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 데님", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님"], matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )
        generated = json.dumps({
            "items": [{"product_id": "MS1", "summary": "세미와이드 핏이라 예산에도 잘 맞아 추천해요.", "evidence_ids": ["MS1-E1"]}]
        }, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        settings = {"FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "test-key"}
        with patch.dict(os.environ, settings, clear=False), patch(
            "recommendation_explanations.urllib.request.urlopen", return_value=FakeResponse()
        ):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)
        self.assertEqual(product.recommendation_reason_source, "rules")
        self.assertNotIn("예산", product.recommendation_reason)

    def test_llm_with_unknown_evidence_id_falls_back(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 팬츠", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )
        generated = json.dumps({"items": [{
            "product_id": "MS1", "summary": "세미와이드 핏이라 추천해요.",
            "evidence_ids": ["MS1-NOT-ALLOWED"],
        }]}, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        settings = {"FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "test-key"}
        with patch.dict(os.environ, settings, clear=False), patch(
            "recommendation_explanations.urllib.request.urlopen", return_value=FakeResponse()
        ):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)
        self.assertEqual(product.recommendation_reason_source, "rules")

    def test_llm_receives_only_allowed_product_evidence(self):
        product = ShoppingProduct(
            "MS1", "세미 와이드 팬츠", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님", "데이트"],
            matched_keywords=["세미와이드"],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"], "purpose": ["데이트"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01", "R-CTX-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"], "데이트": ["R-CTX-01"]}},
        )
        generated = json.dumps({"items": [{
            "product_id": "MS1", "summary": "세미와이드 핏이 상품명과 일치해 추천했어요.",
            "evidence_ids": ["MS1-E1"],
        }]}, ensure_ascii=False)
        captured = {}

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        def fake_open(request, timeout):
            captured.update(json.loads(request.data.decode("utf-8")))
            return FakeResponse()

        settings = {"FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
                    "GEMINI_API_KEY": "test-key"}
        with patch.dict(os.environ, settings, clear=False), patch(
            "recommendation_explanations.urllib.request.urlopen", side_effect=fake_open
        ):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)

        prompt_payload = json.loads(captured["contents"][0]["parts"][0]["text"])
        self.assertEqual(set(prompt_payload), {"products"})
        sent_product = prompt_payload["products"][0]
        self.assertNotIn("search_keywords", sent_product)
        self.assertNotIn("user_context", prompt_payload)
        self.assertNotIn("body_analysis", prompt_payload)
        self.assertEqual(sent_product["allowed_evidence"][0]["matched_keywords"], ["세미와이드"])
        self.assertNotIn("데이트", json.dumps(prompt_payload, ensure_ascii=False))

    def test_unmatched_representative_keywords_do_not_create_reason(self):
        product = ShoppingProduct(
            "MS1", "베이직 팬츠", "브랜드", 59_000,
            "https://image", "https://product", "bottom",
            search_keywords=["세미와이드", "데님", "데이트"], matched_keywords=[],
        )
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["세미와이드"]}},
            sources={"bottom.fit": "user_style_rule"}, applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-SIL-01"]}},
        )
        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            add_product_recommendation_reasons([product], self.profile, self.pose, targets)
        self.assertEqual(product.recommendation_reason, "")
        self.assertEqual(product.fit_evidence, [])

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
