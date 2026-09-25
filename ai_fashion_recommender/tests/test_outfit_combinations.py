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
from live_product_attributes import POLICY_VERSION
from outfit_combination_recommender import recommend_outfit_combinations
from recommendation_keywords import TargetKeywordResult
from schemas import OutfitAnalysis, PoseAnalysis, UserProfile


class FakeRecommender:
    def __init__(self):
        self.harmony_calls = []

    def _garment(self, _product, category, _outfit):
        self.asserted_category = category
        return {
            "category": category,
            "color": "블루",
            "fit": "스트레이트핏",
            "length": "풀렝스",
            "pattern": "무지",
            "material": "데님",
            "formality": 2,
        }

    def _outfit_harmony_score(self, top, bottom, _profile):
        self.harmony_calls.append((top, bottom))
        score = 0.94 if top["fit"] == "오버핏" else 0.82
        return (
            score,
            {"silhouette": score},
            ["현재 하의와 상의의 볼륨 관계가 자연스럽게 이어집니다."],
            ["R-CMP-03", "R-SIL-01"],
        )


def product(product_id: str, category: str, matched: list[str]) -> ShoppingProduct:
    return ShoppingProduct(
        product_id, f"{product_id} 상품", "브랜드", 59_000,
        "https://image", "https://product", category,
        matched_keywords=matched, retrieval_score=10.0,
    )


class OutfitCombinationTests(unittest.TestCase):
    def setUp(self):
        self.profile = UserProfile(desired_style="스트리트", change_categories=["top", "shoes"])
        self.pose = PoseAnalysis(True, 0.9, "사각체형", 1.0, 0.5, 0.48, "정면", 0.8)
        self.outfit = OutfitAnalysis(
            "parser", "화이트", "블루", "보통 조합", ["티셔츠", "데님"], "캐주얼",
            upper_type="티셔츠", lower_type="팬츠", lower_subtype="데님 팬츠",
            fit="레귤러핏", lower_fit="스트레이트핏", bottom_length="풀렝스",
            material="코튼", lower_material="데님", pattern="무지", lower_pattern="무지",
        )
        self.targets = TargetKeywordResult(
            "user_input",
            {
                "top": {"fit": ["오버핏", "레귤러"], "style": ["스트리트"]},
                "shoes": {"item_type": ["스니커즈", "부츠"], "style": ["스트리트"]},
            },
        )

    def test_three_outfits_replace_selected_categories_and_keep_current_bottom(self):
        products = [
            product("T1", "top", ["오버핏", "스트리트"]),
            product("T2", "top", ["레귤러", "스트리트"]),
            product("S1", "shoes", ["스니커즈", "스트리트"]),
            product("S2", "shoes", ["부츠", "스트리트"]),
        ]
        recommender = FakeRecommender()

        with patch.dict(os.environ, {"FASHION_LLM_REASONS": "0"}, clear=False):
            outfits = recommend_outfit_combinations(
                products, self.profile, self.pose, self.outfit, self.targets, recommender, limit=3,
            )

        self.assertEqual(len(outfits), 3)
        self.assertEqual([item.combination_id for item in outfits], ["OUTFIT-1", "OUTFIT-2", "OUTFIT-3"])
        self.assertEqual(len({tuple(item.product_ids) for item in outfits}), 3)
        self.assertTrue(all(len(item.product_ids) == 2 for item in outfits))
        self.assertTrue(all(item.product_ids[0].startswith("T") for item in outfits))
        self.assertTrue(all(item.product_ids[1].startswith("S") for item in outfits))
        self.assertTrue(all([current["category"] for current in item.current_items] == ["bottom"] for item in outfits))
        self.assertTrue(all("현재 하의" in item.reason for item in outfits))
        self.assertTrue(all("현재 하의를" in item.reason for item in outfits))
        self.assertTrue(all("R-CMP-03" in item.public_dict()["rule_ids"] for item in outfits))
        self.assertEqual(len(recommender.harmony_calls), 4)

    def test_missing_selected_category_returns_no_partial_outfit(self):
        products = [product("T1", "top", ["오버핏"])]
        outfits = recommend_outfit_combinations(
            products, self.profile, self.pose, self.outfit, self.targets, FakeRecommender(), limit=3,
        )
        self.assertEqual(outfits, [])

    def test_three_outfits_do_not_reuse_category_items_when_candidates_are_sufficient(self):
        products = [
            product("T1", "top", ["오버핏", "스트리트"]),
            product("T2", "top", ["레귤러", "스트리트"]),
            product("T3", "top", ["레귤러", "스트리트"]),
            product("S1", "shoes", ["스니커즈", "스트리트"]),
            product("S2", "shoes", ["부츠", "스트리트"]),
            product("S3", "shoes", ["로퍼", "스트리트"]),
        ]
        outfits = recommend_outfit_combinations(
            products, self.profile, self.pose, self.outfit, self.targets,
            FakeRecommender(), limit=3,
        )
        top_ids = [next(product_id for product_id in item.product_ids if product_id.startswith("T"))
                   for item in outfits]
        shoe_ids = [next(product_id for product_id in item.product_ids if product_id.startswith("S"))
                    for item in outfits]
        self.assertEqual(len(set(top_ids)), 3)
        self.assertEqual(len(set(shoe_ids)), 3)

    def test_each_public_outfit_contains_at_most_three_verified_facts(self):
        products = [
            product("T1", "top", ["오버핏", "스트리트"]),
            product("S1", "shoes", ["스니커즈", "스트리트"]),
        ]
        outfits = recommend_outfit_combinations(
            products, self.profile, self.pose, self.outfit, self.targets, FakeRecommender(), limit=3,
        )
        payload = outfits[0].public_dict()
        self.assertLessEqual(len(payload["evidence"]), 3)
        self.assertEqual(len(payload["evidence"]), len(payload["evidence_labels"]))
        self.assertNotIn("예산", payload["reason"])

    def test_sporty_outfit_cannot_be_carried_by_running_shoes_alone(self):
        profile = UserProfile(
            purpose="데일리", desired_style="스포티",
            change_categories=["top", "bottom", "shoes"],
        )
        targets = TargetKeywordResult("user_input", {
            "top": {"item_type": ["트랙 재킷"]},
            "bottom": {"item_type": ["트랙팬츠"]},
            "shoes": {"item_type": ["러닝화"]},
        })
        top = product("T1", "top", [])
        top.name = "체크 오버핏 셔츠"
        bottom = product("B1", "bottom", [])
        bottom.name = "레귤러 데님 팬츠"
        shoes = product("S1", "shoes", ["러닝화"])
        shoes.name = "러닝화"

        outfits = recommend_outfit_combinations(
            [top, bottom, shoes], profile, self.pose, self.outfit,
            targets, FakeRecommender(), limit=3,
        )

        self.assertEqual(outfits, [])

    def test_sports_brand_denim_is_compatible_but_two_real_sporty_anchors_are_required(self):
        profile = UserProfile(
            purpose="데일리", desired_style="스포티",
            change_categories=["top", "bottom", "shoes"],
        )
        targets = TargetKeywordResult("user_input", {
            "top": {"item_type": ["트랙 재킷"]},
            "bottom": {"item_type": ["트랙팬츠"]},
            "shoes": {"item_type": ["러닝화"]},
        })
        top = product("T1", "top", ["트랙 재킷"])
        top.name = "사이드라인 트랙 재킷"
        bottom = product("B1", "bottom", [])
        bottom.name = "레귤러 데님 팬츠"
        bottom.brand = "아디다스"
        shoes = product("S1", "shoes", ["러닝화"])
        shoes.name = "러닝화"

        outfits = recommend_outfit_combinations(
            [top, bottom, shoes], profile, self.pose, self.outfit,
            targets, FakeRecommender(), limit=1,
        )

        self.assertEqual(len(outfits), 1)
        self.assertIn("스포티 구성", outfits[0].public_dict()["evidence_labels"])
        self.assertIn("스포츠 브랜드 예외", outfits[0].reason)

    def test_visual_fit_is_used_by_fashion_rule_harmony(self):
        self.targets = TargetKeywordResult(
            "mixed",
            {"top": {"fit": ["오버핏"]}, "shoes": {"item_type": ["스니커즈"]}},
            applied_rules=["R-SIL-01", "R-CTX-01"],
            keyword_rules={
                "top": {"오버핏": ["R-SIL-01"]},
                "shoes": {"스니커즈": ["R-CTX-01"]},
            },
        )
        top = product("T1", "top", [])
        top.photo_attributes = {"fit": {
            "label": "오버핏", "confidence": 0.96, "keyword": "오버핏",
            "source": "product_photo", "policy": POLICY_VERSION,
        }}
        shoe = product("S1", "shoes", ["스니커즈"])
        recommender = FakeRecommender()

        outfits = recommend_outfit_combinations(
            [top, shoe], self.profile, self.pose, self.outfit,
            self.targets, recommender, limit=1,
        )

        self.assertEqual(recommender.harmony_calls[0][0]["fit"], "오버핏")
        self.assertEqual(recommender.harmony_calls[0][0]["fit_source"], "product_photo")
        self.assertIn("상품 핏", outfits[0].public_dict()["evidence_labels"])
        self.assertIn("R-SIL-01", outfits[0].public_dict()["rule_ids"])

    def test_gemini_can_phrase_the_verified_combination_evidence(self):
        products = [
            product("T1", "top", ["오버핏", "스트리트"]),
            product("S1", "shoes", ["스니커즈", "스트리트"]),
        ]
        generated = json.dumps({"items": [{
            "combination_id": "OUTFIT-1",
            "summary": "현재 하의를 유지하면서 오버핏 상의와 스니커즈로 흐름을 연결한 코디예요.",
            "evidence_ids": ["OUTFIT-1-E1", "OUTFIT-1-E2"],
        }]}, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        settings = {
            "FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, settings, clear=False), patch(
            "outfit_combination_recommender.urllib.request.urlopen", return_value=FakeResponse()
        ):
            outfits = recommend_outfit_combinations(
                products, self.profile, self.pose, self.outfit, self.targets, FakeRecommender(), limit=1,
            )

        self.assertEqual(outfits[0].reason_source, "llm")
        self.assertIn("현재 하의", outfits[0].reason)

    def test_llm_combination_reason_that_mentions_budget_is_rejected(self):
        products = [
            product("T1", "top", ["오버핏", "스트리트"]),
            product("S1", "shoes", ["스니커즈", "스트리트"]),
        ]
        generated = json.dumps({"items": [{
            "combination_id": "OUTFIT-1", "summary": "예산에 맞아서 추천해요.",
            "evidence_ids": ["OUTFIT-1-E1"],
        }]}, ensure_ascii=False)

        class FakeResponse:
            def __enter__(self): return self
            def __exit__(self, *_args): return False
            def read(self):
                return json.dumps({"candidates": [{"content": {"parts": [{"text": generated}]}}]},
                                  ensure_ascii=False).encode("utf-8")

        settings = {
            "FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini",
            "GEMINI_API_KEY": "test-key",
        }
        with patch.dict(os.environ, settings, clear=False), patch(
            "outfit_combination_recommender.urllib.request.urlopen", return_value=FakeResponse()
        ):
            outfits = recommend_outfit_combinations(
                products, self.profile, self.pose, self.outfit, self.targets, FakeRecommender(), limit=1,
            )

        self.assertEqual(outfits[0].reason_source, "rules")
        self.assertNotIn("예산", outfits[0].reason)


if __name__ == "__main__":
    unittest.main()
