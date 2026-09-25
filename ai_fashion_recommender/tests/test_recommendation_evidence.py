import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from musinsa_live_search import ShoppingProduct
from recommendation_explanations import add_product_recommendation_reasons, build_product_evidence
from recommendation_keywords import RecommendationKeywordGenerator, TargetKeywordResult
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

    def test_sold_colors_never_become_a_recommendation_reason(self):
        """파는 색 목록으로는 추천하지 않는다 — 카드 사진과 합성은 대표 사진 한 장이다.

        첫 컬러칩과 대표 사진 색이 같은 경우가 345개 중 46%뿐이었다(2026-09-25).
        """
        profile = UserProfile(purpose="데일리", preferred_colors=["블랙"],
                              provided_fields={"purpose", "preferred_colors"})
        pose = PoseAnalysis(True, 0.9, "삼각체형", 0.9, 0.48, 0.46, "정면", 0.82)
        outfit = OutfitAnalysis("test", "화이트", "블루", "보통 조합", [], "캐주얼")
        targets = RecommendationKeywordGenerator().generate(profile, pose, outfit)
        for category in targets.targets:
            targets.targets[category]["color"] = ["블랙"]
        product = ShoppingProduct("MS1", "베이직 반팔 티셔츠", "브랜드", 39000,
                                  "https://image", "https://product", "top",
                                  color_options=["아이보리", "(19)BLACK"])

        evidence = build_product_evidence(product, profile, pose, targets)

        self.assertEqual([item for item in evidence if "색" in item.label], [])
        self.assertNotIn("블랙", " ".join(item.text for item in evidence))

    def test_low_confidence_shape_is_not_stated(self):
        profile = UserProfile(provided_fields=[])
        pose = PoseAnalysis(True, 0.9, "삼각체형", 0.90, 0.48, 0.46, "정면", 0.40)
        targets = TargetKeywordResult(
            "photo_fallback", {"bottom": {"fit": ["세미와이드"]}},
            sources={"body_shape": "photo_fallback"}, applied_rules=["R-BOD-01"],
            keyword_rules={"bottom": {"세미와이드": ["R-BOD-01"]}},
        )
        product = ShoppingProduct(
            "MS1", "세미 와이드 팬츠", "브랜드", 59000,
            "https://image", "https://product", "bottom",
            matched_keywords=["세미와이드"],
        )

        evidence = build_product_evidence(product, profile, pose, targets)

        self.assertFalse(any(item.kind == "body_shape" for item in evidence))
        self.assertFalse(any("삼각체형" in item.text for item in evidence))

    def test_confident_shape_uses_category_specific_evidence(self):
        profile = UserProfile(provided_fields=[])
        pose = PoseAnalysis(True, 0.9, "삼각체형", 0.90, 0.48, 0.46, "정면", 0.88)
        targets = TargetKeywordResult(
            "photo_fallback",
            {"top": {"structure": ["어깨 구조"]}, "bottom": {"fit": ["스트레이트"]}},
            sources={"body_shape": "photo_fallback"}, applied_rules=["R-BOD-01"],
            keyword_rules={
                "top": {"어깨 구조": ["R-BOD-01"]},
                "bottom": {"스트레이트": ["R-BOD-01"]},
            },
        )
        top = ShoppingProduct(
            "T1", "어깨 구조 셔츠", "브랜드", 59000,
            "https://image", "https://product", "top", matched_keywords=["어깨 구조"],
        )
        bottom = ShoppingProduct(
            "B1", "스트레이트 팬츠", "브랜드", 59000,
            "https://image", "https://product", "bottom", matched_keywords=["스트레이트"],
        )

        top_evidence = build_product_evidence(top, profile, pose, targets)
        bottom_evidence = build_product_evidence(bottom, profile, pose, targets)

        self.assertEqual(top_evidence[0].kind, "body_shape")
        self.assertEqual(bottom_evidence[0].kind, "body_shape")
        self.assertIn("상체 라인", top_evidence[0].text)
        self.assertIn("하체 볼륨", bottom_evidence[0].text)

    def test_product_evidence_is_limited_to_three(self):
        profile = UserProfile(desired_style="캐주얼", provided_fields=["desired_style"])
        pose = PoseAnalysis(False, 0.0, "분석 보류", 0.0, 0.0, 0.0, "분석 보류", 0.0)
        targets = TargetKeywordResult(
            "user_input",
            {"top": {
                "fit": ["여유핏"], "style": ["캐주얼"], "material": ["코튼"],
                "color": ["블랙"], "function": ["통기성"],
            }},
            sources={"top.fit": "user_style_rule", "style": "user_input", "material": "user_input",
                     "color": "user_input", "function": "user_input"},
            applied_rules=["R-SIL-01", "R-MAT-01", "R-COL-08", "R-WEA-02"],
            keyword_rules={"top": {
                "여유핏": ["R-SIL-01"], "코튼": ["R-MAT-01"],
                "블랙": ["R-COL-08"], "통기성": ["R-WEA-02"],
            }},
        )
        product = ShoppingProduct(
            "MS1", "캐주얼 여유핏 블랙 코튼 통기성 셔츠", "브랜드", 59000,
            "https://image", "https://product", "top",
            matched_keywords=["여유핏", "캐주얼", "코튼", "블랙", "통기성"],
        )

        evidence = build_product_evidence(product, profile, pose, targets)

        self.assertLessEqual(len(evidence), 3)
        self.assertEqual([item.kind for item in evidence], ["fit", "style", "detail"])


if __name__ == "__main__":
    unittest.main()
