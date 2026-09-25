import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recommendation_keywords import RecommendationKeywordGenerator
from schemas import OutfitAnalysis, PoseAnalysis, UserProfile


class RecommendationKeywordTests(unittest.TestCase):
    def setUp(self):
        self.generator = RecommendationKeywordGenerator()
        self.pose = PoseAnalysis(
            True, 0.95, "삼각체형", 0.88, 0.56, 0.55,
            "정면에 가까움", body_shape_confidence=0.91,
        )
        self.outfit = OutfitAnalysis(
            "test", "레드", "블랙", "대비색 조합", ["top", "pants"], "스트리트",
            upper_type="후드티", lower_type="팬츠", upper_length="롱 기장",
            bottom_length="앵클", fit="오버핏", lower_fit="슬림핏",
            material="니트", lower_material="가죽", attribute_confidence=0.92,
        )

    def test_user_input_wins_over_photo_attributes(self):
        profile = UserProfile(
            purpose="데이트",
            desired_style="미니멀",
            change_scope="하의만 변경",
            season="가을",
            preferred_colors=["네이비"],
            preferred_materials=["데님"],
            provided_fields=[
                "purpose", "desired_style", "change_scope", "season",
                "preferred_colors", "preferred_materials",
            ],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)
        bottom = result.targets["bottom"]

        self.assertEqual(set(result.targets), {"bottom"})
        self.assertEqual(bottom["style"], ["미니멀"])
        self.assertEqual(bottom["color"], ["네이비"])
        self.assertEqual(bottom["material"], ["데님"])
        self.assertNotIn("스트리트", bottom["style"])
        self.assertEqual(result.sources["style"], "user_input")
        self.assertEqual(result.mode, "mixed")  # 체형·비율은 사진에서 보충

    def test_shoes_ignore_body_and_clothing_material(self):
        profile = UserProfile(change_categories=["shoes"], desired_style="미니멀",
                              purpose="데이트", provided_fields=["desired_style", "purpose"])
        first = self.generator.generate(profile, self.pose, self.outfit)
        self.pose.leg_ratio = .9
        self.pose.body_shape = "역삼각체형"
        second = self.generator.generate(profile, self.pose, self.outfit)
        self.assertEqual(first.to_dict(), second.to_dict())
        self.assertEqual(first.targets["shoes"]["item_type"], ["로퍼"])
        self.assertEqual(first.keyword_rules["shoes"]["로퍼"], ["R-CTX-01", "R-ACC-06"])
        self.assertNotIn("material", first.targets["shoes"])
        self.assertFalse(any(rule.startswith("R-BOD") for rule in first.applied_rules))

    def test_checkbox_combinations_override_legacy_scope(self):
        from itertools import combinations
        for count in (1, 2, 3):
            for values in combinations(("top", "bottom", "shoes"), count):
                profile = UserProfile(change_categories=list(values), change_scope="현재 유지")
                result = self.generator.generate(profile, self.pose, self.outfit)
                self.assertEqual(tuple(result.targets), values)
        for invalid in ([], ["hat"], "shoes"):
            with self.assertRaises(ValueError):
                self.generator.generate(UserProfile(change_categories=invalid), self.pose, self.outfit)

    def test_trained_shoe_photo_fallback(self):
        self.outfit.shoes = {"accepted": True, "item_type": "부츠"}
        result = self.generator.generate(UserProfile(change_categories=["shoes"], provided_fields=[]),
                                         self.pose, self.outfit)
        self.assertEqual(result.targets["shoes"]["item_type"], ["부츠"])

    def test_missing_inputs_fall_back_to_photo(self):
        profile = UserProfile(
            purpose="데일리",
            desired_style="캐주얼",
            change_scope="전체 변경",
            season="사계절",
            provided_fields=[],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)

        self.assertEqual(set(result.targets), {"top", "bottom"})
        self.assertEqual(result.targets["top"]["style"], ["스트리트"])
        self.assertEqual(result.sources["style"], "photo_fallback")
        self.assertIn("레드", result.targets["top"]["harmony_reference_color"])
        self.assertIn("풀렝스", result.targets["bottom"]["length"])
        self.assertIn("하이라이즈", result.targets["bottom"]["waistline"])
        self.assertEqual(result.mode, "photo_fallback")

    def test_constraints_are_separate_from_target_keywords(self):
        profile = UserProfile(
            change_scope="상의만 변경",
            min_budget=50_000,
            max_budget=100_000,
            avoided_colors=["베이지"],
            avoided_materials=["가죽"],
            provided_fields=["change_scope", "min_budget", "max_budget"],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)

        self.assertEqual(result.constraints["search_categories"], ["top"])
        self.assertEqual(result.constraints["min_budget"], 50_000)
        self.assertEqual(result.constraints["max_budget"], 100_000)
        self.assertEqual(result.constraints["excluded_colors"], ["베이지"])
        self.assertEqual(result.constraints["excluded_materials"], ["가죽"])
        self.assertNotIn("excluded_colors", result.targets["top"])

    def test_brief_output_contains_no_numeric_score(self):
        result = self.generator.generate(
            UserProfile(provided_fields=[]), self.pose, self.outfit
        )
        lines = result.brief_lines()

        self.assertTrue(lines)
        self.assertTrue(all("점" not in line and "/ 100" not in line for line in lines))

    def test_live_reranking_rules_are_applied_only_to_relevant_contexts(self):
        casual = self.generator.generate(
            UserProfile(change_categories=["top", "bottom"], purpose="데일리",
                        desired_style="캐주얼"), self.pose, self.outfit
        )
        formal = self.generator.generate(
            UserProfile(change_categories=["top", "bottom"], purpose="출근",
                        desired_style="포멀"), self.pose, self.outfit
        )
        self.assertIn("R-DET-01", casual.applied_rules)
        self.assertIn("R-TREND-01", casual.applied_rules)
        self.assertIn("R-DET-01", formal.applied_rules)
        self.assertNotIn("R-TREND-01", formal.applied_rules)
        self.assertIn("R-CTX-01", formal.applied_rules)

    def test_formal_context_collects_formal_item_types_before_photo_material(self):
        profile = UserProfile(
            change_categories=["top", "bottom", "shoes"], purpose="출근",
            desired_style="포멀", provided_fields=["purpose", "desired_style"],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)

        self.assertEqual(result.targets["top"]["item_type"][:2], ["셔츠", "블레이저"])
        self.assertEqual(result.targets["bottom"]["item_type"][:2], ["슬랙스", "치노 팬츠"])
        self.assertEqual(result.targets["shoes"]["item_type"][:2], ["더비슈즈", "로퍼"])
        self.assertNotIn("material", result.targets["top"])
        self.assertNotIn("material", result.targets["bottom"])
        self.assertEqual(result.keyword_rules["top"]["셔츠"], ["R-CTX-01"])

    def test_sporty_context_collects_athletic_item_types_before_photo_material(self):
        profile = UserProfile(
            change_categories=["top", "bottom", "shoes"], purpose="데일리",
            desired_style="스포티", provided_fields=["purpose", "desired_style"],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)

        self.assertEqual(result.targets["top"]["item_type"][:2], ["트랙 재킷", "저지"])
        self.assertEqual(result.targets["bottom"]["item_type"][:2], ["트랙팬츠", "조거팬츠"])
        self.assertEqual(result.targets["shoes"]["item_type"][:2], ["러닝화", "스니커즈"])
        self.assertNotIn("material", result.targets["top"])
        self.assertNotIn("material", result.targets["bottom"])
        self.assertEqual(result.keyword_rules["bottom"]["트랙팬츠"], ["R-CTX-01"])

    def test_photo_material_fallback_stays_in_its_own_category(self):
        profile = UserProfile(
            change_categories=["top", "bottom"], purpose="데일리",
            desired_style="캐주얼", provided_fields=["purpose", "desired_style"],
        )
        result = self.generator.generate(profile, self.pose, self.outfit)

        self.assertEqual(result.targets["top"]["material"], ["니트"])
        self.assertEqual(result.targets["bottom"]["material"], ["가죽"])
        self.assertEqual(result.sources["top.material"], "photo_fallback")
        self.assertEqual(result.sources["bottom.material"], "photo_fallback")

    def test_unreliable_photo_and_no_inputs_still_produce_searchable_keywords(self):
        unknown_outfit = OutfitAnalysis(
            "test", "분석 불가", "분석 불가", "보통 조합", [], "스타일 불확실"
        )
        unreliable_pose = PoseAnalysis(
            False, 0.2, "분석 불확실", 1.0, 0.5, 0.5, "판단 보류",
            body_shape_confidence=0.1,
        )
        result = self.generator.generate(
            UserProfile(provided_fields=[]), unreliable_pose, unknown_outfit
        )

        self.assertTrue(result.targets["top"]["fit"])
        self.assertTrue(result.targets["top"]["length"])
        self.assertTrue(result.targets["bottom"]["fit"])
        self.assertTrue(result.targets["bottom"]["length"])
        self.assertEqual(result.sources["top.fit"], "fashion_rule_default")


if __name__ == "__main__":
    unittest.main()
