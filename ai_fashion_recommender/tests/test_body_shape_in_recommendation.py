"""분석한 체형이 실제로 추천을 바꾸는지 지킨다.

이전에는 사용자가 실루엣 목표를 고른 경우에만 체형 규칙이 돌았다(R-KOR-02).
기본값이 '반영 안 함'이라, 사진을 분석해 체형을 판정해 놓고도 추천에는 전혀
쓰이지 않아 체형 분석이 화면 표시용으로만 남아 있었다.

또 상품 카탈로그의 body_shapes 칼럼은 ProductCatalog 이 읽어 Product 에 넣기만 하고
추천 엔진이 참조하지 않는 죽은 데이터였다. 체형 라벨(역삼각·사각·삼각)과 카탈로그
어휘(상체 강조형·하체 강조형·균형형)가 서로 다른 축이라 섞어 비교할 수 없었기 때문이다.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import recommendation_engine as engine_module  # noqa: E402
from product_catalog import ProductCatalog  # noqa: E402
from recommendation_engine import RecommendationEngine  # noqa: E402
from schemas import (  # noqa: E402
    ALL_BODY_SHAPES,
    CATALOG_FOCUS_LABELS,
    FOCUS_LOWER,
    FOCUS_UPPER,
    GOAL_NONE,
    OutfitAnalysis,
    PoseAnalysis,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_TRIANGLE,
    UserProfile,
)


def sample_outfit() -> OutfitAnalysis:
    return OutfitAnalysis(
        parser_backend="fashn-human-parser",
        upper_color="화이트", lower_color="네이비",
        color_harmony="안정적인 무채색 조합",
        detected_items=["top", "pants"], style="캐주얼",
        upper_type="셔츠", lower_type="청바지",
        fit="레귤러핏", lower_fit="와이드핏",
        pattern="무지", material="코튼",
        lower_pattern="무지", lower_material="데님",
        attribute_confidence=0.8,
    )


def pose(shape: str, confidence: float = 0.9) -> PoseAnalysis:
    return PoseAnalysis(True, 0.96, shape, 1.0, 0.66, 0.44, "정면에 가까움", confidence)


class BodyShapeReachesRecommendationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngine(
            ROOT / "FASHION_RULES_MASTER.md",
            ProductCatalog(ROOT / "data" / "products.csv"),
        )
        cls.outfit = sample_outfit()

    def picks(self, shape, confidence=0.9, goal=GOAL_NONE):
        results = self.engine.recommend(
            UserProfile(purpose="데일리", silhouette_goal=goal),
            pose(shape, confidence), self.outfit,
        )
        return ["+".join(p.product_id for p in r.products) for r in results]

    def test_different_body_shapes_give_different_recommendations(self):
        """목표를 고르지 않아도 체형이 결과를 바꿔야 한다."""
        self.assertNotEqual(self.picks(SHAPE_INVERTED_TRIANGLE), self.picks(SHAPE_TRIANGLE))

    def test_low_confidence_does_not_apply_the_body_shape(self):
        """판정이 흐릿할 때까지 밀어붙이면 잘못된 근거로 추천이 갈린다."""
        self.assertEqual(
            self.picks(SHAPE_INVERTED_TRIANGLE, confidence=0.4),
            self.picks(SHAPE_TRIANGLE, confidence=0.4),
        )

    def test_reason_names_the_body_shape(self):
        """왜 그렇게 추천했는지 사용자가 볼 수 있어야 한다."""
        results = self.engine.recommend(
            UserProfile(purpose="데일리", silhouette_goal=GOAL_NONE),
            pose(SHAPE_INVERTED_TRIANGLE), self.outfit,
        )
        reasons = " ".join(results[0].reasons)
        self.assertIn(SHAPE_INVERTED_TRIANGLE, reasons)

    def test_reason_avoids_language_the_rulebook_forbids(self):
        """R-KOR-02 는 '단점'·'결함'·'가려야' 같은 표현을 금지한다."""
        results = self.engine.recommend(
            UserProfile(purpose="데일리", silhouette_goal=GOAL_NONE),
            pose(SHAPE_TRIANGLE), self.outfit,
        )
        text = " ".join(results[0].reasons)
        for banned in ("단점", "결함", "가려야", "커버"):
            with self.subTest(word=banned):
                self.assertNotIn(banned, text)

    def test_switch_restores_the_previous_behaviour(self):
        """config.ENABLE_AUTO_BODY_SHAPE 로 되돌릴 수 있어야 한다."""
        saved = engine_module.ENABLE_AUTO_BODY_SHAPE
        try:
            engine_module.ENABLE_AUTO_BODY_SHAPE = False
            self.assertEqual(self.picks(SHAPE_INVERTED_TRIANGLE), self.picks(SHAPE_TRIANGLE))
        finally:
            engine_module.ENABLE_AUTO_BODY_SHAPE = saved


class CatalogFocusColumnTests(unittest.TestCase):
    """상품이 밝힌 적합 실루엣이 점수에 들어가는지 본다."""

    @classmethod
    def setUpClass(cls):
        cls.engine = RecommendationEngine(
            ROOT / "FASHION_RULES_MASTER.md",
            ProductCatalog(ROOT / "data" / "products.csv"),
        )

    def garment(self, shapes):
        return {"body_shapes": list(shapes)}

    def test_matching_focus_scores_higher_than_a_mismatch(self):
        match = self.engine._catalog_shape_score(
            self.garment([FOCUS_UPPER]), self.garment(["균형형"]), FOCUS_UPPER)
        mismatch = self.engine._catalog_shape_score(
            self.garment(["균형형"]), self.garment(["균형형"]), FOCUS_UPPER)
        self.assertGreater(match, mismatch)

    def test_a_quiet_counterpart_scores_best(self):
        """시선을 모을 쪽이 맞고 반대쪽이 조용할 때가 가장 좋다."""
        quiet = self.engine._catalog_shape_score(
            self.garment([FOCUS_UPPER]), self.garment(["균형형"]), FOCUS_UPPER)
        noisy = self.engine._catalog_shape_score(
            self.garment([FOCUS_UPPER]), self.garment([FOCUS_LOWER]), FOCUS_UPPER)
        self.assertGreater(quiet, noisy)

    def test_no_focus_means_the_component_is_skipped(self):
        """대응 규칙이 없는 체형(마름모꼴·둥근)에서 억지로 점수를 만들지 않는다."""
        self.assertIsNone(
            self.engine._catalog_shape_score(self.garment([FOCUS_UPPER]), self.garment([]), ""))

    def test_current_outfit_without_the_column_is_skipped(self):
        """지금 입고 있는 옷은 카탈로그 상품이 아니라 이 정보가 없다."""
        self.assertIsNone(
            self.engine._catalog_shape_score(self.garment([]), self.garment([]), FOCUS_UPPER))

    def test_the_two_vocabularies_never_overlap(self):
        """겹치면 어느 축으로 비교하는지 알 수 없어져 규칙이 조용히 잠든다."""
        self.assertEqual(set(ALL_BODY_SHAPES) & set(CATALOG_FOCUS_LABELS), set())


if __name__ == "__main__":
    unittest.main()
