"""상품 CSV가 추천 엔진을 실제로 검증할 만큼 다양한지 지킨다.

쇼핑몰 API로 교체할 때도 같은 기준을 만족해야 규칙이 잠들지 않는다.
"""

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))  # 런타임 모듈은 src/에 있다

from outfit_analyzer import COLOR_PALETTE
from product_catalog import ProductCatalog
from recommendation_engine import PURPOSE_STYLES, STYLE_FORMALITY, RecommendationEngine
from schemas import GOAL_BALANCE, OutfitAnalysis, PoseAnalysis, UserProfile

CATALOG = ProductCatalog(ROOT / "data" / "products.csv")
DEFAULT_BUDGET = UserProfile().budget


def sample_pose(body_shape: str = "사각체형", ratio: float = 1.0) -> PoseAnalysis:
    return PoseAnalysis(True, 0.96, body_shape, ratio, 0.66, 0.44, "정면에 가까움", 0.95)


def sample_outfit() -> OutfitAnalysis:
    return OutfitAnalysis(
        parser_backend="fashn-human-parser",
        upper_color="화이트",
        lower_color="네이비",
        color_harmony="안정적인 무채색 조합",
        detected_items=["top", "pants"],
        style="캐주얼",
        upper_type="셔츠",
        lower_type="청바지",
        fit="레귤러핏",
        lower_fit="와이드핏",
        pattern="무지",
        material="코튼",
        lower_pattern="무지",
        lower_material="데님",
        attribute_confidence=0.8,
    )


class CatalogSchemaTests(unittest.TestCase):
    def test_product_ids_are_unique(self):
        ids = [product.product_id for product in CATALOG.products]
        self.assertEqual(len(ids), len(set(ids)))

    def test_colors_exist_in_the_palette(self):
        """팔레트 밖 색은 색상 조화 계산에서 '보통 조합'으로 뭉개진다."""
        unknown = {p.color for p in CATALOG.products} - set(COLOR_PALETTE)
        self.assertEqual(unknown, set())

    def test_styles_have_a_known_formality(self):
        unknown = {p.style for p in CATALOG.products} - set(STYLE_FORMALITY)
        self.assertEqual(unknown, set())

    def test_purposes_are_known(self):
        unknown = {purpose for p in CATALOG.products for purpose in p.purposes} - set(PURPOSE_STYLES)
        self.assertEqual(unknown, set())

    def test_numeric_columns_stay_in_range(self):
        for product in CATALOG.products:
            for name in ("formality", "warmth", "breathability", "visual_weight", "detail_level"):
                with self.subTest(product=product.product_id, column=name):
                    self.assertIn(getattr(product, name), range(1, 6))

    def test_only_tops_carry_a_neckline(self):
        for product in CATALOG.products:
            with self.subTest(product=product.product_id):
                self.assertEqual(bool(product.neckline), product.category == "top")

    def test_patterned_products_describe_their_scale(self):
        """R-PAT-03은 두 패턴의 크기·대비를 비교하므로 값이 없으면 잠든다."""
        for product in CATALOG.products:
            if product.pattern != "무지":
                with self.subTest(product=product.product_id):
                    self.assertTrue(product.pattern_scale)


class CatalogDiversityTests(unittest.TestCase):
    def values(self, column: str) -> set:
        return {getattr(product, column) for product in CATALOG.products}

    def test_every_purpose_has_stocked_tops_and_bottoms(self):
        for purpose in PURPOSE_STYLES:
            for category in ("top", "bottom"):
                matching = [
                    product
                    for product in CATALOG.available(category)
                    if purpose in product.purposes
                ]
                with self.subTest(purpose=purpose, category=category):
                    self.assertGreaterEqual(len(matching), 3)

    def test_weather_rules_have_something_to_select(self):
        self.assertTrue([p for p in CATALOG.products if p.water_resistant])
        self.assertTrue([p for p in CATALOG.products if p.warmth >= 4])
        self.assertTrue([p for p in CATALOG.products if p.breathability >= 4])
        self.assertTrue([p for p in CATALOG.products if "운동" in p.activity_tags])

    def test_both_patterned_and_plain_products_exist(self):
        patterns = self.values("pattern")
        self.assertIn("무지", patterns)
        self.assertGreaterEqual(len(patterns), 4)

    def test_pattern_clash_rule_can_trigger(self):
        """상·하의가 동시에 패턴인 조합이 있어야 R-PAT-03이 살아난다."""
        patterned_tops = [p for p in CATALOG.available("top") if p.pattern != "무지"]
        patterned_bottoms = [p for p in CATALOG.available("bottom") if p.pattern != "무지"]
        self.assertTrue(patterned_tops)
        self.assertTrue(patterned_bottoms)

    def test_waist_and_length_variety_supports_silhouette_rules(self):
        self.assertIn("하이웨이스트", self.values("waistline"))
        self.assertIn("롱 기장", self.values("length"))  # R-ACC-05
        self.assertIn("크롭 기장", self.values("length"))

    def test_accessory_neckline_branches_are_reachable(self):
        necklines = self.values("neckline")
        self.assertTrue(any("V넥" in value for value in necklines))
        self.assertTrue(any(word in value for value in necklines for word in ("칼라", "터틀", "하이넥")))

    def test_stock_filter_has_something_to_exclude(self):
        self.assertTrue([p for p in CATALOG.products if not p.stock])
        self.assertNotIn(
            False,
            {product.stock for product in CATALOG.available()},
        )


class RecommendationReachabilityTests(unittest.TestCase):
    def setUp(self):
        self.engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", CATALOG)

    def test_every_purpose_returns_recommendations_on_the_default_budget(self):
        """웹 화면의 목적 선택지는 전부 결과가 나와야 한다."""
        for purpose in PURPOSE_STYLES:
            profile = UserProfile(purpose=purpose, budget=DEFAULT_BUDGET)
            with self.subTest(purpose=purpose):
                recommendations = self.engine.recommend(
                    profile, sample_pose(), sample_outfit(), top_k=3
                )
                self.assertEqual(len(recommendations), 3)

    def test_every_season_returns_recommendations(self):
        for season in ("봄", "여름", "가을", "겨울", "사계절"):
            profile = UserProfile(purpose="데일리", budget=DEFAULT_BUDGET, season=season)
            with self.subTest(season=season):
                self.assertTrue(
                    self.engine.recommend(profile, sample_pose(), sample_outfit(), top_k=1)
                )

    def test_weather_inputs_reach_their_rules(self):
        cases = {
            "R-WEA-02": UserProfile(purpose="데일리", budget=250_000, feels_like_c=30.0, humidity=80.0),
            "R-WEA-03": UserProfile(purpose="데일리", budget=250_000, feels_like_c=-3.0),
            "R-WEA-04": UserProfile(purpose="여행", budget=250_000, precipitation_probability=80.0),
        }
        for rule_id, profile in cases.items():
            with self.subTest(rule=rule_id):
                applied = {
                    rule
                    for recommendation in self.engine.recommend(
                        profile, sample_pose(), sample_outfit(), top_k=5
                    )
                    for rule in recommendation.applied_rules
                }
                self.assertIn(rule_id, applied)

    def test_body_shape_rules_reach_their_branches(self):
        cases = {
            "R-BOD-01": sample_pose("삼각체형", 0.85),
            "R-BOD-02": sample_pose("역삼각체형", 1.7),
            "R-BOD-03": sample_pose("사각체형", 1.0),
        }
        profile = UserProfile(purpose="데일리", budget=250_000, silhouette_goal=GOAL_BALANCE)
        for rule_id, pose in cases.items():
            with self.subTest(rule=rule_id):
                applied = {
                    rule
                    for recommendation in self.engine.recommend(profile, pose, sample_outfit(), top_k=5)
                    for rule in recommendation.applied_rules
                }
                self.assertIn(rule_id, applied)


if __name__ == "__main__":
    unittest.main()
