from __future__ import annotations

import colorsys
import hashlib
import itertools
from pathlib import Path
from typing import Callable

from config import ENABLE_AUTO_BODY_SHAPE
from fashion_rules import FashionRuleBook
from outfit_analyzer import COLOR_PALETTE, NEUTRALS, color_harmony
from product_catalog import ProductCatalog
from schemas import (
    CurrentOutfitEvaluation,
    FOCUS_LOWER,
    FOCUS_UPPER,
    SHAPE_HOURGLASS,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_TRIANGLE,
    SHAPE_UNCERTAIN,
    GOAL_BALANCE,
    GOAL_LOWER_FOCUS,
    GOAL_NONE,
    GOAL_UPPER_FOCUS,
    PROPORTION_GOALS,
    OutfitAnalysis,
    PoseAnalysis,
    Product,
    Recommendation,
    UserProfile,
    WardrobeItem,
)


CHANGE_SCOPE_MAP = {
    "현재 유지": [],
    "상의만 변경": ["top"],
    "하의만 변경": ["bottom"],
    "전체 변경": ["top", "bottom"],
}

PURPOSE_STYLES = {
    "데일리": {"캐주얼", "미니멀", "스포티"},
    "데이트": {"로맨틱", "미니멀", "캐주얼", "포멀"},
    "출근": {"포멀", "미니멀"},
    "면접": {"포멀", "미니멀"},
    "결혼식": {"포멀", "로맨틱", "미니멀"},
    "여행": {"캐주얼", "스포티", "스트리트"},
}

PURPOSE_FORMALITY = {
    "데일리": (1, 3),
    "데이트": (2, 4),
    "출근": (3, 5),
    "면접": (4, 5),
    "결혼식": (4, 5),
    "여행": (1, 2),
}

# 상황 적합도는 원하는 미감(desired_style)과 분리해 사용 목적(purpose)만 본다.
# 값은 1차 서비스 가설이며 사용자 평가 데이터로 보정한다.
PURPOSE_TOP_SCORES = {
    "데일리": {"티셔츠": .95, "폴로 셔츠": .85, "셔츠": .85, "블라우스": .80, "니트": .90, "가디건": .85, "후드티": .90, "재킷": .80, "블레이저": .65, "코트": .80, "베스트": .75, "탑": .80},
    "데이트": {"티셔츠": .58, "폴로 셔츠": .55, "셔츠": .90, "블라우스": 1.0, "니트": .86, "가디건": .85, "후드티": .20, "재킷": .86, "블레이저": .90, "코트": .86, "베스트": .70, "탑": .82},
    "출근": {"티셔츠": .35, "폴로 셔츠": .55, "셔츠": .95, "블라우스": .95, "니트": .78, "가디건": .75, "후드티": .10, "재킷": .78, "블레이저": 1.0, "코트": .90, "베스트": .65, "탑": .35},
    "면접": {"티셔츠": .15, "폴로 셔츠": .25, "셔츠": 1.0, "블라우스": .95, "니트": .50, "가디건": .45, "후드티": .03, "재킷": .72, "블레이저": 1.0, "코트": .82, "베스트": .35, "탑": .15},
    "결혼식": {"티셔츠": .20, "폴로 셔츠": .30, "셔츠": .88, "블라우스": .95, "니트": .60, "가디건": .58, "후드티": .03, "재킷": .82, "블레이저": .96, "코트": .88, "베스트": .50, "탑": .55},
    "여행": {"티셔츠": .95, "폴로 셔츠": .82, "셔츠": .65, "블라우스": .55, "니트": .75, "가디건": .75, "후드티": .95, "재킷": .72, "블레이저": .30, "코트": .60, "베스트": .78, "탑": .72},
}

PURPOSE_BOTTOM_SCORES = {
    "데일리": {"팬츠": .85, "슬랙스": .75, "치노 팬츠": .85, "청바지": .95, "카고 팬츠": .85, "조거·스웨트팬츠": .85, "트랙팬츠": .75, "레깅스": .65, "쇼츠": .85, "스커트": .85},
    "데이트": {"팬츠": .75, "슬랙스": .90, "치노 팬츠": .78, "청바지": .72, "카고 팬츠": .45, "조거·스웨트팬츠": .20, "트랙팬츠": .15, "레깅스": .25, "쇼츠": .55, "스커트": .92},
    "출근": {"팬츠": .82, "슬랙스": 1.0, "치노 팬츠": .82, "청바지": .45, "카고 팬츠": .20, "조거·스웨트팬츠": .08, "트랙팬츠": .05, "레깅스": .12, "쇼츠": .15, "스커트": .88},
    "면접": {"팬츠": .75, "슬랙스": 1.0, "치노 팬츠": .68, "청바지": .15, "카고 팬츠": .05, "조거·스웨트팬츠": .02, "트랙팬츠": .02, "레깅스": .05, "쇼츠": .05, "스커트": .85},
    "결혼식": {"팬츠": .72, "슬랙스": .95, "치노 팬츠": .55, "청바지": .15, "카고 팬츠": .05, "조거·스웨트팬츠": .02, "트랙팬츠": .02, "레깅스": .08, "쇼츠": .12, "스커트": .95},
    "여행": {"팬츠": .82, "슬랙스": .38, "치노 팬츠": .72, "청바지": .82, "카고 팬츠": .90, "조거·스웨트팬츠": .95, "트랙팬츠": .92, "레깅스": .88, "쇼츠": .92, "스커트": .45},
}

ITEM_FORMALITY = {
    "티셔츠": 1, "후드티": 1, "탑": 1, "쇼츠": 1,
    "조거·스웨트팬츠": 1, "트랙팬츠": 1, "레깅스": 1,
    "폴로 셔츠": 2, "청바지": 2, "카고 팬츠": 2, "팬츠": 2,
    "니트": 3, "가디건": 3, "치노 팬츠": 3, "스커트": 3,
    "셔츠": 4, "블라우스": 4, "재킷": 4, "슬랙스": 4, "코트": 4,
    "블레이저": 5,
}

STYLE_NEIGHBORS = {
    "캐주얼": {"미니멀", "스포티", "스트리트"},
    "미니멀": {"캐주얼", "포멀"},
    "포멀": {"미니멀", "로맨틱"},
    "스트리트": {"캐주얼", "스포티"},
    "로맨틱": {"미니멀", "포멀"},
    "스포티": {"캐주얼", "스트리트"},
}

DRESS_CODE_FORMALITY = {
    "캐주얼": (1, 2),
    "스마트 캐주얼": (2, 3),
    "비즈니스 캐주얼": (3, 4),
    "포멀": (4, 5),
}

STYLE_FORMALITY = {
    "스포티": 1,
    "스트리트": 1,
    "캐주얼": 2,
    "로맨틱": 3,
    "미니멀": 3,
    "포멀": 5,
}

LARGE_TOP_FITS = {"여유핏", "오버핏", "루즈핏"}
LARGE_BOTTOM_FITS = {"와이드핏", "플레어핏", "여유핏", "오버핏"}
ORDERED_BOTTOM_FITS = {"스트레이트핏", "테이퍼드핏", "슬림핏", "레귤러핏"}
BRIGHT_COLORS = {"화이트", "베이지", "옐로", "핑크", "오렌지"}
DARK_COLORS = {"블랙", "네이비", "브라운", "그레이", "카키", "버건디"}
MAXIMAL_STYLES = {"스트리트", "맥시멀", "페스티벌", "힙합"}
DIAGNOSTIC_PASS_THRESHOLD = 85.0


def _tie_seed(pose: PoseAnalysis) -> str:
    """같은 사진이면 항상 같고, 사람이 다르면 달라지는 문자열을 만든다.

    동점 후보 중 무엇을 고를지 정하는 데만 쓴다. 점수에는 영향을 주지 않는다.
    관절 좌표가 있으면 그것을 쓰고, 없으면(테스트의 합성 포즈 등) 비율로 대신한다.
    """
    landmarks = getattr(pose, "landmarks", None) or {}
    if landmarks:
        return "|".join(
            f"{name}:{values[0]:.4f},{values[1]:.4f}"
            for name, values in sorted(landmarks.items())
        )
    return "|".join(
        f"{value:.4f}"
        for value in (
            pose.shoulder_hip_ratio, pose.upper_lower_ratio,
            pose.leg_ratio, pose.full_body_score,
        )
    )


def _tie_rank(seed: str, products: list) -> str:
    """동점자 사이의 순서. 파이썬의 hash()는 실행마다 달라져 쓸 수 없다."""
    key = seed + "#" + "+".join(product.product_id for product in products)
    return hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()


class NoBudgetMatch(Exception):
    """Raised when no candidate combinations fit the user's min/max budget range."""


class MinGreaterThanMax(Exception):
    """Raised when min_budget > max_budget in profile input."""


class RecommendationEngine:
    """Markdown 규칙을 필터·점수·안전장치·스타일링 안내로 실행한다.

    자연어를 임의로 실행하지 않고 규칙 ID별 Python 계산식을 둔다. 상품
    상세정보는 사진 추정보다 우선하며, 입력이 없는 규칙은 점수에 넣지 않는다.
    """

    SCORING_RULE_IDS = {
        "R-CTX-01", "R-CTX-02", "R-CTX-03", "R-CAT-02",
        "R-SIL-01", "R-SIL-03", "R-SIL-05", "R-SIL-06",
        "R-COL-01", "R-COL-02", "R-COL-03", "R-COL-04", "R-COL-05",
        "R-COL-08", "R-COL-10", "R-COL-11", "R-COL-13",
        "R-PAT-01", "R-PAT-03", "R-PAT-04", "R-MAT-01", "R-MAT-02", "R-CMP-01", "R-CMP-02", "R-CMP-03",
        "R-BOD-01", "R-BOD-02", "R-BOD-03", "R-BOD-04", "R-BOD-05", "R-BOD-06",
        "R-WEA-01", "R-WEA-02", "R-WEA-03", "R-WEA-04",
        "R-OWN-01",
    }
    SAFETY_RULE_IDS = {"R-DAT-01", "R-KOR-01", "R-KOR-02"}
    GUIDANCE_RULE_IDS = {
        "R-CAT-01", "R-PAT-02", "R-ACC-01", "R-ACC-02",
        "R-ACC-04", "R-ACC-05", "R-ACC-06",
    }
    PIPELINE_RULE_IDS = {"R-COL-09", "R-DET-01"}
    EXECUTABLE_RULE_IDS = SCORING_RULE_IDS | SAFETY_RULE_IDS | GUIDANCE_RULE_IDS | PIPELINE_RULE_IDS

    # 실측 사이즈 15점은 아직 계산하지 않는다. 보유 옷 5점은 목록이 있을 때만 활성화한다.
    BASE_WEIGHTS = {
        "purpose_tpo": 0.20,
        "weather_activity": 0.12,
        "silhouette": 0.15,
        "color": 0.15,
        "pattern_material_complexity": 0.10,
        "preference": 0.08,
    }

    # 현재 착장은 후보 상품과 달리 '무난해서 실패하지 않는가'보다 사진에 보이는
    # 스타일링 완성도와 체형-핏 상호작용을 더 강하게 본다. 분석 불가능한 축은
    # evaluate_current_outfit 에서 제외한 뒤 남은 배점만 다시 정규화한다.
    CURRENT_OUTFIT_WEIGHTS = {
        "purpose_tpo": 0.15,
        "silhouette": 0.30,
        "color": 0.15,
        "pattern_material_complexity": 0.10,
        "styling_intent": 0.20,
        "preference": 0.10,
    }

    UNSUPPORTED_RULE_REASONS = {
        "R-SIL-02": "상품별 실측 사이즈와 사용자 신체 치수가 없어 당김·여유량을 판정할 수 없습니다.",
        "R-SIL-04": "이너·아우터 레이어와 밑단 위치 데이터가 없습니다.",
        "R-COL-06": "로고·양말·가방처럼 작은 포인트 영역의 색 데이터가 없습니다.",
        "R-COL-07": "추천 상품 이미지의 아이템별 색 면적 데이터가 없습니다.",
        "R-COL-12": "신발·가방·양말 상품 후보와 색 데이터가 없습니다.",
        "R-ACC-03": "가방과 신발 상품 후보를 아직 함께 생성하지 않습니다.",
        "R-TREND-01": "출처 날짜가 있는 국내 판매·검색 추세 데이터가 없습니다.",
    }

    def __init__(self, rules_path: str | Path, catalog: ProductCatalog) -> None:
        self.rule_book = FashionRuleBook.from_markdown(rules_path)
        self.catalog = catalog

    @property
    def active_rule_ids(self) -> list[str]:
        return sorted(self.EXECUTABLE_RULE_IDS & set(self.rule_book.rules))

    @property
    def scoring_rule_ids(self) -> list[str]:
        return sorted(self.SCORING_RULE_IDS & set(self.rule_book.rules))

    @property
    def documented_rule_ids(self) -> list[str]:
        return self.rule_book.active_rule_ids

    @property
    def unsupported_rule_ids(self) -> list[str]:
        return sorted(set(self.documented_rule_ids) - set(self.active_rule_ids))

    @property
    def rules_source(self) -> Path:
        return self.rule_book.source_path

    @staticmethod
    def _season_matches(product: Product, profile: UserProfile) -> bool:
        if not profile.season or profile.season == "사계절":
            return True
        seasons = {value.strip() for value in product.season.split("|") if value.strip()}
        return "사계절" in seasons or profile.season in seasons

    @staticmethod
    def _contains_any(value: str, needles: list[str]) -> bool:
        normalized = value.replace(" 추정", "")
        return any(needle and needle in normalized for needle in needles)

    @staticmethod
    def _safe_harmony(first: str, second: str) -> str:
        if first not in COLOR_PALETTE or second not in COLOR_PALETTE:
            return "보통 조합"
        return color_harmony(first, second)

    @staticmethod
    def _shrink_to_neutral(score: float, confidence: float) -> float:
        """R-DAT-01: 불확실한 이미지 기반 점수를 중립값 0.5로 수축한다."""
        confidence = max(0.0, min(1.0, confidence))
        return 0.5 + confidence * (score - 0.5)

    def _available_for_profile(self, category: str, profile: UserProfile) -> list[Product]:
        products = self.catalog.available(category)
        # 무신사 카탈로그처럼 성별 정보가 있으면 해당 성별·공용 상품만 남긴다.
        if profile.gender:
            products = [
                product for product in products
                if getattr(product, "gender", "") in ("", "공용", profile.gender)
            ]
        if profile.excluded_item_types:
            products = [
                product for product in products
                if product.category not in profile.excluded_item_types
                and product.item_type not in profile.excluded_item_types
            ]
        if profile.avoided_colors:
            products = [product for product in products if product.color not in profile.avoided_colors]
        if profile.avoided_materials:
            products = [
                product for product in products
                if not self._contains_any(product.material, profile.avoided_materials)
            ]
        if self.rule_book.has("R-CTX-01"):
            products = [product for product in products if profile.purpose in product.purposes]
        if self.rule_book.has("R-MAT-01"):
            products = [product for product in products if self._season_matches(product, profile)]
        return products

    @staticmethod
    def _derived_visual_weight(color: str, fit: str, pattern: str, material: str) -> int:
        weight = 2
        if color in DARK_COLORS:
            weight += 1
        if any(word in fit for word in ("여유", "오버", "루즈", "와이드", "플레어")):
            weight += 1
        if pattern and pattern not in {"무지", "분석 보류", "패턴 불확실"}:
            weight += 1
        if any(word in material for word in ("니트", "데님", "가죽", "울")):
            weight += 1
        return max(1, min(5, weight))

    def _garment(self, product: Product | None, category: str, outfit: OutfitAnalysis) -> dict:
        if product:
            return {
                "category": category,
                "color": product.color,
                "style": product.style,
                "style_confidence": 1.0,
                "purposes": product.purposes,
                "item_type": product.item_type,
                "subtype": product.item_type,
                "fit": product.fit,
                "length": product.length,
                "pattern": product.pattern,
                "material": product.material,
                "neckline": product.neckline,
                "formality": product.formality,
                "activity_tags": product.activity_tags,
                "warmth": product.warmth,
                "breathability": product.breathability,
                "water_resistant": product.water_resistant,
                "visual_weight": product.visual_weight,
                "detail_level": product.detail_level,
                "waistline": product.waistline,
                "pattern_scale": product.pattern_scale,
                "pattern_contrast": product.pattern_contrast,
                # 상품이 밝힌 적합 실루엣. _catalog_shape_score 가 목표와 맞춰 본다.
                "body_shapes": product.body_shapes,
                "palette": [{"name": product.color, "proportion": 1.0}],
                "confidence": 1.0,
                "is_product": True,
            }

        is_top = category == "top"
        color = outfit.upper_color if is_top else outfit.lower_color
        fit = outfit.fit if is_top else outfit.lower_fit
        pattern = outfit.pattern if is_top else outfit.lower_pattern
        material = outfit.material if is_top else outfit.lower_material
        palette = outfit.upper_palette if is_top else outfit.lower_palette
        confidence = max(0.0, min(1.0, outfit.attribute_confidence))
        return {
            "category": category,
            "color": color,
            "style": (
                outfit.upper_style if is_top else outfit.lower_style
            ) if self._usable_analysis_value(
                outfit.upper_style if is_top else outfit.lower_style
            ) else outfit.style,
            "style_confidence": (
                outfit.upper_style_confidence if is_top else outfit.lower_style_confidence
            ) or confidence,
            "purposes": [],
            "item_type": outfit.upper_type if is_top else outfit.lower_type,
            "subtype": outfit.upper_type if is_top else (
                outfit.lower_subtype
                if self._usable_analysis_value(outfit.lower_subtype)
                else outfit.lower_type
            ),
            "fit": fit,
            "length": outfit.upper_length if is_top else outfit.bottom_length,
            "pattern": pattern,
            "material": material,
            "neckline": outfit.neckline if is_top else "",
            "formality": STYLE_FORMALITY.get(outfit.style, 3),
            "activity_tags": [],
            "warmth": 3,
            "breathability": 3,
            "water_resistant": False,
            "visual_weight": self._derived_visual_weight(color, fit, pattern, material),
            "detail_level": 2 if pattern not in {"", "무지", "분석 보류", "패턴 불확실"} else 1,
            # 지금 입고 있는 옷은 카탈로그 상품이 아니라 적합 실루엣 정보가 없다.
            # 빈 값이면 _catalog_shape_score 가 None 을 돌려 점수에서 빠진다.
            "body_shapes": [],
            "waistline": "",
            "pattern_scale": "",
            "pattern_contrast": 0,
            "palette": palette or [{"name": color, "proportion": 1.0}],
            "confidence": confidence,
            "is_product": False,
        }

    @staticmethod
    def _target_formality(profile: UserProfile) -> tuple[int, int]:
        if profile.dress_code != "자동":
            return DRESS_CODE_FORMALITY.get(profile.dress_code, PURPOSE_FORMALITY.get(profile.purpose, (1, 5)))
        return PURPOSE_FORMALITY.get(profile.purpose, (1, 5))

    def _purpose_formality_score(
        self,
        products: list[Product],
        garments: list[dict],
        profile: UserProfile,
    ) -> tuple[float, list[str], list[str]]:
        suitable_styles = PURPOSE_STYLES.get(profile.purpose, set())
        purpose_values = []
        for product in products:
            purpose_match = 1.0 if profile.purpose in product.purposes else 0.0
            style_match = 1.0 if not suitable_styles or product.style in suitable_styles else 0.55
            purpose_values.append(0.72 * purpose_match + 0.28 * style_match)
        purpose_score = sum(purpose_values) / len(purpose_values)

        low, high = self._target_formality(profile)
        formalities = [garment["formality"] for garment in garments]
        formality_values = []
        for value in formalities:
            distance = low - value if value < low else value - high if value > high else 0
            formality_values.append(max(0.35, 1.0 - 0.20 * distance))
        formality_score = sum(formality_values) / len(formality_values)
        if max(formalities) - min(formalities) >= 3 and profile.desired_style not in MAXIMAL_STYLES:
            formality_score = max(0.35, formality_score - 0.15)

        score = 0.58 * purpose_score + 0.42 * formality_score
        reasons = [
            f"{profile.purpose} 목적과 격식도 {low}~{high} 범위를 먼저 비교했습니다."
        ]
        return score, reasons, ["R-CTX-01", "R-CTX-02"]

    def _weather_activity_score(
        self,
        products: list[Product],
        garments: list[dict],
        profile: UserProfile,
    ) -> tuple[float, list[str], list[str]]:
        values = [sum(self._season_matches(product, profile) for product in products) / len(products)]
        reasons = [f"{profile.season} 계절과 상품의 계절 태그를 확인했습니다."]
        rules = ["R-MAT-01", "R-CAT-02"]

        high_activity = profile.activity_level == "높음" or profile.purpose == "여행"
        if high_activity:
            suitable = {"보행", "여행", "운동"}
            activity_values = [1.0 if suitable & set(product.activity_tags) else 0.58 for product in products]
            values.append(sum(activity_values) / len(activity_values))
            reasons.append("보행·여행 활동 태그와 통기성을 활동성 점수에 반영했습니다.")

        weather_values = (
            profile.temperature_c,
            profile.feels_like_c,
            profile.humidity,
            profile.precipitation_probability,
            profile.wind_mps,
            profile.uv_index,
        )
        detailed_weather = any(value is not None for value in weather_values)
        if detailed_weather:
            rules.append("R-WEA-01")

        feels_like = profile.feels_like_c if profile.feels_like_c is not None else profile.temperature_c
        hot_humid = (feels_like is not None and feels_like >= 27) or (profile.humidity is not None and profile.humidity >= 70)
        if hot_humid:
            breathability = sum(garment["breathability"] for garment in garments) / (5 * len(garments))
            loose_bonus = 0.08 if any(self._is_large_fit(garment["fit"], garment["category"]) for garment in garments) else 0.0
            values.append(min(1.0, breathability + loose_bonus))
            rules.append("R-WEA-02")
            reasons.append("덥거나 습한 조건이라 통기성과 몸에 붙지 않는 핏을 우선했습니다.")

        cold_windy = (feels_like is not None and feels_like <= 8) or (profile.wind_mps is not None and profile.wind_mps >= 7)
        if cold_windy:
            warmth = sum(garment["warmth"] for garment in garments) / (5 * len(garments))
            values.append(warmth)
            rules.append("R-WEA-03")
            reasons.append("낮은 체감온도 또는 강한 바람을 고려해 보온성을 비교했습니다.")

        rainy = profile.precipitation_probability is not None and profile.precipitation_probability >= 50
        high_uv = profile.uv_index is not None and profile.uv_index >= 6
        if rainy or high_uv:
            protection = []
            if rainy:
                protection.append(sum(float(garment["water_resistant"]) for garment in garments) / len(garments))
                long_dragging = any(
                    garment["category"] == "bottom" and garment["length"] in {"롱·맥시 기장"}
                    for garment in garments
                )
                if long_dragging:
                    protection[-1] = max(0.0, protection[-1] - 0.20)
            if high_uv:
                protection.append(0.70)
            values.append(sum(protection) / len(protection))
            rules.append("R-WEA-04")
            reasons.append("강수 또는 자외선 조건을 기능성 점수와 추가 준비물에 반영했습니다.")

        score = sum(values) / len(values)
        if any(not garment["is_product"] for garment in garments):
            confidence = min(garment["confidence"] for garment in garments if not garment["is_product"])
            score = self._shrink_to_neutral(score, confidence)
            rules.append("R-DAT-01")
        return score, reasons, list(dict.fromkeys(rules))

    @staticmethod
    def _is_large_fit(fit: str, category: str) -> bool:
        cleaned = fit.replace(" 추정", "")
        known = LARGE_TOP_FITS if category == "top" else LARGE_BOTTOM_FITS
        return cleaned in known or any(word in cleaned for word in ("여유", "오버", "루즈", "와이드", "플레어"))

    @staticmethod
    def _is_ordered_bottom(fit: str) -> bool:
        cleaned = fit.replace(" 추정", "")
        return cleaned in ORDERED_BOTTOM_FITS or any(
            word in cleaned for word in ("스트레이트", "테이퍼드", "슬림", "레귤러")
        )

    @staticmethod
    def _is_bold_color(color: str) -> bool:
        if color in NEUTRALS or color not in COLOR_PALETTE:
            return False
        rgb = COLOR_PALETTE[color]
        saturation = colorsys.rgb_to_hsv(*(value / 255 for value in rgb))[1]
        return saturation >= 0.42

    @staticmethod
    def _is_tight_fit(fit: str) -> bool:
        cleaned = fit.replace(" 추정", "")
        return any(word in cleaned for word in ("슬림", "스키니", "타이트", "바디"))

    @staticmethod
    def _is_short_bottom(length: str) -> bool:
        return any(word in length for word in ("쇼츠", "미니", "반바지", "무릎"))

    @staticmethod
    def _has_upper_structure(top: dict) -> bool:
        structural_types = {"재킷", "블레이저", "셔츠", "블라우스", "코트", "베스트"}
        structural_necklines = ("보트", "스퀘어", "오프숄더", "칼라", "라펠", "세일러")
        return (
            top["item_type"].replace(" 추정", "") in structural_types
            or any(word in top["neckline"] for word in structural_necklines)
            or top["pattern"] not in {"", "무지", "분석 보류", "패턴 불확실"}
            or top["detail_level"] >= 3
            or RecommendationEngine._is_large_fit(top["fit"], "top")
        )

    # 체형이 알려주는 "시선을 나눠 줄 쪽". 상품 카탈로그의 body_shapes 칼럼과 짝이 맞는다.
    # 마름모꼴·둥근체형은 허리가 중심이라 대응하는 R-BOD 규칙이 문서에 아직 없어 비워 둔다.
    BODY_SHAPE_FOCUS = {
        SHAPE_INVERTED_TRIANGLE: FOCUS_LOWER,  # 어깨가 넓으니 하체로 시선을 나눈다
        SHAPE_TRIANGLE: FOCUS_UPPER,           # 골반이 넓으니 상체로 시선을 나눈다
        SHAPE_RECTANGLE: "",                     # 어느 쪽이든 한쪽에 볼륨을 두면 된다
        SHAPE_HOURGLASS: "",
    }

    def _catalog_shape_score(self, top: dict, bottom: dict, focus: str) -> float | None:
        """상품이 스스로 밝힌 body_shapes 가 목표와 맞는지 본다.

        이 칼럼은 지금까지 ProductCatalog 이 읽어 Product 에 넣기만 하고 추천에서는
        아무도 참조하지 않았다. 체형 판정이 화면 표시용으로만 남아 있던 이유 중 하나다.
        """
        if not focus:
            return None
        wanted = top if focus == FOCUS_UPPER else bottom
        other = bottom if focus == FOCUS_UPPER else top
        shapes = wanted.get("body_shapes") or []
        other_shapes = other.get("body_shapes") or []
        if not shapes:
            return None
        if focus in shapes:
            # 시선을 모을 쪽이 맞고, 반대쪽이 조용하면 가장 좋다.
            opposite = FOCUS_LOWER if focus == FOCUS_UPPER else FOCUS_UPPER
            return 1.0 if opposite not in other_shapes else 0.86
        return 0.66

    def _silhouette_score(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
        pose: PoseAnalysis,
    ) -> tuple[float, list[str], list[str]]:
        top_large = self._is_large_fit(top["fit"], "top")
        bottom_large = self._is_large_fit(bottom["fit"], "bottom")
        bottom_ordered = self._is_ordered_bottom(bottom["fit"])
        reasons: list[str] = []
        rules = ["R-SIL-01", "R-SIL-05", "R-KOR-01", "R-KOR-02"]

        if top_large and bottom_large:
            fit_score = 0.84 if profile.desired_style in MAXIMAL_STYLES else 0.48
            reasons.append(
                "선택한 스타일 의도에 맞춰 상·하의의 큰 볼륨을 허용했습니다."
                if profile.desired_style in MAXIMAL_STYLES
                else "상·하의 볼륨이 동시에 커 중심이 약해질 수 있어 실루엣 점수를 낮췄습니다."
            )
        elif (bottom_large and not top_large) or (top_large and bottom_ordered):
            fit_score = 1.0
            reasons.append("한쪽의 볼륨과 다른 쪽의 정돈된 핏을 연결했습니다.")
        else:
            fit_score = 0.84

        top_weight, bottom_weight = top["visual_weight"], bottom["visual_weight"]
        if top_weight >= 4 and bottom_weight >= 4 and profile.desired_style not in MAXIMAL_STYLES:
            weight_score = 0.55
            reasons.append("어두운색·소재·핏을 합친 시각적 무게가 양쪽에 몰렸습니다.")
        elif abs(top_weight - bottom_weight) in {1, 2}:
            weight_score = 0.95
        else:
            weight_score = 0.84

        components = [(fit_score, 0.55), (weight_score, 0.45)]
        goal = profile.silhouette_goal
        body_confident = pose.valid and pose.body_shape_confidence >= 0.65 and pose.body_shape != SHAPE_UNCERTAIN

        # 사용자가 목표를 고르지 않았어도 분석된 체형이 뚜렷하면 균형 목표로 본다.
        # R-KOR-02 는 원래 "목표를 고른 경우에만" 이었지만, 그 기본값에서는 체형을
        # 판정해 놓고 추천에 전혀 쓰지 않아 분석이 표시용으로만 남았다. 대신
        # 가중치를 낮게 두고, 신뢰도가 낮으면 적용하지 않는다.
        auto_body = False
        if (goal == GOAL_NONE and body_confident and ENABLE_AUTO_BODY_SHAPE
                and pose.body_shape in self.BODY_SHAPE_FOCUS):
            goal = GOAL_BALANCE
            auto_body = True
            rules.append("R-KOR-02")
            reasons.append(
                f"고른 목표가 없어 분석한 체형({pose.body_shape})을 참고해 "
                "상·하의 볼륨 배치를 비교했습니다."
            )

        if goal in PROPORTION_GOALS:
            rules.extend(["R-SIL-03", "R-SIL-06"])
            length_score = {"크롭 기장": 1.0, "기본 기장": 0.78, "롱 기장": 0.45}.get(top["length"], 0.65)
            if bottom["waistline"] == "하이웨이스트":
                length_score = min(1.0, length_score + 0.12)
            components.append((length_score, 0.35))
            reasons.append("상의 기장과 하의 허리선이 만드는 분할점을 목표 실루엣에 맞춰 평가했습니다.")

        if goal == GOAL_BALANCE and body_confident:
            body_score = 0.78
            body_component_weight = 0.18 if auto_body else 0.30
            if pose.body_shape == SHAPE_INVERTED_TRIANGLE:
                rules.append("R-BOD-02")
                body_score = 1.0 if bottom_large or bottom_ordered else 0.62
                reasons.append("상·하체 균형 목표에 맞춰 하의의 구조와 볼륨을 비교했습니다.")
            elif pose.body_shape == SHAPE_TRIANGLE:
                rules.append("R-BOD-01")
                top_focus = top["color"] in BRIGHT_COLORS or top["pattern"] not in {"", "무지", "패턴 불확실", "분석 보류"}
                bottom_quiet = bottom["color"] in DARK_COLORS and bottom["pattern"] in {"", "무지"}
                tight_top = self._is_tight_fit(top["fit"])
                upper_structure = self._has_upper_structure(top)
                if tight_top and not upper_structure:
                    body_score = 0.38
                    # 명확한 체형-핏 충돌이 일반 볼륨 점수에 희석되지 않게 한다.
                    body_component_weight = 0.48 if auto_body else 0.55
                    rules.append("R-BOD-06")
                    reasons.append(
                        "어깨가 상대적으로 좁게 분석된 경우 몸에 붙는 상의에 "
                        "어깨 구조·레이어·넥라인 포인트가 있는지 함께 확인했습니다."
                    )
                elif tight_top:
                    body_score = 0.68
                    body_component_weight = 0.32 if auto_body else 0.42
                    rules.append("R-BOD-06")
                    reasons.append("슬림한 상의의 어깨 구조와 상체 포인트가 균형을 만드는지 확인했습니다.")
                else:
                    body_score = 1.0 if (top_focus or upper_structure) and bottom_quiet else 0.66
                    reasons.append("상·하체 균형 목표에 맞춰 상체로 시선을 옮기는 색·패턴 배치를 비교했습니다.")
            elif pose.body_shape in {SHAPE_RECTANGLE, SHAPE_HOURGLASS}:
                # 모래시계체형도 어깨와 엉덩이가 비슷하므로 같은 규칙을 쓴다.
                # 마름모꼴·둥근체형은 허리가 중심이라 대응하는 규칙이 문서에 아직 없다.
                rules.append("R-BOD-03")
                body_score = 0.96 if top_large != bottom_large else 0.80
                reasons.append("상·하체 폭이 비슷한 경우 볼륨을 한쪽씩 배치했는지 확인했습니다.")
            components.append((self._shrink_to_neutral(body_score, pose.body_shape_confidence),
                               body_component_weight))
        elif goal in {GOAL_UPPER_FOCUS, GOAL_LOWER_FOCUS}:
            rules.append("R-BOD-04")
            target = top if goal == GOAL_UPPER_FOCUS else bottom
            other = bottom if goal == GOAL_UPPER_FOCUS else top
            target_focus = self._is_bold_color(target["color"]) or target["pattern"] not in {"", "무지", "분석 보류", "패턴 불확실"}
            other_quiet = other["color"] in NEUTRALS and other["pattern"] in {"", "무지"}
            components.append((1.0 if target_focus and other_quiet else 0.62, 0.30))
            reasons.append(f"사용자가 선택한 '{goal}' 위치에 색이나 패턴의 시선 중심이 생기는지 확인했습니다.")

        # leg_ratio 는 이미 자세 분석에서 계산하지만 이전에는 채점에 전혀 쓰지 않았다.
        # 절대 체형 라벨로 단정하지 않고 0.60 아래에서만 연속적인 보완 필요도로 사용한다.
        if pose.valid and 0.40 <= pose.leg_ratio < 0.60:
            short_leg_evidence = min(1.0, (0.60 - pose.leg_ratio) / 0.10)
            evidence = short_leg_evidence * max(0.0, min(1.0, pose.body_shape_confidence))
            tight_bottom = self._is_tight_fit(bottom["fit"])
            short_bottom = self._is_short_bottom(bottom["length"])
            long_top = "롱" in top["length"]
            cropped_top = "크롭" in top["length"]
            long_bottom = any(word in bottom["length"] for word in ("긴바지", "롱", "풀렝스", "맥시"))

            leg_score = 0.82
            if cropped_top:
                leg_score += 0.14
            if long_bottom and not tight_bottom:
                leg_score += 0.10
            if long_top:
                leg_score -= 0.22
            if tight_bottom:
                leg_score -= 0.18
            if short_bottom:
                leg_score -= 0.16
            if tight_bottom and short_bottom:
                leg_score -= 0.18
            leg_score = max(0.25, min(1.0, leg_score))
            # 비율이 경계에 가깝거나 분석 신뢰도가 낮으면 일반적인 0.82점에 가까워진다.
            leg_score = 0.82 + evidence * (leg_score - 0.82)
            components.append((leg_score, 0.45))
            rules.append("R-BOD-05")
            reasons.append(
                "사진에서 추정한 하체 비율과 상의 밑단·하의 기장·밀착 핏이 "
                "세로선을 이어 주는지 함께 평가했습니다."
            )

        # 상품이 스스로 밝힌 body_shapes 를 목표와 맞춰 본다.
        # 시선을 모을 쪽은 목표에서 정하고, 목표가 균형이면 체형이 정한다.
        focus = ""
        if goal == GOAL_UPPER_FOCUS:
            focus = FOCUS_UPPER
        elif goal == GOAL_LOWER_FOCUS:
            focus = FOCUS_LOWER
        elif goal == GOAL_BALANCE and body_confident:
            focus = self.BODY_SHAPE_FOCUS.get(pose.body_shape, "")
        catalog_score = self._catalog_shape_score(top, bottom, focus)
        if catalog_score is not None:
            rules.append("R-BOD-04")
            components.append((catalog_score, 0.16 if auto_body else 0.22))
            reasons.append(
                f"상품이 밝힌 적합 실루엣('{focus}')이 목표와 맞는지 비교했습니다."
            )

        total_weight = sum(weight for _, weight in components)
        score = sum(value * weight for value, weight in components) / total_weight
        current_confidences = [item["confidence"] for item in (top, bottom) if not item["is_product"]]
        if current_confidences:
            score = self._shrink_to_neutral(score, min(current_confidences))
            rules.append("R-DAT-01")
        return score, reasons, list(dict.fromkeys(rules))

    def _color_score(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
    ) -> tuple[float, list[str], list[str]]:
        first, second = top["color"], bottom["color"]
        harmony = self._safe_harmony(first, second)
        score = {
            "안정적인 무채색 조합": 0.94,
            "톤온톤": 0.88,
            "유사색 조합": 0.90,
            "대비색 조합": 0.74,
            "보통 조합": 0.68,
        }[harmony]
        rules = ["R-COL-03", "R-COL-13"]
        reasons = [f"상의와 하의의 색상 관계는 '{harmony}'입니다."]

        if (first in NEUTRALS) != (second in NEUTRALS):
            score = max(score, 0.96)
            rules.append("R-COL-01")
            reasons.append("한쪽의 색을 무채색·기본 연결색으로 받쳤습니다.")

        same_color = first == second
        materials_differ = top["material"] and bottom["material"] and top["material"] != bottom["material"]
        if same_color:
            rules.extend(["R-COL-02", "R-COL-11"])
            if materials_differ:
                score = max(score, 0.97)
                reasons.append("같은 색 계열에 서로 다른 소재를 배치해 시각적 깊이를 만들었습니다.")
            elif top["material"] == bottom["material"]:
                score = min(score, 0.80)

        denim_present = any("데님" in garment["material"] or "청바지" in garment["item_type"] for garment in (top, bottom))
        if denim_present:
            rules.append("R-COL-05")
            score = max(score, 0.86)

        bold_count = sum(self._is_bold_color(garment["color"]) for garment in (top, bottom))
        rules.append("R-COL-04")
        if bold_count == 2 and profile.desired_style not in MAXIMAL_STYLES:
            score = max(0.0, score - 0.12)
            reasons.append("강한 색 두 개가 경쟁할 수 있어 포인트 위계 점수를 낮췄습니다.")

        major_colors = {
            entry.get("name")
            for garment in (top, bottom)
            for entry in garment["palette"]
            if entry.get("name") and float(entry.get("proportion", 0)) >= 0.15
        }
        rules.append("R-COL-10")
        if len(major_colors) > 3 and profile.desired_style not in MAXIMAL_STYLES:
            score = max(0.0, score - min(0.18, 0.05 * (len(major_colors) - 3)))
            reasons.append("주요 색이 많아 목적과 스타일에 따른 복잡도 조정을 적용했습니다.")

        if profile.preferred_colors or profile.owned_items:
            rules.append("R-COL-08")
            candidate_colors = {first, second}
            if candidate_colors & set(profile.preferred_colors):
                score = min(1.0, score + 0.05)

        current_items = [item for item in (top, bottom) if not item["is_product"]]
        if current_items:
            rules.extend(["R-COL-09", "R-DAT-01"])
            confidence = max(0.60, min(item["confidence"] for item in current_items))
            score = self._shrink_to_neutral(score, confidence)
        return score, reasons, list(dict.fromkeys(rules))

    def _pattern_material_complexity_score(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
    ) -> tuple[float, list[str], list[str]]:
        unknown = {"", "분석 보류", "패턴 불확실", "분석 불가"}
        patterned = [garment for garment in (top, bottom) if garment["pattern"] not in unknown | {"무지"}]
        rules = ["R-PAT-01", "R-PAT-04", "R-MAT-02", "R-CMP-01"]
        reasons: list[str] = []

        if len(patterned) == 0:
            pattern_score = 0.84
        elif len(patterned) == 1:
            pattern_score = 1.0
            reasons.append("강한 패턴을 한 영역에만 배치해 시선의 중심을 정했습니다.")
        elif profile.desired_style in MAXIMAL_STYLES:
            pattern_score = 0.88
            reasons.append("스트리트·맥시멀 스타일 의도를 반영해 패턴 혼합 감점을 완화했습니다.")
        else:
            scales = {garment["pattern_scale"] for garment in patterned if garment["pattern_scale"]}
            contrasts = [garment["pattern_contrast"] for garment in patterned if garment["pattern_contrast"]]
            if scales or contrasts:
                rules.append("R-PAT-03")
            pattern_score = 0.76 if len(scales) >= 2 else 0.52
            reasons.append("두 패턴의 크기·대비 차이가 부족하면 충돌 가능성이 있어 감점했습니다.")

        same_color = top["color"] == bottom["color"]
        material_known = all(garment["material"] not in {"", "분석 보류", "소재 불확실"} for garment in (top, bottom))
        if same_color and material_known and top["material"] != bottom["material"]:
            material_score = 1.0
            reasons.append("같은 색 안에서 소재 차이로 상·하의 경계를 만들었습니다.")
        elif same_color and material_known and top["material"] == bottom["material"]:
            material_score = 0.72
        else:
            material_score = 0.86

        point_count = sum(self._is_bold_color(item["color"]) for item in (top, bottom))
        point_count += len(patterned)
        point_count += sum(item["detail_level"] >= 4 for item in (top, bottom))
        if profile.desired_style in MAXIMAL_STYLES:
            ideal = {2, 3, 4}
        elif profile.desired_style in {"미니멀", "포멀"}:
            ideal = {0, 1}
        else:
            ideal = {1, 2}
        complexity_score = 1.0 if point_count in ideal else max(0.55, 1.0 - 0.15 * min(abs(point_count - value) for value in ideal))
        if point_count >= 3 and profile.desired_style not in MAXIMAL_STYLES:
            reasons.append("색·패턴·디테일 포인트가 동시에 경쟁해 복잡도 점수를 조정했습니다.")

        score = 0.50 * pattern_score + 0.30 * material_score + 0.20 * complexity_score
        current_items = [item for item in (top, bottom) if not item["is_product"]]
        if current_items:
            score = self._shrink_to_neutral(score, min(item["confidence"] for item in current_items))
            rules.append("R-DAT-01")
        return score, reasons, list(dict.fromkeys(rules))

    @staticmethod
    def _preference_score(products: list[Product], profile: UserProfile) -> float:
        preferred_for_purpose = PURPOSE_STYLES.get(profile.purpose, set())
        values = []
        for product in products:
            if product.style == profile.desired_style:
                style_score = 1.0
            elif product.style in preferred_for_purpose:
                style_score = 0.78
            else:
                style_score = 0.55
            components = [(style_score, 0.65)]
            if profile.preferred_colors:
                color_score = 1.0 if product.color in profile.preferred_colors else 0.70
                components.append((color_score, 0.20))
            if profile.preferred_materials:
                material_score = 1.0 if any(
                    value and value in product.material for value in profile.preferred_materials
                ) else 0.70
                components.append((material_score, 0.15))
            total_weight = sum(weight for _, weight in components)
            values.append(sum(score * weight for score, weight in components) / total_weight)
        return sum(values) / len(values)

    def _wardrobe_score(self, products: list[Product], profile: UserProfile) -> float | None:
        if not profile.owned_items:
            return None
        values = []
        for product in products:
            complements = [
                item for item in profile.owned_items
                if item.category != product.category and self._wardrobe_season_matches(item, profile)
            ]
            if not complements:
                values.append(0.50)
                continue
            compatibilities = []
            for item in complements:
                harmony = self._safe_harmony(product.color, item.color)
                color_value = {
                    "안정적인 무채색 조합": 1.0,
                    "톤온톤": 0.95,
                    "유사색 조합": 0.90,
                    "대비색 조합": 0.72,
                    "보통 조합": 0.65,
                }[harmony]
                style_value = 1.0 if not item.style or item.style == product.style else 0.72
                compatibilities.append(0.75 * color_value + 0.25 * style_value)
            values.append(sum(value >= 0.78 for value in compatibilities) / len(compatibilities))
        return sum(values) / len(values)

    @staticmethod
    def _wardrobe_season_matches(item: WardrobeItem, profile: UserProfile) -> bool:
        if not profile.season or profile.season == "사계절":
            return True
        seasons = {value.strip() for value in item.season.split("|") if value.strip()}
        return "사계절" in seasons or profile.season in seasons

    def _weights_for_profile(self, profile: UserProfile, include_wardrobe: bool) -> dict[str, float]:
        weights = dict(self.BASE_WEIGHTS)
        if profile.purpose in {"출근", "면접"}:
            weights.update(purpose_tpo=0.23, color=0.14, pattern_material_complexity=0.09, preference=0.07)
        elif profile.purpose == "여행":
            weights.update(purpose_tpo=0.19, weather_activity=0.16, color=0.13, pattern_material_complexity=0.09)
        elif profile.purpose == "데이트":
            weights.update(purpose_tpo=0.18, weather_activity=0.10, silhouette=0.17, preference=0.10)
        if include_wardrobe:
            weights["wardrobe"] = 0.05
        return weights

    def _styling_guidance(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
    ) -> tuple[list[str], list[str]]:
        tips: list[str] = []
        rules = ["R-CAT-01", "R-ACC-06"]

        if profile.purpose in {"출근", "면접", "결혼식"}:
            tips.append("신발 슬롯은 정돈된 로퍼·부츠·구두부터 확인하세요.")
        elif profile.purpose == "여행" or profile.activity_level == "높음":
            tips.append("신발 슬롯은 장시간 보행이 가능한 쿠셔닝과 안정성을 먼저 확인하세요.")
        elif self._is_large_fit(bottom["fit"], "bottom") and bottom["length"] in {"긴바지", "롱·맥시 기장"}:
            tips.append("넓고 긴 하의의 밑단을 받쳐 줄 적당한 볼륨의 신발을 연결해보세요.")
        else:
            tips.append("상의·하의와 격식이 맞는 신발을 마지막 필수 슬롯으로 확인하세요.")

        strong_detail = any(
            garment["pattern"] not in {"", "무지", "분석 보류", "패턴 불확실"}
            or garment["detail_level"] >= 4
            for garment in (top, bottom)
        )
        simple_outfit = all(
            garment["pattern"] in {"", "무지"} and garment["detail_level"] <= 1
            for garment in (top, bottom)
        )
        if strong_detail:
            tips.append("옷 자체가 포인트이므로 액세서리는 작거나 단색인 한 가지부터 시도하세요.")
            rules.extend(["R-PAT-02", "R-ACC-01"])
        elif simple_outfit and profile.desired_style not in {"미니멀", "포멀"}:
            tips.append("기본 착장에는 컬러 가방·신발·스카프 중 하나만 포인트로 더할 수 있어요.")
            rules.extend(["R-ACC-01", "R-ACC-02"])

        neckline = top["neckline"]
        if neckline and "불확실" not in neckline and "보류" not in neckline:
            if "V넥" in neckline:
                tips.append("V넥의 세로선을 따라가는 작은 펜던트형 목걸이가 자연스럽습니다.")
            elif any(word in neckline for word in ("칼라", "터틀", "하이넥")):
                tips.append("넥라인 구조가 있으므로 목걸이는 생략하거나 작은 형태를 우선하세요.")
            else:
                tips.append("넥라인의 빈 공간을 넘지 않는 크기의 목걸이를 우선하세요.")
            rules.append("R-ACC-04")

        if profile.silhouette_goal in PROPORTION_GOALS and top["length"] == "롱 기장":
            tips.append("긴 상의를 유지한다면 턱인이나 얇은 벨트로 허리 기준점을 만들 수 있어요.")
            rules.append("R-ACC-05")

        if profile.feels_like_c is not None and profile.feels_like_c <= 8:
            tips.append("외출 시간이 길다면 목도리·모자처럼 노출 부위를 줄이는 방한 아이템도 확인하세요.")
            rules.append("R-WEA-03")
        if profile.precipitation_probability is not None and profile.precipitation_probability >= 50:
            tips.append("비 예보가 있어 물에 강한 신발과 바닥에 끌리지 않는 밑단을 확인하세요.")
            rules.append("R-WEA-04")
        if profile.uv_index is not None and profile.uv_index >= 6:
            tips.append("자외선이 높아 모자·선글라스·가벼운 긴소매를 선택지로 둘 수 있어요.")
            rules.append("R-WEA-04")
        return tips, list(dict.fromkeys(rules))

    @staticmethod
    def _usable_analysis_value(value: str) -> bool:
        blocked = ("분석 보류", "분석 불가", "불확실", "해당 없음", "판단 보류")
        return bool(value and not any(marker in value for marker in blocked))

    def _current_purpose_formality_score(
        self,
        garments: list[dict],
        profile: UserProfile,
        outfit: OutfitAnalysis,
    ) -> tuple[float, list[str], list[str]]:
        suitable_styles = PURPOSE_STYLES.get(profile.purpose, set())
        if outfit.style == profile.desired_style:
            style_score = 1.0
        elif not suitable_styles or outfit.style in suitable_styles:
            style_score = 0.88
        else:
            style_score = 0.55

        low, high = self._target_formality(profile)
        formality_values = []
        for garment in garments:
            value = garment["formality"]
            distance = low - value if value < low else value - high if value > high else 0
            formality_values.append(max(0.35, 1.0 - 0.20 * distance))
        formality_score = sum(formality_values) / len(formality_values)
        return (
            0.58 * style_score + 0.42 * formality_score,
            [f"현재 착장의 {outfit.style} 스타일과 {profile.purpose} 목적의 격식도를 비교했습니다."],
            ["R-CTX-01", "R-CTX-02"],
        )

    def _current_preference_score(self, outfit: OutfitAnalysis, profile: UserProfile) -> float:
        preferred_for_purpose = PURPOSE_STYLES.get(profile.purpose, set())
        if outfit.style == profile.desired_style:
            style_score = 1.0
        elif outfit.style in preferred_for_purpose:
            style_score = 0.78
        else:
            style_score = 0.55

        colors = {outfit.upper_color, outfit.lower_color}
        if profile.preferred_colors:
            color_score = 1.0 if colors & set(profile.preferred_colors) else 0.65
            score = 0.78 * style_score + 0.22 * color_score
        else:
            score = style_score
        if colors & set(profile.avoided_colors):
            score -= 0.25
        materials = f"{outfit.material}|{outfit.lower_material}"
        if profile.preferred_materials:
            material_score = 1.0 if any(
                value and value in materials for value in profile.preferred_materials
            ) else 0.70
            score = 0.85 * score + 0.15 * material_score
        if any(value and value in materials for value in profile.avoided_materials):
            score -= 0.20
        return max(0.0, min(1.0, score))

    def _current_styling_intent_score(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
        outfit: OutfitAnalysis,
    ) -> tuple[float, list[str], list[str]]:
        """안전한 기본 조합과 의도가 읽히는 스타일링을 구분한다.

        무채색·무지 자체를 나쁘게 보지 않는다. 대신 비율, 소재 깊이, 구조,
        레이어 또는 명확한 포인트가 하나도 없을 때 자동으로 최고점이 되지 않게 한다.
        """
        score = 0.48
        signals: list[str] = []
        top_large = self._is_large_fit(top["fit"], "top")
        bottom_large = self._is_large_fit(bottom["fit"], "bottom")
        patterned = [
            garment for garment in (top, bottom)
            if garment["pattern"] not in {"", "무지", "분석 보류", "패턴 불확실"}
        ]

        if top_large != bottom_large or "크롭" in top["length"]:
            score += 0.16
            signals.append("상·하의 비율과 볼륨의 중심")
        if len(patterned) == 1:
            score += 0.16
            signals.append("한 영역에 정리된 패턴 포인트")
        bold_count = sum(self._is_bold_color(item["color"]) for item in (top, bottom))
        if bold_count == 1:
            score += 0.12
            signals.append("한 영역에 정리된 색상 포인트")
        elif top["color"] != bottom["color"] and top["color"] in NEUTRALS and bottom["color"] in NEUTRALS:
            # 무채색 조합은 안정적이지만 그 사실만으로 높은 완성도 점수를 주지는 않는다.
            score += 0.04
        if top["material"] and bottom["material"] and top["material"] != bottom["material"]:
            score += 0.10
            signals.append("상·하의 소재의 시각적 깊이")
        if self._has_upper_structure(top):
            score += 0.08
            signals.append("넥라인·어깨의 구조")
        if outfit.layering_state in {"레이어드", "레이어드 가능성"}:
            score += 0.12
            signals.append("의도적인 레이어 구조")

        score = min(1.0, score)
        if not signals:
            reasons = [
                "색과 핏의 충돌은 적지만 비율·소재·구조·포인트에서 "
                "뚜렷한 스타일링 의도가 확인되지 않아 기본 조합과 구분했습니다."
            ]
        else:
            reasons = ["스타일링 의도는 " + ", ".join(signals) + "에서 확인했습니다."]

        # 미니멀은 강한 색·패턴이 없어도 소재와 선이 분명하면 충분히 높은 점수를 받는다.
        if profile.desired_style in {"미니멀", "포멀"} and not patterned and bold_count == 0:
            reasons.append("미니멀·포멀 의도에서는 강한 색이나 패턴의 부재를 별도로 감점하지 않았습니다.")
        confidence = min(top["confidence"], bottom["confidence"])
        return self._shrink_to_neutral(score, confidence), reasons, ["R-CMP-02"]

    @staticmethod
    def _situation_item_key(item: dict) -> str:
        """상황 점수표의 세부 카테고리 어휘로 정규화한다."""
        values = [item.get("subtype", ""), item.get("item_type", "")]
        known = set(ITEM_FORMALITY)
        for value in values:
            cleaned = value.replace(" 추정", "")
            if cleaned in known:
                return cleaned
            for key in sorted(known, key=len, reverse=True):
                if key in cleaned:
                    return key
        return values[0] or values[1]

    def _item_situation_fit(
        self, item: dict, profile: UserProfile
    ) -> tuple[float, list[str], list[str]]:
        key = self._situation_item_key(item)
        table = PURPOSE_TOP_SCORES if item["category"] == "top" else PURPOSE_BOTTOM_SCORES
        base = table.get(profile.purpose, {}).get(key, 0.50)
        # 카탈로그 목적 태그는 보조 증거다. 아이템 종류상 강한 부적합을 태그 하나로
        # 뒤집지 않도록 최대 0.08만 올린다.
        purpose_score = min(1.0, base + (0.08 if profile.purpose in item.get("purposes", []) else 0.0))

        low, high = self._target_formality(profile)
        formality = item["formality"] if item["is_product"] else ITEM_FORMALITY.get(key, item["formality"])
        distance = low - formality if formality < low else formality - high if formality > high else 0
        formality_score = max(0.15, 1.0 - 0.30 * distance)

        if profile.purpose == "여행" or profile.activity_level == "높음":
            suitable = {"보행", "여행", "운동"}
            if item.get("activity_tags"):
                activity_score = 1.0 if suitable & set(item["activity_tags"]) else 0.35
            else:
                activity_score = 0.65
            score = 0.55 * purpose_score + 0.30 * formality_score + 0.15 * activity_score
        else:
            score = 0.65 * purpose_score + 0.35 * formality_score

        confidence = item["confidence"] if not item["is_product"] else 1.0
        score = self._shrink_to_neutral(score, confidence)
        reason = (
            f"{profile.purpose} 상황에서 {key or '해당 아이템'}의 종류와 "
            f"격식도 {formality}가 목적 범위 {low}~{high}에 맞는지 평가했습니다."
        )
        return score, [reason], ["R-CTX-01", "R-CTX-02", "R-CTX-03"]

    def _item_style_fit(
        self, item: dict, profile: UserProfile
    ) -> tuple[float, list[str], list[str]]:
        observed = item.get("style", "")
        if not self._usable_analysis_value(observed):
            return 0.50, ["아이템별 스타일 분석 신뢰도가 낮아 중립 점수를 적용했습니다."], ["R-DAT-01"]
        if observed == profile.desired_style:
            raw = 1.0
        elif observed in STYLE_NEIGHBORS.get(profile.desired_style, set()):
            raw = 0.72
        else:
            raw = 0.38
        confidence = item.get("style_confidence", item["confidence"])
        score = raw if item["is_product"] else self._shrink_to_neutral(raw, confidence)
        return (
            score,
            [f"아이템의 {observed} 스타일과 원하는 {profile.desired_style} 스타일을 비교했습니다."],
            ["R-CMP-02", "R-DAT-01"] if not item["is_product"] else ["R-CMP-02"],
        )

    def _item_body_fit(
        self,
        item: dict,
        counterpart: dict,
        profile: UserProfile,
        pose: PoseAnalysis,
    ) -> tuple[float, list[str], list[str]]:
        """체형 자체가 아니라 해당 아이템이 목표 비율을 만드는 정도를 평가한다."""
        score = 0.90
        rules = ["R-KOR-01", "R-KOR-02"]
        reasons: list[str] = []
        tight = self._is_tight_fit(item["fit"])
        body_confident = (
            pose.valid
            and pose.body_shape_confidence >= 0.65
            and pose.body_shape != SHAPE_UNCERTAIN
        )
        use_body_shape = body_confident and (
            ENABLE_AUTO_BODY_SHAPE or profile.silhouette_goal != GOAL_NONE
        )

        if item["category"] == "top":
            if use_body_shape and pose.body_shape == SHAPE_TRIANGLE:
                reasons.append(f"분석한 체형({pose.body_shape})을 상의 구조 평가에 참고했습니다.")
                if tight and not self._has_upper_structure(item):
                    score = 0.38
                    rules.append("R-BOD-06")
                    reasons.append("상대적으로 좁은 어깨에 밀착 상의와 구조 요소가 함께 있는지 확인했습니다.")
                elif self._has_upper_structure(item):
                    score = 0.95
                    rules.append("R-BOD-06")
            if pose.valid and 0.40 <= pose.leg_ratio < 0.60:
                if "롱" in item["length"]:
                    score -= 0.24
                elif "크롭" in item["length"]:
                    score += 0.12
                rules.append("R-BOD-05")
                reasons.append("상의 밑단이 사진에서 추정한 하체 비율을 어떻게 나누는지 확인했습니다.")
        else:
            large = self._is_large_fit(item["fit"], "bottom")
            ordered = self._is_ordered_bottom(item["fit"])
            if use_body_shape and pose.body_shape == SHAPE_INVERTED_TRIANGLE:
                reasons.append(f"분석한 체형({pose.body_shape})을 하의 볼륨 평가에 참고했습니다.")
                score = 0.95 if large or ordered else 0.58
                rules.append("R-BOD-02")
            if pose.valid and 0.40 <= pose.leg_ratio < 0.60:
                short = self._is_short_bottom(item["length"])
                long_line = any(word in item["length"] for word in ("긴바지", "롱", "풀렝스", "맥시"))
                if tight and short:
                    score = min(score, 0.25)
                elif tight:
                    score = min(score, 0.50)
                elif long_line and (large or ordered):
                    score = max(score, 0.96)
                elif short:
                    score = min(score, 0.60)
                rules.append("R-BOD-05")
                reasons.append("하의의 기장·밀착도·세로선이 하체 비율을 보완하는지 확인했습니다.")

        if not reasons:
            reasons.append("아이템의 핏·기장·구조가 분석된 체형과 목표 비율에 맞는지 평가했습니다.")
        confidence = min(item["confidence"], max(pose.full_body_score, pose.body_shape_confidence))
        return self._shrink_to_neutral(max(0.0, min(1.0, score)), confidence), reasons, list(dict.fromkeys(rules))

    def _outfit_harmony_score(
        self, top: dict, bottom: dict, profile: UserProfile,
    ) -> tuple[float, dict[str, float], list[str], list[str]]:
        """관찰 가능한 네 관계로 상·하의 조화를 평가한다(R-CMP-03)."""
        reasons: list[str] = []
        rules = ["R-CMP-03"]

        top_large = self._is_large_fit(top["fit"], "top")
        bottom_large = self._is_large_fit(bottom["fit"], "bottom")
        bottom_ordered = self._is_ordered_bottom(bottom["fit"])
        top_long = any(word in top["length"] for word in ("롱", "긴 기장", "장기장"))
        bottom_long = any(word in bottom["length"] for word in ("롱", "긴바지", "풀렝스", "맥시", "장기장"))
        if top_large and top_long and bottom_large and bottom_long:
            silhouette = 0.96
            reasons.append("롱 오버핏 상의와 롱 와이드 하의의 연속된 볼륨을 의도적인 트렌드 실루엣으로 평가했습니다.")
        elif top_large and bottom_ordered:
            silhouette = 0.72
            reasons.append("오버핏 상의와 스트레이트 계열 하의는 안정적이지만 무난한 기본 조합이라 평균권으로 평가했습니다.")
        elif bottom_large and not top_large:
            silhouette = 0.90
            reasons.append("정돈된 상의와 볼륨 하의가 선명한 실루엣 대비를 만듭니다.")
        elif top_large and bottom_large:
            silhouette = 0.88
            reasons.append("상·하의의 큰 볼륨이 하나의 의도된 실루엣으로 이어집니다.")
        else:
            silhouette = 0.80
            reasons.append("상·하의의 기본 볼륨 관계가 크게 충돌하지 않습니다.")
        rules.append("R-SIL-01")

        formality_gap = abs(top["formality"] - bottom["formality"])
        formality = {0: 1.00, 1: 0.90, 2: 0.72}.get(formality_gap, 0.40)
        if profile.desired_style in MAXIMAL_STYLES:
            formality = max(formality, 0.72)
        reasons.append(f"상·하의 격식도 차이 {formality_gap}단계를 조화에 반영했습니다.")
        rules.append("R-CTX-02")

        harmony_name = self._safe_harmony(top["color"], bottom["color"])
        color = {
            "안정적인 무채색 조합": 0.92,
            "톤온톤": 0.90,
            "유사색 조합": 0.88,
            "대비색 조합": 0.76,
            "보통 조합": 0.68,
        }[harmony_name]
        if (top["color"] in NEUTRALS) != (bottom["color"] in NEUTRALS):
            color = max(color, 0.94)
        reasons.append(f"색상 관계 '{harmony_name}'를 상·하의 연결감으로 평가했습니다.")
        rules.append("R-COL-03")

        quiet_patterns = {"", "무지", "분석 보류", "패턴 불확실"}
        pattern_count = sum(item["pattern"] not in quiet_patterns for item in (top, bottom))
        materials_differ = bool(top["material"] and bottom["material"] and top["material"] != bottom["material"])
        if pattern_count == 1:
            pattern_material = 0.95
        elif pattern_count == 0:
            pattern_material = 0.90 if materials_differ else 0.78
        else:
            pattern_material = 0.82 if profile.desired_style in MAXIMAL_STYLES else 0.48
        if top["color"] == bottom["color"] and materials_differ:
            pattern_material = max(pattern_material, 0.94)
        reasons.append("패턴의 시선 경쟁과 소재의 질감 차이를 함께 확인했습니다.")
        rules.extend(["R-PAT-01", "R-MAT-02"])

        breakdown = {
            "silhouette": silhouette,
            "formality": formality,
            "color": color,
            "pattern_material": pattern_material,
        }
        score = (
            0.35 * silhouette + 0.25 * formality
            + 0.25 * color + 0.15 * pattern_material
        )
        return score, breakdown, reasons, list(dict.fromkeys(rules))

    def _diagnose_pair(
        self,
        top: dict,
        bottom: dict,
        profile: UserProfile,
        pose: PoseAnalysis,
    ) -> dict:
        top_body, top_body_reasons, top_body_rules = self._item_body_fit(top, bottom, profile, pose)
        bottom_body, bottom_body_reasons, bottom_body_rules = self._item_body_fit(bottom, top, profile, pose)
        top_situation, top_situation_reasons, top_situation_rules = self._item_situation_fit(top, profile)
        bottom_situation, bottom_situation_reasons, bottom_situation_rules = self._item_situation_fit(bottom, profile)
        top_style, top_style_reasons, top_style_rules = self._item_style_fit(top, profile)
        bottom_style, bottom_style_reasons, bottom_style_rules = self._item_style_fit(bottom, profile)

        matrix = {
            "top": {"body_fit": top_body * 100, "situation_fit": top_situation * 100, "style_fit": top_style * 100},
            "bottom": {"body_fit": bottom_body * 100, "situation_fit": bottom_situation * 100, "style_fit": bottom_style * 100},
        }
        top_total = 0.35 * top_body + 0.40 * top_situation + 0.25 * top_style
        bottom_total = 0.35 * bottom_body + 0.40 * bottom_situation + 0.25 * bottom_style
        harmony, harmony_breakdown, harmony_reasons, harmony_rules = self._outfit_harmony_score(
            top, bottom, profile
        )
        item_average = (top_total + bottom_total) / 2
        overall = (0.80 * item_average + 0.20 * harmony) * 100

        flat = {
            "top.body_fit": matrix["top"]["body_fit"],
            "top.situation_fit": matrix["top"]["situation_fit"],
            "top.style_fit": matrix["top"]["style_fit"],
            "bottom.body_fit": matrix["bottom"]["body_fit"],
            "bottom.situation_fit": matrix["bottom"]["situation_fit"],
            "bottom.style_fit": matrix["bottom"]["style_fit"],
            "outfit.harmony": harmony * 100,
        }
        weakest_area = min(flat, key=flat.get)
        pass_matrix = {
            section: {
                name: value >= DIAGNOSTIC_PASS_THRESHOLD
                for name, value in values.items()
            }
            for section, values in matrix.items()
        }
        failed_areas = [name for name, value in flat.items() if value < DIAGNOSTIC_PASS_THRESHOLD]
        top_weak = min(matrix["top"].values())
        bottom_weak = min(matrix["bottom"].values())
        harmony_passed = harmony * 100 >= DIAGNOSTIC_PASS_THRESHOLD
        if top_weak >= DIAGNOSTIC_PASS_THRESHOLD and bottom_weak >= DIAGNOSTIC_PASS_THRESHOLD and harmony_passed:
            target = "keep"
        elif top_weak < 60 and bottom_weak < 60:
            target = "both"
        elif top_weak <= bottom_weak - 4:
            target = "top"
        elif bottom_weak <= top_weak - 4:
            target = "bottom"
        else:
            target = "auto"

        return {
            "overall_score": round(overall, 2),
            "matrix": {section: {name: round(value, 1) for name, value in scores.items()} for section, scores in matrix.items()},
            "pass_matrix": pass_matrix,
            "failed_areas": failed_areas,
            "weakest_area": weakest_area,
            "change_target": target,
            "conflict_penalty": 0.0,
            "harmony_score": round(harmony * 100, 1),
            "harmony_breakdown": {name: round(value * 100, 1) for name, value in harmony_breakdown.items()},
            "harmony_passed": harmony_passed,
            "reasons": list(dict.fromkeys(
                top_body_reasons + bottom_body_reasons + top_situation_reasons + bottom_situation_reasons
                + top_style_reasons + bottom_style_reasons + harmony_reasons
            )),
            "rules": list(dict.fromkeys(
                top_body_rules + bottom_body_rules + top_situation_rules + bottom_situation_rules
                + top_style_rules + bottom_style_rules + harmony_rules
            )),
        }

    def evaluate_current_outfit(
        self,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
        *,
        keep_threshold: float = DIAGNOSTIC_PASS_THRESHOLD,
    ) -> CurrentOutfitEvaluation:
        """추천 상품 점수와 별개로 사용자가 현재 입은 상·하의 조합을 채점한다."""
        keep_threshold = max(0.0, min(100.0, float(keep_threshold)))
        top = self._garment(None, "top", outfit)
        bottom = self._garment(None, "bottom", outfit)
        garments = [top, bottom]
        weights = dict(self.CURRENT_OUTFIT_WEIGHTS)
        breakdown: dict[str, float] = {}
        reasons: list[str] = []
        applied_rules: list[str] = []

        if self._usable_analysis_value(outfit.style):
            value, component_reasons, rules = self._current_purpose_formality_score(
                garments, profile, outfit
            )
            breakdown["purpose_tpo"] = value
            reasons.extend(component_reasons)
            applied_rules.extend(rules)
            breakdown["preference"] = self._current_preference_score(outfit, profile)
            if profile.preferred_colors:
                applied_rules.append("R-COL-08")

        fits_known = all(
            self._usable_analysis_value(value) for value in (outfit.fit, outfit.lower_fit)
        )
        if fits_known:
            value, component_reasons, rules = self._silhouette_score(top, bottom, profile, pose)
            breakdown["silhouette"] = value
            reasons.extend(component_reasons)
            applied_rules.extend(rules)

        colors_known = outfit.upper_color in COLOR_PALETTE and outfit.lower_color in COLOR_PALETTE
        if colors_known:
            value, component_reasons, rules = self._color_score(top, bottom, profile)
            breakdown["color"] = value
            reasons.extend(component_reasons)
            applied_rules.extend(rules)

        patterns_known = all(
            self._usable_analysis_value(value) for value in (outfit.pattern, outfit.lower_pattern)
        )
        materials_known = all(
            self._usable_analysis_value(value) for value in (outfit.material, outfit.lower_material)
        )
        if patterns_known or materials_known:
            value, component_reasons, rules = self._pattern_material_complexity_score(
                top, bottom, profile
            )
            breakdown["pattern_material_complexity"] = value
            reasons.extend(component_reasons)
            applied_rules.extend(rules)

        styling_inputs_known = fits_known and (colors_known or patterns_known or materials_known)
        if styling_inputs_known:
            value, component_reasons, rules = self._current_styling_intent_score(
                top, bottom, profile, outfit
            )
            breakdown["styling_intent"] = value
            reasons.extend(component_reasons)
            applied_rules.extend(rules)

        active_weights = {name: weights[name] for name in breakdown if name in weights}
        active_weight = sum(active_weights.values())
        total = (
            sum(breakdown[name] * weight for name, weight in active_weights.items())
            / active_weight * 100
            if active_weight
            else 0.0
        )
        # 개편된 Current Fashion Score는 상·하의별 체형/상황/스타일 진단을 사용한다.
        # 위의 기존 축 계산은 이전 규칙 설명과 호환을 위해 남겨 두되 최종 표시 점수와
        # 자동 변경 판단에는 동일한 pair 진단기를 사용한다.
        diagnostic = self._diagnose_pair(top, bottom, profile, pose)
        breakdown = {
            f"{section}_{name}": value / 100
            for section, values in diagnostic["matrix"].items()
            for name, value in values.items()
        }
        total = diagnostic["overall_score"]
        reasons = diagnostic["reasons"]
        applied_rules = diagnostic["rules"]
        active_weight = 1.0
        analysis_confidence = max(0.0, min(1.0, outfit.attribute_confidence))
        core_items_known = all(
            self._usable_analysis_value(value) for value in (outfit.upper_type, outfit.lower_type)
        )
        reliable = core_items_known and analysis_confidence >= 0.55 and active_weight >= 0.50
        all_cells_pass = not diagnostic["failed_areas"]
        should_keep = reliable and total >= keep_threshold and all_cells_pass
        if should_keep:
            verdict = "좋은 코디입니다"
        elif reliable:
            verdict = "추천 코디로 보완할 수 있어요"
        else:
            verdict = "현재 코디 판정을 보류합니다"

        applied_rules = [
            rule_id for rule_id in dict.fromkeys(applied_rules)
            if self.rule_book.has(rule_id) and rule_id in self.EXECUTABLE_RULE_IDS
        ]
        return CurrentOutfitEvaluation(
            total_score=round(total, 2),
            score_breakdown={key: round(value * 100, 1) for key, value in breakdown.items()},
            reasons=list(dict.fromkeys(reasons)),
            applied_rules=applied_rules,
            score_coverage=round(active_weight * 100, 1),
            analysis_confidence=round(analysis_confidence, 3),
            reliable=reliable,
            keep_threshold=keep_threshold,
            should_keep=should_keep,
            verdict=verdict,
            diagnostic_matrix=diagnostic["matrix"],
            pass_matrix=diagnostic["pass_matrix"],
            failed_areas=diagnostic["failed_areas"],
            weakest_area=diagnostic["weakest_area"],
            change_target=diagnostic["change_target"],
            conflict_penalty=diagnostic["conflict_penalty"],
            harmony_score=diagnostic["harmony_score"],
            harmony_breakdown=diagnostic["harmony_breakdown"],
            harmony_passed=diagnostic["harmony_passed"],
        )

    def _score_candidate(
        self,
        top_product: Product | None,
        bottom_product: Product | None,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
    ) -> tuple[float, dict[str, float], list[str], list[str], list[str], float]:
        products = [product for product in (top_product, bottom_product) if product]
        top = self._garment(top_product, "top", outfit)
        bottom = self._garment(bottom_product, "bottom", outfit)
        garments = [top, bottom]

        purpose_tpo, purpose_reasons, purpose_rules = self._purpose_formality_score(products, garments, profile)
        weather, weather_reasons, weather_rules = self._weather_activity_score(products, garments, profile)
        silhouette, silhouette_reasons, silhouette_rules = self._silhouette_score(top, bottom, profile, pose)
        color, color_reasons, color_rules = self._color_score(top, bottom, profile)
        pattern, pattern_reasons, pattern_rules = self._pattern_material_complexity_score(top, bottom, profile)
        preference = self._preference_score(products, profile)
        wardrobe = self._wardrobe_score(products, profile)

        breakdown = {
            "purpose_tpo": purpose_tpo,
            "weather_activity": weather,
            "silhouette": silhouette,
            "color": color,
            "pattern_material_complexity": pattern,
            "preference": preference,
        }
        applied_rules = purpose_rules + weather_rules + silhouette_rules + color_rules + pattern_rules
        if profile.preferred_colors:
            applied_rules.append("R-COL-08")
        if wardrobe is not None:
            breakdown["wardrobe"] = wardrobe
            applied_rules.extend(["R-OWN-01", "R-COL-08"])

        if all(garment["item_type"] for garment in garments):
            applied_rules.append("R-DET-01")
        styling_tips, guidance_rules = self._styling_guidance(top, bottom, profile)
        applied_rules.extend(guidance_rules)
        applied_rules = [
            rule_id for rule_id in dict.fromkeys(applied_rules)
            if self.rule_book.has(rule_id) and rule_id in self.EXECUTABLE_RULE_IDS
        ]
        legacy_applied_rules = list(applied_rules)

        weights = self._weights_for_profile(profile, include_wardrobe=wardrobe is not None)
        active_weight = sum(weights.values())
        total = sum(breakdown[name] * weight for name, weight in weights.items()) / active_weight * 100
        reasons = purpose_reasons + weather_reasons + color_reasons + silhouette_reasons + pattern_reasons
        diagnostic = self._diagnose_pair(top, bottom, profile, pose)
        total = diagnostic["overall_score"]
        breakdown = {
            f"{section}_{name}": value
            for section, values in diagnostic["matrix"].items()
            for name, value in values.items()
        }
        breakdown["outfit_harmony"] = diagnostic["harmony_score"]
        breakdown.update({
            f"harmony_{name}": value
            for name, value in diagnostic["harmony_breakdown"].items()
        })
        reasons = diagnostic["reasons"]
        applied_rules = [
            rule_id for rule_id in dict.fromkeys(legacy_applied_rules + diagnostic["rules"])
            if self.rule_book.has(rule_id) and rule_id in self.EXECUTABLE_RULE_IDS
        ]
        return (
            round(total, 2),
            {key: round(value, 1) for key, value in breakdown.items()},
            reasons,
            applied_rules,
            styling_tips,
            round(active_weight * 100, 1),
        )

    def recommend(
        self,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
        top_k: int = 3,
        current_outfit_keep_threshold: float | None = None,
        on_stage: Callable[[str], None] | None = None,
    ) -> list[Recommendation]:
        if on_stage:
            on_stage("candidates")
        current = self.evaluate_current_outfit(
            profile,
            pose,
            outfit,
            keep_threshold=current_outfit_keep_threshold if current_outfit_keep_threshold is not None else 100.0,
        )
        if current_outfit_keep_threshold is not None and current.should_keep:
            return [
                Recommendation(
                    rank=1,
                    products=[],
                    total_score=current.total_score,
                    score_breakdown=current.score_breakdown,
                    reasons=["좋은 코디입니다. 현재 착장을 유지해도 좋아요.", *current.reasons],
                    applied_rules=current.applied_rules,
                    score_coverage=current.score_coverage,
                    styling_tips=["상·하의를 유지하고 작은 액세서리만 선택적으로 조정해보세요."],
                    change_target="keep",
                    current_score=current.total_score,
                    predicted_score=current.total_score,
                    diagnostic_matrix=current.diagnostic_matrix,
                    harmony_score=current.harmony_score,
                    harmony_breakdown=current.harmony_breakdown,
                )
            ]

        kept = set(profile.items_to_keep)
        if kept >= {"top", "bottom"} or profile.change_scope == "현재 유지":
            return [
                Recommendation(
                    rank=1, products=[], total_score=current.total_score,
                    score_breakdown=current.score_breakdown,
                    reasons=["사용자가 지정한 상의·하의 유지 조건을 적용했습니다.", *current.reasons],
                    applied_rules=current.applied_rules, score_coverage=current.score_coverage,
                    change_target="keep", current_score=current.total_score,
                    predicted_score=current.total_score, diagnostic_matrix=current.diagnostic_matrix,
                    harmony_score=current.harmony_score, harmony_breakdown=current.harmony_breakdown,
                )
            ]

        # change_scope의 단일 변경 값은 구버전 UI의 명시적 요청으로 존중한다.
        # 기본값인 '전체 변경'에서는 엔진이 top/bottom/both를 모두 비교해 자동 결정한다.
        if kept == {"top"}:
            actions = ["bottom"]
        elif kept == {"bottom"}:
            actions = ["top"]
        elif profile.change_scope == "상의만 변경":
            actions = ["top"]
        elif profile.change_scope == "하의만 변경":
            actions = ["bottom"]
        else:
            actions = ["top", "bottom", "both"]

        available_tops = self._available_for_profile("top", profile)
        available_bottoms = self._available_for_profile("bottom", profile)
        candidates = []
        min_b = getattr(profile, "min_budget", None)
        max_b = getattr(profile, "max_budget", None)
        # Defensive check: if both bounds provided but invalid, fail fast with clear message
        if (min_b is not None and max_b is not None) and (min_b > max_b):
            raise MinGreaterThanMax("최소 예산이 최대 예산보다 큽니다. 최소/최대 예산을 확인하세요.")

        if on_stage:
            on_stage("scoring")
        for action in actions:
            if action == "top":
                pairs = ((top, None) for top in available_tops)
            elif action == "bottom":
                pairs = ((None, bottom) for bottom in available_bottoms)
            else:
                pairs = itertools.product(available_tops, available_bottoms)
            for top, bottom in pairs:
                products = [product for product in (top, bottom) if product]
                if not products:
                    continue
                total_price = sum(product.price for product in products)
                # min/max가 없을 때는 기존 단일 budget을 상한으로 유지한다.
                if min_b is None and max_b is None and total_price > profile.budget:
                    continue
                score, breakdown, reasons, applied_rules, tips, coverage = self._score_candidate(
                    top, bottom, profile, pose, outfit
                )
                changed_count = 2 if action == "both" else 1
                delta = round(score - current.total_score, 2)
                change_cost = float(4 * changed_count)
                utility = round(delta - change_cost, 2)
                matrix = {
                    "top": {
                        name: breakdown[f"top_{name}"]
                        for name in ("body_fit", "situation_fit", "style_fit")
                    },
                    "bottom": {
                        name: breakdown[f"bottom_{name}"]
                        for name in ("body_fit", "situation_fit", "style_fit")
                    },
                }
                candidates.append((
                    utility, score, action, products, breakdown, reasons,
                    applied_rules, tips, coverage, delta, change_cost, matrix, total_price,
                ))

        if not candidates:
            raise ValueError(
                "재고·예산·목적·계절·사용자 제외 조건을 모두 만족하는 상품이 없습니다. "
                "예산, 계절, 제외 목록 또는 변경 범위를 조정하세요."
            )

        # If the user provided min/max bounds, prefer only candidates within that range.
        if min_b is not None or max_b is not None:
            in_range = []
            for item in candidates:
                total = item[12]
                ok = True
                if min_b is not None and total < min_b:
                    ok = False
                if max_b is not None and total > max_b:
                    ok = False
                if ok:
                    in_range.append(item)
            if not in_range:
                # Explicitly signal that no combinations satisfied the user's budget range.
                raise NoBudgetMatch("예산 범위에 맞는 상품 조합이 없습니다. 다른 예산 범위를 시도해 주세요.")
            candidates = in_range

        # 점수가 정확히 같은 후보가 매우 많다. 실측(2026-08-21)에서 최고점 동점이
        # 남성 777개(71종 상의) / 여성 2,833개(98종 상의)였다. 안정 정렬에 맡기면
        # 카탈로그 CSV 행 순서가 승자를 정해 모든 사람에게 같은 조합이 나간다.
        # 동점끼리는 사진에서 뽑은 씨앗으로 섞어, 같은 사람은 같은 결과를 받되
        # 사람이 다르면 다른 조합을 받게 한다. 점수 자체는 건드리지 않는다.
        seed = _tie_seed(pose)
        candidates.sort(key=lambda item: (-item[0], -item[1], _tie_rank(seed, item[3])))
        # 4점 미만 개선이거나 변경 비용을 이기지 못하면 교체 이득이 작다고 본다.
        if candidates[0][9] < 4.0 or candidates[0][0] <= 0.0:
            return [
                Recommendation(
                    rank=1, products=[], total_score=current.total_score,
                    score_breakdown=current.score_breakdown,
                    reasons=["교체로 얻는 개선폭이 변경 비용보다 작아 현재 착장을 유지합니다.", *current.reasons],
                    applied_rules=current.applied_rules, score_coverage=current.score_coverage,
                    change_target="keep", current_score=current.total_score,
                    predicted_score=current.total_score, diagnostic_matrix=current.diagnostic_matrix,
                    harmony_score=current.harmony_score, harmony_breakdown=current.harmony_breakdown,
                )
            ]
        recommendations = [
            Recommendation(
                rank=index,
                products=products,
                total_score=score,
                score_breakdown=breakdown,
                reasons=[
                    f"현재 {current.total_score:.1f}점에서 예상 {score:.1f}점으로 {delta:+.1f}점 개선됩니다.",
                    *reasons,
                ],
                applied_rules=applied_rules,
                score_coverage=coverage,
                styling_tips=tips,
                change_target=action,
                current_score=current.total_score,
                predicted_score=score,
                delta_score=delta,
                change_cost=change_cost,
                utility_score=utility,
                diagnostic_matrix=matrix,
                harmony_score=breakdown.get("outfit_harmony", 0.0),
                harmony_breakdown={
                    name: breakdown.get(f"harmony_{name}", 0.0)
                    for name in ("silhouette", "formality", "color", "pattern_material")
                },
            )
            for index, (utility, score, action, products, breakdown, reasons, applied_rules,
                        tips, coverage, delta, change_cost, matrix, _total_price)
            in enumerate(candidates[:top_k], start=1)
        ]
        primary_groups: dict[tuple[float, float], list[Recommendation]] = {}
        for recommendation in recommendations:
            key = (recommendation.utility_score, recommendation.total_score)
            primary_groups.setdefault(key, []).append(recommendation)
        for recommendation in recommendations:
            key = (recommendation.utility_score, recommendation.total_score)
            tied = primary_groups[key]
            recommendation.display_rank = min(item.rank for item in tied)
            recommendation.ranking_tied = len(tied) > 1
            if recommendation.ranking_tied:
                recommendation.ranking_reason = (
                    "적합도와 변경 효율이 같은 공동 순위이며, 표시 순서만 "
                    "같은 사진에서 항상 같도록 고정했습니다."
                )
        return recommendations
