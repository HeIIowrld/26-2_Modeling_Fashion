from __future__ import annotations

import colorsys
import itertools
from pathlib import Path

from fashion_rules import FashionRuleBook
from outfit_analyzer import COLOR_PALETTE, NEUTRALS, color_harmony
from product_catalog import ProductCatalog
from schemas import (
    SHAPE_HOURGLASS,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_TRIANGLE,
    SHAPE_UNCERTAIN,
    GOAL_BALANCE,
    GOAL_LOWER_FOCUS,
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


class RecommendationEngine:
    """Markdown 규칙을 필터·점수·안전장치·스타일링 안내로 실행한다.

    자연어를 임의로 실행하지 않고 규칙 ID별 Python 계산식을 둔다. 상품
    상세정보는 사진 추정보다 우선하며, 입력이 없는 규칙은 점수에 넣지 않는다.
    """

    SCORING_RULE_IDS = {
        "R-CTX-01", "R-CTX-02", "R-CAT-02",
        "R-SIL-01", "R-SIL-03", "R-SIL-05", "R-SIL-06",
        "R-COL-01", "R-COL-02", "R-COL-03", "R-COL-04", "R-COL-05",
        "R-COL-08", "R-COL-10", "R-COL-11", "R-COL-13",
        "R-PAT-01", "R-PAT-03", "R-PAT-04", "R-MAT-01", "R-MAT-02", "R-CMP-01",
        "R-BOD-01", "R-BOD-02", "R-BOD-03", "R-BOD-04",
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
                "item_type": product.item_type,
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
            "style": outfit.style,
            "item_type": outfit.upper_type if is_top else outfit.lower_type,
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
        if goal in PROPORTION_GOALS:
            rules.extend(["R-SIL-03", "R-SIL-06"])
            length_score = {"크롭 기장": 1.0, "기본 기장": 0.78, "롱 기장": 0.45}.get(top["length"], 0.65)
            if bottom["waistline"] == "하이웨이스트":
                length_score = min(1.0, length_score + 0.12)
            components.append((length_score, 0.35))
            reasons.append("상의 기장과 하의 허리선이 만드는 분할점을 목표 실루엣에 맞춰 평가했습니다.")

        body_confident = pose.valid and pose.body_shape_confidence >= 0.65 and pose.body_shape != SHAPE_UNCERTAIN
        if goal == GOAL_BALANCE and body_confident:
            body_score = 0.78
            if pose.body_shape == SHAPE_INVERTED_TRIANGLE:
                rules.append("R-BOD-02")
                body_score = 1.0 if bottom_large or bottom_ordered else 0.62
                reasons.append("상·하체 균형 목표에 맞춰 하의의 구조와 볼륨을 비교했습니다.")
            elif pose.body_shape == SHAPE_TRIANGLE:
                rules.append("R-BOD-01")
                top_focus = top["color"] in BRIGHT_COLORS or top["pattern"] not in {"", "무지", "패턴 불확실", "분석 보류"}
                bottom_quiet = bottom["color"] in DARK_COLORS and bottom["pattern"] in {"", "무지"}
                body_score = 1.0 if top_focus and bottom_quiet else 0.66
                reasons.append("상·하체 균형 목표에 맞춰 상체로 시선을 옮기는 색·패턴 배치를 비교했습니다.")
            elif pose.body_shape in {SHAPE_RECTANGLE, SHAPE_HOURGLASS}:
                # 모래시계체형도 어깨와 엉덩이가 비슷하므로 같은 규칙을 쓴다.
                # 마름모꼴·둥근체형은 허리가 중심이라 대응하는 규칙이 문서에 아직 없다.
                rules.append("R-BOD-03")
                body_score = 0.96 if top_large != bottom_large else 0.80
                reasons.append("상·하체 폭이 비슷한 경우 볼륨을 한쪽씩 배치했는지 확인했습니다.")
            components.append((self._shrink_to_neutral(body_score, pose.body_shape_confidence), 0.30))
        elif goal in {GOAL_UPPER_FOCUS, GOAL_LOWER_FOCUS}:
            rules.append("R-BOD-04")
            target = top if goal == GOAL_UPPER_FOCUS else bottom
            other = bottom if goal == GOAL_UPPER_FOCUS else top
            target_focus = self._is_bold_color(target["color"]) or target["pattern"] not in {"", "무지", "분석 보류", "패턴 불확실"}
            other_quiet = other["color"] in NEUTRALS and other["pattern"] in {"", "무지"}
            components.append((1.0 if target_focus and other_quiet else 0.62, 0.30))
            reasons.append(f"사용자가 선택한 '{goal}' 위치에 색이나 패턴의 시선 중심이 생기는지 확인했습니다.")

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
            color_score = 1.0 if product.color in profile.preferred_colors else 0.75
            values.append(0.78 * style_score + 0.22 * color_score if profile.preferred_colors else style_score)
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

        weights = self._weights_for_profile(profile, include_wardrobe=wardrobe is not None)
        active_weight = sum(weights.values())
        total = sum(breakdown[name] * weight for name, weight in weights.items()) / active_weight * 100
        reasons = purpose_reasons + weather_reasons + color_reasons + silhouette_reasons + pattern_reasons
        return (
            round(total, 2),
            {key: round(value * 100, 1) for key, value in breakdown.items()},
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
    ) -> list[Recommendation]:
        scopes = CHANGE_SCOPE_MAP.get(profile.change_scope, ["top", "bottom"])
        if not scopes:
            return [
                Recommendation(
                    rank=1,
                    products=[],
                    total_score=100.0,
                    score_breakdown={},
                    reasons=["사용자가 선택한 현재 코디 유지 조건을 적용했습니다."],
                    applied_rules=[rule for rule in ("R-KOR-02", "R-DAT-01") if self.rule_book.has(rule)],
                    score_coverage=0.0,
                    styling_tips=[],
                )
            ]

        tops = self._available_for_profile("top", profile) if "top" in scopes else [None]
        bottoms = self._available_for_profile("bottom", profile) if "bottom" in scopes else [None]
        candidates = []
        for top, bottom in itertools.product(tops, bottoms):
            products = [product for product in (top, bottom) if product]
            if not products or sum(product.price for product in products) > profile.budget:
                continue
            score, breakdown, reasons, applied_rules, tips, coverage = self._score_candidate(
                top, bottom, profile, pose, outfit
            )
            candidates.append((score, products, breakdown, reasons, applied_rules, tips, coverage))

        if not candidates:
            raise ValueError(
                "재고·예산·목적·계절·사용자 제외 조건을 모두 만족하는 상품이 없습니다. "
                "예산, 계절, 제외 목록 또는 변경 범위를 조정하세요."
            )

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            Recommendation(
                rank=index,
                products=products,
                total_score=score,
                score_breakdown=breakdown,
                reasons=reasons,
                applied_rules=applied_rules,
                score_coverage=coverage,
                styling_tips=tips,
            )
            for index, (score, products, breakdown, reasons, applied_rules, tips, coverage)
            in enumerate(candidates[:top_k], start=1)
        ]
