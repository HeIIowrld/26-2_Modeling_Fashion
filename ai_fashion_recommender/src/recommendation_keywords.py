"""Fashion Rule 기반 추천 키워드 생성.

상품 검색기와 UI에서 재사용할 수 있도록 점수가 아닌 구조화된 속성을 만든다.
사용자가 직접 준 값이 사진 추정보다 항상 우선하며, 빠진 값만 사진으로 보충한다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from schemas import (
    GOAL_BALANCE,
    GOAL_LONGER_LEGS,
    GOAL_LOWER_FOCUS,
    GOAL_NONE,
    GOAL_UPPER_FOCUS,
    GOAL_WAISTLINE,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_TRIANGLE,
    OutfitAnalysis,
    PoseAnalysis,
    UserProfile,
)


UNKNOWN_MARKERS = ("분석 보류", "분석 불가", "불확실", "해당 없음", "자동")
SCOPE_CATEGORIES = {
    "상의만 변경": ("top",),
    "하의만 변경": ("bottom",),
    "전체 변경": ("top", "bottom"),
    "현재 유지": (),
}
CATEGORY_LABELS = {"top": "상의", "bottom": "하의"}
STYLE_DEFAULTS = {
    "스트리트": {
        "top": {"fit": ("오버핏",), "length": ("기본 기장", "롱 기장")},
        "bottom": {"fit": ("와이드",), "length": ("풀렝스",)},
    },
    "미니멀": {
        "top": {"fit": ("레귤러",), "length": ("기본 기장",)},
        "bottom": {"fit": ("스트레이트",), "length": ("풀렝스",)},
    },
    "포멀": {
        "top": {"fit": ("레귤러", "정돈된 핏"), "length": ("기본 기장",)},
        "bottom": {"fit": ("스트레이트", "세미와이드"), "length": ("풀렝스",)},
    },
    "스포티": {
        "top": {"fit": ("여유핏",), "length": ("기본 기장",)},
        "bottom": {"fit": ("조거", "와이드"), "length": ("긴바지",)},
    },
    "로맨틱": {
        "top": {"fit": ("세미핏",), "length": ("허리선", "기본 기장")},
        "bottom": {"fit": ("플레어", "A라인"), "length": ("미디", "풀렝스")},
    },
    "캐주얼": {
        "top": {"fit": ("레귤러", "여유핏"), "length": ("기본 기장",)},
        "bottom": {"fit": ("스트레이트", "세미와이드"), "length": ("풀렝스",)},
    },
}


@dataclass
class TargetKeywordResult:
    """검색 단계가 그대로 소비할 수 있는 추천 속성과 제약 조건."""

    mode: str
    targets: dict[str, dict[str, list[str]]]
    constraints: dict[str, Any] = field(default_factory=dict)
    sources: dict[str, str] = field(default_factory=dict)
    applied_rules: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def brief_lines(self, max_keywords: int = 8) -> list[str]:
        priority = (
            "fit", "length", "waistline", "structure", "silhouette",
            "style", "material", "color", "harmony_reference_color",
            "purpose", "season", "function",
        )
        lines = []
        for category, attributes in self.targets.items():
            values = [
                value
                for attribute in priority
                for value in attributes.get(attribute, [])
            ]
            if values:
                unique = list(dict.fromkeys(values))[:max_keywords]
                lines.append(f"{CATEGORY_LABELS.get(category, category)}: " + ", ".join(unique))
        return lines


class RecommendationKeywordGenerator:
    """사용자 조건과 사진 분석을 Fashion Rule의 검색 키워드로 변환한다."""

    def __init__(self, available_rule_ids: set[str] | None = None) -> None:
        self.available_rule_ids = available_rule_ids

    @staticmethod
    def _usable(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, str):
            stripped = value.strip()
            return bool(stripped and not any(marker in stripped for marker in UNKNOWN_MARKERS))
        return bool(value)

    @staticmethod
    def _provided(profile: UserProfile, field_name: str) -> bool:
        """웹은 provided_fields로 명시 입력을 구분하고, 기존 호출은 값 자체를 존중한다."""
        provided_fields = profile.provided_fields
        if provided_fields is None:
            return RecommendationKeywordGenerator._usable(getattr(profile, field_name, None))
        return field_name in provided_fields and RecommendationKeywordGenerator._usable(
            getattr(profile, field_name, None)
        )

    @staticmethod
    def _add(target: dict[str, list[str]], attribute: str, *values: str) -> None:
        bucket = target.setdefault(attribute, [])
        for value in values:
            if value and value not in bucket:
                bucket.append(value)

    def _rule(self, rules: list[str], rule_id: str) -> None:
        if self.available_rule_ids is None or rule_id in self.available_rule_ids:
            if rule_id not in rules:
                rules.append(rule_id)

    def generate(
        self,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
    ) -> TargetKeywordResult:
        explicit_scope = self._provided(profile, "change_scope")
        categories = SCOPE_CATEGORIES.get(profile.change_scope, ("top", "bottom")) if explicit_scope else ("top", "bottom")
        targets = {category: {"category": [CATEGORY_LABELS[category]]} for category in categories}
        constraints: dict[str, Any] = {"search_categories": list(categories)}
        sources: dict[str, str] = {}
        rules: list[str] = []
        used_input = False
        used_photo = False

        def add_all(attribute: str, values: list[str], source: str) -> None:
            nonlocal used_input, used_photo
            clean = [value for value in values if self._usable(value)]
            if not clean:
                return
            for target in targets.values():
                self._add(target, attribute, *clean)
            sources[attribute] = source
            used_input |= source == "user_input"
            used_photo |= source == "photo_fallback"

        # 입력 조건이 있으면 사진 추정보다 먼저 사용한다.
        if self._provided(profile, "desired_style"):
            add_all("style", [profile.desired_style], "user_input")
        elif self._usable(outfit.style):
            add_all("style", [outfit.style], "photo_fallback")

        if self._provided(profile, "purpose"):
            add_all("purpose", [profile.purpose], "user_input")
            self._rule(rules, "R-CTX-01")

        if self._provided(profile, "season") and profile.season != "사계절":
            add_all("season", [profile.season], "user_input")
            self._rule(rules, "R-MAT-01")

        if self._provided(profile, "preferred_materials"):
            add_all("material", profile.preferred_materials, "user_input")
            self._rule(rules, "R-MAT-01")
        else:
            photo_materials = [outfit.material, outfit.lower_material]
            add_all("material", photo_materials, "photo_fallback")

        if self._provided(profile, "preferred_colors"):
            add_all("color", profile.preferred_colors, "user_input")
            self._rule(rules, "R-COL-08")
        else:
            # 색을 그대로 강제하지 않고 현재 착장과 연결할 기준색으로 저장한다.
            detected_colors = [outfit.upper_color, outfit.lower_color]
            add_all("harmony_reference_color", detected_colors, "photo_fallback")
            self._rule(rules, "R-COL-03")

        if self._provided(profile, "activity_level") and profile.activity_level == "높음":
            add_all("function", ["활동성", "통기성"], "user_input")
            self._rule(rules, "R-WEA-02")

        # 체형·비율은 사진에서만 오는 공통 기반이다. 신뢰 가능한 경우에만 쓴다.
        proportion_reliable = pose.valid and pose.full_body_score >= 0.65
        shape_reliable = pose.valid and pose.body_shape_confidence >= 0.65
        goal = profile.silhouette_goal if self._provided(profile, "silhouette_goal") else GOAL_NONE
        if proportion_reliable or shape_reliable:
            if proportion_reliable and (pose.leg_ratio < 0.60 or goal in {GOAL_LONGER_LEGS, GOAL_WAISTLINE}):
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "length", "허리선", "기본 기장")
                if "bottom" in targets:
                    self._add(targets["bottom"], "waistline", "미드라이즈", "하이라이즈")
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                    self._add(targets["bottom"], "length", "풀렝스")
                sources["proportion"] = "photo_fallback"
                self._rule(rules, "R-BOD-05")

            balance_goal = goal in {GOAL_NONE, GOAL_BALANCE, GOAL_UPPER_FOCUS, GOAL_LOWER_FOCUS}
            if shape_reliable and balance_goal and pose.body_shape == SHAPE_INVERTED_TRIANGLE:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "fit", "레귤러", "정돈된 핏")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드", "와이드")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-02")
            elif shape_reliable and balance_goal and pose.body_shape == SHAPE_TRIANGLE:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "structure", "어깨 구조", "넥라인 포인트")
                    self._add(targets["top"], "fit", "레귤러", "여유핏")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-01")
                self._rule(rules, "R-BOD-06")
            elif shape_reliable and balance_goal and pose.body_shape == SHAPE_RECTANGLE:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "silhouette", "허리 기준점", "세미핏")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-03")

        # 사진 신뢰도가 낮거나 대응 체형 규칙이 없어도 검색 속성이 빈 채로 끝나지
        # 않게 한다. 사용자 스타일 → 사진 스타일 → 캐주얼 순으로 기본 실루엣을 고른다.
        if self._provided(profile, "desired_style"):
            fallback_style = profile.desired_style
            default_source = "user_style_rule"
        elif self._usable(outfit.style):
            fallback_style = outfit.style
            default_source = "photo_style_rule"
        else:
            fallback_style = "캐주얼"
            default_source = "fashion_rule_default"
        style_defaults = STYLE_DEFAULTS.get(fallback_style, STYLE_DEFAULTS["캐주얼"])
        for category, target in targets.items():
            defaults = style_defaults[category]
            if not target.get("fit"):
                self._add(target, "fit", *defaults["fit"])
                sources[f"{category}.fit"] = default_source
                self._rule(rules, "R-SIL-01")
            if not target.get("length"):
                self._add(target, "length", *defaults["length"])
                sources[f"{category}.length"] = default_source
                self._rule(rules, "R-SIL-03")

        # 검색 단계에서 강제할 조건은 키워드와 분리한다.
        if profile.min_budget is not None:
            constraints["min_budget"] = profile.min_budget
        if profile.max_budget is not None:
            constraints["max_budget"] = profile.max_budget
        elif self._provided(profile, "budget"):
            constraints["max_budget"] = profile.budget
        if profile.avoided_colors:
            constraints["excluded_colors"] = list(dict.fromkeys(profile.avoided_colors))
        if profile.avoided_materials:
            constraints["excluded_materials"] = list(dict.fromkeys(profile.avoided_materials))
        if profile.excluded_item_types:
            constraints["excluded_item_types"] = list(dict.fromkeys(profile.excluded_item_types))

        mode = "mixed" if used_input and used_photo else "user_input" if used_input else "photo_fallback"
        return TargetKeywordResult(
            mode=mode,
            targets=targets,
            constraints=constraints,
            sources=sources,
            applied_rules=rules,
        )
