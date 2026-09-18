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
    SHAPE_DIAMOND,
    SHAPE_HOURGLASS,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_ROUND,
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
CATEGORY_LABELS = {"top": "상의", "bottom": "하의", "shoes": "신발"}


def selected_categories(profile: UserProfile) -> tuple[str, ...]:
    """새 체크박스 입력을 우선하고, 기존 노트북의 change_scope도 지원한다."""
    if profile.change_categories is not None:
        values = profile.change_categories
        if not isinstance(values, list) or not values or any(not isinstance(value, str) or value not in CATEGORY_LABELS for value in values):
            raise ValueError("change_categories는 top, bottom, shoes 중 하나 이상을 담은 목록이어야 합니다.")
        return tuple(category for category in CATEGORY_LABELS if category in values)
    return SCOPE_CATEGORIES.get(profile.change_scope, ("top", "bottom"))


SHOE_STYLE_DEFAULTS = {
    "스트리트": "스니커즈", "캐주얼": "스니커즈", "미니멀": "로퍼",
    "포멀": "더비슈즈", "스포티": "러닝화", "로맨틱": "메리제인",
}
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
    keyword_rules: dict[str, dict[str, list[str]]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def brief_lines(self, max_keywords: int = 8) -> list[str]:
        priority = (
            "item_type", "fit", "length", "waistline", "structure", "silhouette",
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
        categories = selected_categories(profile)
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
            applicable = [target for category, target in targets.items()
                          if not (category == "shoes" and attribute == "material" and source == "photo_fallback")]
            if not applicable:
                return
            for target in applicable:
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
        if ("top" in targets or "bottom" in targets) and (proportion_reliable or shape_reliable):
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
            elif shape_reliable and balance_goal and pose.body_shape == SHAPE_HOURGLASS:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "silhouette", "허리 기준점", "세미핏")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-03")
            elif shape_reliable and balance_goal and pose.body_shape == SHAPE_ROUND:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "structure", "V넥", "넥라인 포인트")
                    self._add(targets["top"], "fit", "레귤러", "여유핏")
                    self._add(targets["top"], "length", "기본 기장")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                    self._add(targets["bottom"], "length", "풀렝스")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-07")
            elif shape_reliable and balance_goal and pose.body_shape == SHAPE_DIAMOND:
                used_photo = True
                if "top" in targets:
                    self._add(targets["top"], "structure", "어깨 구조", "넥라인 포인트")
                    self._add(targets["top"], "fit", "레귤러", "여유핏")
                if "bottom" in targets:
                    self._add(targets["bottom"], "fit", "스트레이트", "세미와이드")
                    self._add(targets["bottom"], "length", "풀렝스")
                sources["body_shape"] = "photo_fallback"
                self._rule(rules, "R-BOD-08")

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
            if category == "shoes":
                shoe_type = SHOE_STYLE_DEFAULTS.get(fallback_style, "스니커즈")
                # 상황과 활동성 우선. 신발에는 체형/다리 길이 보정 규칙을 적용하지 않는다.
                if self._provided(profile, "purpose") and profile.purpose in {"면접", "비즈니스", "결혼식", "하객"}:
                    shoe_type = "로퍼"
                if self._provided(profile, "purpose") and profile.purpose in {"운동", "스포츠", "러닝"}:
                    shoe_type = "러닝화"
                if profile.dress_code in {"포멀", "비즈니스 포멀"}:
                    shoe_type = "더비슈즈"
                if profile.purpose == "여행" or profile.activity_level == "높음":
                    self._add(target, "function", "장시간 보행", "착화감")
                # 학습된 신발 종류는 명시 스타일/목적이 없는 경우에만 보충한다.
                if (not self._provided(profile, "desired_style") and not self._provided(profile, "purpose")
                        and not self._provided(profile, "dress_code")
                        and outfit.shoes.get("accepted")):
                    shoe_type = outfit.shoes["item_type"]
                    default_source = "photo_shoe_head"
                    used_photo = True
                self._add(target, "item_type", shoe_type)
                sources["shoes.item_type"] = default_source
                self._rule(rules, "R-CTX-01")
                self._rule(rules, "R-ACC-06")
                continue
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
        keyword_rules = {category: {} for category in targets}
        body_rule_values = {
            "R-BOD-05": {"허리선", "기본 기장", "미드라이즈", "하이라이즈", "스트레이트", "세미와이드", "풀렝스"},
            "R-BOD-01": {"어깨 구조", "넥라인 포인트", "스트레이트", "세미와이드"},
            "R-BOD-06": {"어깨 구조", "넥라인 포인트", "레귤러", "여유핏"},
            "R-BOD-02": {"레귤러", "정돈된 핏", "스트레이트", "세미와이드", "와이드"},
            "R-BOD-03": {"허리 기준점", "세미핏", "스트레이트", "세미와이드"},
            "R-BOD-07": {"V넥", "넥라인 포인트", "레귤러", "여유핏", "기본 기장", "스트레이트", "세미와이드", "풀렝스"},
            "R-BOD-08": {"어깨 구조", "넥라인 포인트", "레귤러", "여유핏", "스트레이트", "세미와이드", "풀렝스"},
        }
        for category, attributes in targets.items():
            for attribute, values in attributes.items():
                for value in values:
                    rule_ids = []
                    if attribute == "purpose" and "R-CTX-01" in rules:
                        rule_ids.append("R-CTX-01")
                    if attribute in {"material", "season"} and "R-MAT-01" in rules:
                        rule_ids.append("R-MAT-01")
                    if attribute == "color" and "R-COL-08" in rules:
                        rule_ids.append("R-COL-08")
                    if attribute == "function" and "R-WEA-02" in rules:
                        rule_ids.append("R-WEA-02")
                    if attribute == "item_type":
                        rule_ids.extend(
                            rule_id for rule_id in ("R-CTX-01", "R-ACC-06")
                            if rule_id in rules
                        )
                    if attribute == "fit" and sources.get(f"{category}.fit") in {"user_style_rule", "photo_style_rule", "fashion_rule_default"} and "R-SIL-01" in rules:
                        rule_ids.append("R-SIL-01")
                    if attribute == "length" and sources.get(f"{category}.length") in {"user_style_rule", "photo_style_rule", "fashion_rule_default"} and "R-SIL-03" in rules:
                        rule_ids.append("R-SIL-03")
                    for rule_id, rule_values in body_rule_values.items():
                        if value in rule_values and rule_id in rules:
                            rule_ids.append(rule_id)
                    if rule_ids:
                        keyword_rules[category][value] = list(dict.fromkeys(rule_ids))
        return TargetKeywordResult(
            mode=mode,
            targets=targets,
            constraints=constraints,
            sources=sources,
            applied_rules=rules,
            keyword_rules=keyword_rules,
        )
