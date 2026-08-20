from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class WardrobeItem:
    """보유 옷 활용도를 계산하기 위한 최소 메타데이터."""

    item_id: str
    category: str
    color: str
    style: str = ""
    season: str = "사계절"
    item_type: str = ""
    fit: str = ""
    pattern: str = "무지"
    material: str = ""


@dataclass
class UserProfile:
    purpose: str = "데일리"
    desired_style: str = "캐주얼"
    budget: int = 150_000
    change_scope: str = "전체 변경"
    height_cm: float | None = None
    weight_kg: float | None = None
    season: str = "사계절"
    silhouette_goal: str = "자동 보정 안 함"
    dress_code: str = "자동"
    activity_level: str = "보통"
    preferred_colors: list[str] = field(default_factory=list)
    avoided_colors: list[str] = field(default_factory=list)
    avoided_materials: list[str] = field(default_factory=list)
    excluded_item_types: list[str] = field(default_factory=list)
    temperature_c: float | None = None
    feels_like_c: float | None = None
    humidity: float | None = None
    precipitation_probability: float | None = None
    wind_mps: float | None = None
    uv_index: float | None = None
    owned_items: list[WardrobeItem] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class PoseAnalysis:
    valid: bool
    full_body_score: float
    body_shape: str
    shoulder_hip_ratio: float
    upper_lower_ratio: float
    leg_ratio: float
    posture: str
    body_shape_confidence: float = 0.0
    warnings: list[str] = field(default_factory=list)
    landmarks: dict[str, tuple[float, float, float]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class OutfitAnalysis:
    parser_backend: str
    upper_color: str
    lower_color: str
    color_harmony: str
    detected_items: list[str]
    style: str
    upper_type: str = "분석 불가"
    lower_type: str = "분석 불가"
    lower_subtype: str = "분석 보류"
    pant_leg_shape: str = "분석 보류"
    pant_length: str = "분석 보류"
    sleeve_length: str = "분석 불가"
    upper_length: str = "분석 불가"
    bottom_length: str = "분석 불가"
    fit: str = "분석 불가"
    lower_fit: str = "분석 불가"
    neckline: str = "분석 보류"
    pattern: str = "분석 보류"
    material: str = "분석 보류"
    lower_pattern: str = "분석 보류"
    lower_material: str = "분석 보류"
    sleeve_shape: str = "분석 보류"
    collar: str = "분석 보류"
    silhouette: str = "분석 보류"
    details: list[str] = field(default_factory=list)
    lower_details: list[str] = field(default_factory=list)
    attribute_sources: dict[str, str] = field(default_factory=dict)
    upper_palette: list[dict[str, Any]] = field(default_factory=list)
    lower_palette: list[dict[str, Any]] = field(default_factory=list)
    attribute_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_summary_dict(self) -> dict[str, str]:
        """Notebook에서 바로 읽을 수 있는 상·하의 두 줄 요약을 만든다."""

        def usable(value: str) -> bool:
            blocked = ("분석 보류", "분석 불가", "불확실", "해당 없음")
            return bool(value and not any(marker in value for marker in blocked))

        def description(color: str, item_type: str, features: list[str]) -> str:
            base = " ".join(value for value in (color, item_type) if usable(value)) or "구분 불가"
            unique_features = list(dict.fromkeys(value for value in features if usable(value)))
            return f"{base} ({', '.join(unique_features)})" if unique_features else base

        lower_name = self.lower_subtype if usable(self.lower_subtype) else self.lower_type
        lower_shape = self.pant_leg_shape
        if not usable(lower_shape) and self.attribute_sources.get("lower_fit") in {
            "trained_head", "fused_agreement",
        }:
            lower_shape = self.lower_fit.replace(" 추정", "")
        lower_length = self.pant_length if usable(self.pant_length) else self.bottom_length

        return {
            "상의": description(
                self.upper_color,
                self.upper_type,
                [self.sleeve_length],
            ),
            "하의": description(
                self.lower_color,
                lower_name,
                [lower_shape, lower_length, *self.lower_details[:2]],
            ),
        }


@dataclass
class Product:
    product_id: str
    name: str
    category: str
    color: str
    style: str
    purposes: list[str]
    body_shapes: list[str]
    price: int
    season: str
    stock: bool
    url: str = ""
    item_type: str = ""
    fit: str = ""
    length: str = ""
    pattern: str = "무지"
    material: str = ""
    neckline: str = ""
    formality: int = 3
    activity_tags: list[str] = field(default_factory=list)
    warmth: int = 3
    breathability: int = 3
    water_resistant: bool = False
    visual_weight: int = 3
    detail_level: int = 1
    waistline: str = ""
    pattern_scale: str = ""
    pattern_contrast: int = 0


@dataclass
class Recommendation:
    rank: int
    products: list[Product]
    total_score: float
    score_breakdown: dict[str, float]
    reasons: list[str]
    applied_rules: list[str] = field(default_factory=list)
    score_coverage: float = 0.0
    styling_tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
