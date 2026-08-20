from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class UserProfile:
    purpose: str = "데일리"
    desired_style: str = "캐주얼"
    budget: int = 150_000
    change_scope: str = "전체 변경"
    height_cm: float | None = None
    weight_kg: float | None = None
    season: str = "사계절"
    gender: str = ""  # "남성"/"여성", 빈 값이면 성별 무관

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
    sleeve_length: str = "분석 불가"
    upper_length: str = "분석 불가"
    bottom_length: str = "분석 불가"
    fit: str = "분석 불가"
    neckline: str = "분석 보류"
    pattern: str = "분석 보류"
    material: str = "분석 보류"
    attribute_confidence: float = 0.0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


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
    brand: str = ""
    gender: str = ""
    image_url: str = ""
    image_path: str = ""


@dataclass
class Recommendation:
    rank: int
    products: list[Product]
    total_score: float
    score_breakdown: dict[str, float]
    reasons: list[str]

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
