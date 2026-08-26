from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


# 체형 이름은 Size Korea 『한국인의 표준체형』의 상반신 분류를 따른다.
# 그 문서는 Rasband(1994)의 8분류를 쓰되, 가슴·허리·엉덩이 '둘레'가 필요한
# 모래시계·마름모꼴·둥근·튜브형은 "추가적인 분석이 필요하다"며 실제로는
# 역삼각·삼각·사각 세 가지로 운용했다(20대 여성: 역삼각 28.8%, 삼각 19.3%, 사각 25.0%).
# 사진에서는 둘레를 잴 수 없으므로 이 세 가지만 사용한다.
#
# 판정 기준도 문서를 따른다. "어깨 폭이 크다면 역삼각형, 어깨 폭에 비해
# 허리가 크다면 삼각체형"이며, 크기는 "표준체형과 비교한 상대값"으로 본다.
# 허리둘레 대신 사진에서 잴 수 있는 골반 폭을 쓴다.
SHAPE_INVERTED_TRIANGLE = "역삼각체형"   # 어깨가 골반보다 넓다
SHAPE_RECTANGLE = "사각체형"             # 어깨와 골반이 비슷하다
SHAPE_TRIANGLE = "삼각체형"              # 골반이 어깨보다 넓다
SHAPE_UNCERTAIN = "분석 불확실"
SHAPE_UNAVAILABLE = "분석 불가"

# 사진으로 판정할 수 있는 세 가지. 사진에는 둘레 정보가 없다.
BODY_SHAPES = (SHAPE_INVERTED_TRIANGLE, SHAPE_RECTANGLE, SHAPE_TRIANGLE)

# 사용자가 가슴·허리·엉덩이 둘레를 입력하면 추가로 판정할 수 있는 체형.
# Size Korea 문서가 "둘레 항목이 필요해 추가 분석이 필요하다"고 미룬 부분이다.
SHAPE_HOURGLASS = "모래시계체형"         # 가슴≈엉덩이, 허리가 뚜렷하게 가늘다
SHAPE_DIAMOND = "마름모꼴체형"           # 허리가 가슴·엉덩이보다 크다
SHAPE_ROUND = "둥근체형"                 # 허리 구분이 거의 없다

CIRCUMFERENCE_SHAPES = (SHAPE_HOURGLASS, SHAPE_DIAMOND, SHAPE_ROUND)

# 상품 카탈로그 body_shapes 칼럼의 어휘. 위의 체형 라벨과 **다른 축**이다.
#   체형 라벨  : 몸이 어떤 형태인가 (역삼각·사각·삼각…)
#   아래 라벨  : 이 상품이 어느 쪽으로 시선을 모으는가
# 둘을 섞어 비교하면 조건이 영영 안 맞아 규칙이 조용히 잠든다. 실제로 이 칼럼은
# 오랫동안 아무도 읽지 않는 죽은 데이터였다.
FOCUS_UPPER = "상체 강조형"
FOCUS_LOWER = "하체 강조형"
FOCUS_BALANCED = "균형형"
CATALOG_FOCUS_LABELS = (FOCUS_UPPER, FOCUS_LOWER, FOCUS_BALANCED)
ALL_BODY_SHAPES = BODY_SHAPES + CIRCUMFERENCE_SHAPES

# 체형을 무엇으로 판정했는지. 화면에 근거를 함께 보여주기 위해 쓴다.
BASIS_PHOTO = "사진 추정"          # 어깨·골반 폭만으로 세 가지 구분
BASIS_MEASUREMENT = "입력한 둘레"   # 사용자가 줄자로 잰 값
BASIS_ESTIMATE = "사진에서 추정한 둘레"  # 3D 체형 복원. 오차가 크므로 구분해 표시한다

# 체형 분석을 추천 점수에 반영할지 정하는 선택지.
# 사용자가 목표를 고를 때만 체형 규칙(R-BOD-*)을 적용한다는 R-KOR-02를 따른다.
GOAL_NONE = "별도 보정 없음"
GOAL_BALANCE = "상·하체 균형 맞추기"
GOAL_LONGER_LEGS = "다리가 길어 보이게"
GOAL_WAISTLINE = "허리선 강조하기"
GOAL_UPPER_FOCUS = "상체에 시선 모으기"
GOAL_LOWER_FOCUS = "하체에 시선 모으기"

# 화면에 보여줄 순서와, 선택지마다 덧붙일 설명.
SILHOUETTE_GOAL_CHOICES = [
    (GOAL_NONE, "별도 보정 없음 (목적·취향만 고려)"),
    (GOAL_BALANCE, GOAL_BALANCE),
    (GOAL_LONGER_LEGS, GOAL_LONGER_LEGS),
    (GOAL_WAISTLINE, GOAL_WAISTLINE),
    (GOAL_UPPER_FOCUS, GOAL_UPPER_FOCUS),
    (GOAL_LOWER_FOCUS, GOAL_LOWER_FOCUS),
]
SILHOUETTE_GOALS = [value for value, _ in SILHOUETTE_GOAL_CHOICES]

# 체형 규칙을 실제로 켜는 목표들.
BODY_SHAPE_GOALS = {GOAL_BALANCE, GOAL_UPPER_FOCUS, GOAL_LOWER_FOCUS}
PROPORTION_GOALS = {GOAL_LONGER_LEGS, GOAL_WAISTLINE}


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
    min_budget: int | None = None
    max_budget: int | None = None
    change_scope: str = "전체 변경"
    height_cm: float | None = None
    weight_kg: float | None = None
    # 둘레를 입력하면 사진으로는 판정할 수 없는 체형까지 분류한다.
    chest_cm: float | None = None
    waist_cm: float | None = None
    hip_cm: float | None = None
    # 사용자가 입력하는 '평소 사이즈' 라벨(예: S/M/L 또는 숫자) — 선택
    usual_top_size: str | None = None
    usual_bottom_size: str | None = None
    season: str = "사계절"
    silhouette_goal: str = GOAL_NONE
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
    gender: str = ""  # "남성"/"여성", 빈 값이면 성별 무관

    @property
    def has_circumferences(self) -> bool:
        """세 둘레가 모두 있어야 체형을 판정할 수 있다."""
        return None not in (self.chest_cm, self.waist_cm, self.hip_cm)

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
    visible_sleeve_length: str = "분석 불가"
    sleeve_state: str = "판단 보류"
    input_valid: bool = True
    input_error_code: str = ""
    input_error_message: str = ""
    layering_state: str = "판단 보류"
    upper_items: list[str] = field(default_factory=list)
    inner_category: str = "해당 없음"
    outer_category: str = "해당 없음"
    wear_state_confidence: dict[str, float] = field(default_factory=dict)
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

        upper_name = self.upper_type
        if self.layering_state in {"레이어드", "레이어드 가능성"} and len(self.upper_items) >= 2:
            upper_name = " + ".join(self.upper_items)
        sleeve_description = self.sleeve_length
        if self.sleeve_state == "걷음 가능성 높음":
            sleeve_description = f"{self.sleeve_length}·소매 걷음"
        elif self.sleeve_state == "좌우 비대칭":
            sleeve_description = f"{self.sleeve_length}·좌우 소매 상태 다름"

        return {
            "상의": description(
                self.upper_color,
                upper_name,
                [sleeve_description],
            ),
            "하의": description(
                self.lower_color,
                lower_name,
                [lower_shape, lower_length, *self.lower_details[:2]],
            ),
        }


@dataclass
class CurrentOutfitEvaluation:
    """현재 착장을 추천 상품 후보와 분리해 평가한 결과."""

    total_score: float
    score_breakdown: dict[str, float]
    reasons: list[str]
    applied_rules: list[str]
    score_coverage: float
    analysis_confidence: float
    reliable: bool
    keep_threshold: float
    should_keep: bool
    verdict: str

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
    # 무신사 카탈로그(musinsa_crawler.py) 확장 필드
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
    applied_rules: list[str] = field(default_factory=list)
    score_coverage: float = 0.0
    styling_tips: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data
