"""사용자 착장과 상품 착용 사진이 공유하는 핏 라벨 계약."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FitVisionTask:
    name: str
    labels: tuple[str, ...]
    minimum_confidence: float = 0.50


FIT_VISION_TASKS = {
    "upper_fit": FitVisionTask(
        "upper_fit", ("슬림핏", "레귤러핏", "여유핏", "오버핏"), 0.55,
    ),
    "upper_length": FitVisionTask(
        "upper_length", ("크롭", "기본 기장", "롱"), 0.55,
    ),
    "bottom_silhouette": FitVisionTask(
        "bottom_silhouette",
        ("스키니", "슬림", "스트레이트", "세미와이드", "와이드", "벌룬", "플레어·부츠컷"),
        0.55,
    ),
    "bottom_length": FitVisionTask(
        "bottom_length", ("쇼츠", "무릎·7부", "크롭·앵클", "풀렝스"), 0.55,
    ),
    "quality": FitVisionTask(
        "quality", ("판정 가능", "가림", "측면·교차 자세", "잘림", "평면 상품", "판정 불가"), 0.60,
    ),
}

CATEGORY_TASKS = {
    "top": ("upper_fit", "upper_length", "quality"),
    "bottom": ("bottom_silhouette", "bottom_length", "quality"),
}

USABLE_QUALITY = "판정 가능"

# 새 세부 클래스를 기존 추천·UI 어휘로 안전하게 연결한다.
BOTTOM_TO_LEGACY_FIT = {
    "스키니": "슬림핏",
    "슬림": "슬림핏",
    "스트레이트": "스트레이트핏",
    "세미와이드": "세미와이드",
    "와이드": "와이드핏",
    "벌룬": "벌룬핏",
    "플레어·부츠컷": "플레어핏",
}
BOTTOM_TO_LEGACY_LENGTH = {
    "쇼츠": "쇼츠·미니 기장",
    "무릎·7부": "미디·7부 기장",
    "크롭·앵클": "크롭·앵클 기장",
    "풀렝스": "롱·긴바지 기장",
}
UPPER_TO_LEGACY_LENGTH = {
    "크롭": "크롭 기장",
    "기본 기장": "기본 기장",
    "롱": "롱 기장",
}


def tasks_for_category(category: str) -> tuple[str, ...]:
    if category not in CATEGORY_TASKS:
        raise ValueError(f"핏 비전 모델이 지원하지 않는 카테고리입니다: {category!r}")
    return CATEGORY_TASKS[category]


def validate_fit_vision_schema() -> None:
    for name, task in FIT_VISION_TASKS.items():
        if not task.labels or len(task.labels) != len(set(task.labels)):
            raise ValueError(f"{name} 핏 라벨이 비었거나 중복되었습니다.")


validate_fit_vision_schema()
