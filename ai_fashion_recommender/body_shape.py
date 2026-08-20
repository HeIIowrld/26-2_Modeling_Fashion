"""입력받은 둘레로 체형을 판정한다.

사진에서는 둘레를 잴 수 없어 어깨·골반 폭만으로 역삼각·사각·삼각 세 가지만 구분한다.
사용자가 가슴·허리·엉덩이 둘레를 입력하면 Size Korea 문서가 "둘레 항목이 필요해
추가 분석이 필요하다"며 미룬 모래시계·마름모꼴·둥근체형까지 판정할 수 있다.

임계값은 `data/body_shape_reference.json`에서 읽는다. Rasband의 분류는 서술적 정의라
출처마다 수치가 다르므로, 지금 값은 정의를 비율로 옮긴 잠정값이고 검수가 필요하다.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from schemas import (
    BASIS_MEASUREMENT,
    SHAPE_DIAMOND,
    SHAPE_HOURGLASS,
    SHAPE_INVERTED_TRIANGLE,
    SHAPE_RECTANGLE,
    SHAPE_ROUND,
    SHAPE_TRIANGLE,
)

REFERENCE_PATH = Path(__file__).resolve().parent / "data" / "body_shape_reference.json"

DEFAULT_THRESHOLDS = {
    "balanced_tolerance": 0.05,
    "defined_waist_ratio": 0.75,
    "undefined_waist_ratio": 0.90,
}


@lru_cache(maxsize=1)
def thresholds() -> dict[str, float]:
    if not REFERENCE_PATH.is_file():
        return dict(DEFAULT_THRESHOLDS)
    rules = json.loads(REFERENCE_PATH.read_text(encoding="utf-8")).get("circumference_rules")
    if not rules:
        return dict(DEFAULT_THRESHOLDS)
    return {**DEFAULT_THRESHOLDS, **rules.get("thresholds", {})}


def classify_from_circumferences(
    chest_cm: float, waist_cm: float, hip_cm: float
) -> tuple[str, str]:
    """(체형, 판정 근거)를 돌려준다. 값이 이상하면 ValueError를 올린다."""
    values = {"가슴": chest_cm, "허리": waist_cm, "엉덩이": hip_cm}
    for name, value in values.items():
        if value is None or value <= 0:
            raise ValueError(f"{name} 둘레는 0보다 커야 합니다.")

    limits = thresholds()
    tolerance = limits["balanced_tolerance"]
    balanced = abs(chest_cm - hip_cm) / max(chest_cm, hip_cm) <= tolerance

    # 순서가 곧 우선순위다. 데이터 파일의 order와 같아야 한다.
    if waist_cm >= chest_cm and waist_cm >= hip_cm:
        return SHAPE_DIAMOND, BASIS_MEASUREMENT
    if (
        waist_cm / chest_cm >= limits["undefined_waist_ratio"]
        and waist_cm / hip_cm >= limits["undefined_waist_ratio"]
    ):
        return SHAPE_ROUND, BASIS_MEASUREMENT
    if balanced and waist_cm / chest_cm <= limits["defined_waist_ratio"]:
        return SHAPE_HOURGLASS, BASIS_MEASUREMENT
    if (chest_cm - hip_cm) / hip_cm >= tolerance:
        return SHAPE_INVERTED_TRIANGLE, BASIS_MEASUREMENT
    if (hip_cm - chest_cm) / chest_cm >= tolerance:
        return SHAPE_TRIANGLE, BASIS_MEASUREMENT
    return SHAPE_RECTANGLE, BASIS_MEASUREMENT


def classify(profile, pose, *, person_image=None, estimator=None) -> tuple[str, str]:
    """체형을 정한다. 근거가 확실한 순서대로 시도한다.

    1. 사용자가 입력한 둘레 — 줄자로 잰 값이라 가장 정확하다.
    2. 사진에서 추정한 둘레 — 모델이 연결되어 있을 때만. 오차가 커서 추정으로 표시한다.
    3. 사진의 어깨·골반 폭 — 둘레가 없어 세 가지까지만 구분한다.
    """
    from schemas import BASIS_ESTIMATE, BASIS_PHOTO

    if getattr(profile, "has_circumferences", False):
        try:
            return classify_from_circumferences(profile.chest_cm, profile.waist_cm, profile.hip_cm)
        except ValueError:
            pass

    if person_image is not None:
        from body_measure import BodyMeasurementEstimator, MeasurementNotReady

        try:
            measured = (estimator or BodyMeasurementEstimator()).estimate(
                person_image, pose, height_cm=getattr(profile, "height_cm", None)
            )
            # 분류는 세 부위의 비율로 갈리므로 둘레 대신 폭을 넣어도 결과가 같다.
            # 폭은 정면 사진에서 직접 잰 값이라 둘레 환산보다 가정이 적다.
            shape, _ = classify_from_circumferences(
                measured.chest_width, measured.waist_width, measured.hip_width
            )
            return shape, BASIS_ESTIMATE
        except (MeasurementNotReady, ValueError):
            pass

    return pose.body_shape, BASIS_PHOTO
