from __future__ import annotations

from dataclasses import dataclass, field

from fashion_attribute_model import AttributePrediction


@dataclass(frozen=True)
class SleeveStateResult:
    designed_length: str
    visible_length: str
    state: str
    confidence: float
    reason: str
    requires_retake: bool = False
    error_code: str = ""
    error_message: str = ""


@dataclass(frozen=True)
class LayeringResult:
    state: str
    upper_items: list[str] = field(default_factory=list)
    inner_category: str = "해당 없음"
    outer_category: str = "해당 없음"
    confidence: float = 0.0
    reason: str = ""


def infer_sleeve_state(
    measurements: dict,
    learned: AttributePrediction | None,
    fallback_length: str,
    *,
    pose_landmarks: dict[str, tuple[float, float, float]] | None = None,
) -> SleeveStateResult:
    """설계상 소매 길이와 사진에서 보이는 길이를 분리한다."""
    visible = str(measurements.get("visible_sleeve_length") or fallback_length)
    ratio = float(measurements.get("sleeve_coverage_ratio", 0.0))
    side_values = measurements.get("sleeve_side_coverage") or {}
    left = float(side_values.get("left", ratio))
    right = float(side_values.get("right", ratio))
    learned_label = (
        learned.labels[0]
        if learned is not None and learned.accepted and learned.labels
        else ""
    )
    learned_confidence = float(learned.confidence) if learned_label else 0.0
    designed = learned_label or fallback_length

    pose_landmarks = pose_landmarks or {}
    arm_landmarks = (
        "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
        "left_wrist", "right_wrist",
    )
    arm_visibility = min(
        (pose_landmarks.get(name, (0.0, 0.0, 0.0))[2] for name in arm_landmarks),
        default=0.0,
    )
    both_arms_visible = arm_visibility >= 0.60

    # 어깨→팔꿈치→손목 경로에서 팔꿈치는 약 0.5다. 마스크가 0.58 이하에서
    # 끝나면 팔꿈치 부근까지 올라온 상태로 본다. 정상 반팔을 막지 않기 위해
    # 설계 길이가 긴팔이라는 강한 근거가 있거나 좌우 차이가 매우 클 때만
    # 재촬영 게이트를 작동시킨다.
    elbow_level = 0.58
    excessive_side = min(left, right) <= elbow_level
    strong_asymmetry = abs(left - right) >= 0.20 and max(left, right) >= 0.72
    confident_long_sleeve = learned_label == "긴팔" and learned_confidence >= 0.65
    if both_arms_visible and excessive_side and (confident_long_sleeve or strong_asymmetry):
        asymmetric = strong_asymmetry
        reason = (
            "좌우 소매 길이 차이가 크고 한쪽 소매 끝이 팔꿈치 부근에 있음"
            if asymmetric
            else "긴팔 예측과 달리 소매 끝이 팔꿈치 부근에 있음"
        )
        message = (
            "한쪽 소매가 팔꿈치 부근까지 올라가 원래 소매 길이를 정확하게 분석할 수 없습니다. "
            "양쪽 소매를 손목까지 내리고 다시 촬영해주세요."
            if asymmetric
            else
            "소매가 팔꿈치 부근까지 올라가 원래 소매 길이를 정확하게 분석할 수 없습니다. "
            "양쪽 소매를 손목까지 내리고 다시 촬영해주세요."
        )
        return SleeveStateResult(
            "긴팔" if confident_long_sleeve else designed,
            visible,
            "재촬영 필요",
            max(learned_confidence, 0.70 if asymmetric else 0.65),
            reason,
            requires_retake=True,
            error_code="SLEEVE_ROLLUP_RETAKE_REQUIRED",
            error_message=message,
        )

    if learned_label == "긴팔" and 0.48 <= ratio < 0.83:
        if both_arms_visible and abs(left - right) >= 0.20:
            return SleeveStateResult(
                "긴팔", visible, "좌우 비대칭", learned_confidence,
                "긴팔 예측과 좌우 소매 노출 길이가 서로 다름",
            )
        return SleeveStateResult(
            "긴팔", visible, "걷음 가능성 높음", learned_confidence,
            "긴팔 예측이지만 사진에서 보이는 소매 끝은 7부 구간",
        )
    if learned_label == "긴팔" and ratio < 0.48:
        return SleeveStateResult(
            "긴팔", visible, "가림 또는 불확실", learned_confidence,
            "긴팔 예측과 매우 짧은 가시 길이가 충돌함",
        )
    if both_arms_visible and abs(left - right) >= 0.20:
        return SleeveStateResult(
            designed, visible, "좌우 비대칭", max(learned_confidence, 0.55),
            "좌우 소매 마스크의 끝점 차이가 큼",
        )
    if learned_label and learned_label == visible:
        return SleeveStateResult(
            designed, visible, "정상", learned_confidence,
            "학습 헤드의 길이와 사진에서 보이는 길이가 일치함",
        )
    if visible == "긴팔" and ratio >= 0.83:
        return SleeveStateResult(
            designed, visible, "정상", max(learned_confidence, 0.55),
            "소매 마스크가 손목 부근까지 이어짐",
        )
    return SleeveStateResult(
        designed, visible, "판단 보류", learned_confidence,
        "원래 소매 길이와 착용 상태를 분리할 근거가 부족함",
    )


def infer_layering_state(
    category: str,
    collar: str,
    neckline: str,
    material: str,
    *,
    trained_prediction=None,
    zero_shot_label: str = "",
    zero_shot_confidence: float = 0.0,
) -> LayeringResult:
    """단일 상의 결과의 속성 모순과 ROI 상대점수로 레이어드를 보조 판정한다."""
    if trained_prediction is not None and trained_prediction.accepted:
        if trained_prediction.state == "단일 상의":
            return LayeringResult(
                state="단일 상의",
                upper_items=[category] if category else [],
                confidence=trained_prediction.confidence,
                reason="학습된 멀티 ROI 레이어드 헤드가 단일 상의로 판정함",
            )
        if trained_prediction.state == "레이어드":
            inner = trained_prediction.inner_category
            outer = trained_prediction.outer_category
            items = [value for value in (inner, outer) if value and value != "종류 불확실"]
            if len(items) < 2:
                items = ["안쪽 상의(종류 불확실)", outer if outer != "종류 불확실" else category]
            return LayeringResult(
                state="레이어드",
                upper_items=items,
                inner_category=inner,
                outer_category=outer,
                confidence=trained_prediction.confidence,
                reason="목·커프스·밑단·앞여밈 멀티 ROI 학습 헤드가 레이어드로 판정함",
            )
    knit_category = category in {"니트", "베스트", "가디건"}
    knit_material = "니트" in material
    shirt_collar = collar in {"셔츠 칼라", "폴로 칼라"}
    vneck_with_collar = neckline == "V넥" and shirt_collar
    # 소재 헤드 하나가 니트로 오인한 실제 폴로셔츠까지 레이어드로 만들지 않는다.
    structural_conflict = shirt_collar and (
        knit_category or (knit_material and neckline == "V넥")
    )
    zero_shot_layered = zero_shot_label == "레이어드" and zero_shot_confidence >= 0.68

    if structural_conflict or (vneck_with_collar and zero_shot_confidence >= 0.50):
        outer = category if category in {"니트", "베스트", "가디건"} else "니트·베스트 계열"
        return LayeringResult(
            state="레이어드",
            upper_items=["셔츠", outer],
            inner_category="셔츠",
            outer_category=outer,
            confidence=max(zero_shot_confidence, 0.75 if structural_conflict else 0.65),
            reason="니트 계열 상의와 셔츠형 칼라가 동시에 관찰됨",
        )
    if zero_shot_layered:
        outer = category if category not in {"분석 불가", "상의"} else "바깥 상의(종류 불확실)"
        return LayeringResult(
            state="레이어드 가능성",
            upper_items=["안쪽 상의(종류 불확실)", outer],
            inner_category="종류 불확실",
            outer_category=outer,
            confidence=zero_shot_confidence,
            reason="상의 전체·목 영역의 FashionSigLIP 레이어드 상대점수가 높음",
        )
    weak_conflict = vneck_with_collar or (
        shirt_collar and zero_shot_label == "레이어드" and zero_shot_confidence >= 0.52
    )
    if weak_conflict:
        return LayeringResult(
            state="판단 보류",
            upper_items=[category] if category else [],
            confidence=max(zero_shot_confidence, 0.50),
            reason="칼라·넥라인 단서가 단일 상의 분류와 충돌함",
        )
    return LayeringResult(
        state="단일 상의",
        upper_items=[category] if category else [],
        confidence=max(0.0, 1.0 - zero_shot_confidence),
        reason="복수 의류를 지지하는 충분한 단서가 없음",
    )
