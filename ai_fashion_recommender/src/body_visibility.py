"""Conservative input policy; clothing visibility is not a body-size estimate.

Uses existing parser/attribute results, without inventing a hidden body outline.
The rules are not calibrated probabilities and cannot certify a true body shape.

실제 사진으로 신호별 성능을 재고(2026-09-23, Fashionpedia 사람 핏 라벨 464장,
reports/body_visibility_2026-09-23.md) 차단 신호를 정했다.

| 신호 | 몸선 드러나는 사진 오차단 | 헐렁한 옷 탐지 | 치마·원피스 탐지 |
| --- | --- | --- | --- |
| 치마·원피스 면적 1% | 1% | 2% | 98% |
| 학습 헤드가 지지하는 넉넉한 핏 | 8% | 38% | 30% |
| 두꺼운 외투 이름 | 9% | 9% | 3% |
| 마스크 폭만 근거인 넉넉한 핏 | **92%** | 79% | 83% |

마지막 신호는 정상 사진의 92%를 막으면서 헐렁한 옷(79%)과 구분하지 못한다. 그래서 차단하지
않고 경고로 남긴다. 로컬 서비스 실험에서는 넉넉한 핏이 일반적인 사용자 사진을 지나치게 많이
막는 문제를 먼저 확인하기 위해, 학습 헤드가 지지한 넉넉한 상·하의도 차단 대신 경고로 낮춘다.
옷에 따른 왜곡(허리 +6%)은 실제로 체형 분류를 25% 뒤집으므로 결과는 참고값으로 표시한다.
치마·원피스와 두꺼운 외투 차단은 유지한다. 측정 잡음 수준(+2%)에서도 8%가 뒤집히므로 통과
자체가 정확도 보장은 아니다.
"""
from __future__ import annotations

import numpy as np


RETAKE = "몸선을 가리지 않는 상의와 일자 또는 슬림한 바지를 입고 다시 촬영해 주세요. 노출이 많은 옷은 필요하지 않습니다."
UNCERTAIN_NOTE = "옷 때문에 체형 판정이 정확하지 않을 수 있습니다. 실제 둘레를 입력하면 그 값을 우선 사용합니다."
LOOSE = ("오버", "여유", "루즈", "와이드", "배기", "벌룬", "플레어")
UNKNOWN = ("불가", "보류", "불확실")
SKIRT_FRACTION_LIMIT = 0.01


def assess_body_visibility(outfit, parsed: dict) -> dict:
    """Reject clear occluders; warn (but continue) when the fit evidence is weak.

    No BMI, gender, body width, or assumed 'normal' body shape is used here.
    A mask-only loose-fit label is uncertainty, not proof of oversized clothing,
    and blocking on it rejected 92% of photos whose clothes do show the body line.
    """
    reasons, uncertain, evidence = [], [], []
    seg = np.asarray(parsed.get("segmentation"))
    if parsed.get("backend") != "fashn-human-parser" or seg.ndim != 2 or not np.any(seg):
        uncertain.append("옷과 몸의 경계를 확인하지 못했습니다.")
    else:
        person_area = int(np.count_nonzero(seg))
        # Ignore isolated parser speckles; record the fraction for later auditing.
        skirt_fraction = float(np.count_nonzero(np.isin(seg, (4, 5))) / person_area)
        evidence.append({"source": "parser", "skirt_or_dress_fraction": round(skirt_fraction, 4)})
        if skirt_fraction >= SKIRT_FRACTION_LIMIT:
            reasons.append("치마·원피스가 골반과 다리 윤곽을 가려 사진 기반 체형 분석에 적합하지 않습니다.")

    sources = getattr(outfit, "attribute_sources", {})
    for key, name in (("fit", "상의"), ("lower_fit", "하의")):
        value = getattr(outfit, key, "")
        source = sources.get(key, "mask")
        evidence.append({"region": key, "label": value, "source": source})
        if not value or any(word in value for word in UNKNOWN):
            uncertain.append(f"{name}가 몸선을 얼마나 가리는지 확인하지 못했습니다.")
        elif any(word in value for word in LOOSE):
            if source in {"trained_head", "fused_agreement"}:
                uncertain.append(
                    f"{name}의 넉넉한 핏이 몸선을 가릴 수 있어 체형 결과를 참고값으로만 제공합니다."
                )
            else:
                uncertain.append(f"{name} 윤곽과 몸선을 구분하기 어렵습니다. 옷의 폭을 체형으로 사용하지 않습니다.")

    outer = getattr(outfit, "outer_category", "")
    upper = getattr(outfit, "upper_type", "")
    if any(word in f"{outer} {upper}" for word in ("코트", "패딩", "다운", "판초")):
        reasons.append("두꺼운 외투가 몸선을 가려 체형 분석에 적합하지 않습니다.")
    status = "occluded" if reasons else "uncertain" if uncertain else "no_obvious_occlusion"
    return {
        "status": status,
        # 근거가 약한 '보류'는 막지 않는다. 막으면 정상 사진 대부분이 거절된다(위 표).
        "passed": status != "occluded",
        "issues": (reasons + [RETAKE]) if reasons else [],
        "warnings": (uncertain + [UNCERTAIN_NOTE]) if uncertain and not reasons else [],
        "evidence": evidence,
        "policy_version": "2026-09-25-loose-fit-warning",
        "calibrated": False,
    }


def with_body_visibility(quality: dict, outfit, parsed: dict) -> dict:
    visibility = assess_body_visibility(outfit, parsed)
    return {**quality, "passed": bool(quality["passed"] and visibility["passed"]),
            "body_visibility": visibility,
            "issues": list(dict.fromkeys(quality.get("issues", []) + visibility["issues"])),
            "warnings": list(dict.fromkeys(quality.get("warnings", []) + visibility["warnings"]))}
