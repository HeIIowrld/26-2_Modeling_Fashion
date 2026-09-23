"""Conservative input policy; clothing visibility is not a body-size estimate.

Uses existing parser/attribute results, without inventing a hidden body outline.
The rules are not calibrated probabilities and cannot certify a true body shape.
"""
from __future__ import annotations

import numpy as np


RETAKE = "몸선을 가리지 않는 상의와 일자 또는 슬림한 바지를 입고 다시 촬영해 주세요. 노출이 많은 옷은 필요하지 않습니다."
LOOSE = ("오버", "여유", "루즈", "와이드", "배기", "벌룬", "플레어")
UNKNOWN = ("불가", "보류", "불확실")


def assess_body_visibility(outfit, parsed: dict) -> dict:
    """Reject clear occluders; abstain when the existing fit evidence is weak.

    No BMI, gender, body width, or assumed 'normal' body shape is used here.
    A mask-only loose-fit label is uncertainty, not proof of oversized clothing.
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
        if skirt_fraction >= 0.01:
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
                reasons.append(f"{name}의 넉넉한 핏이 몸선을 가릴 가능성이 있어 체형 분석을 진행하지 않습니다.")
            else:
                uncertain.append(f"{name} 윤곽과 몸선을 구분하기 어렵습니다. 옷의 폭을 체형으로 사용하지 않습니다.")

    outer = getattr(outfit, "outer_category", "")
    upper = getattr(outfit, "upper_type", "")
    if any(word in f"{outer} {upper}" for word in ("코트", "패딩", "다운", "판초")):
        reasons.append("두꺼운 외투가 몸선을 가려 체형 분석에 적합하지 않습니다.")
    status = "occluded" if reasons else "uncertain" if uncertain else "no_obvious_occlusion"
    return {
        "status": status,
        "passed": status == "no_obvious_occlusion",
        "issues": (reasons + uncertain + [RETAKE]) if reasons or uncertain else [],
        "evidence": evidence,
        "policy_version": "2026-09-23",
        "calibrated": False,
    }


def with_body_visibility(quality: dict, outfit, parsed: dict) -> dict:
    visibility = assess_body_visibility(outfit, parsed)
    return {**quality, "passed": bool(quality["passed"] and visibility["passed"]),
            "body_visibility": visibility,
            "issues": list(dict.fromkeys(quality.get("issues", []) + visibility["issues"]))}
