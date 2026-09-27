"""Conservative admission checks for the offline outfit library.

Geometry is an uncalibrated screening signal, not a physical fit measurement.
An automatic pass still needs an explicit visual review before verified lookup.
This module is deliberately separate from the clothing compatibility scorer.
"""
from __future__ import annotations

import hashlib
import math
from pathlib import Path

import numpy as np

QUALITY_VERSION = 2
UNKNOWN = {"", "분석 불가", "분석 보류", "해당 없음", "스타일 불확실"}


def quality_fingerprint():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def basic_quality_status(reports, input_valid):
    clothing = [r for r in reports if r.get("category") in {"top", "bottom"}]
    if {r.get("category") for r in clothing} != {"top", "bottom"}:
        return "unassessed"
    if any(r.get("skipped") or not r.get("checks") for r in clothing):
        return "unassessed"
    if not input_valid or any(not c.get("passed", False) for r in clothing for c in r["checks"]):
        return "quality_failed"
    shoes = [r for r in reports if r.get("backend") and "outside_mask_preserved" in r]
    if not shoes:
        return "unassessed"
    return "auto_passed" if all(r["outside_mask_preserved"] for r in shoes) else "quality_failed"


def pants_geometry(segmentation):
    """Compare occupied pixels, never the span including the gap between legs.

    Ratios describe the image silhouette. Flat lays and worn photos can differ;
    measurements cannot alone certify a fit. Small disconnected parsing islands
    and clipped hems are excluded.
    """
    import cv2

    seg = np.asarray(segmentation)
    if seg.ndim != 2:
        return {"measurable": False, "reason": "missing_segmentation"}
    mask = (seg == 6).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if count < 2 or stats[1:, cv2.CC_STAT_AREA].max() < 100:
        return {"measurable": False, "reason": "missing_pants"}
    largest = stats[1:, cv2.CC_STAT_AREA].max()
    mask = np.isin(labels, [i for i in range(1, count)
                           if stats[i, cv2.CC_STAT_AREA] >= largest * .05])
    ys = np.flatnonzero(mask.any(axis=1))
    top, bottom = int(ys[0]), int(ys[-1])
    if bottom >= seg.shape[0] - 2 or bottom - top < 40:
        return {"measurable": False, "reason": "clipped_or_small_pants"}
    widths = mask.sum(axis=1)

    def band(a, b):
        values = widths[top + round(a * (bottom-top)):top + round(b * (bottom-top))]
        return float(np.median(values)) if len(values) and np.all(values > 0) else 0.

    thigh, shin, hem = band(.35, .50), band(.65, .78), band(.86, .94)
    if min(thigh, shin, hem) < 6:
        return {"measurable": False, "reason": "occluded_pants"}
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask, dtype=np.uint8)
    cv2.drawContours(filled, contours, -1, 1, thickness=cv2.FILLED)
    holes = (filled > 0) & np.isin(seg, (14, 16))
    return {"measurable": True, "hem_thigh": round(hem / thigh, 5),
            "hem_shin": round(hem / shin, 5), "top": top, "bottom": bottom,
            "thigh_pixels": thigh, "hem_pixels": hem,
            "skin_hole_fraction": round(float(holes.sum() / max(1, filled.sum())), 6)}


def garment_family(item_type):
    if any(s in item_type for s in ("치마", "스커트")):
        return "skirt"
    if any(s in item_type for s in ("원피스", "드레스", "점프수트")):
        return "dress"
    if any(s in item_type for s in ("쇼츠", "반바지")):
        return "shorts"
    if any(s in item_type for s in ("팬츠", "바지", "슬랙스", "진", "레깅스")):
        return "pants"
    return "unknown"


def preservation_checks(products, observed, segmentation, target_geometry):
    """Screen semantic conflicts and substantial silhouette drift; fail unknowns closed."""
    bottom = next(p for p in products if p["category"] == "bottom")
    top = next(p for p in products if p["category"] == "top")
    checks = []

    def add(name, passed, **evidence):
        checks.append({"name": name, "passed": passed, **evidence})

    expected = garment_family(bottom.get("item_type", ""))
    actual = garment_family(observed.get("lower_type", ""))
    add("bottom_family", None if "unknown" in (expected, actual) else expected == actual,
        expected=expected, actual=actual)
    subtype = observed.get("lower_subtype", "")
    target_kind = bottom.get("item_type", "")
    # Subtype heads are only a veto for obvious conflicts, never proof of a good fit.
    structured = any(word in target_kind for word in ("치노", "슬랙스", "청바지", "데님"))
    conflict = structured and any(word in subtype for word in ("하렘", "조거", "트랙"))
    add("bottom_subtype_conflict", not conflict, expected=target_kind, actual=subtype,
        source=observed.get("attribute_sources", {}).get("lower_subtype", "unknown"))
    measured = pants_geometry(segmentation)
    if expected == "pants":
        if not target_geometry.get("measurable") or not measured.get("measurable"):
            add("pants_silhouette", None, target=target_geometry, observed=measured)
        else:
            retention = measured["hem_thigh"] / target_geometry["hem_thigh"]
            cuff_retention = measured["hem_shin"] / target_geometry["hem_shin"]
            add("pants_silhouette", .75 <= retention <= 1.35 and .75 <= cuff_retention <= 1.35,
                hem_thigh_retention=round(retention, 5), hem_shin_retention=round(cuff_retention, 5),
                target=target_geometry, observed=measured,
                thresholds={"min": .75, "max": 1.35}, calibration="provisional_requires_visual_review")
            hole_limit = target_geometry.get("skin_hole_fraction", 0) + .005
            add("pants_skin_holes", measured["skin_hole_fraction"] <= hole_limit,
                actual=measured["skin_hole_fraction"], threshold=hole_limit)
    else:
        add("pants_silhouette", None, reason="shape_gate_not_validated_for_this_family")
    tee = top.get("item_type") == "티셔츠"
    upper = observed.get("upper_type", "")
    if tee:
        add("top_family", None if upper in UNKNOWN else upper == "티셔츠", expected="티셔츠", actual=upper)
        layering = observed.get("layering_state", "")
        add("top_layering", None if layering in UNKNOWN or layering == "판단 보류" else layering == "단일 상의",
            actual=layering)
    return checks


def assess_preservation(products, observed, segmentation, target_geometry, reports):
    checks = preservation_checks(products, observed, segmentation, target_geometry)
    basic = basic_quality_status(reports, observed.get("input_valid", False))
    failed = [c["name"] for c in checks if c["passed"] is False]
    unassessed = [c["name"] for c in checks if c["passed"] is None]
    status = ("quality_failed" if basic == "quality_failed" or failed else
              "unassessed" if basic == "unassessed" or unassessed else "auto_passed")
    return {"version": QUALITY_VERSION, "fingerprint": quality_fingerprint(), "status": status,
            "basic_status": basic, "checks": checks, "failed": failed, "unassessed": unassessed,
            "visual_review_required": True}


def reference_rank(record, products, target_geometry, segmentation):
    """Return a compatibility penalty or reasons to reject a reference.

    Exact body/gender/proportion filtering is done by the caller. The rank never
    substitutes another body bucket when no compatible photograph exists.
    """
    pose, outfit = record.get("pose") or {}, record.get("outfit") or {}
    reasons = []
    if not pose.get("valid") or pose.get("body_shape_confidence", 0) < .65:
        reasons.append("unreliable_pose")
    if record.get("feet_error"):
        reasons.append("hidden_feet")
    if not outfit.get("input_valid", False):
        reasons.append("invalid_outfit")
    if outfit.get("layering_state") != "단일 상의":
        reasons.append("layered_or_unknown_top")
    if outfit.get("upper_type") not in {"티셔츠", "셔츠", "블라우스", "니트", "스웨터", "민소매"}:
        reasons.append("outerwear_or_unsupported_top")
    bottom = next(p for p in products if p["category"] == "bottom")
    family = garment_family(bottom.get("item_type", ""))
    if family != garment_family(outfit.get("lower_type", "")):
        reasons.append("different_bottom_family")
    if np.isin(segmentation, (4, 5)).sum() > max(100, (segmentation == 6).sum() * .10):
        reasons.append("dress_or_skirt_overlap")
    measured = pants_geometry(segmentation)
    distance = 0.
    if family == "pants":
        if not target_geometry.get("measurable") or not measured.get("measurable"):
            reasons.append("unmeasurable_pants")
        else:
            ratios = [measured[k] / target_geometry[k] for k in ("hem_thigh", "hem_shin")]
            if any(r < .75 or r > 1.35 for r in ratios):
                reasons.append("incompatible_pants_silhouette")
            if measured.get("skin_hole_fraction", 0) > target_geometry.get("skin_hole_fraction", 0) + .005:
                reasons.append("distressed_pants_skin_holes")
            distance = sum(abs(math.log(r)) for r in ratios)
        if outfit.get("bottom_length") not in {"긴바지", "롱·긴바지 기장", "풀렝스"}:
            reasons.append("non_full_length_pants")
        subtype = outfit.get("lower_subtype", "")
        if any(w in subtype for w in ("하렘", "조거", "트랙")) and not any(w in bottom.get("item_type", "") for w in ("하렘", "조거", "트랙")):
            reasons.append("incompatible_pants_subtype")
    else:
        reasons.append("unsupported_reference_family")
    penalty = distance + (0 if outfit.get("upper_type") == "티셔츠" else .4) + (1 - pose.get("full_body_score", 0))
    return {"eligible": not reasons, "reasons": reasons, "penalty": round(penalty, 6), "geometry": measured}


def render_is_eligible(record):
    quality = record.get("preservation", {})
    review = record.get("visual_review", {})
    return bool(record.get("status") == "complete"
                and quality.get("version") == QUALITY_VERSION
                and quality.get("fingerprint") == quality_fingerprint()
                and quality.get("status") == "auto_passed"
                and isinstance(review, dict) and review.get("decision") == "accepted"
                and review.get("output_sha256") == record.get("output_sha256")
                and record.get("output_sha256") and record.get("render_fingerprint"))
