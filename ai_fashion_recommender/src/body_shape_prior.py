"""Pose-aligned hidden-body prior for virtual try-on.

This module does not recover a person's true measurements.  It builds a
conservative 2D corridor from visible joints, the parser result and the
observed garment fit.  The corridor is useful for deciding when the source
garment silhouette must be erased before a narrower garment is generated.

``central_mask`` is the most plausible pose-aligned region,
``plausible_mask`` is a wider uncertainty envelope, and ``confidence_map`` is
an uncalibrated per-pixel confidence map.  None of them should be presented as
an actual body measurement.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

import cv2
import numpy as np


_UNKNOWN = ("", "분석 불가", "분석 보류", "불확실", "해당 없음")

_TOP_LEVELS = {
    "슬림": 0,
    "타이트": 0,
    "머슬": 0,
    "레귤러": 1,
    "기본": 1,
    "여유": 2,
    "루즈": 2,
    "오버": 3,
}
_BOTTOM_LEVELS = {
    "스키니": 0,
    "슬림": 0,
    "테이퍼드": 0,
    "스트레이트": 1,
    "일자": 1,
    "레귤러": 1,
    "세미와이드": 2,
    "세미 와이드": 2,
    "와이드": 3,
    "배기": 3,
    "벌룬": 3,
    "플레어": 3,
    "부츠컷": 3,
}

_TOP_NAME = (
    (0, re.compile(r"슬림|타이트|머슬|바디콘|slim|tight|muscle|bodycon", re.I)),
    (3, re.compile(r"오버|oversi[sz]ed|over[ -]?fit", re.I)),
    (2, re.compile(r"루즈|여유|loose|relaxed", re.I)),
    (1, re.compile(r"레귤러|regular|standard", re.I)),
)
_BOTTOM_NAME = (
    (0, re.compile(r"스키니|슬림|테이퍼드|레깅스|제깅스|skinny|slim|tapered|legging|jegging", re.I)),
    (2, re.compile(r"세미\s*와이드|semi[ -]?wide", re.I)),
    (3, re.compile(r"와이드|배기|벌룬|플레어|부츠컷|wide|baggy|balloon|flare|bootcut", re.I)),
    (1, re.compile(r"스트레이트|일자|레귤러|straight|regular|standard", re.I)),
)


@dataclass(frozen=True)
class BodyShapePrior:
    """A conservative, uncalibrated body-location estimate."""

    category: str
    fit_label: str
    fit_level: int | None
    source: str
    central_mask: np.ndarray
    plausible_mask: np.ndarray
    confidence_map: np.ndarray
    confidence: float
    landmarks_used: tuple[str, ...]

    def to_diagnostic(self) -> dict:
        pixels = max(1, int(self.plausible_mask.size))
        return {
            "category": self.category,
            "fit_label": self.fit_label,
            "fit_level": self.fit_level,
            "source": self.source,
            "confidence": round(float(self.confidence), 4),
            "central_fraction": round(float(self.central_mask.sum()) / pixels, 6),
            "plausible_fraction": round(float(self.plausible_mask.sum()) / pixels, 6),
            "landmarks_used": list(self.landmarks_used),
            "calibrated": False,
        }


def _usable(value: str | None) -> bool:
    text = (value or "").strip()
    return bool(text) and not any(marker and marker in text for marker in _UNKNOWN[1:])


def fit_level(category: str, label: str | None) -> int | None:
    """Map project and shopping fit vocabulary to a narrow-to-wide level."""
    text = (label or "").replace(" ", "")
    levels = _TOP_LEVELS if category == "top" else _BOTTOM_LEVELS
    # Match specific labels before their substrings (세미와이드 before 와이드).
    for token, level in sorted(levels.items(), key=lambda item: len(item[0]), reverse=True):
        if token.replace(" ", "") in text:
            return level
    patterns = _TOP_NAME if category == "top" else _BOTTOM_NAME
    return next((level for level, pattern in patterns if pattern.search(label or "")), None)


def current_fit_label(outfit, category: str) -> tuple[str, str]:
    """Return the outermost observed fit label and its evidence source."""
    if outfit is None:
        return "", ""
    sources = getattr(outfit, "attribute_sources", {}) or {}
    if category == "top":
        return getattr(outfit, "fit", "") or "", sources.get("fit", "")
    detailed = getattr(outfit, "pant_leg_shape", "") or ""
    if _usable(detailed):
        return detailed, sources.get("pant_leg_shape", sources.get("lower_fit", ""))
    return getattr(outfit, "lower_fit", "") or "", sources.get("lower_fit", "")


def target_fit_label(product, category: str) -> str:
    """Prefer structured product fit, then conservatively parse the product name."""
    structured = getattr(product, "fit", "") or ""
    if fit_level(category, structured) is not None:
        return structured
    name = getattr(product, "name", "") or ""
    patterns = _TOP_NAME if category == "top" else _BOTTOM_NAME
    for _level, pattern in patterns:
        match = pattern.search(name)
        if match:
            return match.group(0)
    return ""


def fit_transition_needed(category: str, source_fit: str, target_fit: str) -> bool:
    """Whether a narrower target needs the old garment boundary erased."""
    source = fit_level(category, source_fit)
    target = fit_level(category, target_fit)
    # Only narrow/straight targets need this route.  Wide-to-semiwide still has
    # enough intended garment volume that the validated native policies remain safer.
    return source is not None and target is not None and target <= 1 and source > target


def _point(points: dict[str, tuple[float, float]], name: str) -> tuple[int, int] | None:
    value = points.get(name)
    if value is None or len(value) < 2 or not np.isfinite(value[:2]).all():
        return None
    return int(round(value[0])), int(round(value[1]))


def _main_person_mask(segmentation: np.ndarray, anchors: list[tuple[int, int]]) -> np.ndarray:
    person = np.asarray(segmentation) != 0
    count, labels = cv2.connectedComponents(person.astype(np.uint8), connectivity=8)
    if count <= 2:
        return person
    selected: set[int] = set()
    radius = max(2, round(min(person.shape) * 0.025))
    rows, cols = np.ogrid[:person.shape[0], :person.shape[1]]
    for x, y in anchors:
        near = (rows - y) ** 2 + (cols - x) ** 2 <= radius ** 2
        selected.update(int(item) for item in np.unique(labels[near & person]) if item)
    if not selected:
        areas = np.bincount(labels.ravel())[1:]
        selected = {1 + int(np.argmax(areas))}
    return np.isin(labels, sorted(selected))


def _top_core(shape: tuple[int, int], points: dict[str, tuple[float, float]]) -> tuple[np.ndarray, tuple[str, ...]] | None:
    required = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    anchors = [_point(points, name) for name in required]
    if any(value is None for value in anchors):
        return None
    mask = np.zeros(shape, np.uint8)
    polygon = cv2.convexHull(np.asarray(anchors, np.int32))
    cv2.fillConvexPoly(mask, polygon, 1)
    shoulder_span = float(np.linalg.norm(np.asarray(anchors[0]) - np.asarray(anchors[1])))
    used = list(required)
    for side in ("left", "right"):
        chain = [_point(points, f"{side}_{joint}") for joint in ("shoulder", "elbow", "wrist")]
        if chain[1] is not None:
            used.append(f"{side}_elbow")
            cv2.line(mask, chain[0], chain[1], 1, max(3, round(shoulder_span * 0.18)))
        if chain[1] is not None and chain[2] is not None:
            used.append(f"{side}_wrist")
            cv2.line(mask, chain[1], chain[2], 1, max(2, round(shoulder_span * 0.13)))
    return mask.astype(bool), tuple(used)


def _bottom_core(shape: tuple[int, int], points: dict[str, tuple[float, float]]) -> tuple[np.ndarray, tuple[str, ...]] | None:
    required = tuple(f"{side}_{joint}" for side in ("left", "right") for joint in ("hip", "knee", "ankle"))
    anchors = {name: _point(points, name) for name in required}
    if any(value is None for value in anchors.values()):
        return None
    mask = np.zeros(shape, np.uint8)
    hip_span = float(np.linalg.norm(np.asarray(anchors["left_hip"]) - np.asarray(anchors["right_hip"])))
    cv2.line(mask, anchors["left_hip"], anchors["right_hip"], 1, max(4, round(hip_span * 0.38)))
    for side in ("left", "right"):
        hip, knee, ankle = (anchors[f"{side}_{joint}"] for joint in ("hip", "knee", "ankle"))
        cv2.line(mask, hip, knee, 1, max(4, round(hip_span * 0.30)))
        cv2.line(mask, knee, ankle, 1, max(3, round(hip_span * 0.20)))
    return mask.astype(bool), required


def estimate_body_shape_prior(
    segmentation: np.ndarray,
    landmarks_px: dict[str, tuple[float, float]],
    category: str,
    fit_label: str,
    *,
    fit_source: str = "",
    layering_state: str = "",
    observed_mask: np.ndarray | None = None,
) -> BodyShapePrior | None:
    """Build a pose-aligned central body region and uncertainty envelope."""
    segmentation = np.asarray(segmentation)
    if category not in {"top", "bottom"} or segmentation.ndim != 2:
        return None
    built = (_top_core if category == "top" else _bottom_core)(segmentation.shape, landmarks_px)
    if built is None:
        return None
    central, used = built
    anchors = [_point(landmarks_px, name) for name in used]
    main_person = _main_person_mask(segmentation, [value for value in anchors if value is not None])
    central &= main_person
    if not central.any():
        return None

    level = fit_level(category, fit_label)
    if category == "top":
        left, right = _point(landmarks_px, "left_shoulder"), _point(landmarks_px, "right_shoulder")
        scale = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
        margins = {0: 0.05, 1: 0.10, 2: 0.18, 3: 0.27}
    else:
        left, right = _point(landmarks_px, "left_hip"), _point(landmarks_px, "right_hip")
        scale = float(np.linalg.norm(np.asarray(left) - np.asarray(right)))
        margins = {0: 0.07, 1: 0.13, 2: 0.22, 3: 0.32}
    margin = max(2, round(scale * margins.get(level, 0.25)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    plausible = cv2.dilate(central.astype(np.uint8), kernel).astype(bool) & main_person
    # A fitted observation is useful evidence, but never permit parser speckles
    # outside the pose-selected person to enter the prior.
    if level == 0 and observed_mask is not None and np.asarray(observed_mask).shape == central.shape:
        observed = cv2.erode(np.asarray(observed_mask, np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        central |= observed & plausible & main_person

    source_weights = {
        "shared_fit_head": 1.0,
        "trained_head": 0.95,
        "fused_agreement": 0.95,
        "zero_shot": 0.78,
        "mask": 0.65,
    }
    confidence = 0.82 * source_weights.get(fit_source, 0.72)
    if level is None:
        confidence *= 0.65
    if category == "top" and layering_state and layering_state not in {"단일 상의", "해당 없음"}:
        confidence *= 0.72
    expected = 8 if category == "top" else 6
    confidence *= 0.85 + 0.15 * min(1.0, len(used) / expected)
    confidence = float(np.clip(confidence, 0.15, 0.9))

    confidence_map = np.zeros(segmentation.shape, np.float32)
    confidence_map[plausible] = confidence * 0.4
    confidence_map[central] = confidence
    return BodyShapePrior(
        category=category,
        fit_label=fit_label,
        fit_level=level,
        source=fit_source,
        central_mask=central,
        plausible_mask=plausible,
        confidence_map=confidence_map,
        confidence=confidence,
        landmarks_used=used,
    )
