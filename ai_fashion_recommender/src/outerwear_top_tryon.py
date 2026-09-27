"""Experimental outerwear removal and reference-guided top try-on.

CatVTON uses one mask both as the area to erase and as the area where the new
garment may be drawn.  That is a structural problem for bulky outerwear: a
wide mask removes the coat but encourages a wide replacement, while a narrow
mask leaves the old coat around the result.

This module keeps two different pieces of geometry:

``erase_mask``
    Covers the complete existing outerwear, including a small amount of its
    fuzzy boundary.  A general image editor may reconstruct background or the
    person's other clothes in this region.

``target_body_mask``
    A pose-derived torso and arm corridor that does not inherit the coat's
    outer silhouette.  It is used for diagnostics and optional second-pass
    garment editing, not as a claim about the person's hidden body.

The implementation deliberately stays outside the production adapter until a
paired-photo evaluation shows that it improves outerwear inputs without
inventing unacceptable anatomy or background.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

from config import garment_image_path
from virtual_tryon import TryOnNotReady


OUTERWEAR_WORDS = (
    "코트", "재킷", "블레이저", "가디건", "점퍼", "패딩", "다운", "판초",
    "무스탕", "퍼", "플리스", "coat", "jacket", "blazer", "cardigan",
    "puffer", "fleece", "fur",
)
PROTECT_LABELS = (1, 2, 8, 9, 11, 13, 15, 17)
TOP_LABELS = (3, 4, 10)

OUTERWEAR_TOP_PROMPT = (
    "Remove all existing outerwear and replace it with the exact top shown in the "
    "reference product image. The person wears only that top on the upper body: no "
    "coat, jacket, cardigan, padding, fur outerwear or layered garment remains. Follow "
    "the reference top's actual fit, sleeve length, hem, neckline, color, fabric and "
    "details. Do not fill the entire edit mask with clothing. Where the old outerwear "
    "was wider than the person's natural torso and arms, reconstruct the surrounding "
    "background and a plausible natural body silhouette. Preserve the same identity, "
    "face, hair, hands, pose, lower-body clothing, footwear, camera, lighting and "
    "background. One realistic person, anatomically consistent, no extra limbs."
)

OUTERWEAR_TOP_PROMPT_STRICT = (
    "Replace every original upper-body garment, including both the outer coat or jacket "
    "and any shirt beneath it, with only the top in the reference product image. No old "
    "undershirt hem, collar, zipper, coat cord, fur, cuff, lapel or jacket panel remains. "
    "Copy the reference top's sleeve length, neckline, fit, hem, color and distinctive "
    "details. Draw a plausible natural torso and exposed arms for the same pose. Restore "
    "the continuous original background where the outerwear extended beyond the new top "
    "and body; leave no outline or shadow of the old coat. Preserve the same face, hair, "
    "hands, lower-body clothing, footwear, camera angle and lighting. One realistic "
    "person with anatomically consistent arms and exactly one upper-body garment."
)

OUTERWEAR_REMOVAL_PROMPT = (
    "Remove all existing outerwear and every visible upper-body layer inside the masked "
    "region. Reconstruct a plausible natural torso and arms in a plain, close-fitting "
    "neutral base layer. The old coat, jacket, cardigan, padding, fur, collar, lapels and "
    "wide sleeves must not remain. Do not fill the entire edit mask with clothing: where "
    "the old outerwear was wider than the natural torso and arms, reconstruct the original "
    "surrounding background. Preserve the same identity, face, hair, hands, pose, visible "
    "lower-body clothing, footwear, camera and lighting. This is an intermediate cleanup "
    "for a second garment pass, so do not copy the reference garment yet. One realistic "
    "person, anatomically consistent, no extra limbs."
)

OUTERWEAR_TOP_REFINEMENT_PROMPT = (
    "Replace the plain neutral upper-body base layer inside the edit mask with the "
    "exact top in the reference image. Match its neckline, sleeve length, hem, color, "
    "fabric and distinctive details. Fit the garment to the visible shoulders, torso "
    "and arm pose; do not extend it into the former coat silhouette or add another "
    "layer. Preserve the same face, hair, hands, lower-body clothing, body pose, "
    "camera and background. One realistic person with exactly two arms and two hands."
)

BILATERAL_SLEEVE_GUIDANCE = (
    "Treat the reference garment as a complete two-sleeve item. Both sleeves must have "
    "the same length, fabric, color and pattern as the reference, adapted to each arm's "
    "pose. If the reference has long sleeves, cover both arms to the same wrist-level "
    "hem; do not leave either arm bare or short-sleeved. If it has short sleeves, keep "
    "both sleeve hems at the corresponding upper-arm level."
)

OUTER_ONLY_REMOVAL_PROMPT = (
    "Remove only the outer coat or jacket in the masked region. Preserve the visible "
    "inner top exactly where it is already visible, including its neckline, color, "
    "fabric and hem. Continue that same inner top naturally under the removed outerwear "
    "on a plausible torso and arms; do not add a different garment or a second layer. "
    "Reconstruct the original surrounding background where the coat extended beyond "
    "the natural body silhouette. Keep the same face, hair, hands, visible inner top, "
    "trousers, footwear, camera and lighting. One anatomically consistent person."
)


@dataclass(frozen=True)
class OuterwearTopMasks:
    erase_mask: np.ndarray
    target_body_mask: np.ndarray
    outer_ring: np.ndarray
    protected_mask: np.ndarray


def _pose_points(pose, width: int, height: int, minimum_visibility: float) -> dict[str, tuple[int, int]]:
    landmarks = getattr(pose, "landmarks", None) or {}
    required = (
        "left_shoulder", "right_shoulder", "left_hip", "right_hip",
        "left_elbow", "right_elbow", "left_wrist", "right_wrist",
    )
    points: dict[str, tuple[int, int]] = {}
    for name in required:
        value = landmarks.get(name)
        if value is None or len(value) < 3 or not np.isfinite(value).all():
            raise TryOnNotReady("외투 제거에 필요한 어깨·골반·팔 관절을 찾지 못했습니다.")
        x, y, visibility = value[:3]
        if visibility < minimum_visibility or not (0 <= x < 1 and 0 <= y < 1):
            raise TryOnNotReady(
                "외투 제거에는 양쪽 어깨·골반·팔이 보이는 정면 사진이 필요합니다."
            )
        points[name] = (
            int(np.clip(round(x * width), 0, width - 1)),
            int(np.clip(round(y * height), 0, height - 1)),
        )
    return points


def _odd_kernel(shape: tuple[int, int], ratio: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(min(shape) * ratio)))
    return size if size % 2 else size + 1


def _expanded_pair(
    first: tuple[int, int], second: tuple[int, int], scale: float
) -> tuple[tuple[int, int], tuple[int, int]]:
    center = np.mean(np.asarray((first, second), dtype=np.float32), axis=0)
    a = center + (np.asarray(first, dtype=np.float32) - center) * scale
    b = center + (np.asarray(second, dtype=np.float32) - center) * scale
    return tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int))


def build_outerwear_top_masks(
    segmentation: np.ndarray,
    upper_style_mask: np.ndarray,
    pose,
    *,
    erase_dilation_ratio: float = 0.015,
    erase_close_ratio: float = 0.02,
    torso_width_scale: float = 1.18,
    arm_width_ratio: float = 0.24,
    minimum_visibility: float = 0.50,
    preserve_observed_pants: bool = False,
    include_target_in_erase: bool = True,
) -> OuterwearTopMasks:
    """Build a generous erase mask and a narrower pose-derived target prior.

    ``target_body_mask`` is only an editing prior.  It is not a body
    measurement and must never be exposed as the person's actual hidden shape.
    """
    seg = np.asarray(segmentation)
    style = np.asarray(upper_style_mask).astype(bool)
    if seg.ndim != 2 or style.shape != seg.shape:
        raise TryOnNotReady("외투 제거용 분할 마스크의 크기가 올바르지 않습니다.")
    if style.sum() < max(100, int(style.size * 0.005)):
        raise TryOnNotReady("사진에서 지울 상체 의류 영역을 충분히 찾지 못했습니다.")
    height, width = seg.shape
    points = _pose_points(pose, width, height, minimum_visibility)

    left_shoulder, right_shoulder = _expanded_pair(
        points["left_shoulder"], points["right_shoulder"], torso_width_scale
    )
    left_hip, right_hip = _expanded_pair(
        points["left_hip"], points["right_hip"], max(1.0, torso_width_scale - 0.08)
    )
    shoulder_y = (points["left_shoulder"][1] + points["right_shoulder"][1]) / 2
    hip_y = (points["left_hip"][1] + points["right_hip"][1]) / 2
    torso_height = hip_y - shoulder_y
    shoulder_width = float(np.linalg.norm(
        np.asarray(points["left_shoulder"], dtype=float)
        - np.asarray(points["right_shoulder"], dtype=float)
    ))
    if torso_height <= 4 or shoulder_width <= 8:
        raise TryOnNotReady("상체 크기를 안정적으로 정할 수 있는 정면 포즈가 아닙니다.")

    target = np.zeros_like(seg, dtype=np.uint8)
    torso_polygon = np.asarray(
        [left_shoulder, right_shoulder, right_hip, left_hip], dtype=np.int32
    )
    torso_polygon[:, 0] = np.clip(torso_polygon[:, 0], 0, width - 1)
    torso_polygon[:, 1] = np.clip(torso_polygon[:, 1], 0, height - 1)
    # Give a small amount of room above the shoulder line and below the hip.
    torso_polygon[:2, 1] = np.clip(
        torso_polygon[:2, 1] - round(torso_height * 0.06), 0, height - 1
    )
    torso_polygon[2:, 1] = np.clip(
        torso_polygon[2:, 1] + round(torso_height * 0.08), 0, height - 1
    )
    cv2.fillConvexPoly(target, torso_polygon, 1)

    arm_thickness = max(5, int(round(shoulder_width * arm_width_ratio)))
    for side in ("left", "right"):
        chain = [points[f"{side}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
        for start, end in zip(chain, chain[1:]):
            cv2.line(target, start, end, 1, thickness=arm_thickness)

    protected = np.isin(seg, PROTECT_LABELS)
    if preserve_observed_pants:
        # A long coat can overlap real trousers.  This option tests whether
        # protecting pixels confidently parsed as pants avoids invented hems
        # and dark trouser patches; coat panels mislabeled as pants may remain.
        protected |= seg == 6
    target = target.astype(bool) & ~protected

    source = style.astype(np.uint8)
    close_size = _odd_kernel(seg.shape, erase_close_ratio)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    source = cv2.morphologyEx(source, cv2.MORPH_CLOSE, close_kernel)
    dilate_size = _odd_kernel(seg.shape, erase_dilation_ratio)
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_size, dilate_size))
    erase = cv2.dilate(source, dilate_kernel, iterations=1).astype(bool)
    # The new top needs room even where the parser missed an open coat or deep neckline.
    # A coat-only oracle experiment deliberately keeps the visible inner top untouched.
    if include_target_in_erase:
        erase |= target
    erase &= ~protected

    # A small target halo prevents seam pixels from being miscounted as background-restoration area.
    halo_size = _odd_kernel(seg.shape, 0.01)
    target_halo = cv2.dilate(
        target.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (halo_size, halo_size)),
        iterations=1,
    ).astype(bool)
    outer_ring = erase & ~target_halo
    return OuterwearTopMasks(erase, target, outer_ring, protected)


def release_pocket_hand_masks(
    masks: OuterwearTopMasks,
    segmentation: np.ndarray,
    pose,
) -> tuple[OuterwearTopMasks, np.ndarray, int]:
    """Make plausible coat-pocket hands editable in an isolated A/B experiment.

    A visible hand hanging below the hem remains protected.  A substantial
    hand region near the hip and inside the shoulder span may belong to a
    pocket; preserving it while repainting the coat can leave a third hand.
    This geometry is only a hypothesis, not an outerwear/skin classifier.
    """
    seg = np.asarray(segmentation)
    if seg.shape != masks.erase_mask.shape:
        raise TryOnNotReady("손 보호 마스크와 의류 분할의 크기가 일치하지 않습니다.")
    height, width = seg.shape
    points = _pose_points(pose, width, height, 0.50)
    shoulder_min = min(points["left_shoulder"][0], points["right_shoulder"][0])
    shoulder_max = max(points["left_shoulder"][0], points["right_shoulder"][0])
    shoulder_width = shoulder_max - shoulder_min
    hip_y = (points["left_hip"][1] + points["right_hip"][1]) / 2
    min_y = hip_y - height * 0.12
    max_y = hip_y + height * 0.04

    count, components, stats, centroids = cv2.connectedComponentsWithStats(
        (seg == 13).astype(np.uint8), connectivity=8
    )
    minimum_area = max(100, int(seg.size * 0.0008))
    released = np.zeros_like(masks.erase_mask, dtype=bool)
    selected_regions = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] < minimum_area:
            continue
        x, y = centroids[index]
        if not (shoulder_min + shoulder_width * 0.05 <= x <= shoulder_max - shoulder_width * 0.05
                and min_y <= y <= max_y):
            continue
        released |= components == index
        selected_regions += 1
    if not selected_regions:
        return masks, released, 0

    # Let the editor reconnect the forearm and hand, including the old cuff
    # edge.  Never override face, hair, accessories or the other hand.
    size = _odd_kernel(seg.shape, 0.012)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    editable = cv2.dilate(released.astype(np.uint8), kernel).astype(bool)
    editable &= ~np.isin(seg, (1, 2, 8, 9, 11, 15, 17))
    editable &= ~((seg == 6) & masks.protected_mask)
    protected = masks.protected_mask & ~editable
    erase = masks.erase_mask | editable
    # The new hand may be farther from the torso than the pose corridor, so
    # include it in the edited ring rather than silently hiding it in target.
    outer_ring = masks.outer_ring | (editable & ~masks.target_body_mask)
    return OuterwearTopMasks(erase, masks.target_body_mask, outer_ring, protected), editable, selected_regions


def release_hair_labeled_coat_masks(
    masks: OuterwearTopMasks,
    segmentation: np.ndarray,
    coat_proposal: np.ndarray,
    *,
    halo_ratio: float = 0.016,
) -> tuple[OuterwearTopMasks, np.ndarray]:
    """Test coat pixels hidden by FASHN's hair protection, without unmasking hair.

    A second parser's coat proposal is trusted only where FASHN said hair and
    only next to the existing editable coat. LIP/FASHN disagreement is not
    ground truth, so this remains an isolated A/B option.
    """
    seg = np.asarray(segmentation)
    proposal = np.asarray(coat_proposal)
    if (seg.ndim != 2 or proposal.ndim != 2
            or seg.shape != masks.erase_mask.shape or proposal.shape != seg.shape):
        raise TryOnNotReady("외투 제안 마스크와 인물 사진의 크기가 일치하지 않습니다.")
    if not 0 < halo_ratio <= 0.05:
        raise ValueError("halo_ratio must be between 0 and 0.05")
    radius = max(1, round(min(seg.shape) * halo_ratio))
    near_erase = cv2.dilate(
        masks.erase_mask.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1)),
        iterations=1,
    ).astype(bool)
    released = ((proposal != 0) & (seg == 2) & masks.protected_mask
                & near_erase & ~masks.erase_mask)
    if released.mean() > 0.02:
        raise TryOnNotReady("머리카락 경계의 외투 제안 영역이 지나치게 넓습니다.")
    if not released.any():
        return masks, released
    return OuterwearTopMasks(
        masks.erase_mask | released,
        masks.target_body_mask,
        masks.outer_ring | (released & ~masks.target_body_mask),
        masks.protected_mask & ~released,
    ), released


def include_annotated_outerwear_accessories(
    masks: OuterwearTopMasks,
    proposal: np.ndarray,
) -> tuple[OuterwearTopMasks, np.ndarray]:
    """Oracle-test thin coat cords/trim omitted by clothing segmentation.

    This mask must be explicitly annotated for the test image.  It is not an
    automatic cord detector and never overrides the remaining face/hair/hand
    protection.
    """
    candidate = np.asarray(proposal)
    if candidate.ndim != 2 or candidate.shape != masks.erase_mask.shape:
        raise TryOnNotReady("외투 부속품 마스크와 인물 사진의 크기가 일치하지 않습니다.")
    added = (candidate != 0) & ~masks.protected_mask & ~masks.erase_mask
    if added.mean() > 0.025:
        raise TryOnNotReady("외투 부속품 마스크가 지나치게 넓습니다.")
    if not added.any():
        return masks, added
    return OuterwearTopMasks(
        masks.erase_mask | added,
        masks.target_body_mask,
        masks.outer_ring | (added & ~masks.target_body_mask),
        masks.protected_mask,
    ), added


def build_official_layer_masks(
    official_segmentation: np.ndarray,
    pose,
    *,
    torso_width_scale: float = 1.18,
    preserve_observed_pants: bool = False,
    erase_inner: bool = True,
) -> OuterwearTopMasks:
    """Build an oracle mask from DeepFashion-MultiModal human annotations.

    Official 1=top and 2=outer are used only to estimate a segmentation
    ceiling in the offline evaluator.  In the coat-only variant, all visible
    inner-top pixels remain protected even after closing/dilation.  Ordinary
    photos have no such labels; hidden body shape is still not known.
    """
    official = np.asarray(official_segmentation)
    if official.ndim != 2 or official.dtype.kind not in "ui":
        raise TryOnNotReady("공식 상의·외투 분할은 단일 채널 정수 라벨이어야 합니다.")
    if int((official == 2).sum()) < max(100, int(official.size * 0.001)):
        raise TryOnNotReady("공식 분할에서 외투 영역을 찾지 못했습니다.")
    proxy = np.zeros_like(official, dtype=np.uint8)
    proxy[np.isin(official, (1, 2))] = 3
    proxy[official == 5] = 6        # pants
    proxy[official == 12] = 8       # bag
    proxy[official == 7] = 9        # headwear
    proxy[official == 8] = 11       # eyeglasses
    proxy[official == 13] = 2       # hair
    proxy[official == 14] = 1       # face
    proxy[official == 15] = 13      # visible skin, including hands
    proxy[official == 11] = 15      # footwear
    proxy[np.isin(official, (16, 17, 20, 22))] = 17  # small jewelry
    style = np.isin(official, (1, 2)) if erase_inner else official == 2
    masks = build_outerwear_top_masks(
        proxy, style, pose, torso_width_scale=torso_width_scale,
        preserve_observed_pants=preserve_observed_pants,
        include_target_in_erase=erase_inner,
    )
    if not erase_inner:
        inner = official == 1
        return OuterwearTopMasks(
            erase_mask=masks.erase_mask & ~inner,
            target_body_mask=masks.target_body_mask & ~inner,
            outer_ring=masks.outer_ring & ~inner,
            protected_mask=masks.protected_mask | inner,
        )
    return masks


def is_outerwear_input(outfit) -> bool:
    if outfit is None:
        return False
    fields = " ".join(
        str(getattr(outfit, name, "") or "")
        for name in ("upper_type", "outer_category", "material")
    ).casefold()
    if any(word.casefold() in fields for word in OUTERWEAR_WORDS):
        return True
    state = str(getattr(outfit, "layering_state", "") or "")
    outer = str(getattr(outfit, "outer_category", "") or "")
    return state in {"레이어드", "레이어드 가능성"} and outer not in {
        "", "해당 없음", "종류 불확실",
    }


def target_width_scale(product) -> float:
    """Translate explicit target fit metadata into editing room, conservatively."""
    text = " ".join(
        str(getattr(product, name, "") or "") for name in ("fit", "name", "item_type")
    ).casefold()
    if any(word in text for word in ("슬림", "타이트", "skinny", "slim", "tight")):
        return 1.08
    if any(word in text for word in ("오버", "루즈", "oversize", "oversized", "loose")):
        return 1.42
    if any(word in text for word in ("여유", "릴랙스", "relaxed")):
        return 1.30
    return 1.18


def fit_guidance_text(product) -> str:
    """Describe the requested fit to the editor, not just the mask diagnostic."""
    fit = target_width_scale(product)
    if fit <= 1.08:
        return (
            "Fit this top close to the natural shoulders and torso with a slim silhouette. "
            "Do not copy the width of the original jacket."
        )
    if fit >= 1.42:
        return (
            "Fit this top intentionally loose with dropped shoulders and roomy fabric, "
            "while keeping the body proportionate and removing the old jacket silhouette."
        )
    if fit >= 1.30:
        return (
            "Fit this top with relaxed ease around the natural shoulders and torso, "
            "without inheriting the bulk of the removed coat."
        )
    return (
        "Fit this top in a regular straight silhouette with moderate ease at the natural "
        "shoulders and torso, without inheriting the width of the removed jacket."
    )


def _edit_crop(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if not len(xs):
        raise TryOnNotReady("외투 제거 영역이 비어 있습니다.")
    height, width = mask.shape
    margin = max(round(height * 0.07), round((xs.max() - xs.min() + 1) * 0.12))
    return (
        max(0, int(xs.min()) - margin),
        max(0, int(ys.min()) - margin),
        min(width, int(xs.max()) + margin + 1),
        min(height, int(ys.max()) + margin + 1),
    )


def composite_masked(
    original: Image.Image,
    generated: Image.Image,
    mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> Image.Image:
    """Composite inward so every pixel outside ``mask`` stays exactly original."""
    left, top, right, bottom = box
    local = mask[top:bottom, left:right]
    distance = cv2.distanceTransform(local.astype(np.uint8), cv2.DIST_L2, 3)
    alpha = np.clip(distance / max(2.0, original.height * 0.003), 0.0, 1.0)[..., None]
    result = np.asarray(original.convert("RGB")).copy()
    patch = np.asarray(
        generated.convert("RGB").resize((right - left, bottom - top), Image.Resampling.LANCZOS)
    )
    source = result[top:bottom, left:right]
    result[top:bottom, left:right] = np.rint(
        source.astype(np.float32) * (1 - alpha) + patch.astype(np.float32) * alpha
    ).astype(np.uint8)
    return Image.fromarray(result)


def restore_studio_background(
    original: Image.Image,
    edited: Image.Image,
    before_segmentation: np.ndarray,
    after_segmentation: np.ndarray,
    masks: OuterwearTopMasks,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Clear generated coat-shaped ghosts only where a plain backdrop is reliable.

    This is deliberately a studio-photo heuristic, not generic inpainting.  The
    original background is sampled outside the edit mask on each image row; a
    textured or left/right mismatched backdrop fails closed.  Only pixels
    classified as background are eligible, including erroneous blank space
    within the broad pose corridor.  This relies on parser labels rather than
    guaranteeing that white clothing or exposed skin is preserved.  Low
    top-parser support disables cleanup
    to avoid erasing white-on-white garments that the parser missed.
    """
    source = np.asarray(original.convert("RGB"))
    result = np.asarray(edited.convert("RGB")).copy()
    before_seg = np.asarray(before_segmentation)
    after_seg = np.asarray(after_segmentation)
    height, width = masks.erase_mask.shape
    if (source.shape != result.shape or source.shape[:2] != (height, width)
            or before_seg.shape != (height, width) or after_seg.shape != (height, width)):
        raise ValueError("Studio background cleanup requires matching image and mask sizes")

    metrics: dict[str, float | bool] = {
        "studio_background_applied": False,
        "studio_background_restored_fraction": 0.0,
        "studio_background_reliable_rows_fraction": 0.0,
        "studio_background_top_support_fraction": 0.0,
    }
    top_support = float((np.isin(after_seg, TOP_LABELS) & masks.target_body_mask).sum()
                        / max(1, masks.target_body_mask.sum()))
    metrics["studio_background_top_support_fraction"] = round(top_support, 5)
    if top_support < 0.50:
        return edited, metrics
    candidate = masks.erase_mask & ~masks.protected_mask & (after_seg == 0)
    if not candidate.any():
        return edited, metrics

    background = (before_seg == 0) & ~masks.erase_mask
    x = np.arange(width)
    left_side = x < max(1, int(width * 0.25))
    right_side = x >= width - max(1, int(width * 0.25))
    left_colors = np.zeros((height, 3), dtype=np.float32)
    right_colors = np.zeros((height, 3), dtype=np.float32)
    left_positions = np.zeros(height, dtype=np.float32)
    right_positions = np.zeros(height, dtype=np.float32)
    reliable = np.zeros(height, dtype=bool)
    minimum_side_pixels = max(4, int(width * 0.035))
    for y in np.flatnonzero(candidate.any(axis=1)):
        left = source[y, background[y] & left_side]
        right = source[y, background[y] & right_side]
        if len(left) < minimum_side_pixels or len(right) < minimum_side_pixels:
            continue
        left_median = np.median(left, axis=0)
        right_median = np.median(right, axis=0)
        if np.max(np.abs(left_median - right_median)) > 20:
            continue
        pixels = np.concatenate((left, right), axis=0)
        median = np.median(pixels, axis=0)
        if np.median(np.max(np.abs(pixels.astype(np.float32) - median), axis=1)) > 12:
            continue
        left_colors[y] = left_median
        right_colors[y] = right_median
        left_positions[y] = np.median(x[background[y] & left_side])
        right_positions[y] = np.median(x[background[y] & right_side])
        reliable[y] = True

    candidate_rows = candidate.any(axis=1)
    row_fraction = float(reliable.sum() / max(1, candidate_rows.sum()))
    metrics["studio_background_reliable_rows_fraction"] = round(row_fraction, 5)
    # A partly textured image needs a different reconstruction method.  Avoid
    # drawing horizontal stripes by requiring nearly all candidate rows.
    if row_fraction < 0.90:
        return edited, metrics

    selected = candidate & reliable[:, None]
    # Interpolate the original studio gradient across the former coat.  A
    # single row median leaves visible vertical slabs on subtly lit backdrops.
    span = np.maximum(1.0, right_positions - left_positions)
    blend = np.clip((x[None, :] - left_positions[:, None]) / span[:, None], 0, 1)
    replacement = (
        left_colors[:, None, :] * (1 - blend[..., None])
        + right_colors[:, None, :] * blend[..., None]
    )
    result[selected] = np.clip(np.rint(replacement[selected]), 0, 255).astype(np.uint8)
    metrics["studio_background_applied"] = True
    metrics["studio_background_restored_fraction"] = round(float(selected.mean()), 5)
    return Image.fromarray(result), metrics


def _band_width_ratio(mask: np.ndarray, pose) -> float | None:
    landmarks = getattr(pose, "landmarks", None) or {}
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if any(name not in landmarks for name in names):
        return None
    height, width = mask.shape
    ls, rs = landmarks["left_shoulder"], landmarks["right_shoulder"]
    lh, rh = landmarks["left_hip"], landmarks["right_hip"]
    shoulder_width = abs(ls[0] - rs[0]) * width
    if shoulder_width < 5:
        return None
    shoulder_y = (ls[1] + rs[1]) * height / 2
    hip_y = (lh[1] + rh[1]) * height / 2
    start = max(0, int(round(shoulder_y + 0.25 * (hip_y - shoulder_y))))
    end = min(height, int(round(shoulder_y + 0.58 * (hip_y - shoulder_y))))
    if end - start < 2:
        return None
    widths = mask[start:end].sum(axis=1)
    useful = widths[widths > 0]
    return round(float(np.median(useful) / shoulder_width), 4) if len(useful) else None


def hand_region_count(segmentation: np.ndarray) -> int:
    """Count substantial parser hand islands for a conservative anatomy alert."""
    seg = np.asarray(segmentation)
    if seg.ndim != 2:
        raise ValueError("Hand-region count requires a 2-D segmentation")
    count, _, stats, _ = cv2.connectedComponentsWithStats(
        (seg == 13).astype(np.uint8), connectivity=8
    )
    minimum_area = max(100, int(seg.size * 0.0008))
    return sum(int(stats[index, cv2.CC_STAT_AREA]) >= minimum_area
               for index in range(1, count))


def assess_outerwear_top_result(
    before_segmentation: np.ndarray,
    after_segmentation: np.ndarray,
    masks: OuterwearTopMasks,
    pose,
    before: Image.Image,
    after: Image.Image,
) -> dict:
    before_top = np.isin(before_segmentation, TOP_LABELS)
    after_top = np.isin(after_segmentation, TOP_LABELS)
    ring_pixels = int(masks.outer_ring.sum())
    outer_ring_top_fraction = float((after_top & masks.outer_ring).sum() / max(1, ring_pixels))
    before_np = np.asarray(before.convert("RGB"), dtype=np.int16)
    after_np = np.asarray(after.convert("RGB"), dtype=np.int16)
    changed = np.max(np.abs(after_np - before_np), axis=2)
    hand_regions_before = hand_region_count(before_segmentation)
    hand_regions_after = hand_region_count(after_segmentation)
    observed_pants = np.asarray(before_segmentation) == 6
    return {
        "backend": "flux2-klein-outerwear-top-experiment",
        "calibrated": False,
        "erase_fraction": round(float(masks.erase_mask.mean()), 5),
        "target_fraction": round(float(masks.target_body_mask.mean()), 5),
        "outer_ring_fraction": round(float(masks.outer_ring.mean()), 5),
        "outer_ring_top_fraction": round(outer_ring_top_fraction, 5),
        "before_top_width_per_shoulder": _band_width_ratio(before_top, pose),
        "after_top_width_per_shoulder": _band_width_ratio(after_top, pose),
        "target_width_per_shoulder": _band_width_ratio(masks.target_body_mask, pose),
        "changed_inside_fraction": round(float(np.mean(changed[masks.erase_mask] > 8)), 5),
        "outside_mask_preserved": bool(np.all(changed[~masks.erase_mask] == 0)),
        "remaining_protected_pixels_preserved": bool(np.all(changed[masks.protected_mask] == 0)),
        "hand_regions_before": hand_regions_before,
        "hand_regions_after": hand_regions_after,
        "extra_hand_region_needs_review": hand_regions_after > max(2, hand_regions_before),
        "observed_pants_erased_fraction": round(
            float((masks.erase_mask & observed_pants).sum() / max(1, observed_pants.sum())), 5
        ),
    }


def outer_ring_needs_review(report: dict[str, object]) -> bool:
    """Flag likely silhouette overflow without penalizing valid long sleeves.

    The parser can label a correctly rendered long sleeve in ``outer_ring``.
    Treat that occupancy as suspicious only when the measured garment is also
    materially wider than the pose/fit target (or width could not be measured).
    """
    if float(report.get("outer_ring_top_fraction") or 0.0) <= 0.25:
        return False
    after_width = report.get("after_top_width_per_shoulder")
    target_width = report.get("target_width_per_shoulder")
    if after_width is None or target_width is None:
        return True
    return float(after_width) > float(target_width) * 1.10


def save_mask_debug(
    person: Image.Image,
    masks: OuterwearTopMasks,
    directory: str | Path,
    prefix: str,
) -> None:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    for name, mask in (
        ("erase", masks.erase_mask),
        ("target", masks.target_body_mask),
        ("outer_ring", masks.outer_ring),
        ("protected", masks.protected_mask),
    ):
        Image.fromarray(mask.astype(np.uint8) * 255).save(target / f"{prefix}_{name}.png")
    base = np.asarray(person.convert("RGB"), dtype=np.float32)
    color = np.zeros_like(base)
    color[masks.erase_mask] = (230, 70, 70)
    color[masks.target_body_mask] = (60, 200, 100)
    color[masks.outer_ring] = (240, 165, 45)
    overlay = np.where((masks.erase_mask | masks.target_body_mask)[..., None],
                       base * 0.55 + color * 0.45, base)
    Image.fromarray(np.clip(overlay, 0, 255).astype(np.uint8)).save(
        target / f"{prefix}_overlay.jpg", quality=94
    )


def visible_inner_reference(person: Image.Image, official_segmentation: np.ndarray) -> Image.Image:
    """Build a product-like reference from only the oracle-visible inner top."""
    source = np.asarray(person.convert("RGB"))
    inner = np.asarray(official_segmentation) == 1
    if inner.shape != source.shape[:2] or int(inner.sum()) < 100:
        raise TryOnNotReady("참조할 만큼 보이는 공식 이너 영역이 없습니다.")
    ys, xs = np.where(inner)
    left, right = int(xs.min()), int(xs.max()) + 1
    top, bottom = int(ys.min()), int(ys.max()) + 1
    cutout = np.full_like(source[top:bottom, left:right], 240)
    cutout[inner[top:bottom, left:right]] = source[top:bottom, left:right][
        inner[top:bottom, left:right]
    ]
    return ImageOps.pad(Image.fromarray(cutout), (768, 768), color=(240, 240, 240))


def resolve_second_stage_mask(
    masks: OuterwearTopMasks, proposed: np.ndarray | None,
) -> np.ndarray:
    """Add a measured base-garment region without removing pose/body coverage."""
    if proposed is None:
        return masks.target_body_mask
    candidate = np.asarray(proposed)
    if candidate.ndim != 2 or candidate.shape != masks.target_body_mask.shape:
        raise TryOnNotReady("2차 상의 마스크가 인물 사진과 크기가 다릅니다.")
    candidate = (candidate != 0) | masks.target_body_mask
    candidate &= ~masks.protected_mask
    if candidate.sum() < masks.target_body_mask.sum() or candidate.mean() > 0.80:
        raise TryOnNotReady("2차 상의 마스크가 지나치게 넓거나 보호 영역을 침범합니다.")
    return candidate


class OuterwearTopTryOn:
    """Reference-guided outerwear removal for isolated top-only experiments."""

    def __init__(
        self,
        clothing,
        editor,
        parser,
        *,
        debug_dir: str | Path | None = None,
        prompt: str = OUTERWEAR_TOP_PROMPT,
        require_outerwear: bool = True,
        max_side: int = 1024,
        num_inference_steps: int = 4,
        guidance_scale: float = 1.0,
        second_pass_catvton: bool = False,
        second_pass_flux: bool = False,
        fit_guidance: bool = False,
        restore_studio: bool = False,
        release_pocket_hands: bool = False,
        preserve_observed_pants: bool = False,
        oracle_outer_only: bool = False,
        oracle_inner_reference: bool = False,
        bilateral_sleeves: bool = False,
    ) -> None:
        self.clothing = clothing
        self.editor = editor
        self.parser = parser
        self.debug_dir = Path(debug_dir) if debug_dir else None
        self.prompt = prompt
        self.require_outerwear = require_outerwear
        self.max_side = max_side
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.second_pass_catvton = second_pass_catvton
        if second_pass_catvton and second_pass_flux:
            raise ValueError("CatVTON and FLUX second passes are mutually exclusive")
        if oracle_outer_only and (second_pass_catvton or second_pass_flux):
            raise ValueError("Coat-only oracle cannot be combined with a second pass")
        if oracle_outer_only and release_pocket_hands:
            raise ValueError("Coat-only oracle must protect every visible inner-top pixel")
        if oracle_inner_reference and not oracle_outer_only:
            raise ValueError("Visible-inner reference requires the coat-only oracle experiment")
        if bilateral_sleeves and not second_pass_flux:
            raise ValueError("Bilateral sleeve guidance requires the second FLUX pass")
        self.second_pass_flux = second_pass_flux
        self.oracle_outer_only = oracle_outer_only
        self.oracle_inner_reference = oracle_inner_reference
        self.bilateral_sleeves = bilateral_sleeves
        self.fit_guidance = fit_guidance
        self.restore_studio = restore_studio
        self.release_pocket_hands = release_pocket_hands
        self.preserve_observed_pants = preserve_observed_pants
        self.last_warnings: list[str] = []
        self.last_quality_reports: list[dict] = []
        self.last_debug_masks: OuterwearTopMasks | None = None
        self.last_render_kind = ""

    @property
    def available(self) -> bool:
        editor_ready = bool(getattr(self.editor, "available", False))
        return editor_ready and (
            not self.second_pass_catvton or bool(getattr(self.clothing, "available", False))
        )

    @property
    def supported_categories(self) -> set[str]:
        return {"top"}

    @property
    def reference_bottom_lengths(self) -> dict:
        return getattr(self.clothing, "reference_bottom_lengths", {})

    def _reference(self, product, path: Path) -> Image.Image:
        prepare = getattr(self.clothing, "_prepare_garment_reference", None)
        if prepare is not None:
            return prepare(path, "top")
        with Image.open(path) as opened:
            return opened.convert("RGB")

    def generate(self, person_image, recommendation, output_path, context=None) -> Path:
        self.last_warnings = []
        self.last_quality_reports = []
        self.last_debug_masks = None
        self.last_render_kind = ""
        context = context or {}
        products = list(recommendation.products)
        tops = [product for product in products if product.category == "top"]
        if len(tops) != 1 or len(products) != 1:
            raise TryOnNotReady("외투 제거 실험은 한 번에 상의 상품 하나만 지원합니다.")
        if self.require_outerwear and not is_outerwear_input(context.get("outfit")):
            raise TryOnNotReady(
                "외투 또는 레이어드 상의가 확인된 사진에서만 "
                "외투 제거 실험을 실행합니다."
            )
        segmentation = context.get("segmentation")
        upper_style = context.get("upper_style_mask")
        if segmentation is None or upper_style is None:
            raise TryOnNotReady("외투 제거에 필요한 FASHN 상체 마스크가 없습니다.")
        pose = context.get("pose")
        product = tops[0]
        official_segmentation = context.get("official_segmentation")
        if self.oracle_outer_only and official_segmentation is None:
            raise TryOnNotReady("외투만 제거하는 상한선 실험에는 공식 외투 마스크가 필요합니다.")
        if official_segmentation is not None:
            official_segmentation = np.asarray(official_segmentation)
            if official_segmentation.shape != np.asarray(segmentation).shape:
                raise TryOnNotReady("공식 분할과 인물 사진의 크기가 일치하지 않습니다.")
            masks = build_official_layer_masks(
                official_segmentation, pose, torso_width_scale=target_width_scale(product),
                preserve_observed_pants=self.preserve_observed_pants,
                erase_inner=not self.oracle_outer_only,
            )
        else:
            masks = build_outerwear_top_masks(
                segmentation, upper_style, pose, torso_width_scale=target_width_scale(product),
                preserve_observed_pants=self.preserve_observed_pants,
            )
        released_hand_regions = 0
        released_hand_mask = np.zeros_like(masks.erase_mask, dtype=bool)
        if self.release_pocket_hands:
            masks, released_hand_mask, released_hand_regions = release_pocket_hand_masks(
                masks, np.asarray(segmentation), pose
            )
        released_hair_coat_mask = np.zeros_like(masks.erase_mask, dtype=bool)
        if context.get("coat_proposal_mask") is not None:
            masks, released_hair_coat_mask = release_hair_labeled_coat_masks(
                masks, np.asarray(segmentation), context["coat_proposal_mask"]
            )
        added_accessory_mask = np.zeros_like(masks.erase_mask, dtype=bool)
        if context.get("accessory_mask") is not None:
            masks, added_accessory_mask = include_annotated_outerwear_accessories(
                masks, context["accessory_mask"]
            )
        self.last_debug_masks = masks

        path = Path(product.image_path).expanduser() if product.image_path else None
        if path is not None and not path.is_absolute():
            path = garment_image_path(product.image_path)
        if path is None or not path.is_file():
            raise TryOnNotReady("선택한 상의의 실제 상품 이미지가 필요합니다.")

        with Image.open(person_image) as opened:
            person = opened.convert("RGB")
        if person.size != (masks.erase_mask.shape[1], masks.erase_mask.shape[0]):
            raise TryOnNotReady("사진과 외투 마스크의 크기가 일치하지 않습니다.")
        reference = self._reference(product, path).convert("RGB")
        box = _edit_crop(masks.erase_mask)
        crop = person.crop(box)
        scale = self.max_side / max(crop.size)
        size = tuple(max(64, round(dimension * scale / 16) * 16) for dimension in crop.size)
        edit_image = crop.resize(size, Image.Resampling.LANCZOS)
        mask_image = Image.fromarray(masks.erase_mask.astype(np.uint8) * 255).crop(box).resize(
            size, Image.Resampling.NEAREST
        )
        import torch

        pipe = self.editor._load_pipeline()
        # In the two-pass variants FLUX first owns coat removal and hidden-region
        # reconstruction.  The product belongs to the narrow second pass.
        two_stage = self.second_pass_catvton or self.second_pass_flux
        shape_reference = context.get("coatless_shape_reference")
        if shape_reference is not None and not two_stage:
            raise TryOnNotReady("외투 제거 실루엣 참조에는 2단계 생성이 필요합니다.")
        if shape_reference is not None and not isinstance(shape_reference, Image.Image):
            raise TryOnNotReady("외투 제거 실루엣 참조는 PIL 이미지여야 합니다.")
        stage_prompt = (
            OUTER_ONLY_REMOVAL_PROMPT if self.oracle_outer_only else
            OUTERWEAR_REMOVAL_PROMPT if two_stage else self.prompt
        )
        if self.oracle_inner_reference:
            stage_prompt += " The reference image shows the visible part of that same inner top."
        if self.fit_guidance and not two_stage and not self.oracle_outer_only:
            stage_prompt = f"{stage_prompt} {fit_guidance_text(product)}"
        stage_reference = (
            visible_inner_reference(person, official_segmentation)
            if self.oracle_inner_reference else
            ImageOps.pad(shape_reference.convert("RGB"), (768, 768), color="white")
            if shape_reference is not None else
            Image.new("RGB", (768, 768), (127, 127, 127))
            if two_stage or self.oracle_outer_only else
            ImageOps.pad(reference, (768, 768), color="white")
        )
        generated = pipe(
            image=edit_image,
            image_reference=stage_reference,
            mask_image=mask_image,
            prompt=stage_prompt,
            height=size[1],
            width=size[0],
            strength=1.0,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            generator=torch.Generator(device="cuda").manual_seed(self.editor.seed),
        ).images[0]
        if generated.size != size:
            raise RuntimeError("외투 제거 생성 결과 해상도가 요청과 다릅니다.")
        flux_result = composite_masked(person, generated, masks.erase_mask, box)
        parsed_flux = self.parser.parse(np.asarray(flux_result), pose)
        studio_metrics: dict[str, float | bool] = {}
        if self.restore_studio:
            if self.debug_dir is not None:
                self.debug_dir.mkdir(parents=True, exist_ok=True)
                flux_result.save(self.debug_dir / f"{Path(output_path).stem}_before_studio.png")
            flux_result, studio_metrics = restore_studio_background(
                person, flux_result, np.asarray(segmentation),
                np.asarray(parsed_flux["segmentation"]), masks,
            )
            if studio_metrics["studio_background_applied"]:
                parsed_flux = self.parser.parse(np.asarray(flux_result), pose)
        flux_report = assess_outerwear_top_result(
            np.asarray(segmentation), np.asarray(parsed_flux["segmentation"]), masks,
            pose, person, flux_result,
        )
        flux_report["stage"] = (
            "flux_oracle_outer_only" if self.oracle_outer_only else
            "flux_outerwear_removal" if two_stage else
            "flux_outerwear_removal_and_top"
        )
        flux_report.update(studio_metrics)
        flux_report["observed_pants_protection_used"] = self.preserve_observed_pants
        flux_report["official_layer_mask_used"] = official_segmentation is not None
        flux_report["oracle_inner_reference_used"] = self.oracle_inner_reference
        flux_report["coatless_shape_reference_used"] = shape_reference is not None
        if official_segmentation is not None:
            outer = official_segmentation == 2
            inner = official_segmentation == 1
            flux_report["official_outer_erase_recall"] = round(
                float(masks.erase_mask[outer].mean()), 5
            )
            flux_report["official_inner_erase_recall"] = (
                round(float(masks.erase_mask[inner].mean()), 5) if inner.any() else None
            )
            official_pants = official_segmentation == 5
            flux_report["official_pants_erased_fraction"] = (
                round(float(masks.erase_mask[official_pants].mean()), 5)
                if official_pants.any() else None
            )
        flux_report["released_pocket_hand_regions"] = released_hand_regions
        flux_report["released_pocket_hand_fraction"] = round(float(released_hand_mask.mean()), 5)
        flux_report["released_hair_coat_fraction"] = round(float(released_hair_coat_mask.mean()), 5)
        flux_report["added_accessory_fraction"] = round(float(added_accessory_mask.mean()), 5)
        if released_hair_coat_mask.any():
            release_delta = np.max(
                np.abs(np.asarray(flux_result, dtype=np.int16)
                       - np.asarray(person, dtype=np.int16)), axis=2
            )
            flux_report["released_hair_coat_changed_fraction"] = round(
                float(np.mean(release_delta[released_hair_coat_mask] > 8)), 5
            )
        else:
            flux_report["released_hair_coat_changed_fraction"] = 0.0
        if flux_report["changed_inside_fraction"] < 0.02:
            raise TryOnNotReady("외투 영역이 충분히 변경되지 않았습니다.")
        if self.debug_dir is not None:
            # Keep the expensive first-stage evidence even if the optional
            # second stage fails afterwards.
            save_mask_debug(person, masks, self.debug_dir, Path(output_path).stem)
            if official_segmentation is not None:
                for label, name in ((2, "official_outer"), (1, "official_inner")):
                    Image.fromarray((official_segmentation == label).astype(np.uint8) * 255).save(
                        self.debug_dir / f"{Path(output_path).stem}_{name}.png"
                    )
            if self.release_pocket_hands:
                Image.fromarray(released_hand_mask.astype(np.uint8) * 255).save(
                    self.debug_dir / f"{Path(output_path).stem}_released_hand.png"
                )
            if context.get("coat_proposal_mask") is not None:
                Image.fromarray(released_hair_coat_mask.astype(np.uint8) * 255).save(
                    self.debug_dir / f"{Path(output_path).stem}_released_hair_coat.png"
                )
            if context.get("accessory_mask") is not None:
                Image.fromarray(added_accessory_mask.astype(np.uint8) * 255).save(
                    self.debug_dir / f"{Path(output_path).stem}_added_accessory.png"
                )
            flux_result.save(self.debug_dir / f"{Path(output_path).stem}_flux_stage.png")

        result = flux_result
        reports = [flux_report]
        catvton_warnings: list[str] = []
        if self.second_pass_flux:
            # Default: pose/fit corridor.  A measured second-stage mask can
            # additionally cover the neutral garment the first pass actually
            # generated, which may extend beyond that corridor.
            second_mask = resolve_second_stage_mask(
                masks, context.get("second_stage_mask")
            )
            target_box = _edit_crop(second_mask)
            target_crop = flux_result.crop(target_box)
            target_scale = self.max_side / max(target_crop.size)
            target_size = tuple(
                max(64, round(dimension * target_scale / 16) * 16)
                for dimension in target_crop.size
            )
            target_image = target_crop.resize(target_size, Image.Resampling.LANCZOS)
            target_mask_image = Image.fromarray(
                second_mask.astype(np.uint8) * 255
            ).crop(target_box).resize(target_size, Image.Resampling.NEAREST)
            refinement_prompt = OUTERWEAR_TOP_REFINEMENT_PROMPT
            if self.fit_guidance:
                refinement_prompt = f"{refinement_prompt} {fit_guidance_text(product)}"
            if self.bilateral_sleeves:
                refinement_prompt = f"{refinement_prompt} {BILATERAL_SLEEVE_GUIDANCE}"
            refined = pipe(
                image=target_image,
                image_reference=ImageOps.pad(reference, (768, 768), color="white"),
                mask_image=target_mask_image,
                prompt=refinement_prompt,
                height=target_size[1],
                width=target_size[0],
                strength=1.0,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                generator=torch.Generator(device="cuda").manual_seed(self.editor.seed + 1),
            ).images[0]
            if refined.size != target_size:
                raise RuntimeError("2차 FLUX 상의 생성 결과 해상도가 요청과 다릅니다.")
            result = composite_masked(
                flux_result, refined, second_mask, target_box
            )
            if self.debug_dir is not None:
                Image.fromarray(second_mask.astype(np.uint8) * 255).save(
                    self.debug_dir / f"{Path(output_path).stem}_second_stage_mask.png"
                )
                result.save(self.debug_dir / f"{Path(output_path).stem}_flux_target_stage.png")
            parsed_final = self.parser.parse(np.asarray(result), pose)
            final_report = assess_outerwear_top_result(
                np.asarray(segmentation), np.asarray(parsed_final["segmentation"]), masks,
                pose, person, result,
            )
            final_report["stage"] = (
                "flux_measured_garment_refinement"
                if context.get("second_stage_mask") is not None
                else "flux_target_refinement"
            )
            stage_delta = np.max(
                np.abs(np.asarray(result, dtype=np.int16)
                       - np.asarray(flux_result, dtype=np.int16)), axis=2
            )
            final_report["second_stage_changed_target_fraction"] = round(
                float(np.mean(stage_delta[masks.target_body_mask] > 8)), 5
            )
            final_report["second_stage_mask_fraction"] = round(float(second_mask.mean()), 5)
            final_report["bilateral_sleeve_guidance_used"] = self.bilateral_sleeves
            final_report["second_stage_added_fraction"] = round(
                float((second_mask & ~masks.target_body_mask).mean()), 5
            )
            combined_edit = masks.erase_mask | second_mask
            final_delta = np.max(
                np.abs(np.asarray(result, dtype=np.int16)
                       - np.asarray(person, dtype=np.int16)), axis=2
            )
            final_report["outside_combined_edit_preserved"] = bool(
                np.all(final_delta[~combined_edit] == 0)
            )
            final_report["changed_beyond_first_erase_fraction"] = round(
                float(((final_delta > 0) & ~masks.erase_mask).mean()), 5
            )
            final_report["second_stage_outer_ring_preserved"] = bool(
                np.all(stage_delta[masks.outer_ring] == 0)
            )
            reports.append(final_report)
        if self.second_pass_catvton:
            # The broad FLUX pass owns coat removal and background reconstruction.
            # CatVTON then sees only the pose/fit-derived narrower target mask, so it
            # cannot repaint the whole former coat silhouette with the new top.
            import gc

            gc.collect()
            if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
            output_parent = Path(output_path).parent
            output_parent.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix="outerwear_top_", dir=output_parent) as folder:
                first_path = Path(folder) / "flux_stage.png"
                second_path = Path(folder) / "catvton_stage.png"
                flux_result.save(first_path)
                second_context = {
                    "upper_mask": masks.target_body_mask,
                    "upper_style_mask": masks.target_body_mask,
                    "segmentation": parsed_flux["segmentation"],
                    "pose": pose,
                    "classifier": context.get("classifier"),
                    "strict_vton": True,
                }
                self.clothing.generate(first_path, recommendation, second_path, second_context)
                with Image.open(second_path) as opened:
                    catvton_result = opened.convert("RGB")
                if catvton_result.size != flux_result.size:
                    raise RuntimeError("2차 상의 생성 결과 해상도가 원본과 다릅니다.")
                # CatVTON internally feathers its repaint mask.  Re-composite
                # inward so even that feather cannot refill the restored outer
                # ring or modify anything outside the narrow target corridor.
                target_box = _edit_crop(masks.target_body_mask)
                result = composite_masked(
                    flux_result,
                    catvton_result.crop(target_box),
                    masks.target_body_mask,
                    target_box,
                )
            parsed_final = self.parser.parse(np.asarray(result), pose)
            final_report = assess_outerwear_top_result(
                np.asarray(segmentation), np.asarray(parsed_final["segmentation"]), masks,
                pose, person, result,
            )
            final_report["stage"] = "catvton_target_refinement"
            reports.extend(list(getattr(self.clothing, "last_quality_reports", [])))
            reports.append(final_report)
            catvton_warnings = list(getattr(self.clothing, "last_warnings", []))
        report = reports[-1]
        self.last_quality_reports = reports
        self.last_warnings = [
            "외투 제거 상의 합성은 실험 기능입니다. 생성된 몸통·팔·배경과 상품 형태를 "
            "직접 확인해 주세요."
        ]
        if self.second_pass_flux:
            self.last_warnings.append(
                "2단계 FLUX에서는 목표 마스크 바깥에 1차 중립 상의가 남을 수 있습니다. "
                "상의 경계와 외투 잔여를 육안으로 확인해 주세요."
            )
            if context.get("second_stage_mask") is not None:
                self.last_warnings.append(
                    "2차 마스크를 생성된 옷 영역까지 넓혔습니다. 팔·하의·배경의 재편집 여부를 확인해 주세요."
                )
        if official_segmentation is not None:
            self.last_warnings.append(
                "사람이 주석한 공식 외투·이너 마스크를 사용한 상한선 실험입니다. "
                "일반 사진에 자동 적용할 수 있는 분할 모델은 아닙니다."
            )
        if self.oracle_outer_only:
            self.last_warnings.append(
                "외투만 지우고 원래 이너는 보존하는 실험입니다. 상품 상의 착장은 아직 수행하지 않았습니다."
            )
        if self.preserve_observed_pants:
            self.last_warnings.append(
                "원본에서 바지로 분류된 영역을 보호했습니다. 코트 자락이 바지로 오분류된 "
                "경우 잔여물이 남는지 확인해 주세요."
            )
        if released_hand_regions:
            self.last_warnings.append(
                "주머니 근처 손을 편집해 다시 그렸습니다. 원래 손 모양·개수와 팔 연결을 "
                "반드시 비교해 주세요."
            )
        if released_hair_coat_mask.any():
            self.last_warnings.append(
                "다른 파서가 외투로 분류한 머리카락 경계 일부를 편집했습니다. "
                "머리카락·얼굴 손상을 확인해 주세요."
            )
        if context.get("accessory_mask") is not None:
            self.last_warnings.append(
                "사람이 지정한 외투 부속품 영역을 추가 편집한 상한선 실험입니다. "
                "주변 팔·하의·배경을 확인해 주세요."
            )
        self.last_warnings.extend(catvton_warnings)
        if outer_ring_needs_review(report):
            self.last_warnings.append(
                "외투 바깥 부피 영역이 새 상의로 채워진 비율이 높습니다. "
                "상체 실루엣을 직접 확인해 주세요."
            )
        if report.get("extra_hand_region_needs_review"):
            self.last_warnings.append(
                "손으로 분류된 영역이 원본보다 늘었습니다. 팔·손이 중복 생성됐는지 "
                "반드시 확인해 주세요."
            )
        before_width = report["before_top_width_per_shoulder"]
        after_width = report["after_top_width_per_shoulder"]
        target_width = report["target_width_per_shoulder"]
        if (before_width is not None and after_width is not None and target_width is not None
                and before_width > target_width * 1.15 and after_width >= before_width * 0.95):
            self.last_warnings.append("외투 제거 후에도 상의 폭이 충분히 줄지 않았습니다.")

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(f".{output.stem}.outerwear-stage{output.suffix}")
        save_options = {"quality": 95} if output.suffix.lower() in {".jpg", ".jpeg"} else {}
        result.save(temporary, **save_options)
        os.replace(temporary, output)
        self.last_render_kind = "tryon"
        return output
