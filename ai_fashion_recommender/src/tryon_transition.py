"""Clothing transitions: editable background and identity-conditioned skin colour.

The edit envelope is deliberately NOT an estimate of the hidden body. It erases
the old garment boundary so the generator can produce both clothes and background.
Skin correction never paints over clothes or invents limbs from pose landmarks.
"""
from __future__ import annotations

import re

import cv2
import numpy as np


FITTED_NAME = re.compile(
    r"스키니|레깅스|제깅스|슬림\s*핏|타이트\s*핏|머슬\s*핏|"
    r"\b(?:skinny|leggings?|jeggings?|slim[ -]?fit|tight[ -]?fit|muscle[ -]?fit)\b",
    re.I,
)

SHORT_LENGTHS = {
    "top": {"반팔", "민소매"},
    "bottom": {"쇼츠·미니 기장", "미니 기장", "반바지", "무릎 기장", "무릎 기장 바지"},
}


def transition_prompt(category, length, fitted, *, skirt=False):
    garment = "upper-body garment" if category == "top" else "lower-body garment"
    prompt = (
        f"Replace the {garment} with the exact garment in the reference product image. "
        "Match its color, fabric, design, hem, sleeve length and fit. Remove the original garment completely. "
        "The old clothing silhouette is not the person's body outline. Where the new garment is narrower, "
        "reconstruct the background instead of filling the old clothing outline with new fabric. "
        "Keep the same person's identity, natural body proportions, weight, pose and joint positions. "
        "Keep the face, hair, hands, footwear, other clothing and background outside the edit unchanged. "
        "Newly visible arms or legs must have natural anatomy, shading and the same skin tone as the "
        "person's visible skin. Do not copy the reference model's body or skin. Photorealistic clothing edit. "
    )
    if fitted:
        prompt += "The target garment is close-fitting, following the limbs or torso, without the original baggy volume. "
    if length in {"쇼츠·미니 기장", "미니 기장", "반바지", "무릎 기장", "무릎 기장 바지"}:
        if skirt:
            prompt += "The target is a short skirt: expose bare lower legs below its hem, without added trousers or leggings. "
        else:
            prompt += "The target is shorts: expose bare lower legs below the shorts hem, without added trousers or leggings. "
    elif length in {"반팔", "민소매"}:
        prompt += "Expose bare arms below the reference sleeve opening; remove the old long sleeves. "
    return prompt


def exposed_limb_coverage(before_seg, after_seg, edit, points, category, length):
    """Return the worst measurable limb's skin coverage inside the edit.

    Include originally visible skin: short-to-short edits can add unwanted
    leggings too. Keep limbs separate so one good limb cannot hide a bad one.
    Source background, hands and footwear are excluded as occlusions.
    """
    if length not in SHORT_LENGTHS[category] or before_seg is None:
        return None
    joints = ("elbow", "wrist") if category == "top" else ("knee", "ankle")
    if any(f"{side}_{joint}" not in points for side in ("left", "right") for joint in joints):
        return None
    limb = 12 if category == "top" else 14
    labels = (3, 4, 10, limb) if category == "top" else (4, 5, 6, 7, limb)
    coverages = []
    for side in ("left", "right"):
        core = np.zeros(after_seg.shape, np.uint8)
        a, b = (np.array(points[f"{side}_{joint}"]) for joint in joints)
        start, end = a + 0.2 * (b - a), a + 0.8 * (b - a)
        cv2.line(core, tuple(np.rint(start).astype(int)), tuple(np.rint(end).astype(int)),
                 1, max(2, round(min(core.shape) * 0.012)))
        region = core.astype(bool) & edit.astype(bool) & np.isin(before_seg, labels)
        if region.sum() >= 32:
            coverages.append(float(np.mean(after_seg[region] == limb)))
    return min(coverages) if coverages else None


def transition_envelope(mask, segmentation, points, category):
    """Free the garment outline, including background, without shrinking the person.

    Only add background; existing visible skin, the other garment, accessories,
    and other people are never added. The reference editor must reconstruct the
    background in this envelope; passing it to CatVTON produces fabric artifacts.
    """
    names = (("left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle")
             if category == "bottom" else
             ("left_shoulder", "right_shoulder", "left_hip", "right_hip",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"))
    if segmentation is None or any(name not in points for name in names):
        return None
    original = np.asarray(mask, dtype=bool)
    if not original.any():
        return None
    h, w = original.shape
    envelope = np.zeros((h, w), np.uint8)
    # Spatial margin removes the original boundary from the inpainting seam.
    margin = max(3, round(min(h, w) * 0.06))
    if category == "bottom":
        ys, xs = np.where(original)
        envelope[max(0, ys.min() - margin):min(h, ys.max() + margin + 1),
                 max(0, xs.min() - margin):min(w, xs.max() + margin + 1)] = 1
    else:
        torso_points = np.array([points[n] for n in names[:4]])
        x1, y1 = np.floor(torso_points.min(axis=0)).astype(int)
        x2, y2 = np.ceil(torso_points.max(axis=0)).astype(int)
        # Include the old torso's side seams even when the coat is wider than pose.
        torso = original[max(0, y1):min(h, y2 + 1)]
        if torso.any():
            tx = np.where(torso)[1]
            x1, x2 = min(x1, tx.min()), max(x2, tx.max())
        envelope[max(0, y1 - margin):min(h, y2 + margin + 1),
                 max(0, x1 - margin):min(w, x2 + margin + 1)] = 1
        for side in ("left", "right"):
            chain = [points[f"{side}_{joint}"] for joint in ("shoulder", "elbow", "wrist")]
            for a, b in zip(chain, chain[1:]):
                cv2.line(envelope, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)),
                         1, 2 * margin + 1)
        # Retain all old sleeves/hem in the erase mask, including their seam.
        envelope |= cv2.dilate(original.astype(np.uint8),
                               cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1,) * 2))
    return original | (envelope.astype(bool) & (segmentation == 0))


def _skin_sample(lab, segmentation, preferred):
    """Median of interior skin pixels; use local limbs before hands or face.

    No fixed RGB skin thresholds: they discard dark skin and coloured lighting.
    Reject tiny/noisy samples instead of silently substituting a default colour.
    """
    for labels in ((preferred,), (12, 14, 16), (13,), (1,)):
        region = np.isin(segmentation, labels).astype(np.uint8)
        core = cv2.erode(region, np.ones((3, 3), np.uint8)).astype(bool)
        pixels = lab[core]
        if len(pixels) < 64:
            continue
        low, high = np.quantile(pixels[:, 0], (0.15, 0.85))
        pixels = pixels[(pixels[:, 0] >= low) & (pixels[:, 0] <= high)]
        center = np.median(pixels, axis=0)
        distance = np.linalg.norm(pixels[:, 1:] - center[1:], axis=1)
        pixels = pixels[distance <= max(3.0, float(np.quantile(distance, 0.7)))]
        if len(pixels) >= 32 and np.linalg.norm(np.std(pixels[:, 1:], axis=0)) < 12:
            return np.median(pixels, axis=0)
    return None


def harmonize_exposed_skin(source, result, before_seg, after_seg, edit_mask, category):
    """Match newly exposed limbs to source skin, preserving generated shading.

    Uses floating-point CIELAB, robust colour offsets, and an inward feather.
    Pixels outside newly generated limb skin remain bit-for-bit unchanged.
    Returns (RGB, diagnostic); unavailable evidence is explicitly reported.
    """
    limb = 12 if category == "top" else 14
    old_clothes = (3, 4, 10) if category == "top" else (4, 5, 6, 7)
    exposed = (after_seg == limb) & np.isin(before_seg, old_clothes) & edit_mask.astype(bool)
    if exposed.sum() < 64:
        return result, {"status": "no_new_skin", "pixels": int(exposed.sum())}
    source_lab = cv2.cvtColor(source.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    target = _skin_sample(source_lab, before_seg, limb)
    if target is None:
        return result, {"status": "no_reliable_source_skin", "pixels": int(exposed.sum())}
    generated_lab = cv2.cvtColor(result.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)
    corrected = generated_lab.copy()
    count, components = cv2.connectedComponents(exposed.astype(np.uint8))
    changed = np.zeros(exposed.shape, bool)
    shifts = []
    for index in range(1, count):
        region = components == index
        core = cv2.erode(region.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
        if core.sum() < 32:
            continue
        median = np.median(generated_lab[core], axis=0)
        shift = target - median
        # Keep directional illumination; avoid crushing shadows/highlights.
        shift[0] = np.clip(shift[0], -15, 15)
        shift[1:] = np.clip(shift[1:], -20, 20)
        feather = max(1.0, min(result.shape[:2]) * 0.005)
        alpha = np.clip(cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 3) / feather, 0, 1)
        corrected[region] += alpha[region, None] * shift
        changed |= region
        shifts.append([round(float(v), 3) for v in shift])
    rgb = np.clip(np.rint(cv2.cvtColor(corrected, cv2.COLOR_LAB2RGB) * 255), 0, 255).astype(np.uint8)
    output = result.copy()
    output[changed] = rgb[changed]
    return output, {"status": "corrected" if changed.any() else "insufficient_skin_area",
                    "pixels": int(changed.sum()), "lab_shifts": shifts}
