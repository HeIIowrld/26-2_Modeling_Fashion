#!/usr/bin/env python3
"""Research-only three-zone source lock for coat-removal candidates.

Generated pixels are used near the removed coat and newly visible body;
observed identity/pants pixels and confidently distant original backdrop are
kept exact. This is not a quality claim about hidden anatomy or garment fit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from outerwear_top_tryon import composite_masked  # noqa: E402
from composite_layered_vton import unpad_generated  # noqa: E402


IDENTITY_LABELS = (1, 2, 9, 11, 13, 14, 15, 17)


def displaced_pocket_hand_release(
    labels: np.ndarray, generated_labels: np.ndarray, coat_seed: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Release an old pocket hand only when a nearby generated hand replaces it.

    The source hand must be sparse/partly occluded, overlap the coat proposal,
    sit at the trouser waist, and have a substantially displaced generated
    counterpart on the same side. A parser count alone cannot detect the fused
    old/new hand in this case. Any uncertain case keeps the observed hand.
    """
    source_seg = np.asarray(labels)
    generated_seg = np.asarray(generated_labels)
    coat = np.asarray(coat_seed).astype(bool)
    if (source_seg.ndim != 2 or generated_seg.shape != source_seg.shape
            or coat.shape != source_seg.shape):
        raise ValueError("Pocket-hand release requires matching label and coat maps")
    height, width = source_seg.shape
    result = np.zeros_like(coat)
    pants_rows = np.flatnonzero((source_seg == 6).any(axis=1))
    if len(pants_rows) == 0:
        return result, 0
    pants_top = int(pants_rows[0])
    minimum_area = max(100, round(source_seg.size * 0.0008))
    source_count, source_components, source_stats, source_centers = cv2.connectedComponentsWithStats(
        (source_seg == 13).astype(np.uint8), connectivity=8
    )
    generated_count, _, generated_stats, generated_centers = cv2.connectedComponentsWithStats(
        (generated_seg == 13).astype(np.uint8), connectivity=8
    )
    source_ids = [i for i in range(1, source_count)
                  if source_stats[i, cv2.CC_STAT_AREA] >= minimum_area]
    generated_ids = [i for i in range(1, generated_count)
                     if generated_stats[i, cv2.CC_STAT_AREA] >= minimum_area]
    # If the parser sees more or fewer than two substantial hands, do not
    # guess which generated hand belongs to which observed hand.
    if len(source_ids) != 2 or len(generated_ids) != 2:
        return result, 0

    selected = 0
    for source_id in source_ids:
        _, _, box_width, box_height, area = map(int, source_stats[source_id])
        source_center_x, source_center_y = source_centers[source_id]
        region = source_components == source_id
        fill_fraction = area / max(1, box_width * box_height)
        coat_fraction = float(np.mean(coat[region]))
        if (box_width < width * 0.08 or fill_fraction > 0.50
                or coat_fraction < 0.10
                or abs(source_center_y - pants_top) > height * 0.08):
            continue
        matches = []
        for generated_id in generated_ids:
            candidate_x, candidate_y = generated_centers[generated_id]
            x_shift = abs(candidate_x - source_center_x)
            if ((candidate_x >= width / 2) != (source_center_x >= width / 2)
                    or not width * 0.05 <= x_shift <= width * 0.18
                    or abs(candidate_y - source_center_y) > height * 0.12):
                continue
            matches.append((x_shift, generated_id))
        if len(matches) != 1:
            continue
        size = max(3, round(min(height, width) * 0.012))
        size += size % 2 == 0
        halo = cv2.dilate(region.astype(np.uint8), cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (size, size)
        )).astype(bool)
        other_hands = (source_seg == 13) & ~region
        halo &= ~np.isin(source_seg, (1, 2, 6, 9, 11, 14, 15, 17))
        halo &= ~other_hands
        result |= halo
        selected += 1
    # Two displaced candidates are ambiguous: this option is intended for a
    # single coat-pocket hand, not a wholesale replacement of both hands.
    if selected != 1:
        return np.zeros_like(coat), 0
    return result, selected


def validate_carried_pocket_hand_mask(
    labels: np.ndarray, coat_seed: np.ndarray, carried_mask: np.ndarray,
) -> np.ndarray:
    """Accept a prior hand release only at the original coat-pocket hand."""
    seg = np.asarray(labels)
    coat = np.asarray(coat_seed).astype(bool)
    carried = np.asarray(carried_mask).astype(bool)
    if seg.ndim != 2 or coat.shape != seg.shape or carried.shape != seg.shape:
        raise ValueError("Carried pocket-hand mask must match source labels")
    if not carried.any():
        return carried
    if carried.sum() > max(256, seg.size * 0.01):
        raise ValueError("Carried pocket-hand mask is too broad")
    source_hand = seg == 13
    if (carried & source_hand).sum() < carried.sum() * 0.5:
        raise ValueError("Carried pocket-hand mask is not centered on a source hand")
    if (carried & np.isin(seg, (1, 2, 6, 9, 11, 14, 15, 17))).any():
        raise ValueError("Carried pocket-hand mask overlaps protected identity or pants")
    count, components, stats, centers = cv2.connectedComponentsWithStats(
        source_hand.astype(np.uint8), connectivity=8
    )
    touched = [i for i in range(1, count)
               if (carried & (components == i)).sum() >= stats[i, cv2.CC_STAT_AREA] * 0.5]
    if len(touched) != 1:
        raise ValueError("Carried pocket-hand mask must cover one source hand")
    index = touched[0]
    area = int(stats[index, cv2.CC_STAT_AREA])
    width = int(stats[index, cv2.CC_STAT_WIDTH])
    height = int(stats[index, cv2.CC_STAT_HEIGHT])
    pants_rows = np.flatnonzero((seg == 6).any(axis=1))
    if (not len(pants_rows)
            or abs(centers[index][1] - pants_rows[0]) > seg.shape[0] * 0.08
            or area / max(1, width * height) > 0.5
            or float(coat[components == index].mean()) < 0.1):
        raise ValueError("Carried hand is not a partially occluded coat-pocket hand")
    return carried


def build_source_zones(
    source: np.ndarray, labels: np.ndarray, coat_seed: np.ndarray, *,
    generated_labels: np.ndarray | None = None,
    background_distance_ratio: float = 0.06,
    release_displaced_pocket_hand: bool = False,
    carried_pocket_hand_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict]:
    """Return editable area; source-locked masks are disjoint from that area."""
    rgb = np.asarray(source)
    seg = np.asarray(labels)
    coat = np.asarray(coat_seed).astype(bool)
    if (rgb.ndim != 3 or rgb.shape[2] != 3 or seg.shape != rgb.shape[:2]
            or coat.shape != seg.shape):
        raise ValueError("Source, parser labels and coat seed must have matching sizes")
    if not 0.01 <= background_distance_ratio <= 0.15:
        raise ValueError("Background distance ratio is out of bounds")
    if coat.mean() < 0.01:
        raise ValueError("No credible coat seed for three-zone compositing")
    if release_displaced_pocket_hand and generated_labels is None:
        raise ValueError("Generated labels are required for pocket-hand release")
    if release_displaced_pocket_hand and carried_pocket_hand_mask is not None:
        raise ValueError("Detecting and carrying pocket-hand release are exclusive")
    height, width = seg.shape
    protect_identity = np.isin(seg, IDENTITY_LABELS)
    released_hand = np.zeros_like(coat)
    released_hand_regions = 0
    if release_displaced_pocket_hand:
        released_hand, released_hand_regions = displaced_pocket_hand_release(
            seg, generated_labels, coat
        )
        protect_identity &= ~released_hand
    elif carried_pocket_hand_mask is not None:
        released_hand = validate_carried_pocket_hand_mask(seg, coat, carried_pocket_hand_mask)
        released_hand_regions = int(released_hand.any())
        protect_identity &= ~released_hand
    source_pants = seg == 6
    occluded_pants = np.zeros_like(source_pants)
    if generated_labels is not None:
        generated_seg = np.asarray(generated_labels)
        if generated_seg.shape != seg.shape:
            raise ValueError("Generated parser labels must match source dimensions")
        generated_top = np.isin(generated_seg, (3, 4)).astype(np.uint8)
        # Leave a small boundary band to the generated top. Locking original
        # trousers underneath a newly longer top creates a hard blue patch.
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        occluded_pants = source_pants & cv2.dilate(generated_top, kernel).astype(bool)
    protect_pants = source_pants & ~occluded_pants

    # Only a bright, nearly uniform *source* studio background is lockable.
    # Avoid assuming a white top or coat fur is background merely by color;
    # FASHN must also label the pixel as background and it must be well away
    # from the coat proposal.
    edge_width = max(8, round(width * 0.12))
    edge_pixels = np.concatenate((rgb[:, :edge_width], rgb[:, -edge_width:]), axis=1)
    edge_median = np.median(edge_pixels.reshape(-1, 3), axis=0)
    edge_deviation = np.median(np.max(
        np.abs(edge_pixels.astype(np.float32) - edge_median), axis=2
    ))
    studio_reliable = bool(edge_median.min() >= 210 and edge_deviation <= 18)
    protect_background = np.zeros_like(coat)
    if studio_reliable:
        diameter = max(3, round(min(height, width) * background_distance_ratio * 2))
        diameter += diameter % 2 == 0
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
        near_coat = cv2.dilate(coat.astype(np.uint8), kernel).astype(bool)
        source_bright = np.min(rgb, axis=2) >= 200
        protect_background = (seg == 0) & source_bright & ~near_coat

    protected = protect_identity | protect_pants | protect_background
    editable = ~protected
    zones = {
        "identity": protect_identity,
        "released_pocket_hand": released_hand,
        "observed_pants": protect_pants,
        "occluded_pants": occluded_pants,
        "distant_background": protect_background,
        "protected": protected,
    }
    metrics = {
        "editable_fraction": round(float(editable.mean()), 6),
        "identity_fraction": round(float(protect_identity.mean()), 6),
        "released_pocket_hand_fraction": round(float(released_hand.mean()), 6),
        "released_pocket_hand_regions": released_hand_regions,
        "carried_pocket_hand_release": carried_pocket_hand_mask is not None,
        "observed_pants_fraction": round(float(protect_pants.mean()), 6),
        "occluded_pants_fraction": round(float(occluded_pants.mean()), 6),
        "distant_background_fraction": round(float(protect_background.mean()), 6),
        "studio_background_reliable": studio_reliable,
        "background_distance_ratio": background_distance_ratio,
    }
    return editable, zones, metrics


def composite_zones(
    source: Image.Image, generated: Image.Image, labels: np.ndarray,
    coat_seed: np.ndarray, *, generated_labels: np.ndarray | None = None,
    release_displaced_pocket_hand: bool = False,
    carried_pocket_hand_mask: np.ndarray | None = None,
) -> tuple[Image.Image, dict, dict[str, np.ndarray]]:
    original = source.convert("RGB")
    if generated.size == original.size:
        candidate = generated.convert("RGB")
    else:
        candidate = unpad_generated(generated, original.size)
    source_pixels = np.asarray(original)
    editable, zones, metrics = build_source_zones(
        source_pixels, labels, coat_seed, generated_labels=generated_labels,
        release_displaced_pocket_hand=release_displaced_pocket_hand,
        carried_pocket_hand_mask=carried_pocket_hand_mask,
    )
    result = composite_masked(original, candidate, editable,
                              (0, 0, original.width, original.height))
    final = np.asarray(result)
    changed = np.any(final != source_pixels, axis=2)
    metrics.update({
        "outside_edit_exact": bool(not changed[~editable].any()),
        "identity_exact": bool(not changed[zones["identity"]].any()),
        "observed_pants_exact": bool(not changed[zones["observed_pants"]].any()),
        "distant_background_exact": bool(not changed[zones["distant_background"]].any()),
    })
    return result, metrics, zones


def load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.size != size:
            raise ValueError(f"Mask {path} size differs from source")
        return np.asarray(opened.convert("L")) >= 128


def unpad_generated_labels(labels: Image.Image, source_size: tuple[int, int]) -> np.ndarray:
    """Project a 512x896 LVTON parser map back with nearest-label sampling."""
    if labels.size == source_size:
        return np.asarray(labels, dtype=np.uint8)
    if labels.size != (512, 896):
        raise ValueError("Generated labels must match source or LVTON 512x896 canvas")
    width, height = source_size
    scale = min(512 / width, 896 / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    left, top = (512 - resized_width) // 2, (896 - resized_height) // 2
    projected = labels.crop((left, top, left + resized_width, top + resized_height))
    return np.asarray(projected.resize(source_size, Image.Resampling.NEAREST), dtype=np.uint8)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--original", type=Path, required=True)
    cli.add_argument("--generated", type=Path, required=True)
    cli.add_argument("--fashn-labels", type=Path, required=True)
    cli.add_argument("--coat-seed", type=Path, required=True)
    cli.add_argument("--generated-labels", type=Path,
                     help="Optional parser map for occlusion-aware observed-pants protection")
    cli.add_argument("--release-displaced-pocket-hand", action="store_true",
                     help="Research-only: use a generated hand instead of a displaced coat-pocket hand")
    cli.add_argument("--carried-pocket-hand-mask", type=Path,
                     help="Reuse a validated hand-release mask from the first stage")
    cli.add_argument("--save-released-pocket-hand-mask", type=Path,
                     help="Save the exact hand-release region for a later source lock")
    cli.add_argument("--official-segmentation", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    if args.release_displaced_pocket_hand and not args.generated_labels:
        cli.error("--release-displaced-pocket-hand requires --generated-labels")
    inputs = tuple(path for path in (
        args.original, args.generated, args.fashn_labels, args.coat_seed,
        args.generated_labels, args.official_segmentation, args.carried_pocket_hand_mask,
    ) if path is not None)
    outputs = (args.output, args.save_released_pocket_hand_mask)
    output_paths = [path.resolve() for path in outputs if path is not None]
    if len(output_paths) != len(set(output_paths)) or set(output_paths) & {
            path.resolve() for path in inputs}:
        cli.error("Outputs must differ from each other and all inputs")
    with Image.open(args.original) as opened:
        original = opened.convert("RGB")
    with Image.open(args.generated) as opened:
        generated = opened.convert("RGB")
    with Image.open(args.fashn_labels) as opened:
        if opened.size != original.size:
            cli.error("FASHN label size differs from source")
        labels = np.asarray(opened, dtype=np.uint8)
    coat = load_mask(args.coat_seed, original.size)
    generated_labels = None
    if args.generated_labels:
        with Image.open(args.generated_labels) as opened:
            generated_labels = unpad_generated_labels(opened, original.size)
    carried_pocket_hand_mask = None
    if args.carried_pocket_hand_mask:
        carried_pocket_hand_mask = load_mask(args.carried_pocket_hand_mask, original.size)
    result, metrics, zones = composite_zones(
        original, generated, labels, coat, generated_labels=generated_labels,
        release_displaced_pocket_hand=args.release_displaced_pocket_hand,
        carried_pocket_hand_mask=carried_pocket_hand_mask,
    )
    if args.official_segmentation:
        with Image.open(args.official_segmentation) as opened:
            if opened.size != original.size:
                cli.error("Official label size differs from source")
            official = np.asarray(opened, dtype=np.uint8)
        source_pixels = np.asarray(original)
        final_pixels = np.asarray(result)
        for label, key in ((5, "official_pants"), (13, "official_hair"), (14, "official_face")):
            region = official == label
            if region.any():
                metrics[f"{key}_exact_fraction"] = round(float(np.mean(
                    np.all(source_pixels[region] == final_pixels[region], axis=1)
                )), 6)
        if generated_labels is not None:
            still_visible = (official == 5) & ~np.isin(generated_labels, (3, 4))
            if still_visible.any():
                metrics["official_unoccluded_pants_exact_fraction"] = round(float(np.mean(
                    np.all(source_pixels[still_visible] == final_pixels[still_visible], axis=1)
                )), 6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    if args.save_released_pocket_hand_mask:
        args.save_released_pocket_hand_mask.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(zones["released_pocket_hand"].astype(np.uint8) * 255).save(
            args.save_released_pocket_hand_mask
        )
    record = {
        "original": str(args.original), "original_sha256": sha256(args.original),
        "generated": str(args.generated), "generated_sha256": sha256(args.generated),
        "fashn_labels_sha256": sha256(args.fashn_labels),
        "generated_labels_sha256": sha256(args.generated_labels) if args.generated_labels else None,
        "carried_pocket_hand_mask_sha256": sha256(args.carried_pocket_hand_mask)
        if args.carried_pocket_hand_mask else None,
        "saved_pocket_hand_mask_sha256": sha256(args.save_released_pocket_hand_mask)
        if args.save_released_pocket_hand_mask else None,
        "coat_seed_sha256": sha256(args.coat_seed),
        "official_segmentation": str(args.official_segmentation) if args.official_segmentation else None,
        "output": str(args.output), "output_sha256": sha256(args.output),
        "metrics": metrics, "production_approved": False,
        "release_displaced_pocket_hand": args.release_displaced_pocket_hand,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
