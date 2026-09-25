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


def build_source_zones(
    source: np.ndarray, labels: np.ndarray, coat_seed: np.ndarray, *,
    generated_labels: np.ndarray | None = None,
    background_distance_ratio: float = 0.06,
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
    height, width = seg.shape
    protect_identity = np.isin(seg, IDENTITY_LABELS)
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
        "observed_pants": protect_pants,
        "occluded_pants": occluded_pants,
        "distant_background": protect_background,
        "protected": protected,
    }
    metrics = {
        "editable_fraction": round(float(editable.mean()), 6),
        "identity_fraction": round(float(protect_identity.mean()), 6),
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
) -> tuple[Image.Image, dict, dict[str, np.ndarray]]:
    original = source.convert("RGB")
    if generated.size == original.size:
        candidate = generated.convert("RGB")
    else:
        candidate = unpad_generated(generated, original.size)
    source_pixels = np.asarray(original)
    editable, zones, metrics = build_source_zones(
        source_pixels, labels, coat_seed, generated_labels=generated_labels
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
    cli.add_argument("--official-segmentation", type=Path)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    inputs = (args.original, args.generated, args.fashn_labels, args.coat_seed)
    if args.output.resolve() in {path.resolve() for path in inputs}:
        cli.error("Output must differ from all inputs")
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
    result, metrics, zones = composite_zones(
        original, generated, labels, coat, generated_labels=generated_labels
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
    record = {
        "original": str(args.original), "original_sha256": sha256(args.original),
        "generated": str(args.generated), "generated_sha256": sha256(args.generated),
        "fashn_labels_sha256": sha256(args.fashn_labels),
        "generated_labels_sha256": sha256(args.generated_labels) if args.generated_labels else None,
        "coat_seed_sha256": sha256(args.coat_seed),
        "official_segmentation": str(args.official_segmentation) if args.official_segmentation else None,
        "output": str(args.output), "output_sha256": sha256(args.output),
        "metrics": metrics, "production_approved": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
