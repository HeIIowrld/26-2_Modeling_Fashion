#!/usr/bin/env python3
"""A/B-test our source-preservation mask on a layered VTON result.

The external generator edits the whole frame.  This experiment accepts a
source-sized output or projects a 512x896 padded output back to the photo, then
accepts generated pixels only inside our precomputed outerwear erase mask.
The source parser's protected
face/hair/hands regions remain exact source pixels.  Mask errors (for example,
fur classified as hair) are not fixed by this operation and require review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from outerwear_top_tryon import composite_masked  # noqa: E402


def unpad_generated(generated: Image.Image, source_size: tuple[int, int]) -> Image.Image:
    """Invert the exact centered 512x896 resize used by the LVTON evaluator."""
    canvas_width, canvas_height = 512, 896
    if generated.size != (canvas_width, canvas_height):
        raise ValueError(f"Expected a 512x896 model output, got {generated.size}")
    source_width, source_height = source_size
    if source_width < 1 or source_height < 1:
        raise ValueError("Source photo must have positive dimensions")
    scale = min(canvas_width / source_width, canvas_height / source_height)
    resized_width = max(1, round(source_width * scale))
    resized_height = max(1, round(source_height * scale))
    left = (canvas_width - resized_width) // 2
    top = (canvas_height - resized_height) // 2
    return generated.crop((left, top, left + resized_width, top + resized_height)).resize(
        source_size, Image.Resampling.LANCZOS
    )


def _load_mask(path: Path, size: tuple[int, int]) -> np.ndarray:
    with Image.open(path) as opened:
        if opened.size != size:
            raise ValueError(f"Mask {path} has size {opened.size}; expected {size}")
        return np.asarray(opened.convert("L")) >= 128


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def composite_layered_result(
    source: Image.Image,
    generated: Image.Image,
    erase_mask: np.ndarray,
    protected_mask: np.ndarray,
) -> tuple[Image.Image, dict[str, float | bool]]:
    """Return source-locked result and exact preservation diagnostics.

    This is an upper bound on safe reuse of our mask, not proof of a good
    silhouette: coat pixels outside the mask will also be faithfully retained.
    """
    source = source.convert("RGB")
    generated = generated.convert("RGB")
    if generated.size == source.size:
        restored = generated
    else:
        restored = unpad_generated(generated, source.size)
    source_pixels = np.asarray(source)
    generated_pixels = np.asarray(restored)
    expected_shape = source_pixels.shape[:2]
    erase = np.asarray(erase_mask)
    protected = np.asarray(protected_mask)
    if erase.shape != expected_shape or protected.shape != expected_shape:
        raise ValueError("Source, erase mask and protected mask dimensions must agree")
    erase = erase.astype(bool)
    protected = protected.astype(bool)
    editable = erase & ~protected
    if not editable.any():
        raise ValueError("No editable pixels remain after protecting source regions")
    # The existing outerwear composite function feather-blends inward from
    # editable edges and guarantees exact source pixels everywhere else.
    composited = composite_masked(
        source, restored, editable, (0, 0, source.width, source.height)
    )
    final_pixels = np.asarray(composited)
    raw_changed = np.any(generated_pixels != source_pixels, axis=2)
    final_changed = np.any(final_pixels != source_pixels, axis=2)
    outside = ~editable
    metrics: dict[str, float | bool] = {
        "editable_fraction": round(float(editable.mean()), 5),
        "raw_outside_changed_fraction": round(float(raw_changed[outside].mean()), 5),
        "raw_protected_changed_fraction": round(
            float(raw_changed[protected].mean()), 5
        ) if protected.any() else 0.0,
        "composite_outside_exact": bool(not final_changed[outside].any()),
        "composite_protected_exact": bool(not final_changed[protected].any()),
        "composite_inside_changed_fraction": round(
            float(final_changed[editable].mean()), 5
        ),
        "calibrated_visual_quality": False,
    }
    return composited, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--generated", type=Path, required=True)
    parser.add_argument("--erase-mask", type=Path, required=True)
    parser.add_argument("--protected-mask", type=Path, required=True)
    parser.add_argument(
        "--edit-policy", choices=("outerwear", "full-frame-except-protected"),
        default="outerwear",
        help="Research A/B: full frame keeps identity regions but may change jeans/background",
    )
    parser.add_argument(
        "--official-segmentation", type=Path,
        help="Evaluation-only DeepFashion-MM label map; never available for ordinary inputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    inputs = (args.original, args.generated, args.erase_mask, args.protected_mask)
    if args.output in inputs:
        parser.error("Output path must differ from every input path")
    for path in inputs:
        if not path.is_file():
            parser.error(f"Missing input: {path}")
    with Image.open(args.original) as opened:
        original = opened.convert("RGB")
    with Image.open(args.generated) as opened:
        generated = opened.convert("RGB")
    erase = _load_mask(args.erase_mask, original.size)
    protected = _load_mask(args.protected_mask, original.size)
    if args.edit_policy == "full-frame-except-protected":
        erase = np.ones_like(erase, dtype=bool)
    result, metrics = composite_layered_result(original, generated, erase, protected)
    if args.official_segmentation is not None:
        with Image.open(args.official_segmentation) as opened:
            if opened.size != original.size:
                parser.error("Official segmentation and original size differ")
            official = np.asarray(opened)
        if official.ndim != 2:
            parser.error("Official segmentation must be a single-channel label map")
        outer = official == 2
        editable = erase & ~protected
        metrics["oracle_outer_locked_fraction"] = round(
            float((outer & ~editable).sum() / max(1, outer.sum())), 5
        )
        metrics["oracle_outer_pixels"] = int(outer.sum())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.save(args.output)
    record = {
        "original": str(args.original),
        "original_sha256": sha256(args.original),
        "generated": str(args.generated),
        "generated_sha256": sha256(args.generated),
        "erase_mask": str(args.erase_mask),
        "erase_mask_sha256": sha256(args.erase_mask),
        "protected_mask": str(args.protected_mask),
        "protected_mask_sha256": sha256(args.protected_mask),
        "official_segmentation": str(args.official_segmentation) if args.official_segmentation else None,
        "official_segmentation_sha256": sha256(args.official_segmentation) if args.official_segmentation else None,
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "metrics": metrics,
        "edit_policy": args.edit_policy,
        "production_approved": False,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, ensure_ascii=False))


if __name__ == "__main__":
    main()
