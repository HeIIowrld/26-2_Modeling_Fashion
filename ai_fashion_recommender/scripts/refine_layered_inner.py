#!/usr/bin/env python3
"""Research-only removal of a source inner top retained by a layered VTON result.

This is an *oracle-mask pilot*: ``--source-labels`` must explicitly identify the
old visible inner top and excluded source regions. Ordinary user photos do not
have those labels. First run ``--mask-only`` and inspect the proposed mask; a
GPU refinement may then be compared with the unmodified layered-model output.
Neither a color match nor a smaller residual score proves garment correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageOps

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

from outerwear_top_tryon import _edit_crop, composite_masked  # noqa: E402


PROMPT = (
    "The person must wear only the top in the reference product image. "
    "Replace the visible old undershirt and its protruding lower hem inside "
    "the masked area with a natural continuation of that same reference top. "
    "Match the reference garment's color, fabric, print and hem; do not add "
    "another layer or a new body part. Keep the face, arms, hands, trousers, "
    "pose and background unchanged outside the masked region."
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def coarse_product_contrast(
    source: np.ndarray, old_inner: np.ndarray, product: np.ndarray,
) -> float:
    """Conservative color gate; uncertain/white-background products fail closed.

    The full product-image median is intentional. A small dark garment on a
    large white background may be rejected, but cannot turn a same-color target
    into a false claim that the remaining garment is an old layer.
    """
    if source.ndim != 3 or source.shape[2] != 3 or old_inner.shape != source.shape[:2]:
        raise ValueError("Source RGB image and old-inner mask dimensions differ")
    if product.ndim != 3 or product.shape[2] != 3:
        raise ValueError("Product must be an RGB image")
    if old_inner.sum() < 100:
        raise ValueError("Too few observed old-inner pixels for color comparison")
    source_median = np.median(source[old_inner], axis=0)
    product_median = np.median(product.reshape(-1, 3), axis=0)
    return float(np.mean(np.abs(source_median - product_median)))


def residual_inner_mask(
    original: np.ndarray,
    candidate: np.ndarray,
    old_inner: np.ndarray,
    forbidden: np.ndarray,
    *,
    similarity_threshold: float = 20.0,
    min_component_fraction: float = 0.0005,
    margin_ratio: float = 0.012,
    fill_component_boxes: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | int | bool]]:
    """Localize large unchanged old-inner components, then add an inward seam margin.

    The source label is indispensable: white generated background, a white logo
    and a white target blouse must not be classified as old inner based on color
    alone. The method is deliberately conservative and can still be wrong when
    the product resembles the old top; visual review is required.
    """
    if original.shape != candidate.shape or original.ndim != 3 or original.shape[2] != 3:
        raise ValueError("Original and candidate must be equal-sized RGB images")
    height, width = original.shape[:2]
    if old_inner.shape != (height, width) or forbidden.shape != (height, width):
        raise ValueError("Masks must match the image size")
    if not 0 < similarity_threshold <= 255 or not 0 < min_component_fraction < 1:
        raise ValueError("Invalid similarity or component threshold")
    if not 0 <= margin_ratio <= 0.05:
        raise ValueError("margin_ratio must be between 0 and 0.05")
    old_inner = old_inner.astype(bool)
    forbidden = forbidden.astype(bool)
    difference = np.abs(original.astype(np.int16) - candidate.astype(np.int16)).mean(axis=2)
    matching = old_inner & ~forbidden & (difference < similarity_threshold)
    count, components, stats, _ = cv2.connectedComponentsWithStats(
        matching.astype(np.uint8), connectivity=8
    )
    minimum = max(100, round(height * width * min_component_fraction))
    selected = np.zeros((height, width), dtype=bool)
    coverage = np.zeros((height, width), dtype=bool)
    selected_components = 0
    for index in range(1, count):
        if stats[index, cv2.CC_STAT_AREA] >= minimum:
            selected |= components == index
            if fill_component_boxes:
                x, y, box_width, box_height = stats[index, :4]
                coverage[y:y + box_height, x:x + box_width] = True
            selected_components += 1
    if not selected.any():
        raise ValueError("No sufficiently large unchanged old-inner region")
    if not fill_component_boxes:
        coverage = selected
    radius = max(2, round(min(height, width) * margin_ratio))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1,) * 2)
    edit = cv2.dilate(coverage.astype(np.uint8), kernel).astype(bool) & ~forbidden
    # A tiny detected component may still have a wide margin. Refuse to repaint
    # a significant fraction of the photograph under this targeted policy.
    if edit.mean() > 0.10:
        raise ValueError("Proposed inner-only edit area is too large")
    metrics: dict[str, float | int | bool] = {
        "old_inner_pixels": int(old_inner.sum()),
        "unchanged_old_inner_pixels": int(matching.sum()),
        "selected_pixels": int(selected.sum()),
        "selected_components": selected_components,
        "edit_fraction": round(float(edit.mean()), 5),
        "margin_pixels": radius,
        "fill_component_boxes": fill_component_boxes,
    }
    return edit, selected, metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--source-labels", type=Path, required=True)
    parser.add_argument(
        "--label-source", choices=("official-oracle", "predicted-parser"), required=True,
        help="Predicted parser labels are audit-only: they do not separate coat and inner top",
    )
    parser.add_argument("--inner-label", type=int, required=True)
    parser.add_argument("--exclude-label", type=int, action="append", default=[])
    parser.add_argument("--protected-mask", type=Path)
    parser.add_argument("--product", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--mask-only", action="store_true")
    parser.add_argument("--min-product-contrast", type=float, default=45.0)
    parser.add_argument("--margin-ratio", type=float, default=0.012)
    parser.add_argument(
        "--no-repeat-graphics", action="store_true",
        help="Tell the refiner to continue plain fabric without duplicating an existing print",
    )
    parser.add_argument(
        "--fill-component-box", action="store_true",
        help="Include the full old-inner component box, including newly exposed garment pixels",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=4)
    args = parser.parse_args()
    inputs = [args.original, args.candidate, args.source_labels]
    if args.protected_mask:
        inputs.append(args.protected_mask)
    inputs.append(args.product)
    if args.output in inputs or any(not path.is_file() for path in inputs):
        parser.error("Input files must exist and output must differ from inputs")
    if args.output.suffix.lower() != ".png":
        parser.error("--output must be a lossless PNG")
    if not args.mask_only and not args.model:
        parser.error("GPU refinement requires --model")
    if not args.mask_only and args.label_source != "official-oracle":
        parser.error("Predicted parser labels cannot authorize GPU inner-only refinement")
    if args.steps < 1:
        parser.error("--steps must be positive")
    with Image.open(args.original) as opened:
        source = opened.convert("RGB")
    with Image.open(args.candidate) as opened:
        candidate = opened.convert("RGB")
    with Image.open(args.source_labels) as opened:
        labels = np.asarray(opened)
    if labels.ndim != 2 or labels.shape != (source.height, source.width):
        parser.error("Source labels must be single-channel and source-sized")
    if source.size != candidate.size:
        parser.error("Candidate must be projected to the source photo size first")
    with Image.open(args.product) as opened:
        product = opened.convert("RGB")
    old_inner = labels == args.inner_label
    try:
        contrast = coarse_product_contrast(
            np.asarray(source), old_inner, np.asarray(product)
        )
    except ValueError as exc:
        parser.error(str(exc))
    if not 0 < args.min_product_contrast <= 255:
        parser.error("--min-product-contrast must be between 0 and 255")
    if contrast < args.min_product_contrast:
        parser.error(
            f"Old inner and target product are not distinguishable by the "
            f"conservative color gate ({contrast:.1f} < {args.min_product_contrast:.1f}); "
            "do not infer residual inner pixels from source similarity"
        )
    forbidden = np.isin(labels, args.exclude_label)
    if args.protected_mask:
        with Image.open(args.protected_mask) as opened:
            if opened.size != source.size:
                parser.error("Protected mask size differs from source")
            forbidden |= np.asarray(opened.convert("L")) >= 128
    try:
        edit, selected, metrics = residual_inner_mask(
            np.asarray(source), np.asarray(candidate),
            old_inner, forbidden,
            margin_ratio=args.margin_ratio,
            fill_component_boxes=args.fill_component_box,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mask_path = args.output if args.mask_only else args.output.with_name(
        f"{args.output.stem}_inner_edit_mask.png"
    )
    Image.fromarray(edit.astype(np.uint8) * 255).save(mask_path)
    record = {
        "original_sha256": sha256(args.original),
        "candidate_sha256": sha256(args.candidate),
        "source_labels_sha256": sha256(args.source_labels),
        "label_source": args.label_source,
        "protected_mask_sha256": sha256(args.protected_mask) if args.protected_mask else None,
        "product_sha256": sha256(args.product),
        "product_color_contrast": round(contrast, 3),
        "inner_label": args.inner_label,
        "exclude_labels": args.exclude_label,
        "no_repeat_graphics": args.no_repeat_graphics,
        "mask_path": str(mask_path),
        "mask_sha256": sha256(mask_path),
        "metrics": metrics,
        "oracle_mask_pilot": args.label_source == "official-oracle",
        "production_approved": False,
    }
    if not args.mask_only:
        import torch

        from shoe_tryon import ShoeTryOn

        if not args.model.is_dir():
            parser.error(f"Missing model directory: {args.model}")
        model = ShoeTryOn(args.model, seed=args.seed)
        pipe = model._load_pipeline()
        box = _edit_crop(edit)
        crop = candidate.crop(box)
        scale = 768 / max(crop.size)
        size = tuple(max(64, round(dimension * scale / 16) * 16) for dimension in crop.size)
        generated = pipe(
            image=crop.resize(size, Image.Resampling.LANCZOS),
            image_reference=ImageOps.pad(product, (768, 768), color="white"),
            mask_image=Image.fromarray(edit.astype(np.uint8) * 255).crop(box).resize(
                size, Image.Resampling.NEAREST
            ),
            prompt=(
                PROMPT + " Continue only plain garment fabric in the masked lower hem. "
                "Do not duplicate any existing logo, lettering, print or buttons."
                if args.no_repeat_graphics else PROMPT
            ),
            height=size[1], width=size[0], strength=1.0,
            num_inference_steps=args.steps, guidance_scale=1.0,
            generator=torch.Generator(device="cuda").manual_seed(args.seed),
        ).images[0]
        if generated.size != size:
            raise RuntimeError("Refiner returned an unexpected image size")
        result = composite_masked(candidate, generated, edit, box)
        before = np.asarray(candidate)
        after = np.asarray(result)
        record["outside_edit_exact"] = bool(np.array_equal(before[~edit], after[~edit]))
        record["selected_source_match_after_fraction"] = round(float(
            (np.abs(np.asarray(source).astype(np.int16) - after.astype(np.int16))
             .mean(axis=2)[selected] < 20).mean()
        ), 5)
        record["selected_source_match_before_fraction"] = round(float(
            (np.abs(np.asarray(source).astype(np.int16) - before.astype(np.int16))
             .mean(axis=2)[selected] < 20).mean()
        ), 5)
        result.save(args.output)
        record["output_sha256"] = sha256(args.output)
        record["model"] = str(args.model)
        record["seed"] = args.seed
        record["steps"] = args.steps
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
