#!/usr/bin/env python3
"""Isolated CatVTON A/B of upper-mask closing/dilation on an outerwear photo.

All production defaults are untouched. The only variable between paired runs
is the top-mask morphology; person, garment, pose, parser, seed and checkpoint
remain the same. These examples are non-commercial research assets.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import catvton_tryon  # noqa: E402
from catvton_tryon import CatVTONTryOn, PROTECT_LABELS  # noqa: E402
from clothing_parser import ClothingParser  # noqa: E402
from pose_analyzer import PoseAnalyzer  # noqa: E402
from schemas import Product, Recommendation  # noqa: E402
from eval_upper_mask_morphology import morph_mask  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def experimental_close(mask: np.ndarray) -> np.ndarray:
    return morph_mask(mask, 0.01, 0.0).astype(np.uint8)


def experimental_dilate(mask: np.ndarray) -> np.ndarray:
    return morph_mask(mask, 0.0, 0.01).astype(np.uint8) * 255


def build_custom_target_mask(
    pose_target: np.ndarray, old_inner: np.ndarray, protected: np.ndarray,
) -> np.ndarray:
    """Union independent pose/old-inner proposals while keeping identity locked."""
    if (pose_target.ndim != 2 or old_inner.shape != pose_target.shape
            or protected.shape != pose_target.shape):
        raise ValueError("Custom masks must be equal-sized 2-D arrays")
    custom = (pose_target.astype(bool) | old_inner.astype(bool)) & ~protected.astype(bool)
    if custom.mean() >= 0.35 or not custom.any():
        raise ValueError("Custom target mask is empty or too large")
    return custom


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--person", type=Path, required=True)
    cli.add_argument("--garment", type=Path, required=True)
    cli.add_argument("--catvton-repo", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    cli.add_argument(
        "--variant",
        choices=("standard", "narrow", "custom-target", "custom-target-standard"),
        required=True,
    )
    cli.add_argument("--pose-target-mask", type=Path)
    cli.add_argument("--old-inner-proposal-mask", type=Path)
    cli.add_argument("--proposal-source", choices=("official-oracle", "predicted-parser"))
    cli.add_argument("--protected-mask", type=Path)
    cli.add_argument("--steps", type=int, default=50)
    cli.add_argument("--seed", type=int, default=42)
    cli.add_argument("--garment-cache-dir", type=Path, required=True)
    args = cli.parse_args()
    if args.steps < 1:
        cli.error("--steps must be positive")
    custom_paths = (
        args.pose_target_mask, args.old_inner_proposal_mask, args.protected_mask
    )
    custom_variant = args.variant.startswith("custom-target")
    if custom_variant and any(path is None for path in custom_paths):
        cli.error("custom target requires pose target, old-inner proposal and protected masks")
    if custom_variant and args.proposal_source != "official-oracle":
        cli.error("custom target currently requires an explicit official-oracle proposal")
    if not custom_variant and any(path is not None for path in custom_paths):
        cli.error("Custom masks require --variant custom-target")
    if not custom_variant and args.proposal_source is not None:
        cli.error("--proposal-source requires a custom target variant")
    for name in ("person", "garment"):
        if not getattr(args, name).is_file():
            cli.error(f"missing {name}: {getattr(args, name)}")
    if not args.catvton_repo.is_dir():
        cli.error(f"missing CatVTON repository: {args.catvton_repo}")
    if custom_variant and any(not path.is_file() for path in custom_paths):
        cli.error("A custom target mask file is missing")
    inputs = (args.person, args.garment) + tuple(
        path for path in custom_paths if path is not None
    )
    if args.out.resolve() in {path.resolve() for path in inputs}:
        cli.error("output must not overwrite an input")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    catvton_tryon.CATVTON_REPO = args.catvton_repo.resolve()

    started = time.perf_counter()
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    try:
        pose = pose_analyzer.analyze(args.person)
        if not pose.landmarks:
            raise RuntimeError("no person pose landmarks")
        parser = ClothingParser(use_fashn=True)
        parsed = parser.parse(args.person, pose)
        segmentation = np.asarray(parsed["segmentation"], dtype=np.uint8)
        product = Product(
            product_id="CATVTON_MORPH_AB", name="외투 제거 상의 마스크 비교",
            category="top", color="", style="", purposes=[], body_shapes=[],
            price=0, season="사계절", stock=True, image_path=str(args.garment),
        )
        recommendation = Recommendation(
            rank=1, products=[product], total_score=0.0,
            score_breakdown={}, reasons=["격리 마스크 비교"],
        )
        adapter = CatVTONTryOn(
            seed=args.seed, num_inference_steps=args.steps, max_retries=0,
            upper_mask_policy="native" if custom_variant else "agnostic",
            post_quality_gate=False,
            garment_cache_dir=args.garment_cache_dir,
        )
        adapter._garment_parser = parser
        context = {
            "upper_mask": parsed["upper_mask"],
            "upper_style_mask": parsed["upper_style_mask"],
            "lower_mask": parsed["lower_mask"],
            "lower_style_mask": parsed["lower_style_mask"],
            "segmentation": segmentation,
            "pose": pose,
            "strict_vton": True,
        }
        custom_fraction = None
        if custom_variant:
            def load_binary(path: Path) -> np.ndarray:
                with Image.open(path) as opened:
                    if opened.size != (segmentation.shape[1], segmentation.shape[0]):
                        raise ValueError(f"Custom mask size differs from person: {path}")
                    return np.asarray(opened.convert("L")) >= 128

            target = load_binary(args.pose_target_mask)
            proposal = load_binary(args.old_inner_proposal_mask)
            protected = load_binary(args.protected_mask)
            custom = build_custom_target_mask(target, proposal, protected)
            context["upper_mask"] = custom.astype(np.uint8)
            context["upper_style_mask"] = custom.astype(np.uint8)
            custom_fraction = round(float(custom.mean()), 6)
        if args.variant in {"narrow", "custom-target"}:
            with patch.object(catvton_tryon, "_solidify_mask", experimental_close), patch.object(
                catvton_tryon, "_dilate_mask", experimental_dilate
            ):
                output = adapter.generate(args.person, recommendation, args.out, context)
        else:
            output = adapter.generate(args.person, recommendation, args.out, context)
        raw = adapter.last_raw_masks["top"]
        close_ratio, dilate_ratio = (
            (0.05, 0.03) if args.variant in {"standard", "custom-target-standard"}
            else (0.01, 0.01)
        )
        edit = morph_mask(raw, close_ratio, dilate_ratio) & ~np.isin(segmentation, PROTECT_LABELS)
        mask_path = args.out.with_name(args.out.stem + "_edit_mask.png")
        Image.fromarray(edit.astype(np.uint8) * 255).save(mask_path)
        record = {
            "variant": args.variant,
            "person": str(args.person.resolve()),
            "person_sha256": sha256(args.person),
            "garment": str(args.garment.resolve()),
            "garment_sha256": sha256(args.garment),
            "catvton_repo": str(args.catvton_repo.resolve()),
            "steps": args.steps,
            "seed": args.seed,
            "upper_mask_policy": adapter.upper_mask_policy,
            "close_ratio": close_ratio,
            "dilate_ratio": dilate_ratio,
            "raw_mask_fraction": round(float(raw.mean()), 6),
            "custom_target_fraction": custom_fraction,
            "pose_target_mask_sha256": sha256(args.pose_target_mask) if args.pose_target_mask else None,
            "old_inner_proposal_mask_sha256": (
                sha256(args.old_inner_proposal_mask) if args.old_inner_proposal_mask else None
            ),
            "protected_mask_sha256": sha256(args.protected_mask) if args.protected_mask else None,
            "proposal_source": args.proposal_source,
            "oracle_inner_proposal_pilot": args.proposal_source == "official-oracle",
            "edit_mask_fraction": round(float(edit.mean()), 6),
            "edit_mask_sha256": sha256(mask_path),
            "result": str(output),
            "result_sha256": sha256(output),
            "warnings": adapter.last_warnings,
            "seconds": round(time.perf_counter() - started, 2),
        }
        args.out.with_suffix(".json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)
    finally:
        pose_analyzer.close()


if __name__ == "__main__":
    main()
