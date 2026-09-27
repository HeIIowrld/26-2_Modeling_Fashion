#!/usr/bin/env python3
"""Isolated FASHN VTON v1.5 comparison on outerwear-to-top examples.

The upstream package and weights are supplied through PYTHONPATH/--weights-dir;
this script never changes the application's production try-on route.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hair_coat_release_mask(
    seg_pred: np.ndarray,
    base_edit_mask: np.ndarray,
    coat_proposal: np.ndarray,
    *,
    near_ratio: float = 0.016,
    max_fraction: float = 0.02,
) -> np.ndarray:
    """Select only proposed coat pixels that the parser protected as hair.

    This is an experimental release proposal, not a claim that the protected
    pixels are all outerwear. A coverage cap prevents broad hair deletion.
    """
    if seg_pred.ndim != 2 or base_edit_mask.shape != seg_pred.shape or coat_proposal.shape != seg_pred.shape:
        raise ValueError("segmentation, edit mask and coat proposal must have identical dimensions")
    if not (0 < near_ratio <= 0.1 and 0 < max_fraction <= 0.1):
        raise ValueError("near_ratio and max_fraction must be within (0, 0.1]")
    radius = max(1, int(round(min(seg_pred.shape) * near_ratio)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    near_edit = cv2.dilate(base_edit_mask.astype(np.uint8), kernel).astype(bool)
    release = (seg_pred == 2) & coat_proposal.astype(bool) & near_edit & ~base_edit_mask
    if release.mean() > max_fraction:
        raise ValueError("hair/coat release exceeds the configured image-area cap")
    return release


def studio_background_condition(
    agnostic: np.ndarray,
    original: np.ndarray,
    base_edit_mask: np.ndarray,
    target_body_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Replace the agnostic outer ring with a simple studio-background prior.

    This is only a controlled diagnostic: the target corridor is not a
    measurement of hidden body shape, and studio backgrounds must be reviewed.
    """
    if agnostic.shape != original.shape or agnostic.ndim != 3 or agnostic.shape[2] != 3:
        raise ValueError("agnostic and original must be matching RGB images")
    if base_edit_mask.shape != original.shape[:2] or target_body_mask.shape != base_edit_mask.shape:
        raise ValueError("edit and target masks must match the person image")
    ring = base_edit_mask.astype(bool) & ~target_body_mask.astype(bool)
    height, width = base_edit_mask.shape
    edge_width = max(1, int(round(width * 0.08)))
    left = np.median(original[:, :edge_width], axis=1).astype(np.float32)
    right = np.median(original[:, -edge_width:], axis=1).astype(np.float32)
    blend = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :, None]
    background = np.rint(left[:, None] * (1.0 - blend) + right[:, None] * blend)
    conditioned = agnostic.copy()
    conditioned[ring] = np.clip(background, 0, 255).astype(np.uint8)[ring]
    return conditioned, ring


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--garment", type=Path, required=True)
    parser.add_argument("--garment-photo-type", choices=("model", "flat-lay"), required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--masked", action="store_true", help="Use parser masking instead of maskless mode")
    parser.add_argument("--coat-proposal-mask", type=Path, help="Optional SCHP coat mask for a hair-boundary oracle test")
    parser.add_argument("--target-body-mask", type=Path, help="Optional pose corridor used only with --studio-background-outside-target")
    parser.add_argument("--studio-background-outside-target", action="store_true", help="Condition outer agnostic ring as studio background")
    parser.add_argument("--debug-preprocess-dir", type=Path, help="Save the upstream clothing-agnostic input and labels")
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    if args.coat_proposal_mask and not args.masked:
        parser.error("--coat-proposal-mask requires --masked")
    if args.studio_background_outside_target and (not args.masked or not args.target_body_mask):
        parser.error("--studio-background-outside-target requires --masked and --target-body-mask")
    for path in (args.person, args.garment, args.weights_dir):
        if not path.exists():
            parser.error(f"missing input: {path}")
    if args.coat_proposal_mask and not args.coat_proposal_mask.exists():
        parser.error(f"missing coat proposal: {args.coat_proposal_mask}")
    if args.target_body_mask and not args.target_body_mask.exists():
        parser.error(f"missing target body mask: {args.target_body_mask}")
    args.out.parent.mkdir(parents=True, exist_ok=True)

    from fashn_vton import TryOnPipeline
    from fashn_vton.dwpose import DWposeDetector
    import fashn_vton.pipeline as pipeline_module
    import onnxruntime as ort

    class CpuPoseTryOnPipeline(TryOnPipeline):
        """Keep the VTON network on CUDA while using portable CPU ONNX pose."""

        def _setup_pose_model(self) -> None:
            # Upstream DWPose opens ONNX sessions without explicit CPU threads;
            # on Slurm this can create hundreds of threads outside our cpuset.
            make_session = ort.InferenceSession

            def bounded_session(*args, **kwargs):
                options = ort.SessionOptions()
                options.intra_op_num_threads = min(8, os.cpu_count() or 8)
                options.inter_op_num_threads = 1
                kwargs.setdefault("sess_options", options)
                return make_session(*args, **kwargs)

            ort.InferenceSession = bounded_session
            try:
                self.pose_model = DWposeDetector(
                    checkpoints_dir=str(Path(self.weights_dir) / "dwpose"), device="cpu"
                )
            finally:
                ort.InferenceSession = make_session

    started = time.perf_counter()
    pipeline = CpuPoseTryOnPipeline(weights_dir=str(args.weights_dir), device="cuda")
    loaded_seconds = time.perf_counter() - started
    with Image.open(args.person) as image:
        person = image.convert("RGB")
    with Image.open(args.garment) as image:
        garment = image.convert("RGB")
    preprocess_info = {}
    make_agnostic = pipeline_module.create_clothing_agnostic_image

    def observed_agnostic(*positional, **keyword):
        original = keyword["img_np"].copy()
        seg_pred = keyword["seg_pred"]
        agnostic = make_agnostic(*positional, **keyword)
        base_mask = np.any(agnostic != original, axis=2)
        release = np.zeros_like(base_mask)
        if args.coat_proposal_mask:
            with Image.open(args.coat_proposal_mask) as image:
                proposal = np.asarray(
                    image.convert("L").resize(
                        (original.shape[1], original.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 127
            release = hair_coat_release_mask(seg_pred, base_mask, proposal)
            agnostic[release] = 127
        ring = np.zeros_like(base_mask)
        if args.studio_background_outside_target:
            with Image.open(args.target_body_mask) as image:
                target = np.asarray(
                    image.convert("L").resize(
                        (original.shape[1], original.shape[0]), Image.Resampling.NEAREST
                    )
                ) > 127
            agnostic, ring = studio_background_condition(
                agnostic, original, base_mask | release, target
            )
        preprocess_info.update({
            "agnostic_base_fraction": round(float(base_mask.mean()), 6),
            "released_hair_coat_fraction": round(float(release.mean()), 6),
            "released_hair_coat_pixels": int(release.sum()),
            "studio_background_ring_fraction": round(float(ring.mean()), 6),
            "preprocessed_size": [int(original.shape[1]), int(original.shape[0])],
        })
        if args.debug_preprocess_dir:
            args.debug_preprocess_dir.mkdir(parents=True, exist_ok=True)
            Image.fromarray(original).save(args.debug_preprocess_dir / "person_resized.png")
            Image.fromarray(seg_pred.astype(np.uint8)).save(args.debug_preprocess_dir / "person_labels.png")
            Image.fromarray(base_mask.astype(np.uint8) * 255).save(args.debug_preprocess_dir / "agnostic_base_mask.png")
            Image.fromarray(release.astype(np.uint8) * 255).save(args.debug_preprocess_dir / "released_hair_coat_mask.png")
            Image.fromarray(ring.astype(np.uint8) * 255).save(args.debug_preprocess_dir / "studio_background_ring.png")
            Image.fromarray(agnostic).save(args.debug_preprocess_dir / "agnostic_input.png")
        return agnostic

    if args.debug_preprocess_dir or args.coat_proposal_mask or args.studio_background_outside_target:
        pipeline_module.create_clothing_agnostic_image = observed_agnostic
    try:
        result = pipeline(
            person_image=person,
            garment_image=garment,
            category="tops",
            garment_photo_type=args.garment_photo_type,
            num_samples=1,
            num_timesteps=args.steps,
            guidance_scale=args.guidance,
            seed=args.seed,
            segmentation_free=not args.masked,
        )
    finally:
        pipeline_module.create_clothing_agnostic_image = make_agnostic
    generated_seconds = time.perf_counter() - started - loaded_seconds
    result.images[0].save(args.out)
    metadata = {
        "model": "fashn-ai/fashn-vton-1.5",
        "person": str(args.person.resolve()),
        "person_sha256": file_sha256(args.person),
        "garment": str(args.garment.resolve()),
        "garment_sha256": file_sha256(args.garment),
        "garment_photo_type": args.garment_photo_type,
        "segmentation_free": not args.masked,
        "steps": args.steps,
        "seed": args.seed,
        "guidance": args.guidance,
        "input_size": list(person.size),
        "output_size": list(result.images[0].size),
        "pipeline_load_seconds": round(loaded_seconds, 2),
        "generation_seconds": round(generated_seconds, 2),
        "output_sha256": file_sha256(args.out),
        "coat_proposal_mask": str(args.coat_proposal_mask.resolve()) if args.coat_proposal_mask else None,
        "target_body_mask": str(args.target_body_mask.resolve()) if args.target_body_mask else None,
        "studio_background_outside_target": args.studio_background_outside_target,
        **preprocess_info,
    }
    args.out.with_suffix(".json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(metadata, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
