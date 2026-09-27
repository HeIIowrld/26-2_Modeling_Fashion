"""Research-only layered VTON evaluation on a coat-on person and target top.

This runner does not integrate the external model into the service.  The
third-party repository has no explicit top-level license at the pinned commit;
review that before any redistribution or production use.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image


def pad_image(image: Image.Image, *, black: bool = False) -> Image.Image:
    width, height = 512, 896
    scale = min(width / image.width, height / image.height)
    resized = image.resize(
        (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
        Image.Resampling.LANCZOS,
    )
    color = (0, 0, 0) if black else (255, 255, 255)
    result = Image.new("RGB", (width, height), color)
    result.paste(resized, ((width - resized.width) // 2, (height - resized.height) // 2))
    return result


def draw_original_pose(person: Image.Image, weights_dir: Path) -> Image.Image:
    import onnxruntime as ort
    from easy_dwpose import DWposeDetector

    for name in ("yolox_l.onnx", "dw-ll_ucoco_384.onnx"):
        if not weights_dir.joinpath(name).is_file():
            raise FileNotFoundError(weights_dir / name)
    if weights_dir.name != "checkpoints":
        raise ValueError("easy_dwpose requires a directory named checkpoints")

    original_session = ort.InferenceSession

    def bounded_session(*args, **kwargs):
        options = ort.SessionOptions()
        options.intra_op_num_threads = min(8, os.cpu_count() or 8)
        options.inter_op_num_threads = 1
        kwargs.setdefault("sess_options", options)
        return original_session(*args, **kwargs)

    ort.InferenceSession = bounded_session
    original_cwd = Path.cwd()
    try:
        os.chdir(weights_dir.parent)
        detector = DWposeDetector(device="cpu")
        pose = detector(
            person, output_type="pil", include_hands=True, include_face=True
        )
    finally:
        os.chdir(original_cwd)
        ort.InferenceSession = original_session
    if np.count_nonzero(np.asarray(pose)) < 100:
        raise RuntimeError("Could not detect a usable pose in the original image")
    return pose.convert("RGB")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", type=Path, required=True)
    parser.add_argument("--garment", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--external-source-dir", type=Path, required=True)
    parser.add_argument("--pose-weights-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-device", default="cuda:1")
    args = parser.parse_args()
    if not 1 <= args.steps <= 60:
        parser.error("steps must be in 1..60")
    if not args.external_source_dir.joinpath("pipeline.py").is_file():
        parser.error("external model pipeline.py is missing")
    for path in (args.person, args.garment, args.model_dir, args.pose_weights_dir):
        if not path.exists():
            parser.error(f"missing input: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(args.person) as opened:
        person = opened.convert("RGB")
    with Image.open(args.garment) as opened:
        garment = opened.convert("RGB")

    started = time.perf_counter()
    pose = draw_original_pose(person, args.pose_weights_dir)
    prepared = {
        "person": pad_image(person),
        "garment": pad_image(garment),
        "pose": pad_image(pose, black=True),
    }
    for name, image in prepared.items():
        image.save(args.output.with_name(f"{args.output.stem}_{name}.png"))

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This pilot requires at least one CUDA GPU")
    if args.text_device != "cpu" and torch.cuda.device_count() < 2:
        raise RuntimeError("A CUDA text encoder requires a second visible GPU")
    sys.path.insert(0, str(args.external_source_dir.resolve()))
    from pipeline import LayeringVTONPipeline

    model = LayeringVTONPipeline(
        str(args.model_dir), str(args.external_source_dir / "weights"),
        device="cuda:0", text_device=args.text_device,
    )
    loaded_seconds = time.perf_counter() - started
    result = model(
        person_img=prepared["person"],
        garment_img=prepared["garment"],
        pose_img=prepared["pose"],
        description=args.description,
        num_inference_steps=args.steps,
        true_cfg_scale=4.0,
        guidance_scale=1.0,
        seed=args.seed,
    )
    result.save(args.output)
    args.output.with_suffix(".json").write_text(
        json.dumps({
            "person": str(args.person),
            "person_sha256": sha256(args.person),
            "garment": str(args.garment),
            "garment_sha256": sha256(args.garment),
            "output": str(args.output),
            "output_sha256": sha256(args.output),
            "description": args.description,
            "steps": args.steps,
            "seed": args.seed,
            "text_device": args.text_device,
            "loaded_seconds": round(loaded_seconds, 2),
            "total_seconds": round(time.perf_counter() - started, 2),
            "model": "Layering-Virtual-Try-On 102b11c2 + Qwen-Image-Edit-2509",
            "production_approved": False,
        }, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
