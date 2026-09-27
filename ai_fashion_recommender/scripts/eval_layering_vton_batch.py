#!/usr/bin/env python3
"""Research-only LVTON batch: reuse one expensive model load across cases."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from PIL import Image

from eval_layering_vton_outerwear import draw_original_pose, pad_image, sha256


def load_cases(manifest: Path) -> list[dict[str, Path | str]]:
    content = json.loads(manifest.read_text(encoding="utf-8"))
    if not isinstance(content, list) or not 1 <= len(content) <= 8:
        raise ValueError("Manifest must contain one to eight cases")
    cases: list[dict[str, Path | str]] = []
    outputs: set[Path] = set()
    for index, item in enumerate(content):
        if not isinstance(item, dict) or set(item) != {"person", "garment", "output", "description"}:
            raise ValueError(f"Case {index} needs person, garment, output, description")
        case: dict[str, Path | str] = {"description": str(item["description"])}
        if len(case["description"]) < 20:
            raise ValueError(f"Case {index} description is too short")
        for name in ("person", "garment", "output"):
            path = Path(item[name])
            case[name] = (path if path.is_absolute() else manifest.parent / path).resolve()
        person, garment, output = (case[name] for name in ("person", "garment", "output"))
        if not person.is_file() or not garment.is_file():
            raise FileNotFoundError(f"Case {index} is missing an input image")
        if output in {person, garment} or output in outputs:
            raise ValueError(f"Case {index} output collides with an input or another output")
        if output.suffix.lower() != ".png":
            raise ValueError(f"Case {index} output must be PNG")
        outputs.add(output)
        cases.append(case)
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--external-source-dir", type=Path, required=True)
    parser.add_argument("--pose-weights-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-device", default="cpu")
    args = parser.parse_args()
    if not 1 <= args.steps <= 60:
        parser.error("steps must be in 1..60")
    cases = load_cases(args.manifest)
    for path in (args.model_dir, args.external_source_dir, args.pose_weights_dir):
        if not path.is_dir():
            parser.error(f"Missing model/input directory: {path}")

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This batch requires at least one CUDA GPU")
    if args.text_device != "cpu" and torch.cuda.device_count() < 2:
        raise RuntimeError("A CUDA text encoder requires a second visible GPU")
    sys.path.insert(0, str(args.external_source_dir.resolve()))
    from pipeline import LayeringVTONPipeline

    started = time.perf_counter()
    model = LayeringVTONPipeline(
        str(args.model_dir), str(args.external_source_dir / "weights"),
        device="cuda:0", text_device=args.text_device,
    )
    loaded_seconds = round(time.perf_counter() - started, 2)
    print(f"Model loaded in {loaded_seconds}s; running {len(cases)} cases", flush=True)
    for index, case in enumerate(cases, 1):
        person_path = case["person"]
        garment_path = case["garment"]
        output_path = case["output"]
        description = case["description"]
        print(f"Case {index}/{len(cases)}: {output_path.name}", flush=True)
        case_started = time.perf_counter()
        with Image.open(person_path) as opened:
            person = opened.convert("RGB")
        with Image.open(garment_path) as opened:
            garment = opened.convert("RGB")
        pose = draw_original_pose(person, args.pose_weights_dir)
        prepared = {
            "person": pad_image(person),
            "garment": pad_image(garment),
            "pose": pad_image(pose, black=True),
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for name, image in prepared.items():
            image.save(output_path.with_name(f"{output_path.stem}_{name}.png"))
        result = model(
            person_img=prepared["person"],
            garment_img=prepared["garment"],
            pose_img=prepared["pose"],
            description=description,
            num_inference_steps=args.steps,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            seed=args.seed,
        )
        result.save(output_path)
        output_path.with_suffix(".json").write_text(
            json.dumps({
                "person": str(person_path),
                "person_sha256": sha256(person_path),
                "garment": str(garment_path),
                "garment_sha256": sha256(garment_path),
                "output": str(output_path),
                "output_sha256": sha256(output_path),
                "description": description,
                "steps": args.steps,
                "seed": args.seed,
                "text_device": args.text_device,
                "loaded_seconds": loaded_seconds,
                "case_seconds": round(time.perf_counter() - case_started, 2),
                "model": "Layering-Virtual-Try-On 102b11c2 + Qwen-Image-Edit-2509",
                "production_approved": False,
            }, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Saved {output_path}", flush=True)


if __name__ == "__main__":
    main()
