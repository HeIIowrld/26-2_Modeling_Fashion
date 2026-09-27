#!/usr/bin/env python3
"""Export actual FASHN labels for an isolated source-mask feasibility audit.

Uses a local cached checkpoint. The output is a parser prediction, not an
official outer/inner annotation. It must not be treated as layer ground truth.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--image", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = cli.parse_args()
    if not args.image.is_file() or args.output.suffix.lower() != ".png":
        cli.error("Existing image and PNG output are required")
    if args.image.resolve() == args.output.resolve():
        cli.error("Output must differ from the input photo")
    from fashn_human_parser import FashnHumanParser

    with Image.open(args.image) as opened:
        rgb = np.asarray(opened.convert("RGB"))
    parser = FashnHumanParser(device=args.device)
    labels = np.asarray(parser.predict(rgb), dtype=np.uint8)
    if labels.shape != rgb.shape[:2]:
        raise RuntimeError("FASHN labels do not match source photo dimensions")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(args.output)
    ids, counts = np.unique(labels, return_counts=True)
    record = {
        "image_sha256": sha256(args.image),
        "labels_sha256": sha256(args.output),
        "shape": list(labels.shape),
        "label_counts": {str(int(k)): int(v) for k, v in zip(ids, counts)},
        "device": args.device,
        "not_layer_ground_truth": True,
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(record, ensure_ascii=False))


if __name__ == "__main__":
    main()
