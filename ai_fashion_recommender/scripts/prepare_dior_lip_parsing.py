#!/usr/bin/env python3
"""Save an isolated SCHP-LIP prediction for DiOr's missing-label pilot.

Predicted parsing is not DeepFashion-MM ground truth. Do not mix its results
with official-mask cases when estimating segmentation-model performance.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from outerwear_layer_parser import OuterwearLayerParser  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--image", type=Path, required=True)
    cli.add_argument("--checkpoint", type=Path, required=True)
    cli.add_argument("--out", type=Path, required=True)
    args = cli.parse_args()
    if not args.image.is_file() or not args.checkpoint.is_file():
        cli.error("image and checkpoint must exist")
    parser = OuterwearLayerParser(args.checkpoint)
    with Image.open(args.image) as image:
        labels = parser.predict(image)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(labels).save(args.out)
    values, counts = np.unique(labels, return_counts=True)
    record = {
        "taxonomy": "LIP-20 predicted by SCHP, not official segmentation",
        "image": str(args.image),
        "image_sha256": sha256(args.image),
        "checkpoint_sha256": sha256(args.checkpoint),
        "label_pixels": {str(int(value)): int(count) for value, count in zip(values, counts)},
        "coat_fraction": round(float(np.mean(labels == 7)), 6),
        "top_fraction": round(float(np.mean(labels == 5)), 6),
    }
    args.out.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, indent=2), flush=True)


if __name__ == "__main__":
    main()
