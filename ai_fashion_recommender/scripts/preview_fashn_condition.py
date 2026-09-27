#!/usr/bin/env python3
"""Preview a studio-background prior from saved FASHN agnostic diagnostics."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

from eval_fashn_vton15_outerwear import studio_background_condition


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-dir", type=Path, required=True)
    parser.add_argument("--target-body-mask", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    with Image.open(args.diagnostic_dir / "person_resized.png") as image:
        original = np.asarray(image.convert("RGB"))
    with Image.open(args.diagnostic_dir / "agnostic_input.png") as image:
        agnostic = np.asarray(image.convert("RGB"))
    with Image.open(args.diagnostic_dir / "agnostic_base_mask.png") as image:
        base = np.asarray(image.convert("L")) > 127
    with Image.open(args.target_body_mask) as image:
        target = np.asarray(
            image.convert("L").resize((original.shape[1], original.shape[0]), Image.Resampling.NEAREST)
        ) > 127
    conditioned, ring = studio_background_condition(agnostic, original, base, target)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(conditioned).save(args.out)
    print(f"studio_background_ring_fraction={ring.mean():.6f}")


if __name__ == "__main__":
    main()
