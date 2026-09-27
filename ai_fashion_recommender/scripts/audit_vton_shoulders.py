#!/usr/bin/env python3
"""Research-only shoulder-regression guard for matched VTON candidates.

Pose landmarks follow clothing and do not reveal the hidden anatomical
shoulder. This audit only rejects further narrowing relative to a supplied
baseline; garment fidelity and other defects still need visual review.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from pose_analyzer import PoseAnalyzer  # noqa: E402


def shoulder_width(landmarks: dict) -> float | None:
    try:
        left = landmarks["left_shoulder"]
        right = landmarks["right_shoulder"]
        if left[2] < 0.7 or right[2] < 0.7:
            return None
        width = abs(float(left[0]) - float(right[0]))
        return width if 0.05 <= width <= 0.8 else None
    except (KeyError, IndexError, TypeError, ValueError):
        return None


def shoulder_gate(width: float | None, baseline_width: float | None,
                  maximum_additional_narrowing: float = 0.04) -> bool:
    if not 0 <= maximum_additional_narrowing < 0.25:
        raise ValueError("Additional narrowing tolerance must be in [0, 0.25)")
    return bool(width is not None and baseline_width is not None
                and width >= baseline_width * (1 - maximum_additional_narrowing))


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--original", type=Path, required=True)
    cli.add_argument("--baseline", type=Path, required=True)
    cli.add_argument("--candidate", type=Path, action="append", required=True)
    cli.add_argument("--max-additional-narrowing", type=float, default=0.04)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    inputs = (args.original, args.baseline, *args.candidate)
    if any(not path.is_file() for path in inputs):
        cli.error("Every input image must exist")
    if args.output.resolve() in {path.resolve() for path in inputs}:
        cli.error("Output must not overwrite an input")
    sizes = []
    for path in inputs:
        with Image.open(path) as image:
            sizes.append(image.size)
    if len(set(sizes)) != 1:
        cli.error("All images must use the same original canvas")

    analyzer = PoseAnalyzer(model_complexity=1)
    try:
        widths = [shoulder_width(analyzer.analyze(path).landmarks) for path in inputs]
    finally:
        analyzer.close()
    source_width, baseline_width, *candidate_widths = widths
    record = {
        "original": str(args.original.resolve()),
        "baseline": str(args.baseline.resolve()),
        "source_shoulder_width_fraction": source_width,
        "baseline_shoulder_width_fraction": baseline_width,
        "baseline_to_original": baseline_width / source_width if source_width and baseline_width else None,
        "maximum_additional_narrowing": args.max_additional_narrowing,
        "candidates": [
            {
                "path": str(path.resolve()),
                "shoulder_width_fraction": width,
                "candidate_to_baseline": width / baseline_width if width and baseline_width else None,
                "passes_shoulder_guard": shoulder_gate(
                    width, baseline_width, args.max_additional_narrowing
                ),
                "quality_not_assessed": True,
            }
            for path, width in zip(args.candidate, candidate_widths)
        ],
        "landmarks_are_not_anatomical_ground_truth": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
