"""Cache image evidence for offline rendering audits, without generating images.

Each cache entry includes the actual image, analysis code and model hashes. Runs
in an isolated GPU job; source render records are never overwritten.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))
from offline_outfit_library import atomic_json, digest


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--people", type=Path, required=True)
    args = cli.parse_args()
    from config import FASHION_ATTRIBUTE_HEADS_PATH
    from clothing_parser import ClothingParser
    from fashion_model import FashionClassifier
    from outfit_analyzer import OutfitAnalyzer
    from pose_analyzer import PoseAnalyzer
    from shoe_tryon import foot_edit_mask
    from schemas import PoseAnalysis

    parser, poses = ClothingParser(use_fashn=True), PoseAnalyzer()
    classifier = FashionClassifier(enabled=True, attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH)
    analyzer = OutfitAnalyzer(parser, classifier)
    fingerprint = digest({"checkpoint": sha(FASHION_ATTRIBUTE_HEADS_PATH),
        "code": {name: sha(PROJECT / "src" / name) for name in
                 ("outfit_analyzer.py", "clothing_parser.py", "garment_attribute_analyzer.py",
                  "pose_analyzer.py", "fashion_model.py", "fashion_attribute_model.py", "config.py")},
        "script": sha(__file__)})
    jobs = [("people", p, None, None) for p in sorted(args.people.rglob("*"))
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    records = []
    for path in sorted((args.root / "renders").rglob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        output = path.with_suffix(".png")
        if output.is_file() and record.get("reference_pose"):
            jobs.append(("renders", output, PoseAnalysis(**record["reference_pose"]), record))
    catalog = json.loads((args.root / "catalog.json").read_text(encoding="utf-8"))
    for row in catalog["products"]:
        p = row["product"]
        if p["category"] != "shoes":
            jobs.append(("products", Path(p["image_path"]), False, row))
    try:
        for i, (kind, path, pose, original) in enumerate(jobs, 1):
            target = args.output / kind / f"{path.stem}.json"
            key = digest({"image": sha(path), "analysis": fingerprint,
                          "pose": asdict(pose) if pose else pose})
            if target.exists() and target.with_suffix(".npz").exists():
                cached = json.loads(target.read_text(encoding="utf-8"))
                if cached.get("key") == key:
                    continue
            try:
                if kind == "products":
                    parsed = parser.parse(path, None)
                    outfit = None
                else:
                    pose = pose or poses.analyze(path)
                    outfit, parsed = analyzer.analyze(path, pose)
                feet_error = ""
                if kind == "people":
                    try:
                        foot_edit_mask(parsed["segmentation"], pose)
                    except Exception as exc:
                        feet_error = str(exc)
                result = {"key": key, "image_sha256": sha(path), "analysis_fingerprint": fingerprint,
                    "path": str(path), "kind": kind, "name": path.name,
                    "pose": asdict(pose) if pose else None,
                    "outfit": asdict(outfit) if outfit else None,
                    "feet_error": feet_error, "original": original,
                    "gender": "여성" if "WOMEN" in path.name else "남성" if "MEN" in path.name else ""}
                target.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target.with_suffix(".npz"), segmentation=parsed["segmentation"])
                atomic_json(target, result)
                print(f"inspected {i}/{len(jobs)} {kind}/{path.name}", flush=True)
            except Exception as exc:
                records.append({"path": str(path), "error": repr(exc)})
                print(f"failed {path.name}: {exc}", flush=True)
        atomic_json(args.output / "inspection.json", {"analysis_fingerprint": fingerprint,
                    "requested": len(jobs), "errors": records})
    finally:
        poses.close()


if __name__ == "__main__":
    main()
