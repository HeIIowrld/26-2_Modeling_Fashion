"""Validate a reviewed shoe manifest, generate inference-compatible crops, cache embeddings."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shoe_model import SHOE_LABELS, shoe_crop


def checksum(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def validate_manifest(rows):
    seen = {}
    if not rows:
        raise ValueError("Empty manifest")
    for row in rows:
        if row["shoe_label"] not in (*SHOE_LABELS, "-1"):
            raise ValueError(f"Unknown shoe_label: {row['shoe_label']}")
        if row["split"] not in ("train", "val", "test"):
            raise ValueError("Explicit train/val/test split required")
        if row.get("domain") not in ("catalog", "outfit"):
            raise ValueError("domain must be catalog or outfit")
        if not row.get("product_id") or (row["domain"] == "outfit" and not row.get("subject_id")):
            raise ValueError("Product ID required; outfit rows also require subject ID")
        for field in ("subject_id", "session_id", "product_id", "image_path"):
            value = row.get(field, "")
            if not value:
                continue
            key = (field, value)
            if key in seen and seen[key] != row["split"]:
                raise ValueError(f"Cross-split leakage: {key}")
            seen[key] = row["split"]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model-id", default="Marqo/marqo-fashionSigLIP")
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--verify-images", action="store_true")
    p.add_argument("--dedupe-phash", action="store_true")
    p.add_argument("--phash-distance", type=int, default=4)
    p.add_argument("--allow-pseudo-labels", action="store_true",
                   help="Include pseudo_ok rows in a research-only cache")
    args = p.parse_args()
    import numpy as np
    import torch
    from PIL import Image
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        p.error("CUDA unavailable")
    if args.batch_size <= 0 or not 0 <= args.phash_distance <= 16:
        p.error("Invalid batch size or pHash distance")
    with args.manifest.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    validate_manifest(rows)
    args.output.mkdir(parents=True, exist_ok=True)
    crop_dir = args.output.parent / "crops"
    crop_dir.mkdir(exist_ok=True)
    parser, accepted, rejected, hashes, exact = None, [], [], [], set()
    for i, row in enumerate(rows):
        row = dict(row)
        accepted_qualities = {"ok", "human_reviewed"}
        if args.allow_pseudo_labels:
            accepted_qualities.add("pseudo_ok")
        if row.get("quality", "ok") not in accepted_qualities:
            rejected.append({**row, "reason": "quality_not_ok"}); continue
        source = Path(row["image_path"])
        if not source.is_absolute(): source = args.manifest.parent / source
        crop_path = Path(row["crop_path"]) if row.get("crop_path") else None
        if crop_path and not crop_path.is_absolute(): crop_path = args.manifest.parent / crop_path
        try:
            if args.verify_images:
                with Image.open(source) as img: img.verify()
            row["source_sha256"] = checksum(source)
            if crop_path:
                allowed_policies = {"fashn_feet15_bbox_v1"}
                if args.allow_pseudo_labels:
                    allowed_policies |= {"fashionpedia_shoe_bbox_union_v1", "fashionpedia_socks_bbox_union_v1",
                                         "deepfashion_footwear_mask_bbox_v1"}
                if row["domain"] == "outfit" and row.get("crop_policy") not in allowed_policies:
                    raise ValueError("Existing outfit crop lacks approved crop provenance")
                with Image.open(crop_path) as img: crop = img.convert("RGB")
            elif row["domain"] == "catalog":
                with Image.open(source) as img: crop = img.convert("RGB")
                crop_path = source
            else:
                if parser is None:
                    from clothing_parser import ClothingParser
                    parser = ClothingParser(use_fashn=True)
                with Image.open(source) as img: rgb = np.asarray(img.convert("RGB"))
                segmentation = parser.parse(rgb, None)["segmentation"]
                crop = shoe_crop(rgb, segmentation)
                if crop is None: raise ValueError("not_visible")
                crop_path = crop_dir / f"{row['source_sha256']}.png"
                crop.save(crop_path)
            if min(crop.size) < 8 or max(crop.size) / min(crop.size) > 12:
                raise ValueError("invalid_crop_geometry")
            digest = checksum(crop_path)
            if digest in exact: raise ValueError("duplicate_sha256")
            phash = None
            if args.dedupe_phash:
                import imagehash
                phash = int(str(imagehash.phash(crop)), 16)
                if any((phash ^ old).bit_count() <= args.phash_distance for old in hashes):
                    raise ValueError("near_duplicate_phash")
            exact.add(digest)
            if phash is not None: hashes.append(phash)
            row.update(crop_path=str(crop_path.resolve()), crop_sha256=digest, width=crop.width, height=crop.height)
            accepted.append(row)
        except (OSError, ValueError) as exc:
            rejected.append({**row, "reason": str(exc)})
        if (i + 1) % 500 == 0: print(f"Verified {i+1}/{len(rows)}; accepted={len(accepted)}", flush=True)
    report_dir = args.output.parent / "reports"
    report_dir.mkdir(exist_ok=True)
    for name, records in (("rejected.csv", rejected), ("prepared.csv", accepted)):
        fields = sorted({k for r in records for k in r})
        with (args.output.parent / name).open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(records)
    summary = {"input_count": len(rows), "accepted": len(accepted), "rejected": len(rejected),
               "rejection_reasons": dict(Counter(r['reason'] for r in rejected)),
               "source_counts": dict(Counter(r['source'] for r in accepted)),
               "split_counts": dict(Counter(r['split'] for r in accepted)),
               "manifest_sha256": checksum(args.manifest), "phash_distance": args.phash_distance if args.dedupe_phash else None}
    (report_dir / "preparation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if not accepted: raise ValueError("No accepted images")
    # Same open_clip model and squash path as FashionClassifier._encode_image.
    import open_clip
    from fashion_attribute_model import apply_preprocess_mode
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{args.model_id}", device=args.device)
    model.eval().requires_grad_(False)
    for split in ("train", "val", "test"):
        part = [r for r in accepted if r["split"] == split]
        if not part: continue
        features = []
        for start in range(0, len(part), args.batch_size):
            batch = []
            for row in part[start:start+args.batch_size]:
                with Image.open(row["crop_path"]) as img:
                    batch.append(preprocess(apply_preprocess_mode(img.convert("RGB"), "squash")))
            with torch.inference_mode():
                emb = model.encode_image(torch.stack(batch).to(args.device), normalize=True).float().cpu()
            features.append(emb)
            if start % (args.batch_size * 10) == 0: print(f"Embedding {split}: {start}/{len(part)}", flush=True)
        cache = {"features": torch.cat(features),
                 "labels": torch.tensor([SHOE_LABELS.index(r['shoe_label']) if r['shoe_label'] != '-1' else -1 for r in part], dtype=torch.long),
                 "paths": [r["crop_path"] for r in part], "subject_ids": [r.get("subject_id", "") for r in part],
                 "product_ids": [r["product_id"] for r in part], "session_ids": [r.get("session_id", "") for r in part],
                 "checksums": [r["crop_sha256"] for r in part], "domains": [r["domain"] for r in part],
                 "label_sources": [r.get("label_source", "") for r in part],
                 "label_names": SHOE_LABELS, "backbone_model_id": args.model_id, "preprocessing": "squash",
                 "manifest_sha256": summary["manifest_sha256"]}
        torch.save(cache, args.output / f"{split}.pt")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
