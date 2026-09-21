"""Build conservative catalog warm-up labels from the official UT-Zap50K files.

Academic/non-commercial use only. Other categories need manual review: notably
Sneakers and Athletic Shoes cannot distinguish sneakers from running shoes.
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--metadata", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    images = {f.stem.replace(".", "-"): f.resolve() for f in args.images.rglob("*.jpg")}
    mapping = {
        ("Shoes", "Loafers"): "로퍼",
        ("Slippers", "Slipper Flats"): "슬리퍼",
        # These source categories intentionally fall outside the ten specific
        # FITTA shoe types. Oxford is kept separate from Derby per SHOE_HEAD.md.
        **{("Shoes", name): "기타 신발" for name in (
            "Oxfords", "Clogs and Mules", "Boat Shoes", "Firstwalker",
            "Prewalker", "Crib Shoes", "Flats",
        )},
        **{("Slippers", name): "기타 신발" for name in ("Boot", "Slipper Heels")},
    }
    rows, review, counts, missing = [], [], Counter(), []
    with args.metadata.open(encoding="utf-8-sig", newline="") as f:
        for item in csv.DictReader(f):
            image = images.get(item["CID"])
            if image is None:
                missing.append(item["CID"])
                continue
            product = "utzap50k:" + item["CID"].split("-")[0]
            bucket = int(hashlib.sha256(product.encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 80 else "val" if bucket < 90 else "test"
            label = mapping.get((item["Category"], item["SubCategory"]))
            row = {"image_path": str(image), "crop_path": "", "source": "UT-Zap50K",
                   "source_id": item["CID"], "subject_id": "", "session_id": "",
                   "product_id": product, "shoe_label": label or "-1", "split": split,
                   "quality": "ok" if label else "needs_review", "domain": "catalog",
                   "license": "academic-noncommercial-only",
                   "label_source": "source_metadata_ground_truth" if label else "",
                   "notes": f"Source category: {item['Category']}/{item['SubCategory']}"}
            (rows if label else review).append(row)
            if label: counts[label] += 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for path, data in ((args.output, rows), (args.output.with_name("needs_review.csv"), review)):
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0])); w.writeheader(); w.writerows(data)
    summary = {"warmup_count": len(rows), "needs_review": len(review), "labels": dict(counts), "missing_images": missing}
    args.output.with_name("import_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
