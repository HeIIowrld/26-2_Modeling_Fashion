"""Create weakly labelled outfit shoe crops from research datasets.

The output is deliberately marked ``pseudo_label``. It can bootstrap the shoe
head and produce a review queue, but it is not human-reviewed ground truth.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from shoe_model import SHOE_LABELS, shoe_crop


PROMPTS = {
    "스니커즈": ("casual lifestyle sneakers", "canvas sneakers", "fashion sneakers"),
    "러닝화": ("technical running shoes", "athletic running shoes", "performance running sneakers"),
    "로퍼": ("leather loafers", "penny loafers", "slip-on loafers"),
    "더비슈즈": ("derby dress shoes with open lacing", "formal lace-up leather shoes", "derby shoes"),
    "메리제인": ("mary jane shoes with an instep strap", "mary jane flats", "mary jane heels"),
    "펌프스": ("closed-toe pump heels", "women's pumps", "high heel pumps"),
    "샌들": ("sandals with heel straps", "open-toe sandals", "strappy sandals"),
    "슬리퍼": ("backless slippers", "slide sandals", "flip flops"),
    "부츠": ("fashion boots", "ankle boots", "knee-high boots"),
    "워커": ("combat boots", "work boots", "hiking boots with thick soles"),
    "기타 신발": ("clogs or unusual footwear", "specialized footwear", "other shoes"),
    "맨발": ("bare feet without shoes", "a person wearing only socks", "barefoot feet"),
}


def stable_split(group_id: str) -> str:
    bucket = int(hashlib.sha256(group_id.encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 80 else "val" if bucket < 90 else "test"


def fashionpedia_records(annotation_path: Path, image_root: Path, crop_root: Path):
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    shoe_ids = {c["id"] for c in data["categories"] if c["name"] == "shoe"}
    sock_ids = {c["id"] for c in data["categories"] if c["name"] == "sock"}
    images = {item["id"]: item for item in data["images"]}
    shoe_boxes, sock_boxes = defaultdict(list), defaultdict(list)
    for annotation in data["annotations"]:
        if annotation["category_id"] in shoe_ids and annotation.get("area", 0) >= 64:
            shoe_boxes[annotation["image_id"]].append(annotation["bbox"])
        elif annotation["category_id"] in sock_ids and annotation.get("area", 0) >= 64:
            sock_boxes[annotation["image_id"]].append(annotation["bbox"])
    # Socks without any shoe annotation are a useful negative class. The
    # service intentionally treats socks-only and barefoot as one safe class.
    for image_id in shoe_boxes.keys() | sock_boxes.keys():
        has_shoes = bool(shoe_boxes[image_id])
        image_boxes = shoe_boxes[image_id] if has_shoes else sock_boxes[image_id]
        info = images[image_id]
        image_path = image_root / info["file_name"]
        if not image_path.is_file():
            continue
        from PIL import Image
        import numpy as np
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        segmentation = np.zeros(rgb.shape[:2], dtype=np.uint8)
        for x, y, width, height in image_boxes:
            x1, y1 = max(0, int(x)), max(0, int(y))
            x2 = min(rgb.shape[1], int(x + width + .5))
            y2 = min(rgb.shape[0], int(y + height + .5))
            segmentation[y1:y2, x1:x2] = 15
        crop = shoe_crop(rgb, segmentation)
        if crop is None:
            continue
        suffix = "shoe" if has_shoes else "socks"
        crop_path = crop_root / "fashionpedia" / f"{image_id}_{suffix}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=95)
        row = {
            "image_path": str(image_path.resolve()), "crop_path": str(crop_path.resolve()),
            "source": "Fashionpedia", "source_id": str(image_id), "group_id": f"fashionpedia:{image_id}",
            "domain": "outfit",
            "crop_policy": ("fashionpedia_shoe_bbox_union_v1" if has_shoes
                            else "fashionpedia_socks_bbox_union_v1"),
            "license": str(info.get("license", "source-specific")), "notes": info.get("original_url", ""),
        }
        if not has_shoes:
            row.update(shoe_label="맨발", pseudo_confidence=1.0, pseudo_margin=1.0,
                       second_label="", scores_json="{}",
                       label_source="fashionpedia_socks_without_shoe_v1")
        yield row


def deepfashion_records(image_root: Path, mask_root: Path, crop_root: Path, *, derive_barefoot: bool, device: str):
    from PIL import Image
    import numpy as np
    masks = {path.stem.removesuffix("_segm"): path for path in mask_root.rglob("*.png")}
    fashn_parser = None
    for image_path in image_root.rglob("*.jpg"):
        mask_path = masks.get(image_path.stem)
        if mask_path is None:
            continue
        with Image.open(image_path) as source:
            rgb = np.asarray(source.convert("RGB"))
        with Image.open(mask_path) as source:
            mask = np.asarray(source)
        if mask.shape != rgb.shape[:2]:
            continue
        has_footwear = bool((mask == 11).any())
        has_socks = bool((mask == 18).any())
        source_label = 11 if has_footwear else 18 if has_socks else None
        if source_label is not None:
            segmentation = np.where(mask == source_label, 15, 0).astype(np.uint8)
        elif derive_barefoot:
            if fashn_parser is None:
                from fashn_human_parser import FashnHumanParser
                fashn_parser = FashnHumanParser(device=device)
            segmentation = np.asarray(fashn_parser.predict(rgb), dtype=np.uint8)
        else:
            continue
        crop = shoe_crop(rgb, segmentation)
        if crop is None:
            continue
        crop_path = crop_root / "deepfashion_multimodal" / f"{image_path.stem}.jpg"
        crop_path.parent.mkdir(parents=True, exist_ok=True)
        crop.save(crop_path, quality=95)
        match = re.search(r"id_\d+", image_path.stem)
        group = match.group(0) if match else image_path.stem
        row = {
            "image_path": str(image_path.resolve()), "crop_path": str(crop_path.resolve()),
            "source": "DeepFashion-MultiModal", "source_id": image_path.stem,
            "group_id": f"deepfashion:{group}", "domain": "outfit",
            "crop_policy": ("deepfashion_footwear_mask_bbox_v1" if source_label is not None
                            else "fashn_feet15_bbox_v1"),
            "license": "non-commercial-research-only",
            "notes": (f"official {'footwear' if has_footwear else 'socks-without-footwear'} mask class {source_label}"
                      if source_label is not None else "no official footwear/socks mask; FASHN feet crop"),
        }
        if not has_footwear:
            row.update(shoe_label="맨발", pseudo_confidence=1.0, pseudo_margin=1.0,
                       second_label="", scores_json="{}",
                       label_source=("deepfashion_socks_without_footwear_v1" if has_socks
                                     else "deepfashion_no_footwear_fashn_feet_v1"))
        yield row


def classify(records, model_id: str, device: str, batch_size: int):
    import numpy as np
    import torch
    import open_clip
    from PIL import Image
    from fashion_attribute_model import apply_preprocess_mode
    model, _, preprocess = open_clip.create_model_and_transforms(f"hf-hub:{model_id}", device=device)
    tokenizer = open_clip.get_tokenizer(f"hf-hub:{model_id}")
    model.eval().requires_grad_(False)
    # These datasets provide a shoe/footwear annotation. Barefoot is therefore
    # excluded from zero-shot candidates; DeepFashion socks-only rows are added
    # explicitly above as the service's barefoot/socks negative class.
    labels = list(SHOE_LABELS[:-1])
    text_features = []
    with torch.inference_mode():
        for label in labels:
            tokens = tokenizer([f"a full-body fashion photo showing {prompt}" for prompt in PROMPTS[label]]).to(device)
            features = model.encode_text(tokens, normalize=True).float().mean(0)
            text_features.append(features / features.norm())
    text_features = torch.stack(text_features)
    unresolved = [row for row in records if not row.get("shoe_label")]
    for start in range(0, len(unresolved), batch_size):
        part = unresolved[start:start + batch_size]
        tensors = []
        for row in part:
            with Image.open(row["crop_path"]) as image:
                tensors.append(preprocess(apply_preprocess_mode(image.convert("RGB"), "squash")))
        with torch.inference_mode():
            image_features = model.encode_image(torch.stack(tensors).to(device), normalize=True).float()
            similarities = image_features @ text_features.T
            probabilities = torch.softmax(similarities * 20.0, dim=-1)
            values, indices = probabilities.topk(2, dim=-1)
        for row, probability, value, index in zip(part, probabilities.cpu(), values.cpu(), indices.cpu()):
            row["shoe_label"] = labels[int(index[0])]
            row["pseudo_confidence"] = float(value[0])
            row["pseudo_margin"] = float(value[0] - value[1])
            row["second_label"] = labels[int(index[1])]
            row["scores_json"] = json.dumps({label: round(float(score), 6) for label, score in zip(labels, probability)}, ensure_ascii=False)
            row["label_source"] = "fashion_siglip_zero_shot_v1"
        print(f"Classified {min(start + batch_size, len(unresolved))}/{len(unresolved)}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("fashionpedia", "deepfashion"), required=True)
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path)
    parser.add_argument("--crop-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-id", default="Marqo/marqo-fashionSigLIP")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--minimum-confidence", type=float, default=.18)
    parser.add_argument("--minimum-margin", type=float, default=.06)
    parser.add_argument("--derive-barefoot-with-fashn", action="store_true",
                        help="Use FASHN feet crops as barefoot when official masks contain neither footwear nor socks")
    args = parser.parse_args()
    if args.dataset == "fashionpedia":
        if not args.annotations: parser.error("--annotations is required")
        records = list(fashionpedia_records(args.annotations, args.images, args.crop_root))
    else:
        if not args.masks: parser.error("--masks is required")
        records = list(deepfashion_records(args.images, args.masks, args.crop_root,
                                           derive_barefoot=args.derive_barefoot_with_fashn,
                                           device=args.device))
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    except ImportError:
        pass
    if not records:
        raise ValueError("No usable shoe crops found")
    classify(records, args.model_id, args.device, args.batch_size)
    for row in records:
        row["split"] = stable_split(row["group_id"])
        row["subject_id"] = row["group_id"]
        row["session_id"] = row["group_id"]
        row["product_id"] = row["group_id"]
        accepted = (row["label_source"] in {"deepfashion_socks_without_footwear_v1",
                                             "fashionpedia_socks_without_shoe_v1"}
                    or (row["pseudo_confidence"] >= args.minimum_confidence
                        and row["pseudo_margin"] >= args.minimum_margin))
        row["quality"] = "pseudo_ok" if accepted else "needs_review"
    fields = list(records[0])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as output:
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader(); writer.writerows(records)
    summary = {
        "dataset": args.dataset, "rows": len(records),
        "pseudo_ok": sum(row["quality"] == "pseudo_ok" for row in records),
        "needs_review": sum(row["quality"] == "needs_review" for row in records),
        "label_counts": {label: sum(row["shoe_label"] == label for row in records) for label in SHOE_LABELS},
        "thresholds": {"confidence": args.minimum_confidence, "margin": args.minimum_margin},
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
