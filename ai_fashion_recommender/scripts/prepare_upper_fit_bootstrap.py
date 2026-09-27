"""Build a research-only three-class upper-fit bootstrap dataset.

Fashionpedia contributes explicit fit annotations and instance masks.  A
strict single-upper filter removes ambiguous layered outfits.  DeepFashion-
MultiModal contributes user-domain photos with its official upper-clothing
mask; those rows receive FashionSigLIP pseudo labels and remain marked for
human review.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import pickle
import random
import re
import shutil
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools import mask as coco_mask

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import FASHION_SIGLIP_MODEL_ID
from fit_vision_model import _crop_views, fit_geometry_vector


LABELS = ("슬림핏", "레귤러핏", "오버핏")
PROMPTS = {
    "슬림핏": (
        "a person wearing a fitted slim top close to the torso and arms",
        "a tight fitted shirt that follows the body line",
        "a close fitting top with little ease around the chest and waist",
    ),
    "레귤러핏": (
        "a person wearing a regular fit top with standard ease",
        "a classic regular shirt neither tight nor oversized",
        "a standard fit top following the body without clinging",
    ),
    "오버핏": (
        "a person wearing an oversized loose top with a wide body",
        "an oversized sweatshirt with dropped shoulders and extra volume",
        "a loose relaxed top substantially wider than the torso",
    ),
}

FASHIONPEDIA_TOP_CATEGORIES = {0, 1, 2, 3, 4, 5, 9, 12}
FASHIONPEDIA_FIT_LABELS = {
    135: "슬림핏",       # tight (fit)
    136: "레귤러핏",     # regular (fit)
    137: "오버핏",       # loose (fit)
    138: "오버핏",       # oversized
}


@dataclass
class Candidate:
    dataset: str
    domain: str
    group_id: str
    image_path: Path
    mask: np.ndarray
    metadata_label: str = ""
    predicted_label: str = ""
    probabilities: dict[str, float] | None = None
    geometry_score: float = 0.0
    geometry_percentile: float = 0.0
    rank_score: float = 0.0


def _mask_quality(mask: np.ndarray, image_size: tuple[int, int]) -> bool:
    binary = np.asarray(mask, dtype=bool)
    height, width = binary.shape
    if (width, height) != image_size or int(binary.sum()) < 350:
        return False
    ys, xs = np.where(binary)
    box_h = int(ys.max() - ys.min() + 1)
    box_w = int(xs.max() - xs.min() + 1)
    area_ratio = float(binary.mean())
    fill_ratio = float(binary.sum()) / max(1, box_h * box_w)
    return (
        box_h >= height * 0.12
        and box_w >= width * 0.12
        and ys.min() <= height * 0.55
        and ys.max() >= height * 0.25
        and 0.012 <= area_ratio <= 0.48
        and 0.20 <= fill_ratio <= 0.96
    )


def _geometry_score(mask: np.ndarray) -> float:
    vector = fit_geometry_vector(mask)
    # Upper-body looseness is weakly reflected by width/height, occupied area,
    # and lower-torso width.  It is used for ranking, never as ground truth.
    return float(vector[0] + 0.16 * vector[1] + 0.12 * np.mean(vector[2:6]))


def _decode_coco_segmentation(segmentation, height: int, width: int) -> np.ndarray:
    if isinstance(segmentation, list):
        rles = coco_mask.frPyObjects(segmentation, height, width)
        decoded = coco_mask.decode(rles)
    else:
        rle = dict(segmentation)
        counts = rle.get("counts")
        if isinstance(counts, str):
            rle["counts"] = counts.encode("ascii")
        decoded = coco_mask.decode(rle)
    if decoded.ndim == 3:
        decoded = decoded.any(axis=2)
    return np.asarray(decoded, dtype=bool)


def collect_fashionpedia(
    annotation_json: Path,
    image_dir: Path,
    *,
    seed: int,
    pool_per_class: int,
) -> list[Candidate]:
    payload = json.loads(annotation_json.read_text(encoding="utf-8"))
    images = {int(row["id"]): row for row in payload["images"]}
    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in payload["annotations"]:
        if int(annotation["category_id"]) in FASHIONPEDIA_TOP_CATEGORIES:
            by_image[int(annotation["image_id"])].append(annotation)

    eligible: dict[str, list[tuple[int, dict]]] = defaultdict(list)
    for image_id, upper_annotations in by_image.items():
        # Single visible upper item only: layered and open-outerwear cases are
        # deferred until the outermost-garment policy has human-reviewed data.
        if len(upper_annotations) != 1:
            continue
        annotation = upper_annotations[0]
        fit_attributes = [
            int(value) for value in annotation.get("attribute_ids", [])
            if int(value) in FASHIONPEDIA_FIT_LABELS
        ]
        if len(fit_attributes) != 1:
            continue
        label = FASHIONPEDIA_FIT_LABELS[fit_attributes[0]]
        eligible[label].append((image_id, annotation))

    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        rows = eligible[label]
        rng.shuffle(rows)
        available = len(rows)
        kept = 0
        for image_id, annotation in rows:
            if kept >= pool_per_class:
                break
            image_row = images[image_id]
            image_path = image_dir / image_row["file_name"]
            if not image_path.is_file():
                continue
            mask = _decode_coco_segmentation(
                annotation["segmentation"], int(image_row["height"]), int(image_row["width"])
            )
            if not _mask_quality(mask, (int(image_row["width"]), int(image_row["height"]))):
                continue
            selected.append(Candidate(
                dataset="fashionpedia",
                domain="shop",
                group_id=f"fashionpedia_{image_id}",
                image_path=image_path,
                mask=mask,
                metadata_label=label,
                geometry_score=_geometry_score(mask),
            ))
            kept += 1
        print(f"[Fashionpedia] {label}: eligible={available}, scored={kept}", flush=True)
    return selected


def _deepfashion_group(path: Path) -> str:
    match = re.search(r"id_(\d+)", path.name)
    return f"dfmm_{match.group(1)}" if match else f"dfmm_{path.stem}"


def collect_deepfashion(
    image_dir: Path,
    mask_dir: Path,
    *,
    seed: int,
    pool_size: int,
) -> list[Candidate]:
    best_by_group: dict[str, Candidate] = {}
    paths = sorted(mask_dir.glob("*.png"))
    random.Random(seed).shuffle(paths)
    for index, mask_path in enumerate(paths, 1):
        raw = np.asarray(Image.open(mask_path))
        # Official DFMM labels: 1=upper clothing, 2=outer clothing, 5=pants.
        # Excluding label 2 avoids assigning the inner top in layered outfits.
        mask = raw == 1
        if not mask.any() or bool((raw == 2).any()):
            continue
        image_path = image_dir / mask_path.name.replace("_segm.png", ".jpg")
        if not image_path.is_file():
            continue
        with Image.open(image_path) as image:
            size = image.size
        if not _mask_quality(mask, size):
            continue
        candidate = Candidate(
            dataset="deepfashion_multimodal",
            domain="user",
            group_id=_deepfashion_group(image_path),
            image_path=image_path,
            mask=mask,
            geometry_score=_geometry_score(mask),
        )
        previous = best_by_group.get(candidate.group_id)
        if previous is None or int(mask.sum()) > int(previous.mask.sum()):
            best_by_group[candidate.group_id] = candidate
        if index % 3000 == 0:
            print(f"[DeepFashion] {index}/{len(paths)} masks scanned, {len(best_by_group)} groups", flush=True)
    rows = list(best_by_group.values())
    random.Random(seed + 1).shuffle(rows)
    rows = rows[:pool_size]
    print(f"[DeepFashion] candidates selected for scoring: {len(rows)}", flush=True)
    return rows


def _normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def score_candidates(
    candidates: list[Candidate], *, device: str, batch_size: int,
    tokenizer_dir: Path | None = None,
) -> None:
    import open_clip
    import torch

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    model_name = f"hf-hub:{FASHION_SIGLIP_MODEL_ID}"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=selected_device)
    if tokenizer_dir is None:
        tokenizer = open_clip.get_tokenizer(model_name)
    else:
        from open_clip.tokenizer import HFTokenizer
        tokenizer = HFTokenizer(
            str(tokenizer_dir), context_length=64, clean="canonicalize",
            local_files_only=True,
        )
    model.eval()
    class_text = []
    with torch.inference_mode():
        for label in LABELS:
            tokens = tokenizer(list(PROMPTS[label])).to(selected_device)
            class_text.append(_normalize(model.encode_text(tokens, normalize=True).float().mean(0, keepdim=True)))
        text_features = torch.cat(class_text)

    for start in range(0, len(candidates), batch_size):
        batch = candidates[start:start + batch_size]
        views = []
        for candidate in batch:
            image = Image.open(candidate.image_path).convert("RGB")
            context, isolated, _ = _crop_views(image, candidate.mask)
            views.extend((context, isolated))
        pixels = torch.stack([preprocess(view) for view in views]).to(selected_device)
        with torch.inference_mode():
            encoded = _normalize(model.encode_image(pixels, normalize=True).float())
            encoded = _normalize((encoded[0::2] + encoded[1::2]) / 2)
            probabilities = (encoded @ text_features.T * 18.0).softmax(dim=-1).cpu().numpy()
        for candidate, values in zip(batch, probabilities):
            candidate.probabilities = {label: float(values[i]) for i, label in enumerate(LABELS)}
            candidate.predicted_label = LABELS[int(np.argmax(values))]
        print(f"[FashionSigLIP] {min(start + batch_size, len(candidates))}/{len(candidates)}", flush=True)


def _set_geometry_percentiles(candidates: list[Candidate]) -> None:
    for domain in {candidate.domain for candidate in candidates}:
        rows = sorted(
            (candidate for candidate in candidates if candidate.domain == domain),
            key=lambda candidate: candidate.geometry_score,
        )
        denominator = max(1, len(rows) - 1)
        for index, candidate in enumerate(rows):
            candidate.geometry_percentile = index / denominator


def _rank(candidate: Candidate, label: str) -> float:
    probability = (candidate.probabilities or {}).get(label, 0.0)
    target_percentile = {"슬림핏": 0.18, "레귤러핏": 0.50, "오버핏": 0.82}[label]
    geometry_agreement = 1.0 - abs(candidate.geometry_percentile - target_percentile)
    metadata_bonus = 0.60 if candidate.metadata_label == label else 0.0
    contradiction_penalty = 0.40 if candidate.metadata_label and candidate.predicted_label != label else 0.0
    return probability + 0.15 * geometry_agreement + metadata_bonus - contradiction_penalty


def select_balanced(candidates: list[Candidate], per_domain_class: int) -> list[Candidate]:
    _set_geometry_percentiles(candidates)
    selected = []
    used_groups = set()
    for domain in ("shop", "user"):
        for label in LABELS:
            pool = [candidate for candidate in candidates if candidate.domain == domain]
            if domain == "shop":
                pool = [candidate for candidate in pool if candidate.metadata_label == label]
            else:
                pool = [candidate for candidate in pool if candidate.predicted_label == label]
            for candidate in pool:
                candidate.rank_score = _rank(candidate, label)
            pool.sort(key=lambda candidate: candidate.rank_score, reverse=True)
            chosen = [row for row in pool if row.group_id not in used_groups][:per_domain_class]
            if len(chosen) < per_domain_class:
                raise RuntimeError(
                    f"not enough {domain}/{label} candidates: {len(chosen)} < {per_domain_class}"
                )
            for candidate in chosen:
                candidate.predicted_label = label
                used_groups.add(candidate.group_id)
            selected.extend(chosen)
            print(f"[selection] {domain}/{label}: {len(chosen)}", flush=True)
    return selected


def _stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def assign_splits(selected: list[Candidate], *, seed: int) -> dict[str, str]:
    assignments = {}
    for domain in ("user", "shop"):
        for label in LABELS:
            rows = [row for row in selected if row.domain == domain and row.predicted_label == label]
            rows.sort(key=lambda row: _stable_key(seed, row.group_id))
            count = len(rows)
            train_end = int(round(count * 0.70))
            val_end = train_end + int(round(count * 0.15))
            for index, row in enumerate(rows):
                assignments[row.group_id] = "train" if index < train_end else ("val" if index < val_end else "test")
    return assignments


def _review_sheet(rows: list[dict], dataset_root: Path, output: Path) -> None:
    rows = rows[:24]
    cell_w, cell_h = 240, 300
    canvas = Image.new("RGB", (cell_w * 6, cell_h * 4), "white")
    draw = ImageDraw.Draw(canvas)
    for index, row in enumerate(rows):
        image = Image.open(dataset_root / row["image_path"]).convert("RGB")
        image.thumbnail((cell_w - 12, cell_h - 48), Image.Resampling.LANCZOS)
        x = (index % 6) * cell_w + (cell_w - image.width) // 2
        y = (index // 6) * cell_h
        canvas.paste(image, (x, y))
        caption = f"{row['split']} p={float(row['pseudo_confidence']):.2f} g={float(row['geometry_percentile']):.2f}"
        draw.text(((index % 6) * cell_w + 5, y + cell_h - 42), caption, fill="black")
        draw.text(((index % 6) * cell_w + 5, y + cell_h - 24), row["source_dataset"], fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, quality=92)


def materialize(
    selected: list[Candidate], output_root: Path, *, seed: int, overwrite: bool = False,
) -> dict:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"output root already exists: {output_root} (use --overwrite to replace it)"
            )
        shutil.rmtree(output_root)
    (output_root / "images").mkdir(parents=True)
    (output_root / "masks").mkdir(parents=True)
    assignments = assign_splits(selected, seed=seed)
    fieldnames = [
        "image_path", "mask_path", "split", "category", "source_domain", "group_id",
        "upper_fit", "upper_length", "bottom_silhouette", "bottom_length", "quality",
        "source_dataset", "label_source", "pseudo_confidence", "geometry_score",
        "geometry_percentile", "layering_state", "review_status", "source_path",
    ]
    output_rows = []
    for candidate in selected:
        extension = candidate.image_path.suffix.lower()
        if extension not in {".jpg", ".jpeg", ".png", ".webp"}:
            extension = ".jpg"
        token = hashlib.sha1(str(candidate.image_path).encode()).hexdigest()[:10]
        basename = f"{candidate.dataset}_{token}{extension}"
        image_relative = Path("images") / candidate.domain / basename
        mask_relative = Path("masks") / candidate.domain / f"{Path(basename).stem}.png"
        (output_root / image_relative).parent.mkdir(parents=True, exist_ok=True)
        (output_root / mask_relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate.image_path, output_root / image_relative)
        Image.fromarray(candidate.mask.astype(np.uint8) * 255, mode="L").save(output_root / mask_relative)
        probability = (candidate.probabilities or {}).get(candidate.predicted_label, 0.0)
        output_rows.append({
            "image_path": image_relative.as_posix(),
            "mask_path": mask_relative.as_posix(),
            "split": assignments[candidate.group_id],
            "category": "top",
            "source_domain": candidate.domain,
            "group_id": candidate.group_id,
            "upper_fit": candidate.predicted_label,
            "upper_length": "", "bottom_silhouette": "", "bottom_length": "", "quality": "",
            "source_dataset": candidate.dataset,
            "label_source": (
                "fashionpedia_explicit_fit+official_instance_mask+fashionsiglip_filter"
                if candidate.metadata_label else
                "fashionsiglip_zero_shot+official_upper_mask+geometry_rank"
            ),
            "pseudo_confidence": f"{probability:.6f}",
            "geometry_score": f"{candidate.geometry_score:.6f}",
            "geometry_percentile": f"{candidate.geometry_percentile:.6f}",
            "layering_state": "single_upper_only",
            "review_status": "pending_human_review",
            "source_path": str(candidate.image_path),
        })
    with (output_root / "annotations.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    for domain in ("user", "shop"):
        for label in LABELS:
            rows = [row for row in output_rows if row["source_domain"] == domain and row["upper_fit"] == label]
            rows.sort(key=lambda row: _stable_key(seed + 1, row["group_id"]))
            _review_sheet(rows, output_root, output_root / "review" / f"{domain}_{label}.jpg")

    counts = Counter((row["split"], row["source_domain"], row["upper_fit"]) for row in output_rows)
    manifest = {
        "version": 1,
        "purpose": "research-only three-class upper fit bootstrap",
        "deployment_status": "blocked_until_human_review",
        "labels": list(LABELS),
        "rows": len(output_rows),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "fit_mapping": {
            "Fashionpedia tight (fit)": "슬림핏",
            "Fashionpedia regular (fit)": "레귤러핏",
            "Fashionpedia loose (fit) or oversized": "오버핏",
        },
        "layering_policy": "train v1 uses single-upper images; runtime display follows the outermost visible garment",
        "label_provenance": {
            "fashionpedia": "explicit fit attribute + official instance mask; FashionSigLIP contradiction ranking only",
            "deepfashion_multimodal": "FashionSigLIP zero-shot + official upper mask + geometry ranking",
        },
        "human_review": "pending",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--fashionpedia-json", required=True)
    value.add_argument("--fashionpedia-images", required=True)
    value.add_argument("--deepfashion-images", required=True)
    value.add_argument("--deepfashion-masks", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--per-domain-class", type=int, default=100)
    value.add_argument("--fashionpedia-pool-per-class", type=int, default=400)
    value.add_argument("--deepfashion-pool", type=int, default=8000)
    value.add_argument("--batch-size", type=int, default=24)
    value.add_argument("--device", default="auto")
    value.add_argument("--tokenizer-dir")
    value.add_argument("--overwrite", action="store_true")
    value.add_argument("--seed", type=int, default=20260927)
    return value


def _load_candidate_cache(path: Path, signature: dict) -> list[Candidate] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rb") as handle:
        payload = pickle.load(handle)
    if payload.get("signature") != signature:
        return None
    print(f"[cache] loaded {len(payload['candidates'])} candidates from {path}", flush=True)
    return payload["candidates"]


def _save_candidate_cache(path: Path, signature: dict, candidates: list[Candidate]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wb", compresslevel=3) as handle:
        pickle.dump({"signature": signature, "candidates": candidates}, handle, protocol=5)
    temporary.replace(path)
    print(f"[cache] saved {len(candidates)} candidates to {path}", flush=True)


def main() -> None:
    args = parser().parse_args()
    output_root = Path(args.output_root).resolve()
    signature = {
        "version": 1,
        "labels": list(LABELS),
        "fashionpedia_json": str(Path(args.fashionpedia_json).resolve()),
        "fashionpedia_images": str(Path(args.fashionpedia_images).resolve()),
        "deepfashion_images": str(Path(args.deepfashion_images).resolve()),
        "deepfashion_masks": str(Path(args.deepfashion_masks).resolve()),
        "fashionpedia_pool_per_class": args.fashionpedia_pool_per_class,
        "deepfashion_pool": args.deepfashion_pool,
        "seed": args.seed,
    }
    cache_root = output_root.parent / "_upper_fit_bootstrap_cache"
    collected_path = cache_root / "candidates_collected.pkl.gz"
    scored_path = cache_root / "candidates_scored.pkl.gz"
    candidates = _load_candidate_cache(scored_path, signature)
    if candidates is None:
        candidates = _load_candidate_cache(collected_path, signature)
        if candidates is None:
            fashionpedia = collect_fashionpedia(
                Path(args.fashionpedia_json), Path(args.fashionpedia_images),
                seed=args.seed, pool_per_class=args.fashionpedia_pool_per_class,
            )
            deepfashion = collect_deepfashion(
                Path(args.deepfashion_images), Path(args.deepfashion_masks),
                seed=args.seed, pool_size=args.deepfashion_pool,
            )
            candidates = fashionpedia + deepfashion
            print(f"candidate totals: Fashionpedia={len(fashionpedia)}, DeepFashion={len(deepfashion)}", flush=True)
            _save_candidate_cache(collected_path, signature, candidates)
        score_candidates(
            candidates, device=args.device, batch_size=args.batch_size,
            tokenizer_dir=Path(args.tokenizer_dir) if args.tokenizer_dir else None,
        )
        _save_candidate_cache(scored_path, signature, candidates)
    selected = select_balanced(candidates, args.per_domain_class)
    manifest = materialize(selected, output_root, seed=args.seed, overwrite=args.overwrite)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
