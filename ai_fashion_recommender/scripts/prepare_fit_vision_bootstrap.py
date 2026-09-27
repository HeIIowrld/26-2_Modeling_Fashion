"""Build a research-only 4-class bottom-fit bootstrap dataset.

The script combines two non-commercial/research image pools:

* DeepFashion-MultiModal outfit photos, using its official pants mask (label 5)
* Fashion200K shopping photos, using explicit straight/wide metadata where present

FashionSigLIP and mask geometry are used only to rank/filter candidates.  Every
row records its weak-label provenance and remains ``pending_human_review``.
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

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config import FASHION_SIGLIP_MODEL_ID
from fit_vision_model import _crop_views, fit_geometry_vector


LABELS = ("슬림", "스트레이트", "세미와이드", "와이드")
PROMPTS = {
    "슬림": (
        "a person wearing slim-fit pants, close through the thigh and calf but not leggings",
        "slim tapered trousers with a narrow leg opening",
        "fitted slim pants that are narrower than straight-leg pants",
    ),
    "스트레이트": (
        "a person wearing straight-leg pants with nearly constant leg width",
        "classic straight cut trousers, neither slim nor wide",
        "straight jeans with a regular leg opening",
    ),
    "세미와이드": (
        "a person wearing semi-wide pants with moderately relaxed legs",
        "relaxed straight trousers, wider than straight but not wide-leg",
        "moderately loose pants with a medium-wide silhouette",
    ),
    "와이드": (
        "a person wearing very wide-leg pants with broad loose legs",
        "palazzo pants or oversized wide trousers",
        "wide-leg trousers with a large leg opening",
    ),
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
    if (width, height) != image_size or int(binary.sum()) < 500:
        return False
    ys, xs = np.where(binary)
    box_h = int(ys.max() - ys.min() + 1)
    box_w = int(xs.max() - xs.min() + 1)
    area_ratio = float(binary.mean())
    fill_ratio = float(binary.sum()) / max(1, box_h * box_w)
    return (
        box_h >= height * 0.30
        and box_w >= width * 0.08
        and ys.max() >= height * 0.58
        and 0.025 <= area_ratio <= 0.65
        and 0.12 <= fill_ratio <= 0.92
    )


def _geometry_score(mask: np.ndarray) -> float:
    vector = fit_geometry_vector(mask)
    # Aspect dominates absolute looseness; lower-leg width breaks ties.
    return float(vector[0] + 0.18 * np.mean(vector[4:7]) + 0.08 * vector[1])


def _deepfashion_group(path: Path) -> str:
    match = re.search(r"id_(\d+)", path.name)
    return f"dfmm_{match.group(1)}" if match else f"dfmm_{path.stem}"


def collect_deepfashion(image_dir: Path, mask_dir: Path) -> list[Candidate]:
    best_by_group: dict[str, Candidate] = {}
    for index, mask_path in enumerate(sorted(mask_dir.glob("*.png")), 1):
        raw = np.asarray(Image.open(mask_path))
        mask = raw == 5
        if not mask.any():
            continue
        image_name = mask_path.name.replace("_segm.png", ".jpg")
        image_path = image_dir / image_name
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
        if index % 2000 == 0:
            print(f"[DeepFashion] {index} masks scanned, {len(best_by_group)} groups", flush=True)
    return list(best_by_group.values())


def _fashion_label(value: str) -> str:
    value = value.strip()
    if value in {"스키니", "테이퍼드"}:
        return "슬림"
    if value == "스트레이트":
        return "스트레이트"
    if value in {"와이드", "팔라초"}:
        return "와이드"
    return ""


def collect_fashion200k(
    annotation_csv: Path,
    image_root: Path,
    *,
    seed: int,
    explicit_pool: int,
    unlabeled_pool: int,
) -> list[Candidate]:
    with annotation_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    pools: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        label = _fashion_label(row.get("pant_leg_shape", ""))
        pools[label or "unlabeled"].append(row)
    rng = random.Random(seed)
    selected_rows = []
    for key in ("슬림", "스트레이트", "와이드"):
        rng.shuffle(pools[key])
        selected_rows.extend(pools[key][:explicit_pool])
    rng.shuffle(pools["unlabeled"])
    selected_rows.extend(pools["unlabeled"][:unlabeled_pool])

    # Import the parser backend directly.  This preparation command does not
    # need MediaPipe pose estimation, so it should not require that optional
    # runtime dependency merely through ClothingParser's fallback import.
    from fashn_human_parser import FashnHumanParser

    parser = FashnHumanParser()
    candidates = []
    for index, row in enumerate(selected_rows, 1):
        image_path = (image_root / row["image_path"]).resolve()
        try:
            with Image.open(image_path) as image:
                rgb = np.asarray(image.convert("RGB"))
                size = image.size
            segmentation = np.asarray(parser.predict(rgb), dtype=np.uint8)
            mask = np.isin(segmentation, [4, 5, 6])
            # A worn-photo bootstrap set must contain at least one human region.
            human_area = int(np.isin(segmentation, [1, 2, 12, 13, 14, 15, 16]).sum())
            if human_area < mask.size * 0.008 or not _mask_quality(mask, size):
                continue
            stem = Path(row["image_path"]).stem
            product_id = stem.rsplit("_", 1)[0]
            candidates.append(Candidate(
                dataset="fashion200k",
                domain="shop",
                group_id=f"fashion200k_{product_id}",
                image_path=image_path,
                mask=mask,
                metadata_label=_fashion_label(row.get("pant_leg_shape", "")),
                geometry_score=_geometry_score(mask),
            ))
        except Exception as error:
            print(f"[Fashion200K] skip {image_path.name}: {error}", flush=True)
        if index % 50 == 0:
            print(f"[Fashion200K] parsed {index}/{len(selected_rows)}, kept {len(candidates)}", flush=True)
    return candidates


def _normalize(tensor):
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def score_candidates(candidates: list[Candidate], *, device: str, batch_size: int) -> None:
    import open_clip
    import torch

    selected_device = "cuda" if device == "auto" and torch.cuda.is_available() else (
        "cpu" if device == "auto" else device
    )
    model_name = f"hf-hub:{FASHION_SIGLIP_MODEL_ID}"
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, device=selected_device)
    tokenizer = open_clip.get_tokenizer(model_name)
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


def _candidate_label(candidate: Candidate) -> str:
    if candidate.metadata_label in {"스트레이트", "와이드"}:
        return candidate.metadata_label
    return candidate.predicted_label


def _eligible(candidate: Candidate, label: str, relaxed: bool = False) -> bool:
    probability = (candidate.probabilities or {}).get(label, 0.0)
    percentile = candidate.geometry_percentile
    if candidate.metadata_label:
        # Explicit product metadata remains weak supervision; FashionSigLIP is
        # only a gross contradiction filter.
        return True if relaxed else probability >= 0.16
    if candidate.predicted_label != label:
        return False
    if relaxed:
        # With four neighboring classes the top softmax score can be below the
        # old three-class threshold.  Argmax + geometry rank is retained and
        # every such row remains pending human review.
        return True
    if probability < 0.40:
        return False
    return {
        "슬림": percentile <= 0.55,
        "스트레이트": 0.08 <= percentile <= 0.72,
        "세미와이드": 0.20 <= percentile <= 0.84,
        "와이드": percentile >= 0.32,
    }[label]


def _rank(candidate: Candidate, label: str) -> float:
    probability = (candidate.probabilities or {}).get(label, 0.0)
    percentile = candidate.geometry_percentile
    geometry_agreement = {
        "슬림": 1.0 - abs(percentile - 0.15),
        "스트레이트": 1.0 - abs(percentile - 0.38),
        "세미와이드": 1.0 - abs(percentile - 0.62),
        "와이드": percentile,
    }[label]
    metadata_bonus = 0.35 if candidate.metadata_label == label else 0.0
    return probability + 0.22 * geometry_agreement + metadata_bonus


def select_balanced(candidates: list[Candidate], per_domain_class: int) -> list[Candidate]:
    _set_geometry_percentiles(candidates)
    selected = []
    used_groups = set()
    for label in LABELS:
        eligible_by_domain = {}
        for domain in ("user", "shop"):
            pool = [candidate for candidate in candidates if candidate.domain == domain and _candidate_label(candidate) == label]
            for candidate in pool:
                candidate.rank_score = _rank(candidate, label)
            ordered = sorted(pool, key=lambda candidate: candidate.rank_score, reverse=True)
            chosen = [candidate for candidate in ordered if _eligible(candidate, label) and candidate.group_id not in used_groups]
            if len(chosen) < per_domain_class:
                chosen = [candidate for candidate in ordered if _eligible(candidate, label, True) and candidate.group_id not in used_groups]
            eligible_by_domain[domain] = chosen

        targets = {
            domain: min(per_domain_class, len(eligible_by_domain[domain]))
            for domain in ("user", "shop")
        }
        total_target = per_domain_class * 2
        remaining = total_target - sum(targets.values())
        while remaining > 0:
            domains = sorted(
                ("user", "shop"),
                key=lambda domain: len(eligible_by_domain[domain]) - targets[domain],
                reverse=True,
            )
            domain = domains[0]
            if len(eligible_by_domain[domain]) <= targets[domain]:
                raise RuntimeError(
                    f"not enough {label} candidates across domains: "
                    f"user={len(eligible_by_domain['user'])}, shop={len(eligible_by_domain['shop'])}, "
                    f"target={total_target}"
                )
            targets[domain] += 1
            remaining -= 1

        for domain in ("user", "shop"):
            chosen = eligible_by_domain[domain][:targets[domain]]
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


def materialize(selected: list[Candidate], output_root: Path, *, seed: int) -> dict:
    if output_root.exists():
        shutil.rmtree(output_root)
    image_root = output_root / "images"
    mask_root = output_root / "masks"
    review_root = output_root / "review"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    assignments = assign_splits(selected, seed=seed)
    fieldnames = [
        "image_path", "mask_path", "split", "category", "source_domain", "group_id",
        "upper_fit", "upper_length", "bottom_silhouette", "bottom_length", "quality",
        "source_dataset", "label_source", "pseudo_confidence", "geometry_score",
        "geometry_percentile", "review_status", "source_path",
    ]
    output_rows = []
    for index, candidate in enumerate(selected):
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
        label_source = (
            "fashion200k_explicit_metadata+fashionsiglip_filter+mask_geometry"
            if candidate.metadata_label
            else "fashionsiglip_zero_shot+mask_geometry"
        )
        output_rows.append({
            "image_path": image_relative.as_posix(),
            "mask_path": mask_relative.as_posix(),
            "split": assignments[candidate.group_id],
            "category": "bottom",
            "source_domain": candidate.domain,
            "group_id": candidate.group_id,
            "upper_fit": "", "upper_length": "",
            "bottom_silhouette": candidate.predicted_label,
            "bottom_length": "", "quality": "",
            "source_dataset": candidate.dataset,
            "label_source": label_source,
            "pseudo_confidence": f"{probability:.6f}",
            "geometry_score": f"{candidate.geometry_score:.6f}",
            "geometry_percentile": f"{candidate.geometry_percentile:.6f}",
            "review_status": "pending_human_review",
            "source_path": str(candidate.image_path),
        })
    csv_path = output_root / "annotations.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

    for domain in ("user", "shop"):
        for label in LABELS:
            rows = [row for row in output_rows if row["source_domain"] == domain and row["bottom_silhouette"] == label]
            rows.sort(key=lambda row: _stable_key(seed + 1, row["group_id"]))
            _review_sheet(rows, output_root, review_root / f"{domain}_{label}.jpg")

    counts = Counter((row["split"], row["source_domain"], row["bottom_silhouette"]) for row in output_rows)
    manifest = {
        "version": 1,
        "purpose": "research-only bottom silhouette bootstrap",
        "deployment_status": "blocked_until_human_review",
        "labels": list(LABELS),
        "rows": len(output_rows),
        "counts": {"|".join(key): value for key, value in sorted(counts.items())},
        "label_provenance": {
            "deepfashion_multimodal": "FashionSigLIP zero-shot + official pants mask geometry",
            "fashion200k_slim": "skinny/tapered metadata merged as slim + FashionSigLIP contradiction filter + FASHN mask geometry",
            "fashion200k_straight_wide": "explicit product metadata + FashionSigLIP contradiction filter + FASHN mask geometry",
            "fashion200k_semiwide": "FashionSigLIP zero-shot + FASHN mask geometry",
        },
        "human_review": "pending",
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    value.add_argument("--deepfashion-images", required=True)
    value.add_argument("--deepfashion-masks", required=True)
    value.add_argument("--fashion200k-csv", required=True)
    value.add_argument("--fashion200k-root", required=True)
    value.add_argument("--output-root", required=True)
    value.add_argument("--per-domain-class", type=int, default=100)
    value.add_argument("--fashion-explicit-pool", type=int, default=260)
    value.add_argument("--fashion-unlabeled-pool", type=int, default=700)
    value.add_argument("--batch-size", type=int, default=24)
    value.add_argument("--device", default="auto")
    value.add_argument("--seed", type=int, default=20260925)
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
        "version": 2,
        "labels": list(LABELS),
        "deepfashion_images": str(Path(args.deepfashion_images).resolve()),
        "deepfashion_masks": str(Path(args.deepfashion_masks).resolve()),
        "fashion200k_csv": str(Path(args.fashion200k_csv).resolve()),
        "fashion200k_root": str(Path(args.fashion200k_root).resolve()),
        "fashion_explicit_pool": args.fashion_explicit_pool,
        "fashion_unlabeled_pool": args.fashion_unlabeled_pool,
        "seed": args.seed,
    }
    cache_root = output_root.parent / "_fit_bootstrap_cache"
    collected_path = cache_root / "candidates_collected.pkl.gz"
    scored_path = cache_root / "candidates_scored.pkl.gz"
    candidates = _load_candidate_cache(scored_path, signature)
    if candidates is None:
        candidates = _load_candidate_cache(collected_path, signature)
        if candidates is None:
            deepfashion = collect_deepfashion(Path(args.deepfashion_images), Path(args.deepfashion_masks))
            fashion200k = collect_fashion200k(
                Path(args.fashion200k_csv), Path(args.fashion200k_root), seed=args.seed,
                explicit_pool=args.fashion_explicit_pool, unlabeled_pool=args.fashion_unlabeled_pool,
            )
            candidates = deepfashion + fashion200k
            print(f"candidate totals: DeepFashion={len(deepfashion)}, Fashion200K={len(fashion200k)}", flush=True)
            _save_candidate_cache(collected_path, signature, candidates)
        score_candidates(candidates, device=args.device, batch_size=args.batch_size)
        _save_candidate_cache(scored_path, signature, candidates)
    selected = select_balanced(candidates, args.per_domain_class)
    manifest = materialize(selected, output_root, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
