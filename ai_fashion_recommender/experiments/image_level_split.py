"""이미지 단위 fit/dev 분할 — 기존 crop 단위 분할의 누수를 고친다.

기존 동작 (eval_lib.split_train_dev)
    crop(=CSV 행) 단위로 나눈다. 사진 한 장에서 crop이 여러 개 나오므로
    같은 이미지가 fit과 dev 양쪽에 들어간다. 2차 학습에서는 dev crop의 60.0%가
    fit과 같은 원본 이미지를 공유했다. dev는 early stopping·임계값 선택에만 쓰였으므로
    보고된 val 점수는 영향받지 않지만, dev 점수 자체는 낙관적이었다.

이 모듈 (split_by_image)
    crop을 **원본 이미지 canonical key**로 묶은 뒤 key 단위로 나눈다.
    따라서 같은 이미지의 crop은 반드시 한쪽에만 들어간다.

canonical key
    Fashionpedia : fashionpedia::{official_split}::{image_id}
    Fashion200K  : fashion200k::{source_subset}::{product_id}::{image_number}
    → used_image_manifest.csv / fashionpedia_train_r3_images.csv 에서 가져온다.
      basename 이나 절대경로를 직접 분할 키로 쓰지 않는다.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MANIFESTS = PROJECT_DIR / "reports" / "manifests"

DEV_FRACTION = 0.15
SPLIT_SEED = 20260819


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_key_map() -> tuple[dict[str, str], dict[str, str]]:
    """original_filename → (canonical key, 출처 그룹).

    파일명이 여러 이미지에 매핑되면 즉시 중단한다. 분할 키가 흔들리면 안 되기 때문이다.
    """
    filename_to_key: dict[str, str] = {}
    filename_to_group: dict[str, str] = {}
    collisions: list[str] = []

    for row in _read_csv(MANIFESTS / "used_image_manifest.csv"):
        name = row["original_filename"]
        key = row["unique_key"]
        if row["dataset"] == "fashionpedia":
            group = (f"fashionpedia_r{row['augmentation_round']}"
                     if row["augmentation_round"] not in ("", "0") else "fashionpedia_seed")
        else:
            group = row["source_subset"]
        if name in filename_to_key and filename_to_key[name] != key:
            collisions.append(name)
        filename_to_key[name] = key
        filename_to_group[name] = group

    for row in _read_csv(MANIFESTS / "fashionpedia_train_r3_images.csv"):
        name = row["original_filename"]
        if not name:
            continue
        key = f"fashionpedia::{row['official_split']}::{row['image_id']}"
        if name in filename_to_key and filename_to_key[name] != key:
            collisions.append(name)
        filename_to_key[name] = key
        filename_to_group[name] = "fashionpedia_r3"

    if collisions:
        raise SystemExit(f"파일명이 여러 이미지에 매핑됩니다({len(collisions)}건): {collisions[:5]}")
    return filename_to_key, filename_to_group


def crop_keys(cache: dict, filename_to_key: dict[str, str]) -> tuple[list[str], list[str]]:
    """캐시의 각 crop을 canonical key로 바꾼다. 매핑 실패는 즉시 중단."""
    keys, unmapped = [], []
    for path in cache["image_paths"]:
        name = Path(str(path).replace("\\", "/")).name
        key = filename_to_key.get(name)
        if key is None:
            unmapped.append(name)
            keys.append("")
        else:
            keys.append(key)
    return keys, unmapped


def split_by_image(cache: dict, *, dev_fraction: float = DEV_FRACTION,
                   seed: int = SPLIT_SEED):
    """이미지 canonical key 단위로 fit/dev 인덱스를 만든다.

    출처 그룹별로 같은 비율을 떼어내 dev가 한쪽 데이터에 쏠리지 않게 한다.
    """
    import torch

    filename_to_key, filename_to_group = canonical_key_map()
    keys, unmapped = crop_keys(cache, filename_to_key)
    if unmapped:
        raise SystemExit(f"canonical key 매핑 실패 {len(unmapped)}건: {sorted(set(unmapped))[:5]}")

    key_to_rows: dict[str, list[int]] = defaultdict(list)
    key_to_group: dict[str, str] = {}
    for index, (key, path) in enumerate(zip(keys, cache["image_paths"])):
        key_to_rows[key].append(index)
        name = Path(str(path).replace("\\", "/")).name
        key_to_group[key] = filename_to_group[name]

    groups: dict[str, list[str]] = defaultdict(list)
    for key, group in key_to_group.items():
        groups[group].append(key)

    generator = torch.Generator().manual_seed(seed)
    fit_keys, dev_keys = [], []
    for group in sorted(groups):
        members = sorted(groups[group])
        order = torch.randperm(len(members), generator=generator).tolist()
        shuffled = [members[i] for i in order]
        cut = int(round(len(shuffled) * dev_fraction))
        dev_keys.extend(shuffled[:cut])
        fit_keys.extend(shuffled[cut:])

    fit_rows = sorted(i for key in fit_keys for i in key_to_rows[key])
    dev_rows = sorted(i for key in dev_keys for i in key_to_rows[key])
    info = {
        "dev_fraction": dev_fraction,
        "seed": seed,
        "total_crops": len(keys),
        "total_images": len(key_to_rows),
        "fit_crops": len(fit_rows),
        "dev_crops": len(dev_rows),
        "fit_images": len(fit_keys),
        "dev_images": len(dev_keys),
        "group_counts": {g: len(v) for g, v in sorted(groups.items())},
        "unmapped_crops": 0,
    }
    return torch.tensor(fit_rows), torch.tensor(dev_rows), set(fit_keys), set(dev_keys), info


def verify(cache: dict, fit_rows, dev_rows, fit_keys: set[str], dev_keys: set[str]) -> list[tuple[str, bool, str]]:
    """분할 무결성 검사 결과 목록을 돌려준다."""
    filename_to_key, _ = canonical_key_map()
    keys, unmapped = crop_keys(cache, filename_to_key)
    used = _read_csv(MANIFESTS / "used_image_manifest.csv")
    old_val = {r["unique_key"] for r in used if r["usage_split"] == "old_val"}
    new_val = {r["unique_key"] for r in used if r["usage_split"] == "new_val"}

    multi = sum(1 for k in keys if k == "")
    checks = [
        ("캐시 crop 중 manifest 매핑 실패", len(unmapped) == 0, f"{len(unmapped)}건"),
        ("하나의 crop이 복수 이미지에 매핑", multi == 0, f"{multi}건"),
        ("fit image key ∩ dev image key", len(fit_keys & dev_keys) == 0,
         f"{len(fit_keys & dev_keys)}건"),
        ("fit/dev crop 합계 == 전체 train crop",
         len(fit_rows) + len(dev_rows) == len(keys),
         f"{len(fit_rows):,} + {len(dev_rows):,} = {len(fit_rows) + len(dev_rows):,} / {len(keys):,}"),
        ("fit ∩ old validation", len(fit_keys & old_val) == 0, f"{len(fit_keys & old_val)}건"),
        ("dev ∩ old validation", len(dev_keys & old_val) == 0, f"{len(dev_keys & old_val)}건"),
        ("fit ∩ new validation", len(fit_keys & new_val) == 0, f"{len(fit_keys & new_val)}건"),
        ("dev ∩ new validation", len(dev_keys & new_val) == 0, f"{len(dev_keys & new_val)}건"),
    ]
    return checks


def compare_with_crop_level(cache: dict) -> dict:
    """기존 crop 단위 분할과의 차이를 수치로 보여준다 (문서화·테스트용)."""
    from eval_lib import split_train_dev

    filename_to_key, _ = canonical_key_map()
    keys, _ = crop_keys(cache, filename_to_key)
    old_fit, old_dev = split_train_dev(cache)
    old_fit_keys = {keys[int(i)] for i in old_fit}
    old_dev_keys = {keys[int(i)] for i in old_dev}
    shared = old_fit_keys & old_dev_keys
    contaminated = sum(1 for i in old_dev if keys[int(i)] in shared)
    return {
        "crop_level_shared_images": len(shared),
        "crop_level_dev_crops": len(old_dev),
        "crop_level_contaminated_dev_crops": contaminated,
        "crop_level_contaminated_ratio": round(contaminated / max(len(old_dev), 1), 4),
    }
