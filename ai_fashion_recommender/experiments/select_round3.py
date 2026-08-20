"""3차 보강 선택 — 탐욕적 집합 커버.

우선순위(현재 학습 표본 수 기준 5단계)와 기존 모델의 라벨별 F1을 함께 반영한다.
    <30 → <100 → <300 → <800 → <1500
같은 단계 안에서는 F1이 낮은 라벨에 더 큰 가중치를 준다.
한 이미지가 여러 부족 라벨을 채우면 점수가 합산되어 자연히 높아진다.

제외 규칙
  - Fashionpedia train2020 에 존재하지 않는 라벨은 점수에서 제외한다.
    (lower_subtype · pant_leg_shape · pant_length · lower_detail 4개 태스크 전체 포함)
  - 이미 1,500개 이상인 라벨은 가중치 0 → 그런 라벨만 채우는 이미지는 선택되지 않는다.
  - 1·2차에서 선택된 image_id 전체(10,500개, crop 0이던 11개 포함)를 후보에서 뺀다.
  - old/new validation 이미지도 뺀다.

라벨 매핑은 기존 scan 결과(train2020_label_scan.json)를 그대로 쓴다. 새 규칙을 만들지 않는다.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data"
MANIFESTS = PROJECT_DIR / "reports" / "manifests"
REPORTS = PROJECT_DIR / "reports"

SEED = 20260819
MAX_IMAGES = 6000
# (상한, 가중치). 표본이 적을수록 압도적으로 큰 가중치를 준다.
TIERS = ((30, 100.0), (100, 40.0), (300, 12.0), (800, 4.0), (1500, 1.0))
EXCLUDED_TASKS = ("lower_subtype", "pant_leg_shape", "pant_length", "lower_detail")

CSV_COLUMNS = [
    "dataset", "official_split", "image_id", "original_filename", "source_shard",
    "augmentation_round", "selected", "contributed_labels", "selection_score",
    "valid_crop_count", "usage_split", "used_for_training",
]


def tier_weight(count: int) -> float:
    for limit, weight in TIERS:
        if count < limit:
            return weight
    return 0.0


def tier_name(count: int) -> str:
    for index, (limit, _) in enumerate(TIERS, start=1):
        if count < limit:
            return f"T{index}(<{limit})"
    return "충분"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-images", type=int, default=MAX_IMAGES)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--sample-size", type=int, default=600,
                        help="탐욕 각 단계에서 검토할 후보 표본 수(근사 탐욕)")
    args = parser.parse_args()

    inputs = json.loads((REPORTS / "_round3_inputs.json").read_text(encoding="utf-8"))
    counts: dict[str, int] = dict(inputs["label_counts"])
    available = set(inputs["available_labels"])
    label_f1: dict[str, float] = inputs["label_f1"]

    scan = json.loads(
        (DATA / "provenance" / "fashionpedia" / "train2020_label_scan.json").read_text(encoding="utf-8"))
    image_labels = {int(k): set(v) for k, v in scan["image_labels"].items()}
    annotation_count = {int(k): v for k, v in scan["image_annotation_count"].items()}
    shard_index = json.loads(
        (DATA / "provenance" / "fashionpedia" / "shard_index.json").read_text(encoding="utf-8"))
    id_to_shard = {int(k): v for k, v in shard_index["id_to_shard"].items()}
    id_to_name = {}
    for round_name in ("r1", "r2"):
        subset = json.loads(
            (DATA / "provenance" / "fashionpedia" / round_name / "instances_subset.json").read_text(encoding="utf-8"))
        for item in subset["images"]:
            id_to_name[int(item["id"])] = item["file_name"]

    excluded = set(json.loads(
        (MANIFESTS / "fashionpedia_train_excluded_image_ids.json").read_text(encoding="utf-8"))["image_ids"])
    used = read_csv(MANIFESTS / "used_image_manifest.csv")
    used_train2020 = {int(r["image_id"]) for r in used
                      if r["official_split"] == "train2020" and r["image_id"]}
    exclusion = excluded | used_train2020

    # 점수 대상 라벨: Fashionpedia에 존재 + 제외 태스크가 아님
    scorable = {
        label for label in available
        if not label.startswith(tuple(f"{t}|" for t in EXCLUDED_TASKS))
    }

    pool = [i for i in image_labels if i not in exclusion and image_labels[i]]
    print(f"후보 풀: 전체 {len(image_labels):,} − 제외 {len(exclusion):,} = {len(pool):,}")
    print(f"점수 대상 라벨: {len(scorable):,} (Fashionpedia 미존재·하의 4축 제외)")

    def score(image_id: int, current: dict[str, int]) -> tuple[float, list[str]]:
        total, contributed = 0.0, []
        for label in image_labels[image_id]:
            if label not in scorable:
                continue
            weight = tier_weight(current.get(label, 0))
            if weight <= 0:
                continue
            # F1이 낮을수록 최대 2배까지 가중. F1 정보가 없으면 중립(1.5배).
            f1 = label_f1.get(label)
            boost = 1.0 + (1.0 - f1) if f1 is not None else 1.5
            total += weight * boost
            contributed.append(label)
        return total, contributed

    rng = random.Random(args.seed)
    rng.shuffle(pool)
    current = dict(counts)
    selected: list[dict] = []
    gained: Counter[str] = Counter()
    stop_reason = "최대 이미지 수 도달"

    while len(selected) < args.max_images and pool:
        sample = pool if len(pool) <= args.sample_size else rng.sample(pool, args.sample_size)
        best_id, best_score, best_labels = None, 0.0, []
        for image_id in sample:
            value, labels = score(image_id, current)
            if value > best_score:
                best_id, best_score, best_labels = image_id, value, labels
        if best_id is None:
            stop_reason = "부족 라벨을 더 채울 수 있는 후보가 표본에 없음"
            break
        pool.remove(best_id)
        for label in best_labels:
            current[label] = current.get(label, 0) + 1
            gained[label] += 1
        selected.append({
            "image_id": best_id,
            "contributed_labels": best_labels,
            "selection_score": round(best_score, 3),
        })
        if len(selected) % 1000 == 0:
            print(f"  {len(selected):,}장 선택  (최근 점수 {best_score:.1f})", flush=True)

    print(f"\n선택 {len(selected):,}장 — 중단 사유: {stop_reason}")
    annotations = sum(annotation_count[e["image_id"]] for e in selected)
    estimated_crops = annotations * 8513 / 36999
    print(f"주석 {annotations:,} → 예상 crop 약 {estimated_crops:,.0f}")

    rows = []
    for entry in sorted(selected, key=lambda e: e["image_id"]):
        image_id = entry["image_id"]
        rows.append({
            "dataset": "fashionpedia",
            "official_split": "train2020",
            "image_id": image_id,
            "original_filename": id_to_name.get(image_id, ""),  # 추출 후 채워짐
            "source_shard": id_to_shard[image_id],
            "augmentation_round": 3,
            "selected": "true",
            "contributed_labels": "|".join(entry["contributed_labels"]),
            "selection_score": entry["selection_score"],
            "valid_crop_count": "",     # 추출·CSV 생성 후 채운다
            "usage_split": "",
            "used_for_training": "",
        })
    MANIFESTS.mkdir(parents=True, exist_ok=True)
    out_csv = MANIFESTS / "fashionpedia_train_r3_images.csv"
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"저장: {out_csv.relative_to(PROJECT_DIR).as_posix()} ({len(rows):,}행)")

    # ---- 라벨별 전후 비교
    print(f"\n{'라벨':<30}{'전':>7}{'추가':>7}{'후':>7}{'단계(전)':>12}")
    print("-" * 66)
    changed = sorted(gained.items(), key=lambda kv: counts.get(kv[0], 0))
    for label, add in changed[:35]:
        before = counts.get(label, 0)
        print(f"{label:<30}{before:>7}{add:>7}{before + add:>7}{tier_name(before):>12}")
    if len(changed) > 35:
        print(f"  … 외 {len(changed) - 35}개 라벨")

    report = {
        "round": 3,
        "seed": args.seed,
        "max_images": args.max_images,
        "sample_size": args.sample_size,
        "tiers": [{"limit": l, "weight": w} for l, w in TIERS],
        "excluded_tasks": list(EXCLUDED_TASKS),
        "candidate_pool": len(pool) + len(selected),
        "excluded_image_ids": len(exclusion),
        "selected_images": len(selected),
        "stop_reason": stop_reason,
        "annotations": annotations,
        "estimated_crops": round(estimated_crops),
        "label_before": {k: counts.get(k, 0) for k in gained},
        "label_gained": dict(gained),
        "label_after": {k: counts.get(k, 0) + v for k, v in gained.items()},
        "scorable_labels": len(scorable),
    }
    (REPORTS / "16_round3_selection.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"저장: reports/16_round3_selection.json")

    # ---- 즉시 교집합 검사
    print()
    print("=" * 70)
    print("선택 직후 교집합 검사")
    print("=" * 70)
    ids3 = {e["image_id"] for e in selected}
    r1 = {int(r["image_id"]) for r in read_csv(MANIFESTS / "fashionpedia_train_r1_images.csv")}
    r2 = {int(r["image_id"]) for r in read_csv(MANIFESTS / "fashionpedia_train_r2_images.csv")}
    old_val = {r["unique_key"] for r in used if r["usage_split"] == "old_val"}
    new_val = {r["unique_key"] for r in used if r["usage_split"] == "new_val"}
    keys3 = {f"fashionpedia::train2020::{i}" for i in ids3}
    checks = [
        ("r3 ∩ r1", len(ids3 & r1)),
        ("r3 ∩ r2", len(ids3 & r2)),
        ("r3 ∩ old validation", len(keys3 & old_val)),
        ("r3 ∩ new validation", len(keys3 & new_val)),
        ("r3 내부 image_id 중복", len(selected) - len(ids3)),
    ]
    failed = False
    for name, value in checks:
        print(f"  [{'PASS' if value == 0 else 'FAIL'}] {name}: {value}")
        failed |= value != 0
    if failed:
        raise SystemExit("\n교집합 검사 실패 — 중단합니다.")
    print("\n전부 0 — 다음 단계 진행 가능")


if __name__ == "__main__":
    main()
