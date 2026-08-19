"""7단계 — 생성된 매니페스트를 전수 검증한다. 매니페스트를 수정하지 않는다."""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA = PROJECT_DIR / "data"
MANIFESTS = PROJECT_DIR / "reports" / "manifests"

ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]|\\\\|/Users/|/home/|AppData|scratchpad|OneDrive")

results: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, ok, detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    r1 = read_csv(MANIFESTS / "fashionpedia_train_r1_images.csv")
    r2 = read_csv(MANIFESTS / "fashionpedia_train_r2_images.csv")
    read_csv(MANIFESTS / "fashionpedia_val_seed_images.csv")
    read_csv(MANIFESTS / "fashion200k_images.csv")
    used = read_csv(MANIFESTS / "used_image_manifest.csv")

    print("=" * 82)
    print("A. 요구된 개수 검증")
    print("=" * 82)
    check("r1 selected image_id", len(r1) == 4500, f"{len(r1):,} (기대 4,500)")
    r1_valid = [r for r in r1 if int(r["valid_crop_count"]) > 0]
    check("r1 valid-crop image_id", len(r1_valid) == 4496, f"{len(r1_valid):,} (기대 4,496)")
    check("r2 selected image_id", len(r2) == 6000, f"{len(r2):,} (기대 6,000)")
    r2_valid = [r for r in r2 if int(r["valid_crop_count"]) > 0]
    check("r2 valid-crop image_id", len(r2_valid) == 5993, f"{len(r2_valid):,} (기대 5,993)")
    ids1 = {int(r["image_id"]) for r in r1}
    ids2 = {int(r["image_id"]) for r in r2}
    check("r1 ∩ r2 선택 교집합", len(ids1 & ids2) == 0, f"{len(ids1 & ids2)}건 (기대 0)")

    zero1 = [r for r in r1 if int(r["valid_crop_count"]) == 0]
    zero2 = [r for r in r2 if int(r["valid_crop_count"]) == 0]
    ok_zero = all(
        r["usage_split"] == "selected_no_valid_crop" and r["used_for_training"] == "false"
        for r in zero1 + zero2
    )
    check("crop 0 이미지 기록 규칙", ok_zero,
          f"r1 {len(zero1)}건 + r2 {len(zero2)}건 모두 "
          f"selected_no_valid_crop / used_for_training=false")

    print()
    print("=" * 82)
    print("B. 데이터 출처별 고유 이미지 수")
    print("=" * 82)
    by_source: Counter[str] = Counter()
    for row in used:
        if row["dataset"] == "fashionpedia":
            key = f"fashionpedia / {row['official_split']}"
            if row["augmentation_round"] not in ("", "0"):
                key += f" (보강 {row['augmentation_round']}차)"
            by_source[key] += 1
        else:
            by_source[f"fashion200k / {row['source_subset']}"] += 1
    total_images = 0
    for name, count in sorted(by_source.items()):
        print(f"  {name:<48}{count:>8,}")
        total_images += count
    print(f"  {'합계':<48}{total_images:>8,}")
    check("통합 매니페스트 행 수", len(used) == total_images, f"{len(used):,}행")

    print()
    print("=" * 82)
    print("C. usage_split 별 이미지·crop 수")
    print("=" * 82)
    images_by_split: Counter[str] = Counter()
    crops_by_split: Counter[str] = Counter()
    for row in used:
        images_by_split[row["usage_split"]] += 1
        crops_by_split[row["usage_split"]] += int(row["crop_count"])
    print(f"  {'usage_split':<28}{'이미지':>10}{'crop':>10}")
    for split in ("train", "old_val", "new_val", "selected_no_valid_crop"):
        print(f"  {split:<28}{images_by_split[split]:>10,}{crops_by_split[split]:>10,}")
    print(f"  {'합계':<28}{sum(images_by_split.values()):>10,}{sum(crops_by_split.values()):>10,}")

    check("train crop 22,341 재현", crops_by_split["train"] == 22341, f"{crops_by_split['train']:,}")
    check("old validation crop 1,195", crops_by_split["old_val"] == 1195, f"{crops_by_split['old_val']:,}")
    check("new validation crop 1,716", crops_by_split["new_val"] == 1716, f"{crops_by_split['new_val']:,}")
    expected_total = 5984 + 8513 + 10755
    check("전체 crop 합계 = 원본 CSV 합계", sum(crops_by_split.values()) == expected_total,
          f"{sum(crops_by_split.values()):,} (CSV 합계 {expected_total:,})")

    print()
    print("=" * 82)
    print("D. 무결성")
    print("=" * 82)
    absolute_rows = 0
    for path in sorted(MANIFESTS.glob("*.csv")):
        rows = read_csv(path)
        hits = sum(1 for r in rows if any(ABSOLUTE_PATH.search(str(v)) for v in r.values()))
        absolute_rows += hits
        if hits:
            print(f"    {path.name}: {hits}행에 절대경로 흔적")
    check("절대경로가 남은 행", absolute_rows == 0, f"{absolute_rows}행")

    keys = Counter(r["unique_key"] for r in used)
    duplicates = {k: v for k, v in keys.items() if v > 1}
    check("고유키 중복", len(duplicates) == 0,
          f"{len(duplicates)}건" + (f" 예: {list(duplicates)[:3]}" if duplicates else ""))

    train_keys = {r["unique_key"] for r in used if r["usage_split"] == "train"}
    old_val_keys = {r["unique_key"] for r in used if r["usage_split"] == "old_val"}
    new_val_keys = {r["unique_key"] for r in used if r["usage_split"] == "new_val"}
    check("train ∩ old validation", len(train_keys & old_val_keys) == 0,
          f"{len(train_keys & old_val_keys)}건")
    check("train ∩ new validation", len(train_keys & new_val_keys) == 0,
          f"{len(train_keys & new_val_keys)}건")
    check("old validation ∩ new validation", len(old_val_keys & new_val_keys) == 0,
          f"{len(old_val_keys & new_val_keys)}건")

    unrecovered = [r for r in used if r["image_id_recovered"] == "false"]
    check("image_id 미복원", len(unrecovered) == 0,
          f"{len(unrecovered)}건" + (f" 예: {unrecovered[0]['unique_key']}" if unrecovered else ""))

    print()
    print("=" * 82)
    print("E. manifest_r2 교정본 검증")
    print("=" * 82)
    corrected = json.loads(
        (DATA / "fashionpedia_train" / "manifest_r2_corrected.json").read_text(encoding="utf-8"))
    shard_index = json.loads(
        (DATA / "provenance" / "fashionpedia" / "shard_index.json").read_text(encoding="utf-8"))
    id_to_shard = {int(k): v for k, v in shard_index["id_to_shard"].items()}
    selection = json.loads((DATA / "fashionpedia_train" / "selection_r2.json").read_text())
    actual_shards = sorted({id_to_shard[int(i)] for i in selection["image_ids"]})
    check("shards_used == selection_r2 실측", corrected["shards_used"] == actual_shards,
          f"{corrected['shards_used']} == {actual_shards}")
    check("selection target == selection_r2.target",
          corrected["selection_target"] == selection["target"],
          f"{corrected['selection_target']} == {selection['target']}")
    r2_csv = read_csv(DATA / "fashionpedia_train" / "fashion_attribute_annotations_r2.csv")
    check("csv_rows == 실제 CSV 행", corrected["csv_rows"] == len(r2_csv),
          f"{corrected['csv_rows']:,} == {len(r2_csv):,}")
    check("images == selection_r2 개수", corrected["images"] == len(selection["image_ids"]),
          f"{corrected['images']:,} == {len(selection['image_ids']):,}")
    original = json.loads(
        (DATA / "fashionpedia_train" / "manifest_r2.json").read_text(encoding="utf-8"))
    check("원본 manifest_r2.json 미변경", original["shards_used"] == [0, 1],
          "원본 그대로 (shards_used=[0,1] 유지)")

    print()
    print("=" * 82)
    print("F. 향후 후보 선정에서 제외할 image_id")
    print("=" * 82)
    exclude = ids1 | ids2
    print(f"  Fashionpedia train2020 선택분(사용 여부 무관): {len(exclude):,}개")
    print(f"    = r1 {len(ids1):,} + r2 {len(ids2):,}, 교집합 {len(ids1 & ids2)}")
    total_pool = sum(v["rows"] for v in shard_index["shards"].values())
    print(f"  train2020 전체 {total_pool:,} → 남은 후보 {total_pool - len(exclude):,}")
    exclude_path = MANIFESTS / "fashionpedia_train_excluded_image_ids.json"
    exclude_path.write_text(json.dumps({
        "dataset": "fashionpedia",
        "official_split": "train2020",
        "note": "1·2차에서 선택된 전체 image_id. crop 0이었던 11개도 포함해 향후 후보에서 제외한다.",
        "count": len(exclude),
        "image_ids": sorted(exclude),
    }), encoding="utf-8")
    print(f"  저장: {exclude_path.relative_to(PROJECT_DIR).as_posix()}")

    print()
    print("=" * 82)
    failed = [name for name, ok, _ in results if not ok]
    print(f"검증 {len(results)}건 중 통과 {len(results) - len(failed)}건, 실패 {len(failed)}건")
    for name in failed:
        print(f"  FAIL: {name}")
    print("=" * 82)


if __name__ == "__main__":
    main()
