"""5단계 — 라벨 자체를 고치고 캐시를 다시 만든다.

두 가지 변경을 각각 분리해 만든다.

D 변형: `디테일 없음` 음성 표본 확보
  Fashionpedia는 전문가가 속성을 전수 주석하므로, Fashionpedia 행에서 detail이 비어 있다는 건
  "미주석"보다 "실제로 그 디테일이 없다"에 가깝다. 지금은 그 행이 전부 valid=False로
  손실에서 빠져서 detail 헤드가 "디테일이 하나라도 있는 옷"만 보고 학습된다.
  → Fashionpedia 행의 빈 detail을 `디테일 없음`으로 채운다. Fashion200K 약지도 행은 손대지 않는다.

K 변형: material 니트 하드코딩 제거
  fashion_attribute_dataset.py:254 가 category가 니트/가디건이면 material에 니트를 자동 주입한다.
  이게 정말 해로운지는 측정으로 확인한다. (선험적으로는 올바른 라벨이라 도움이 될 수도 있다)

FashionSigLIP 임베딩은 재계산하지 않는다. 캐시의 image_paths 순서가 CSV 레코드 순서와
같다는 것을 검증한 뒤 targets/valid 텐서만 다시 만든다.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from eval_lib import PROJECT_DIR, TRAIN_CACHE, VAL_CACHE, load_cache
from fashion_attribute_dataset import MULTI_VALUE_SEPARATOR, load_attribute_csv
from fashion_attribute_schema import ATTRIBUTE_TASKS, label_index

ANNOTATION_CSV = PROJECT_DIR / "data" / "fashion_attribute_annotations.csv"
IMAGE_ROOT = PROJECT_DIR / "data"
CACHE_DIR = PROJECT_DIR / "data" / "cache"
FASHIONPEDIA_MARKER = "fashionpedia_seed"


def read_rows() -> list[dict[str, str]]:
    with ANNOTATION_CSV.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def aligned_rows(split: str, cache: dict) -> list[dict[str, str]]:
    """캐시 행과 CSV 행의 1:1 대응을 검증한다.

    load_attribute_csv 는 CSV 순서를 유지하고 build_embedding_cache 는 그 레코드 순서로
    image_paths 를 저장한다. 실제로 같은지 경로 비교로 확인한다.
    """
    records = load_attribute_csv(ANNOTATION_CSV, IMAGE_ROOT, split=split, require_images=False)
    if len(records) != len(cache["features"]):
        raise SystemExit(
            f"{split}: 레코드 {len(records)}개와 캐시 {len(cache['features'])}개가 다릅니다."
        )
    # 캐시는 다른 팀원 PC에서 만들어져 절대경로 앞부분이 다르다.
    # <출처>/images/<파일명> 세 조각만 비교한다.
    def tail(path: str | Path) -> str:
        parts = Path(str(path).replace("\\", "/")).parts
        return "/".join(parts[-3:]).lower()

    for index, (record, cached) in enumerate(zip(records, cache["image_paths"])):
        if tail(record.image_path) != tail(cached):
            raise SystemExit(f"{split}:{index} 경로 불일치 {tail(record.image_path)} != {tail(cached)}")

    rows = [row for row in read_rows() if (row.get("split") or "train").strip().lower() == split]
    if len(rows) != len(records):
        raise SystemExit(f"{split}: CSV 행 {len(rows)}개와 레코드 {len(records)}개가 다릅니다.")
    return rows


def parse_labels(row: dict[str, str], task_name: str) -> list[str]:
    raw = (row.get(task_name) or "").strip()
    return [value.strip() for value in raw.split(MULTI_VALUE_SEPARATOR) if value.strip()]


def apply_variant(
    rows: list[dict[str, str]],
    *,
    detail_none: bool,
    drop_knit: bool,
    detail_none_ratio: float = 1.0,
    seed: int = 7,
) -> dict:
    """변형 라벨을 적용하고 detail/material 타깃과 valid 마스크를 다시 만든다.

    detail_none_ratio < 1 이면 `디테일 없음` 음성 표본을 그 비율만큼만 채운다.
    전부 채우면 음성 행이 학습 detail 행의 42%가 되어 단추·지퍼 같은 강한 라벨의
    결정경계가 "없음" 쪽으로 밀린다. 비율로 그 트레이드오프를 조절한다.
    """
    import random as _random

    changed = {"detail_none": 0, "knit_removed": 0, "material_emptied": 0}
    eligible = [
        index for index, row in enumerate(rows)
        if FASHIONPEDIA_MARKER in row["image_path"].replace("\\", "/")
        and not parse_labels(row, "detail")
    ]
    chosen = set(eligible)
    if detail_none and detail_none_ratio < 1.0:
        picker = _random.Random(seed)
        keep = round(len(eligible) * detail_none_ratio)
        chosen = set(picker.sample(eligible, keep))
    detail_labels = ATTRIBUTE_TASKS["detail"].labels
    material_labels = ATTRIBUTE_TASKS["material"].labels
    detail_index = label_index("detail")
    material_index = label_index("material")

    detail_target = torch.zeros(len(rows), len(detail_labels))
    detail_valid = torch.zeros(len(rows), dtype=torch.bool)
    material_target = torch.zeros(len(rows), len(material_labels))
    material_valid = torch.zeros(len(rows), dtype=torch.bool)

    for index, row in enumerate(rows):
        is_fashionpedia = FASHIONPEDIA_MARKER in row["image_path"].replace("\\", "/")
        details = parse_labels(row, "detail")
        if detail_none and is_fashionpedia and not details and index in chosen:
            details = ["디테일 없음"]
            changed["detail_none"] += 1
        for value in details:
            detail_target[index, detail_index[value]] = 1.0
        detail_valid[index] = bool(details)

        materials = parse_labels(row, "material")
        if drop_knit and row.get("category", "").strip() in {"니트", "가디건"} and "니트" in materials:
            materials = [value for value in materials if value != "니트"]
            changed["knit_removed"] += 1
            if not materials:
                changed["material_emptied"] += 1
        for value in materials:
            material_target[index, material_index[value]] = 1.0
        material_valid[index] = bool(materials)

    return {
        "detail": (detail_target, detail_valid),
        "material": (material_target, material_valid),
        "changed": changed,
    }


def build(name: str, *, detail_none: bool, drop_knit: bool, detail_none_ratio: float = 1.0) -> dict:
    summary = {
        "variant": name, "detail_none": detail_none, "drop_knit": drop_knit,
        "detail_none_ratio": detail_none_ratio, "splits": {},
    }
    for split, source in (("train", TRAIN_CACHE), ("val", VAL_CACHE)):
        cache = load_cache(source)
        rows = aligned_rows(split, cache)
        # val 은 항상 전량 채운다. 평가 정답을 학습 비율에 따라 바꾸면 비교가 깨진다.
        variant = apply_variant(
            rows, detail_none=detail_none, drop_knit=drop_knit,
            detail_none_ratio=1.0 if split == "val" else detail_none_ratio,
        )
        targets = dict(cache["targets"])
        valid = dict(cache["valid"])
        for task_name in ("detail", "material"):
            targets[task_name], valid[task_name] = variant[task_name]
        output = CACHE_DIR / f"fashion_attributes_{split}_{name}.pt"
        torch.save(
            {
                "version": 1,
                "backbone_model_id": cache["backbone_model_id"],
                "features": cache["features"],
                "targets": targets,
                "valid": valid,
                "image_paths": cache["image_paths"],
            },
            output,
        )
        summary["splits"][split] = {
            "rows": len(rows),
            "changed": variant["changed"],
            "detail_valid": int(valid["detail"].sum()),
            "detail_valid_before": int(cache["valid"]["detail"].sum()),
            "material_valid": int(valid["material"].sum()),
            "material_valid_before": int(cache["valid"]["material"].sum()),
            "detail_none_positives": int(
                targets["detail"][:, ATTRIBUTE_TASKS["detail"].labels.index("디테일 없음")].sum()
            ),
            "cache": str(output),
        }
        print(
            f"  {name:<6} {split:<5} rows={len(rows):<5} "
            f"detail valid {summary['splits'][split]['detail_valid_before']} -> "
            f"{summary['splits'][split]['detail_valid']} "
            f"(디테일 없음 {variant['changed']['detail_none']}건 추가) | "
            f"material valid {summary['splits'][split]['material_valid_before']} -> "
            f"{summary['splits'][split]['material_valid']} "
            f"(니트 제거 {variant['changed']['knit_removed']})"
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variants", nargs="*", default=["D", "K", "DK"])
    arguments = parser.parse_args()

    torch.set_num_threads(2)
    options = {
        "D": dict(detail_none=True, drop_knit=False),
        "K": dict(detail_none=False, drop_knit=True),
        "DK": dict(detail_none=True, drop_knit=True),
        "D25": dict(detail_none=True, drop_knit=False, detail_none_ratio=0.25),
        "D50": dict(detail_none=True, drop_knit=False, detail_none_ratio=0.50),
    }
    summaries = []
    for name in arguments.variants:
        if name not in options:
            raise SystemExit(f"알 수 없는 변형: {name}")
        summaries.append(build(name, **options[name]))

    output = PROJECT_DIR / "reports" / "08_relabel_summary.json"
    output.write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
