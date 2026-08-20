"""최종 검증 — 같은 원본 이미지의 crop이 학습과 평가 양쪽에 들어갔는지 확인한다.

한 장의 사진에서 상의·하의 crop이 여러 개 나오므로, crop 단위로 나누면
같은 이미지가 학습과 평가에 동시에 들어가 성능이 부풀 수 있다.
여기서는 세 가지를 각각 확인한다.

  A. 학습 캐시 vs 평가 캐시 (기존 val 1,195 / 새 val 1,716)
  B. 1차·2차 보강분끼리의 중복
  C. split_train_dev() 가 만드는 fit/dev (early stopping·임계값 선택용)

이미지 동일성은 경로 꼬리 <출처>/images/<파일명> 으로 판단한다.
캐시가 다른 PC에서 만들어져 절대경로 앞부분이 다르기 때문이다.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch

from eval_lib import PROJECT_DIR, REPORT_DIR, TRAIN_CACHE, VAL_CACHE, load_cache, split_train_dev

CACHE_DIR = PROJECT_DIR / "data" / "cache"
R1_TRAIN = CACHE_DIR / "fashion_attributes_fp_train.pt"
R1_VAL = CACHE_DIR / "fashion_attributes_fp_val.pt"
R2_TRAIN = CACHE_DIR / "fashion_attributes_fp_train_r2.pt"


def image_key(path: str) -> str:
    parts = Path(str(path).replace("\\", "/")).parts
    return "/".join(parts[-3:]).lower()


def keys(cache) -> list[str]:
    return [image_key(p) for p in cache["image_paths"]]


def report_overlap(name_a, keys_a, name_b, keys_b, findings):
    set_a, set_b = set(keys_a), set(keys_b)
    shared = set_a & set_b
    status = "OK" if not shared else "LEAK"
    print(
        f"  [{status:4}] {name_a:<22} ∩ {name_b:<22} "
        f"이미지 {len(set_a):>6,} / {len(set_b):>6,} → 공통 {len(shared):,}"
    )
    findings.append({
        "a": name_a, "b": name_b,
        "images_a": len(set_a), "images_b": len(set_b),
        "shared": len(shared),
        "examples": sorted(shared)[:5],
    })
    return shared


def main() -> None:
    torch.set_num_threads(2)
    findings: dict = {"image_level": [], "fit_dev": {}, "crops_per_image": {}}

    caches = {
        "old_train(4789)": load_cache(TRAIN_CACHE),
        "old_val(1195)": load_cache(VAL_CACHE),
        "r1_train(6797)": load_cache(R1_TRAIN),
        "r1_val(1716)": load_cache(R1_VAL),
        "r2_train(10755)": load_cache(R2_TRAIN),
    }
    key_map = {name: keys(cache) for name, cache in caches.items()}

    print("=" * 92)
    print("A. 학습 캐시 vs 평가 캐시 — 같은 이미지가 양쪽에 있으면 안 된다")
    print("=" * 92)
    train_names = ["old_train(4789)", "r1_train(6797)", "r2_train(10755)"]
    eval_names = ["old_val(1195)", "r1_val(1716)"]
    total_leak = 0
    for train_name in train_names:
        for eval_name in eval_names:
            total_leak += len(
                report_overlap(train_name, key_map[train_name], eval_name, key_map[eval_name],
                               findings["image_level"])
            )

    print()
    print("=" * 92)
    print("B. 보강분끼리 / 평가셋끼리 중복")
    print("=" * 92)
    total_leak += len(report_overlap(
        "r1_train(6797)", key_map["r1_train(6797)"], "r2_train(10755)", key_map["r2_train(10755)"],
        findings["image_level"]))
    total_leak += len(report_overlap(
        "old_val(1195)", key_map["old_val(1195)"], "r1_val(1716)", key_map["r1_val(1716)"],
        findings["image_level"]))
    total_leak += len(report_overlap(
        "old_train(4789)", key_map["old_train(4789)"], "r1_train(6797)", key_map["r1_train(6797)"],
        findings["image_level"]))
    total_leak += len(report_overlap(
        "old_train(4789)", key_map["old_train(4789)"], "r2_train(10755)", key_map["r2_train(10755)"],
        findings["image_level"]))

    print()
    print("=" * 92)
    print("C. split_train_dev() 가 만드는 fit/dev — early stopping·임계값 선택에 쓰인 분할")
    print("=" * 92)
    combined_paths = (
        caches["old_train(4789)"]["image_paths"]
        + caches["r1_train(6797)"]["image_paths"]
        + caches["r2_train(10755)"]["image_paths"]
    )
    combined = {
        "version": 1,
        "backbone_model_id": caches["old_train(4789)"]["backbone_model_id"],
        "features": torch.cat([caches[n]["features"] for n in train_names], dim=0),
        "targets": {}, "valid": {},
        "image_paths": combined_paths,
    }
    fit_indices, dev_indices = split_train_dev(combined)
    fit_keys = [image_key(combined_paths[int(i)]) for i in fit_indices]
    dev_keys = [image_key(combined_paths[int(i)]) for i in dev_indices]
    shared = set(fit_keys) & set(dev_keys)
    dev_shared_rows = sum(1 for k in dev_keys if k in shared)
    print(f"  fit crop {len(fit_keys):,} / dev crop {len(dev_keys):,}")
    print(f"  fit 이미지 {len(set(fit_keys)):,} / dev 이미지 {len(set(dev_keys)):,}")
    print(f"  양쪽에 걸친 이미지: {len(shared):,}")
    print(f"  그 때문에 오염된 dev crop: {dev_shared_rows:,} / {len(dev_keys):,} "
          f"({dev_shared_rows / max(len(dev_keys), 1) * 100:.1f}%)")
    findings["fit_dev"] = {
        "fit_crops": len(fit_keys), "dev_crops": len(dev_keys),
        "fit_images": len(set(fit_keys)), "dev_images": len(set(dev_keys)),
        "shared_images": len(shared), "contaminated_dev_crops": dev_shared_rows,
        "contaminated_ratio": round(dev_shared_rows / max(len(dev_keys), 1), 4),
    }

    print()
    print("=" * 92)
    print("D. 이미지당 crop 수 (crop 단위 분할이 왜 문제가 되는지의 근거)")
    print("=" * 92)
    for name, key_list in key_map.items():
        counts = Counter(key_list)
        multi = sum(1 for v in counts.values() if v > 1)
        print(f"  {name:<22} crop {len(key_list):>6,}  이미지 {len(counts):>6,}  "
              f"crop/이미지 {len(key_list)/len(counts):.2f}  2개 이상인 이미지 {multi:,}")
        findings["crops_per_image"][name] = {
            "crops": len(key_list), "images": len(counts),
            "crops_per_image": round(len(key_list) / len(counts), 3),
            "multi_crop_images": multi,
        }

    print()
    print("=" * 92)
    verdict = "PASS" if total_leak == 0 else "FAIL"
    print(f"결론: 학습셋 ↔ 평가셋 이미지 누수 = {total_leak}건  →  {verdict}")
    if findings["fit_dev"]["shared_images"]:
        print("주의: fit/dev 는 crop 단위로 나뉘어 같은 이미지가 양쪽에 걸쳐 있습니다.")
        print("      dev 는 early stopping·임계값 선택에만 쓰였고 최종 성능 보고에는 쓰이지 않았습니다.")
        print("      따라서 보고된 val 점수는 부풀지 않지만, dev 점수 자체는 낙관적입니다.")
    print("=" * 92)

    findings["verdict"] = verdict
    findings["train_eval_leak_images"] = total_leak
    output = REPORT_DIR / "13_split_integrity.json"
    output.write_text(json.dumps(findings, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
