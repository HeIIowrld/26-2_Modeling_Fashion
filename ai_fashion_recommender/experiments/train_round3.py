"""3차 재학습 — R2_mixup 설정 그대로, image-level dev로만 모델을 선택한다.

old/new validation 은 학습·early stopping·임계값 선택 어디에도 쓰지 않는다.
최종 체크포인트는 validation 점수가 아니라 **dev composite** 가 가장 좋은 seed 로 고른다.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from eval_lib import PROJECT_DIR, REPORT_DIR, TRAIN_CACHE, load_cache, subset
from evaluate_relabel import label_support
from fashion_attribute_model import build_attribute_heads, save_attribute_checkpoint
from fashion_attribute_schema import ATTRIBUTE_TASKS
from finetune import HIDDEN_DIM, RECIPES, train_once
from image_level_split import compare_with_crop_level, split_by_image, verify
from train_with_train_split import concat_caches, tune_all_thresholds

CACHE_DIR = PROJECT_DIR / "data" / "cache"
MODELS = PROJECT_DIR / "models"
BASELINE_R2 = MODELS / "fashion_attribute_heads_augmented.pt"
MINIMUM_LABEL_EXAMPLES = 5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    began = time.time()

    before_hash = sha256(BASELINE_R2)
    print(f"학습 전 2차 모델 sha256: {before_hash}")

    caches = [
        load_cache(TRAIN_CACHE),
        load_cache(CACHE_DIR / "fashion_attributes_fp_train.pt"),
        load_cache(CACHE_DIR / "fashion_attributes_fp_train_r2.pt"),
        load_cache(CACHE_DIR / "fashion_attribute_embeddings_r3.pt"),
    ]
    combined = concat_caches(caches)
    sizes = " + ".join(f"{len(c['features']):,}" for c in caches)
    print(f"train crop: {sizes} = {len(combined['features']):,}")

    fit_rows, dev_rows, fit_keys, dev_keys, info = split_by_image(combined)
    print(f"\nimage-level 분할: fit {info['fit_crops']:,} crop / {info['fit_images']:,} 이미지, "
          f"dev {info['dev_crops']:,} crop / {info['dev_images']:,} 이미지")

    print("\n분할 무결성 검사")
    checks = verify(combined, fit_rows, dev_rows, fit_keys, dev_keys)
    failed = False
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        failed |= not ok
    if failed:
        raise SystemExit("분할 검사 실패 — 학습을 시작하지 않습니다.")

    legacy = compare_with_crop_level(combined)
    print(f"\n참고: 같은 데이터를 기존 crop 단위로 나누면 dev crop의 "
          f"{legacy['crop_level_contaminated_ratio'] * 100:.1f}%가 fit과 이미지를 공유합니다 "
          f"(이미지 {legacy['crop_level_shared_images']:,}장). image-level 분할은 0%입니다.")

    fit = subset(combined, fit_rows)
    dev = subset(combined, dev_rows)
    recipe = next(r for r in RECIPES if r.name == "R2_mixup")
    print(f"\n레시피: {asdict(recipe)}")

    support = label_support(fit)
    seed_results = {}
    states = {}
    for seed in args.seeds:
        started = time.time()
        run = train_once(recipe, seed, fit, dev, "cpu")
        composite = sum(run["dev_selection"].values()) / len(run["dev_selection"])
        seed_results[seed] = {
            "dev_composite": round(composite, 6),
            "dev_selection": {k: round(v, 4) for k, v in run["dev_selection"].items()},
            "best_epoch": run["dev_epoch"],
            "seconds": round(time.time() - started),
        }
        states[seed] = run["state"]
        print(f"  seed={seed}  dev_composite={composite:.4f}  {time.time() - started:.0f}s", flush=True)

        heads = build_attribute_heads(int(fit["features"].shape[1]), HIDDEN_DIM, recipe.dropout)
        for task_name, state in run["state"].items():
            heads.heads[task_name].load_state_dict(state)
        heads.eval()
        thresholds = tune_all_thresholds(heads, dev, support, MINIMUM_LABEL_EXAMPLES)
        save_attribute_checkpoint(
            MODELS / f"fashion_attribute_heads_augmented_r3_seed{seed}.pt",
            heads, backbone_model_id=fit["backbone_model_id"], thresholds=thresholds,
            training_summary={
                "round": 3, "seed": seed, "recipe": asdict(recipe),
                "dev_composite": composite, "best_epoch": run["dev_epoch"],
                "split": "image-level (canonical key)", "split_info": info,
                "note": "old/new validation 은 학습·선택에 사용하지 않음",
            },
            label_support=support, minimum_label_examples=MINIMUM_LABEL_EXAMPLES)

    best_seed = max(seed_results, key=lambda s: seed_results[s]["dev_composite"])
    print(f"\ndev composite 최고 seed = {best_seed} "
          f"({seed_results[best_seed]['dev_composite']:.4f})")

    heads = build_attribute_heads(int(fit["features"].shape[1]), HIDDEN_DIM, recipe.dropout)
    for task_name, state in states[best_seed].items():
        heads.heads[task_name].load_state_dict(state)
    heads.eval()
    thresholds = tune_all_thresholds(heads, dev, support, MINIMUM_LABEL_EXAMPLES)
    final_path = MODELS / "fashion_attribute_heads_augmented_r3.pt"
    save_attribute_checkpoint(
        final_path, heads, backbone_model_id=fit["backbone_model_id"], thresholds=thresholds,
        training_summary={
            "round": 3, "selected_seed": best_seed, "selection_rule": "dev composite 최고",
            "recipe": asdict(recipe), "seed_results": seed_results,
            "split": "image-level (canonical key)", "split_info": info,
            "train_crops": len(combined["features"]),
            "note": "old/new validation 은 학습·선택 어디에도 사용하지 않음",
        },
        label_support=support, minimum_label_examples=MINIMUM_LABEL_EXAMPLES)

    after_hash = sha256(BASELINE_R2)
    print(f"\n학습 후 2차 모델 sha256: {after_hash}")
    print(f"2차 모델 미변경: {'PASS' if before_hash == after_hash else 'FAIL'}")

    payload = {
        "round": 3,
        "recipe": asdict(recipe),
        "train_crops": len(combined["features"]),
        "cache_sizes": [len(c["features"]) for c in caches],
        "split": info,
        "split_checks": [{"name": n, "passed": o, "detail": d} for n, o, d in checks],
        "legacy_crop_level_comparison": legacy,
        "seed_results": seed_results,
        "selected_seed": best_seed,
        "thresholds": thresholds,
        "baseline_r2_sha256_before": before_hash,
        "baseline_r2_sha256_after": after_hash,
        "final_checkpoint_sha256": sha256(final_path),
        "seed_checkpoint_sha256": {
            str(s): sha256(MODELS / f"fashion_attribute_heads_augmented_r3_seed{s}.pt")
            for s in args.seeds
        },
        "elapsed_seconds": round(time.time() - began),
    }
    (REPORT_DIR / "17_round3_split_integrity.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (MODELS / "fashion_attribute_heads_augmented_r3.metrics.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n저장: reports/17_round3_split_integrity.json, "
          f"models/fashion_attribute_heads_augmented_r3.metrics.json")
    print(f"총 소요 {(time.time() - began) / 60:.1f}분")


if __name__ == "__main__":
    main()
