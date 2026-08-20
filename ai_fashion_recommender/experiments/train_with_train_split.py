"""7단계 E — Fashionpedia train split 보강분을 합쳐 재학습하고 두 평가셋에서 비교한다.

평가셋 2개
  1) 기존 val 1,195 crop : 지금까지의 모든 비교와 같은 자. "같은 잣대로 좋아졌나"
  2) 새 val 1,716 crop   : train split에서 20% 떼어낸 것. 표본이 없어 측정조차 못했던
                           희소 라벨(니트 val 4장 등)을 처음으로 제대로 측정한다.

대조군은 기존 R2_mixup 실행(구 데이터만 학습)이다. 같은 레시피·같은 시드.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from eval_lib import (
    REPORT_DIR,
    TRAIN_CACHE,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
    split_train_dev,
    subset,
)
from evaluate_relabel import label_support, tune_multi_threshold
from fashion_attribute_model import build_attribute_heads, save_attribute_checkpoint
from fashion_attribute_schema import ATTRIBUTE_TASKS
from finetune import HIDDEN_DIM, RECIPES, train_once

PANTS_TASKS = ("lower_subtype", "pant_leg_shape", "pant_length")
MINIMUM_LABEL_EXAMPLES = 5


def concat_caches(caches):
    return {
        "version": 1,
        "backbone_model_id": caches[0]["backbone_model_id"],
        "features": torch.cat([c["features"] for c in caches], dim=0),
        "targets": {
            name: torch.cat([c["targets"][name] for c in caches], dim=0)
            for name in ATTRIBUTE_TASKS
        },
        "valid": {
            name: torch.cat([c["valid"][name] for c in caches], dim=0)
            for name in ATTRIBUTE_TASKS
        },
        "image_paths": [p for c in caches for p in c["image_paths"]],
    }


def assemble(runs, input_dim, dropout=0.4):
    heads = build_attribute_heads(input_dim, HIDDEN_DIM, dropout)
    picked = {}
    for task_name in ATTRIBUTE_TASKS:
        candidates = [r for r in runs if task_name in r["state"]]
        if not candidates:
            continue
        winner = max(candidates, key=lambda r: r["dev_selection"][task_name])
        heads.heads[task_name].load_state_dict(winner["state"][task_name])
        picked[task_name] = {
            "seed": winner["seed"],
            "dev_epoch": winner["dev_epoch"][task_name],
            "dev_selection": round(winner["dev_selection"][task_name], 4),
        }
    heads.eval()
    return heads, picked


def tune_all_thresholds(heads, dev, support, minimum, device="cpu"):
    """기존 코드와 같은 임계값 정책. 다중분류=micro-F1 최대, 하의 3축=정확도 0.80 목표."""
    logits = head_logits(heads, dev, device)
    thresholds = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task.multi_label:
            thresholds[task_name] = tune_multi_threshold(logits, dev, task_name, support, minimum)
            continue
        if task_name not in PANTS_TASKS:
            thresholds[task_name] = task.minimum_confidence
            continue
        mask = dev["valid"][task_name]
        if not bool(mask.any()):
            thresholds[task_name] = task.minimum_confidence
            continue
        expected = dev["targets"][task_name][mask]
        selected = logits[task_name][mask].float()
        allowed = [
            i for i, label in enumerate(task.labels)
            if int(support.get(task_name, {}).get(label, 0)) >= minimum
        ]
        masked = selected.clone()
        blocked = sorted(set(range(len(task.labels))) - set(allowed))
        if blocked:
            masked[:, blocked] = float("-inf")
        confidence, predicted = masked.softmax(dim=-1).max(dim=-1)
        chosen = task.minimum_confidence
        for threshold in (0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            if threshold < task.minimum_confidence:
                continue
            accepted = confidence >= threshold
            if float(accepted.float().mean()) < 0.20:
                continue
            if float((predicted[accepted] == expected[accepted]).float().mean()) >= 0.80:
                chosen = threshold
                break
        thresholds[task_name] = chosen
    return thresholds


def print_comparison(title, control_metrics, augmented_metrics):
    header = (
        f"{'task':<16}{'kind':<7}{'control':>9}{'augment':>9}{'delta':>8}"
        f"{'cMacro':>8}{'aMacro':>8}{'delta':>8}{'n':>7}"
    )
    print()
    print("=" * len(header))
    print(title)
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = []
    for task_name in ATTRIBUTE_TASKS:
        control = control_metrics.get(task_name, {})
        augmented = augmented_metrics.get(task_name, {})
        if control.get("score") is None or augmented.get("score") is None:
            continue
        rows.append((task_name, control, augmented))
    rows.sort(key=lambda row: -(row[2]["score"] - row[1]["score"]))
    for task_name, control, augmented in rows:
        kind = "multi" if augmented.get("multi_label") else "single"
        control_macro = control.get("macro_f1")
        augmented_macro = augmented.get("macro_f1")
        macro_delta = (
            augmented_macro - control_macro
            if control_macro is not None and augmented_macro is not None else None
        )
        print(
            f"{task_name:<16}{kind:<7}{control['score']:>9.3f}{augmented['score']:>9.3f}"
            f"{augmented['score'] - control['score']:>+8.3f}"
            f"{(f'{control_macro:.3f}' if control_macro is not None else '-'):>8}"
            f"{(f'{augmented_macro:.3f}' if augmented_macro is not None else '-'):>8}"
            f"{(f'{macro_delta:+.3f}' if macro_delta is not None else '-'):>8}"
            f"{augmented['samples']:>7}"
        )
    print(f"\n  control  : {overall_summary(control_metrics)}")
    print(f"  augmented: {overall_summary(augmented_metrics)}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--new-train-cache", required=True, nargs="+")
    parser.add_argument("--new-val-cache", required=True)
    parser.add_argument("--control-runs", required=True, help="기존 R2_mixup.pt 경로")
    parser.add_argument("--runs-output", required=True)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--threads", type=int, default=6)
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    device = "cpu"
    minimum = MINIMUM_LABEL_EXAMPLES

    old_train = load_cache(TRAIN_CACHE)
    new_trains = [load_cache(path) for path in args.new_train_cache]
    old_val = load_cache(VAL_CACHE)
    new_val = load_cache(args.new_val_cache)
    combined = concat_caches([old_train, *new_trains])
    sizes = " + ".join(f"{len(c['features']):,}" for c in new_trains)
    print(
        f"train: 기존 {len(old_train['features']):,} + 신규 {sizes}"
        f" = {len(combined['features']):,}",
        flush=True,
    )
    print(
        f"평가: 기존 val {len(old_val['features']):,} / 새 val {len(new_val['features']):,}",
        flush=True,
    )

    fit_indices, dev_indices = split_train_dev(combined)
    fit = subset(combined, fit_indices)
    dev = subset(combined, dev_indices)
    print(f"fit {len(fit['features']):,} / dev {len(dev['features']):,}", flush=True)

    recipe = next(r for r in RECIPES if r.name == "R2_mixup")
    runs_path = Path(args.runs_output)
    if runs_path.is_file():
        print("기존 학습 결과 재사용", flush=True)
        runs = torch.load(runs_path, map_location="cpu", weights_only=False)["runs"]
    else:
        runs = []
        for seed in range(args.seeds):
            started = time.time()
            run = train_once(recipe, seed, fit, dev, device)
            mean_selection = sum(run["dev_selection"].values()) / len(run["dev_selection"])
            print(
                f"  seed={seed} dev_mean={mean_selection:.4f} {time.time() - started:.0f}s",
                flush=True,
            )
            runs.append(run)
        runs_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(recipe), "runs": runs}, runs_path)

    input_dim = int(fit["features"].shape[1])
    heads, picked = assemble(runs, input_dim)
    support = label_support(fit)
    thresholds = tune_all_thresholds(heads, dev, support, minimum)

    control_runs = torch.load(args.control_runs, map_location="cpu", weights_only=False)["runs"]
    old_fit_indices, old_dev_indices = split_train_dev(old_train)
    old_fit = subset(old_train, old_fit_indices)
    old_dev = subset(old_train, old_dev_indices)
    control_heads, _ = assemble(control_runs, input_dim)
    control_support = label_support(old_fit)
    control_thresholds = tune_all_thresholds(control_heads, old_dev, control_support, minimum)

    results = {}
    for eval_name, cache in (("old_val", old_val), ("new_val", new_val)):
        results[eval_name] = {
            "control": evaluate_logits(
                head_logits(control_heads, cache, device), cache, control_thresholds,
                label_support=control_support, minimum_label_examples=minimum,
            ),
            "augmented": evaluate_logits(
                head_logits(heads, cache, device), cache, thresholds,
                label_support=support, minimum_label_examples=minimum,
            ),
        }

    print_comparison(
        "기존 val 1,195 crop — 같은 잣대 비교",
        results["old_val"]["control"], results["old_val"]["augmented"],
    )
    print_comparison(
        "새 val 1,716 crop — 희소 라벨 측정용 (Fashionpedia train split의 20%)",
        results["new_val"]["control"], results["new_val"]["augmented"],
    )

    print("\n" + "=" * 74)
    print("희소 라벨 회복 — 새 val 기준 (기존 train 표본 30장 미만이던 라벨)")
    print("=" * 74)
    print(f"{'task':<15}{'label':<14}{'old n':>7}{'new n':>7}{'val n':>7}{'ctrl':>8}{'aug':>8}")
    print("-" * 74)
    for task_name in ATTRIBUTE_TASKS:
        control = results["new_val"]["control"].get(task_name, {})
        augmented = results["new_val"]["augmented"].get(task_name, {})
        if not augmented.get("per_label"):
            continue
        for label, stats in augmented["per_label"].items():
            old_count = int(control_support.get(task_name, {}).get(label, 0))
            if old_count >= 30 or stats["support"] < 3:
                continue
            new_count = int(support.get(task_name, {}).get(label, 0))
            control_stats = (control.get("per_label") or {}).get(label, {})
            control_f1 = control_stats.get("f1")
            new_f1 = stats.get("f1")
            control_text = (
                "blocked" if control_stats.get("blocked")
                else (f"{control_f1:.3f}" if control_f1 is not None else "-")
            )
            print(
                f"{task_name:<15}{label:<14}{old_count:>7}{new_count:>7}{stats['support']:>7}"
                f"{control_text:>8}{(f'{new_f1:.3f}' if new_f1 is not None else '-'):>8}"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "11_train_split_results.json"
    output.write_text(
        json.dumps(
            {
                "sizes": {
                    "old_train": len(old_train["features"]),
                    "new_train": [len(c["features"]) for c in new_trains],
                    "combined": len(combined["features"]),
                    "fit": len(fit["features"]),
                    "dev": len(dev["features"]),
                    "old_val": len(old_val["features"]),
                    "new_val": len(new_val["features"]),
                },
                "picked": picked,
                "thresholds": thresholds,
                "label_support_fit": support,
                "label_support_control": control_support,
                "results": results,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    save_attribute_checkpoint(
        REPORT_DIR.parent / "models" / "fashion_attribute_heads_augmented.pt",
        heads,
        backbone_model_id=fit["backbone_model_id"],
        thresholds=thresholds,
        training_summary={
            "note": "Fashionpedia train split 보강(이미지 4,500장 / crop 8,513개) 후 재학습",
            "picked": picked,
            "metrics": results,
        },
        label_support=support,
        minimum_label_examples=minimum,
    )
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
