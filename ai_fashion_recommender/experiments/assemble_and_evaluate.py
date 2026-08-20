"""3단계 — 태스크별 최적 레시피를 조립하고 val에서 기준선과 비교한다.

정직성 규칙
- 레시피/시드/epoch/임계값 선택은 전부 train_dev(718)에서만 한다.
- val(1,195)은 이 스크립트의 마지막 평가에서 단 한 번만 쓴다.
- 임계값 탐색 목표는 기존 코드와 동일하게 둔다(다중분류=micro-F1 최대,
  하의 3축=정확도 0.80 목표, 나머지=스키마 최소 신뢰도). 개선분이 학습 레시피에서만
  나오도록 하기 위한 조건이다.

출력
- reports/03_finetune_selection.json
- reports/04_comparison.json
- models/fashion_attribute_heads_finetuned.pt   (기존 체크포인트는 건드리지 않는다)
"""

from __future__ import annotations

import json

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
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
from fashion_attribute_model import build_attribute_heads, load_attribute_heads, save_attribute_checkpoint
from fashion_attribute_schema import ATTRIBUTE_TASKS
from finetune import HIDDEN_DIM, MULTI_THRESHOLD_GRID, RUN_DIR

PANTS_TASKS = ("lower_subtype", "pant_leg_shape", "pant_length")
OUTPUT_CHECKPOINT = BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_finetuned.pt"


def label_support(cache) -> dict[str, dict[str, int]]:
    support = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        target = cache["targets"][task_name][mask]
        if task.multi_label:
            counts = target.float().sum(dim=0) if len(target) else torch.zeros(len(task.labels))
        else:
            counts = (
                torch.bincount(target.long(), minlength=len(task.labels))
                if len(target) else torch.zeros(len(task.labels), dtype=torch.long)
            )
        support[task_name] = {label: int(counts[i]) for i, label in enumerate(task.labels)}
    return support


def tune_thresholds(logits, cache, *, support, minimum) -> dict[str, float]:
    """기존 fashion_attribute_training.py 의 임계값 정책을 dev 셋에서 그대로 적용."""
    thresholds: dict[str, float] = {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        mask = cache["valid"][task_name]
        if not bool(mask.any()):
            thresholds[task_name] = task.minimum_confidence
            continue
        expected = cache["targets"][task_name][mask]
        selected = logits[task_name][mask].float()
        allowed = [
            i for i, label in enumerate(task.labels)
            if int(support.get(task_name, {}).get(label, 0)) >= minimum
        ]
        if task.multi_label:
            probabilities = selected.sigmoid()
            blocked = sorted(set(range(len(task.labels))) - set(allowed))
            if blocked:
                probabilities[:, blocked] = 0.0
            expected_bool = expected.bool()
            best, best_f1 = task.minimum_confidence, -1.0
            for threshold in MULTI_THRESHOLD_GRID:
                predicted = probabilities >= threshold
                tp = int((predicted & expected_bool).sum())
                fp = int((predicted & ~expected_bool).sum())
                fn = int((~predicted & expected_bool).sum())
                denominator = 2 * tp + fp + fn
                f1 = 2 * tp / denominator if denominator else 0.0
                if f1 > best_f1:
                    best, best_f1 = threshold, f1
            thresholds[task_name] = best
            continue
        if task_name not in PANTS_TASKS:
            thresholds[task_name] = task.minimum_confidence
            continue
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


def main() -> None:
    torch.set_num_threads(4)
    device = "cpu"
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(train_cache)
    fit = subset(train_cache, fit_indices)
    dev = subset(train_cache, dev_indices)

    run_files = sorted(RUN_DIR.glob("*.pt"))
    if not run_files:
        raise SystemExit(f"학습 결과가 없습니다: {RUN_DIR}")
    runs = []
    configs = {}
    for path in run_files:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        configs[path.stem] = payload["config"]
        runs.extend(payload["runs"])
    print(f"loaded {len(runs)} runs from {len(run_files)} recipes: {[p.stem for p in run_files]}")

    # ---- 태스크별 선택: 레시피는 시드 평균으로, 시드는 그 안에서 최고 dev 점수로
    selection: dict[str, dict] = {}
    for task_name in ATTRIBUTE_TASKS:
        candidates = [run for run in runs if task_name in run["state"]]
        if not candidates:
            continue
        by_recipe: dict[str, list] = {}
        for run in candidates:
            by_recipe.setdefault(run["recipe"], []).append(run)
        recipe_means = {
            name: sum(r["dev_selection"][task_name] for r in group) / len(group)
            for name, group in by_recipe.items()
        }
        winner_recipe = max(recipe_means, key=recipe_means.get)
        winner = max(by_recipe[winner_recipe], key=lambda r: r["dev_selection"][task_name])
        selection[task_name] = {
            "recipe": winner_recipe,
            "seed": winner["seed"],
            "dev_epoch": winner["dev_epoch"][task_name],
            "dev_selection": round(winner["dev_selection"][task_name], 4),
            "recipe_means": {k: round(v, 4) for k, v in sorted(recipe_means.items())},
            "state": winner["state"][task_name],
        }

    heads = build_attribute_heads(int(fit["features"].shape[1]), HIDDEN_DIM, 0.5).to(device)
    for task_name, value in selection.items():
        heads.heads[task_name].load_state_dict(value["state"])
    heads.eval()

    support = label_support(fit)
    minimum = 5
    dev_logits = head_logits(heads, dev, device)
    thresholds = tune_thresholds(dev_logits, dev, support=support, minimum=minimum)

    # ---- 기준선
    baseline_heads, baseline_payload = load_attribute_heads(BASELINE_CHECKPOINT, device)
    baseline_thresholds = baseline_payload.get("thresholds", {})
    baseline_support = baseline_payload.get("label_support", {})
    baseline_minimum = int(baseline_payload.get("minimum_label_examples", 5))

    baseline_val = evaluate_logits(
        head_logits(baseline_heads, val_cache, device), val_cache, baseline_thresholds,
        label_support=baseline_support, minimum_label_examples=baseline_minimum,
    )
    tuned_val = evaluate_logits(
        head_logits(heads, val_cache, device), val_cache, thresholds,
        label_support=support, minimum_label_examples=minimum,
    )
    tuned_dev = evaluate_logits(
        dev_logits, dev, thresholds, label_support=support, minimum_label_examples=minimum
    )
    tuned_fit = evaluate_logits(
        head_logits(heads, fit, device), fit, thresholds,
        label_support=support, minimum_label_examples=minimum,
    )

    print()
    header = (
        f"{'task':<16}{'kind':<7}{'base':>8}{'tuned':>8}{'Δ':>8}"
        f"{'baseMacro':>11}{'newMacro':>10}{'Δ':>8}{'cov':>7}{'Δcov':>8}{'n':>7}"
    )
    print("=" * len(header))
    print("val 1,195 crop — 기준선 vs 미세조정 (개선폭 큰 순)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = []
    for task_name in ATTRIBUTE_TASKS:
        base = baseline_val.get(task_name, {})
        new = tuned_val.get(task_name, {})
        if base.get("score") is None or new.get("score") is None:
            continue
        rows.append((task_name, base, new))
    rows.sort(key=lambda row: -(row[2]["score"] - row[1]["score"]))
    for task_name, base, new in rows:
        kind = "multi" if new.get("multi_label") else "single"
        delta = new["score"] - base["score"]
        base_macro, new_macro = base.get("macro_f1"), new.get("macro_f1")
        macro_delta = (new_macro - base_macro) if (base_macro is not None and new_macro is not None) else None
        base_cov, new_cov = base.get("accepted_coverage"), new.get("accepted_coverage")
        cov_delta = (new_cov - base_cov) if (base_cov is not None and new_cov is not None) else None
        print(
            f"{task_name:<16}{kind:<7}{base['score']:>8.3f}{new['score']:>8.3f}{delta:>+8.3f}"
            f"{(f'{base_macro:.3f}' if base_macro is not None else '-'):>11}"
            f"{(f'{new_macro:.3f}' if new_macro is not None else '-'):>10}"
            f"{(f'{macro_delta:+.3f}' if macro_delta is not None else '-'):>8}"
            f"{(f'{new_cov:.3f}' if new_cov is not None else '-'):>7}"
            f"{(f'{cov_delta:+.3f}' if cov_delta is not None else '-'):>8}"
            f"{new['samples']:>7}"
        )

    base_overall = overall_summary(baseline_val)
    new_overall = overall_summary(tuned_val)
    print()
    print(f"{'baseline val':<16}", base_overall)
    print(f"{'finetuned val':<16}", new_overall)
    print(f"{'finetuned dev':<16}", overall_summary(tuned_dev))
    print(f"{'finetuned fit':<16}", overall_summary(tuned_fit))

    print()
    print("붕괴 라벨 회복 (val support >= 3, 기준선 F1 < 0.40)")
    for task_name, base, new in rows:
        for label, stats in (base.get("per_label") or {}).items():
            if stats["support"] < 3:
                continue
            base_f1 = stats.get("f1")
            if base_f1 is not None and base_f1 >= 0.40:
                continue
            new_stats = (new.get("per_label") or {}).get(label, {})
            new_f1 = new_stats.get("f1")
            print(
                f"  {task_name:<16}{label:<14} n={stats['support']:>3}"
                f"  {(f'{base_f1:.3f}' if base_f1 is not None else 'None'):>6}"
                f" -> {(f'{new_f1:.3f}' if new_f1 is not None else 'None'):>6}"
                f"{'  [blocked]' if new_stats.get('blocked') else ''}"
            )

    print()
    print("태스크별 선택된 레시피")
    for task_name, value in selection.items():
        print(
            f"  {task_name:<16}{value['recipe']:<14} seed={value['seed']} epoch={value['dev_epoch']:<4}"
            f" dev_sel={value['dev_selection']:.4f}   " +
            " ".join(f"{k.split('_')[0]}={v:.3f}" for k, v in value["recipe_means"].items())
        )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "03_finetune_selection.json").write_text(
        json.dumps(
            {
                "recipes": configs,
                "selection": {
                    name: {k: v for k, v in value.items() if k != "state"}
                    for name, value in selection.items()
                },
                "thresholds": thresholds,
                "label_support_train_fit": support,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    (REPORT_DIR / "04_comparison.json").write_text(
        json.dumps(
            {
                "overall": {
                    "baseline_val": base_overall,
                    "finetuned_val": new_overall,
                    "finetuned_train_dev": overall_summary(tuned_dev),
                    "finetuned_train_fit": overall_summary(tuned_fit),
                },
                "baseline_val": baseline_val,
                "finetuned_val": tuned_val,
                "finetuned_train_dev": tuned_dev,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    save_attribute_checkpoint(
        OUTPUT_CHECKPOINT,
        heads,
        backbone_model_id=fit["backbone_model_id"],
        thresholds=thresholds,
        training_summary={
            "note": "train_fit(4071)만 학습, train_dev(718)에서 레시피·epoch·임계값 선택, val(1195)은 평가 전용",
            "recipes": configs,
            "selection": {
                name: {k: v for k, v in value.items() if k != "state"}
                for name, value in selection.items()
            },
            "metrics": tuned_val,
            "baseline_metrics": baseline_val,
            "overall": {"baseline_val": base_overall, "finetuned_val": new_overall},
        },
        label_support=support,
        minimum_label_examples=minimum,
    )
    print(f"\nsaved checkpoint: {OUTPUT_CHECKPOINT}")
    print(f"saved reports   : {REPORT_DIR / '03_finetune_selection.json'}, {REPORT_DIR / '04_comparison.json'}")


if __name__ == "__main__":
    main()
