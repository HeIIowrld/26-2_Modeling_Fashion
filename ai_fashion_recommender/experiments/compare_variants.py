"""3단계(대조) — 누출 없는 동일 조건에서 "레시피 효과"만 분리한다.

문제
- 배포된 기준선 체크포인트는 val로 태스크별 early stopping을 했다. 그래서 val 점수가
  낙관 편향돼 있고, 내 모델(val 미사용)과 직접 비교하면 기준선에 유리하게 기울어진다.

해결
- 기존 레시피(R0_replica)를 내 프로토콜(train_dev로만 선택 → 전체 train refit)로 다시 학습해
  진짜 대조군을 만든다. 두 모델 모두 val을 한 번도 보지 않았으므로 차이는 레시피 효과다.

비교 대상
- BASE_SHIPPED : 배포된 models/fashion_attribute_heads.pt (val 누출 있음, 참고용)
- V0_oldrecipe : R0_replica를 전 태스크에 적용 + 전체 train refit  ← 진짜 대조군
- V1_bestglobal: dev 전체 평균 1위 단일 레시피를 전 태스크에 적용 + refit
- V2_pertask   : 태스크별 선택(refit_full.py와 동일) + refit
"""

from __future__ import annotations

import json
import statistics

import torch

from assemble_and_evaluate import label_support, tune_thresholds
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
from finetune import RECIPES, Recipe
from refit_full import MINIMUM_LABEL_EXAMPLES, load_runs, select, train_full

RECIPE_BY_NAME = {recipe.name: recipe for recipe in RECIPES}


def median_epochs(runs, recipe_name: str) -> dict[str, int]:
    epochs: dict[str, list[int]] = {}
    for run in runs:
        if run["recipe"] != recipe_name:
            continue
        for task_name, epoch in run["dev_epoch"].items():
            epochs.setdefault(task_name, []).append(epoch)
    return {
        name: max(1, int(round(statistics.median(values))))
        for name, values in epochs.items()
    }


def best_global_recipe(runs, configs, hidden_dim: int) -> str:
    """해당 hidden_dim 안에서 dev 전 태스크 평균이 가장 높은 단일 레시피."""
    scores: dict[str, float] = {}
    for name, config in configs.items():
        if int(config["hidden_dim"]) != hidden_dim:
            continue
        group = [run for run in runs if run["recipe"] == name]
        if not group:
            continue
        per_task = [
            sum(run["dev_selection"][task] for run in group if task in run["dev_selection"])
            / max(1, sum(1 for run in group if task in run["dev_selection"]))
            for task in ATTRIBUTE_TASKS
        ]
        scores[name] = sum(per_task) / len(per_task)
    return max(scores, key=scores.get)


def best_fit_state(runs, recipe_name: str, task_name: str):
    pool = [r for r in runs if r["recipe"] == recipe_name and task_name in r["state"]]
    if not pool:
        return None
    return max(pool, key=lambda r: r["dev_selection"][task_name])["state"][task_name]


def build_variant(name, plan, runs, fit, dev, train_cache, device="cpu"):
    """plan: {task -> (recipe_name, target_epoch)}

    1) train_fit 모델을 조립해 dev에서 임계값을 뽑는다(표본 외 선택).
    2) 같은 설정으로 전체 train에 refit 한다.
    """
    hidden = {RECIPE_BY_NAME[recipe].hidden_dim for recipe, _ in plan.values()}
    if len(hidden) != 1:
        raise ValueError(f"{name}: hidden_dim이 섞였습니다 {hidden}")
    hidden_dim = hidden.pop()

    fit_heads = build_attribute_heads(int(fit["features"].shape[1]), hidden_dim, 0.4).to(device)
    for task_name, (recipe_name, _) in plan.items():
        state = best_fit_state(runs, recipe_name, task_name)
        if state is not None:
            fit_heads.heads[task_name].load_state_dict(state)
    fit_heads.eval()
    fit_support = label_support(fit)
    thresholds = tune_thresholds(
        head_logits(fit_heads, dev, device), dev, support=fit_support, minimum=MINIMUM_LABEL_EXAMPLES
    )

    by_recipe: dict[str, dict[int, list[str]]] = {}
    for task_name, (recipe_name, epoch) in plan.items():
        by_recipe.setdefault(recipe_name, {}).setdefault(epoch, []).append(task_name)
    heads = build_attribute_heads(int(train_cache["features"].shape[1]), hidden_dim, 0.4).to(device)
    for recipe_name, targets_by_epoch in by_recipe.items():
        snapshots = train_full(RECIPE_BY_NAME[recipe_name], train_cache, targets_by_epoch, device=device)
        for task_name, state in snapshots.items():
            heads.heads[task_name].load_state_dict(state)
    heads.eval()
    print(f"  built {name:<16} hidden={hidden_dim} recipes={sorted(by_recipe)}", flush=True)
    return heads, thresholds, label_support(train_cache), hidden_dim


def main() -> None:
    torch.set_num_threads(8)
    device = "cpu"
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(train_cache)
    fit = subset(train_cache, fit_indices)
    dev = subset(train_cache, dev_indices)

    runs, configs = load_runs()
    hidden, selection = select(runs, configs, dev)
    global_best = best_global_recipe(runs, configs, hidden)
    print(f"hidden_dim={hidden}  dev 전체 평균 1위 단일 레시피 = {global_best}\n")

    plans = {
        "V0_oldrecipe": {
            task: ("R0_replica", median_epochs(runs, "R0_replica").get(task, 1))
            for task in ATTRIBUTE_TASKS
        },
        "V1_bestglobal": {
            task: (global_best, median_epochs(runs, global_best).get(task, 1))
            for task in ATTRIBUTE_TASKS
        },
        "V2_pertask": {
            task: (value["recipe"], value["target_epoch"])
            for task, value in selection.items()
        },
    }

    results: dict[str, dict] = {}
    baseline_heads, baseline_payload = load_attribute_heads(BASELINE_CHECKPOINT, device)
    results["BASE_SHIPPED"] = evaluate_logits(
        head_logits(baseline_heads, val_cache, device), val_cache,
        baseline_payload.get("thresholds", {}),
        label_support=baseline_payload.get("label_support", {}),
        minimum_label_examples=int(baseline_payload.get("minimum_label_examples", 5)),
    )
    artifacts = {}
    for name, plan in plans.items():
        heads, thresholds, support, hidden_dim = build_variant(
            name, plan, runs, fit, dev, train_cache, device
        )
        results[name] = evaluate_logits(
            head_logits(heads, val_cache, device), val_cache, thresholds,
            label_support=support, minimum_label_examples=MINIMUM_LABEL_EXAMPLES,
        )
        artifacts[name] = (heads, thresholds, support, hidden_dim)

    names = list(results)
    print()
    header = f"{'task':<16}" + "".join(f"{n.split('_')[0]:>14}" for n in names) + f"{'n':>7}"
    print("=" * len(header))
    print("val 1,195 crop — 대표 지표 (single=accuracy, multi=micro-F1)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for task in ATTRIBUTE_TASKS:
        if results["BASE_SHIPPED"].get(task, {}).get("score") is None:
            continue
        line = f"{task:<16}"
        for n in names:
            line += f"{results[n][task]['score']:>14.3f}"
        print(line + f"{results['BASE_SHIPPED'][task]['samples']:>7}")

    print()
    header = f"{'task':<16}" + "".join(f"{n.split('_')[0]:>14}" for n in names) + f"{'n':>7}"
    print("=" * len(header))
    print("val 1,195 crop — macro-F1 (소수 라벨 붕괴가 드러나는 지표)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    for task in ATTRIBUTE_TASKS:
        if results["BASE_SHIPPED"].get(task, {}).get("macro_f1") is None:
            continue
        line = f"{task:<16}"
        for n in names:
            value = results[n][task].get("macro_f1")
            line += f"{(f'{value:.3f}' if value is not None else '-'):>14}"
        print(line + f"{results['BASE_SHIPPED'][task]['samples']:>7}")

    print()
    print("=" * 96)
    print("전체 요약")
    print("=" * 96)
    overall = {name: overall_summary(value) for name, value in results.items()}
    print(f"{'variant':<16}{'mean_score':>12}{'weighted':>12}{'mean_macroF1':>14}   설명")
    notes = {
        "BASE_SHIPPED": "배포 체크포인트 (val로 early stop → val 점수 낙관 편향)",
        "V0_oldrecipe": "기존 레시피 + 정직한 프로토콜  ← 진짜 대조군",
        "V1_bestglobal": f"{global_best} 전 태스크 + 정직한 프로토콜",
        "V2_pertask": "태스크별 선택 + 정직한 프로토콜",
    }
    for name in names:
        value = overall[name]
        print(
            f"{name:<16}{value['mean_score']:>12.4f}{value['sample_weighted_score']:>12.4f}"
            f"{value['mean_macro_f1']:>14.4f}   {notes[name]}"
        )

    print()
    print("레시피 효과만 분리 (V2 − V0, 둘 다 val 미사용)")
    print(f"  mean_score   {overall['V2_pertask']['mean_score'] - overall['V0_oldrecipe']['mean_score']:+.4f}")
    print(f"  weighted     {overall['V2_pertask']['sample_weighted_score'] - overall['V0_oldrecipe']['sample_weighted_score']:+.4f}")
    print(f"  mean_macroF1 {overall['V2_pertask']['mean_macro_f1'] - overall['V0_oldrecipe']['mean_macro_f1']:+.4f}")
    gains = []
    for task in ATTRIBUTE_TASKS:
        if results["V0_oldrecipe"].get(task, {}).get("score") is None:
            continue
        gains.append((
            task,
            results["V2_pertask"][task]["score"] - results["V0_oldrecipe"][task]["score"],
            (results["V2_pertask"][task].get("macro_f1") or 0) - (results["V0_oldrecipe"][task].get("macro_f1") or 0),
        ))
    gains.sort(key=lambda row: -row[1])
    for task, delta, macro_delta in gains:
        print(f"    {task:<16}{delta:>+8.3f} (macroF1 {macro_delta:+.3f})")

    # 최종 채택: mean_score / macro-F1 을 함께 보고 V2를 저장한다(선택은 dev에서 끝났다).
    heads, thresholds, support, hidden_dim = artifacts["V2_pertask"]
    output = BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_finetuned.pt"
    save_attribute_checkpoint(
        output,
        heads,
        backbone_model_id=train_cache["backbone_model_id"],
        thresholds=thresholds,
        training_summary={
            "note": (
                "설정(레시피·hidden_dim·epoch·임계값)은 train_dev(718)에서만 선택하고 "
                "전체 train(4789)으로 refit. val(1195)은 평가 전용."
            ),
            "hidden_dim": hidden_dim,
            "selection": selection,
            "recipes": configs,
            "metrics": results["V2_pertask"],
            "baseline_metrics": results["BASE_SHIPPED"],
            "control_metrics": results["V0_oldrecipe"],
            "overall": overall,
        },
        label_support=support,
        minimum_label_examples=MINIMUM_LABEL_EXAMPLES,
    )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "06_variant_comparison.json").write_text(
        json.dumps(
            {
                "hidden_dim": hidden,
                "global_best_recipe": global_best,
                "plans": {
                    name: {task: list(value) for task, value in plan.items()}
                    for name, plan in plans.items()
                },
                "selection": selection,
                "overall": overall,
                "per_task": results,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nsaved: {output}")
    print(f"saved: {REPORT_DIR / '06_variant_comparison.json'}")


if __name__ == "__main__":
    main()
