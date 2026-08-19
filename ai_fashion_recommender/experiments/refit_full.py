"""3단계(본선) — dev에서 고른 설정으로 전체 train(4,789)에 다시 학습하고 val에서 비교한다.

왜 refit이 필요한가
- 1차 조립 모델은 train_fit(4,071)만 봤다. 기준선은 4,789를 다 봤다.
  같은 데이터량으로 맞추지 않으면 "레시피 효과"와 "데이터 15% 손실"이 섞인다.
- 그래서 설정 선택은 train_dev에서 끝내고, 그 설정으로 전체 train에 다시 학습한다.
  하이퍼파라미터를 held-out에서 고른 뒤 전체 데이터로 refit 하는 표준 절차다.

호환성 제약
- build_attribute_heads()는 17개 헤드에 같은 hidden_dim을 쓴다.
  → hidden_dim은 먼저 전역으로 하나 고르고, 태스크별 레시피 선택은 그 안에서만 한다.
- hidden_dim은 체크포인트에 저장·복원되므로 256이 아니어도 기존 로더와 호환된다.

출력
- reports/05_refit_comparison.json
- models/fashion_attribute_heads_finetuned.pt
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict

import torch

from assemble_and_evaluate import OUTPUT_CHECKPOINT, label_support, tune_thresholds
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
from finetune import RECIPES, RUN_DIR, Recipe, class_weights, positive_weights, batch_loss

MINIMUM_LABEL_EXAMPLES = 5
# dev 표본이 작은 태스크에서 우연한 1등을 그대로 믿지 않기 위한 여유.
# 표본 n의 점수 표준오차가 대략 0.5/sqrt(n)이므로 그 30% 수준을 요구한다.
MARGIN_SCALE = 0.15


def load_runs() -> tuple[list[dict], dict[str, dict]]:
    runs: list[dict] = []
    configs: dict[str, dict] = {}
    for path in sorted(RUN_DIR.glob("*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        config = dict(payload["config"])
        # 1라운드 실행분은 hidden_dim 필드가 추가되기 전에 저장됐다. 당시 값은 256이다.
        config.setdefault("hidden_dim", 256)
        configs[path.stem] = config
        runs.extend(payload["runs"])
    return runs, configs


def dev_support(dev, task_name: str) -> int:
    return int(dev["valid"][task_name].sum())


def select(runs, configs, dev) -> tuple[int, dict[str, dict]]:
    """hidden_dim을 전역으로 먼저 고르고, 그 안에서 태스크별 레시피를 고른다."""
    means: dict[str, dict[str, float]] = {}
    for task_name in ATTRIBUTE_TASKS:
        by_recipe: dict[str, list[float]] = {}
        for run in runs:
            if task_name in run["dev_selection"]:
                by_recipe.setdefault(run["recipe"], []).append(run["dev_selection"][task_name])
        means[task_name] = {name: sum(v) / len(v) for name, v in by_recipe.items()}

    hidden_dims = sorted({int(config["hidden_dim"]) for config in configs.values()})
    group_score: dict[int, float] = {}
    group_best: dict[int, str] = {}
    for hidden in hidden_dims:
        names = [name for name, config in configs.items() if int(config["hidden_dim"]) == hidden]
        # 그룹 대표 = 전 태스크 평균이 가장 높은 단일 레시피 (태스크별 최대값을 쓰면 낙관 편향)
        candidate_means = {
            name: sum(means[t].get(name, 0.0) for t in ATTRIBUTE_TASKS) / len(ATTRIBUTE_TASKS)
            for name in names
        }
        best_name = max(candidate_means, key=candidate_means.get)
        group_score[hidden] = candidate_means[best_name]
        group_best[hidden] = best_name
    winning_hidden = max(group_score, key=group_score.get)
    anchor = group_best[winning_hidden]
    print("hidden_dim 그룹별 최고 단일 레시피 (dev 전 태스크 평균)")
    for hidden in hidden_dims:
        mark = " <=" if hidden == winning_hidden else ""
        print(f"  h={hidden:<4} {group_best[hidden]:<18} {group_score[hidden]:.4f}{mark}")

    allowed = [name for name, config in configs.items() if int(config["hidden_dim"]) == winning_hidden]
    selection: dict[str, dict] = {}
    for task_name in ATTRIBUTE_TASKS:
        scores = {name: means[task_name][name] for name in allowed if name in means[task_name]}
        if not scores:
            continue
        anchor_score = scores.get(anchor, max(scores.values()))
        margin = MARGIN_SCALE / max(dev_support(dev, task_name), 1) ** 0.5
        best_name = max(scores, key=scores.get)
        chosen = best_name if scores[best_name] > anchor_score + margin else anchor
        epochs = [
            run["dev_epoch"][task_name]
            for run in runs
            if run["recipe"] == chosen and task_name in run["dev_epoch"]
        ]
        selection[task_name] = {
            "recipe": chosen,
            "target_epoch": int(round(statistics.median(epochs))) if epochs else 1,
            "dev_mean": round(scores[chosen], 4),
            "dev_best_alternative": best_name,
            "dev_best_alternative_mean": round(scores[best_name], 4),
            "margin_required": round(margin, 4),
            "dev_support": dev_support(dev, task_name),
            "recipe_means": {k: round(v, 4) for k, v in sorted(scores.items())},
        }
    return winning_hidden, selection


def train_full(recipe: Recipe, cache, targets_by_epoch: dict[int, list[str]], *, seed: int = 0, device="cpu"):
    """전체 train으로 학습하면서 태스크별 지정 epoch에서 헤드를 스냅샷한다."""
    torch.manual_seed(seed)
    heads = build_attribute_heads(int(cache["features"].shape[1]), recipe.hidden_dim, recipe.dropout).to(device)
    optimizer = torch.optim.AdamW(heads.parameters(), lr=recipe.learning_rate, weight_decay=recipe.weight_decay)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=recipe.epochs)
        if recipe.cosine_schedule else None
    )
    weights = class_weights(cache, recipe.class_weight_power, device)
    pos_weights = positive_weights(cache, recipe.class_weight_power, device)
    features_all = cache["features"].to(device)
    targets_all = {name: value.to(device) for name, value in cache["targets"].items()}
    valid_all = {name: value.to(device) for name, value in cache["valid"].items()}
    size = len(features_all)
    last_epoch = max(targets_by_epoch)
    snapshots: dict[str, dict] = {}

    for epoch in range(1, last_epoch + 1):
        heads.train()
        permutation = torch.randperm(size)
        for start in range(0, size, recipe.batch_size):
            indices = permutation[start:start + recipe.batch_size]
            features = features_all[indices]
            targets = {name: value[indices] for name, value in targets_all.items()}
            valid = {name: value[indices] for name, value in valid_all.items()}
            mix = None
            if recipe.noise_sigma > 0:
                features = torch.nn.functional.normalize(
                    features + torch.randn_like(features) * recipe.noise_sigma, dim=-1
                )
            if recipe.mixup_alpha > 0:
                lam = float(torch.distributions.Beta(recipe.mixup_alpha, recipe.mixup_alpha).sample())
                lam = max(lam, 1 - lam)
                partner = indices[torch.randperm(len(indices))]
                features = torch.nn.functional.normalize(
                    lam * features + (1 - lam) * features_all[partner], dim=-1
                )
                mix = (
                    {name: value[partner] for name, value in targets_all.items()},
                    {name: value[partner] for name, value in valid_all.items()},
                    lam,
                )
            optimizer.zero_grad(set_to_none=True)
            loss = batch_loss(heads, features, targets, valid, recipe, weights, pos_weights, mix=mix)
            if loss is None:
                continue
            loss.backward()
            optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for task_name in targets_by_epoch.get(epoch, []):
            heads.eval()
            snapshots[task_name] = {
                key: tensor.detach().cpu().clone()
                for key, tensor in heads.heads[task_name].state_dict().items()
            }
    return snapshots


def main() -> None:
    torch.set_num_threads(8)
    device = "cpu"
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(train_cache)
    fit = subset(train_cache, fit_indices)
    dev = subset(train_cache, dev_indices)

    runs, configs = load_runs()
    print(f"{len(runs)} runs / {len(configs)} recipes: {sorted(configs)}\n")
    hidden, selection = select(runs, configs, dev)

    print(f"\n선택된 hidden_dim = {hidden}")
    print("태스크별 최종 설정 (dev 기준, 표본 작은 태스크는 여유 미달 시 대표 레시피 유지)")
    for task_name, value in selection.items():
        note = "" if value["recipe"] == value["dev_best_alternative"] else (
            f"  (1등 {value['dev_best_alternative']} {value['dev_best_alternative_mean']:.3f}"
            f" 여유 {value['margin_required']:.3f} 미달 → 대표 유지)"
        )
        print(
            f"  {task_name:<16}{value['recipe']:<18} epoch={value['target_epoch']:<4}"
            f" dev={value['dev_mean']:.4f} devN={value['dev_support']:<5}{note}"
        )

    # ---- 1) train_fit 조립 모델(설정 검증용) — dev에서 임계값을 뽑는 데 사용
    recipes = {r.name: r for r in RECIPES}
    fit_heads = build_attribute_heads(int(fit["features"].shape[1]), hidden, 0.4).to(device)
    for task_name, value in selection.items():
        pool = [
            run for run in runs
            if run["recipe"] == value["recipe"] and task_name in run["state"]
        ]
        winner = max(pool, key=lambda run: run["dev_selection"][task_name])
        fit_heads.heads[task_name].load_state_dict(winner["state"][task_name])
    fit_heads.eval()

    fit_support = label_support(fit)
    dev_logits = head_logits(fit_heads, dev, device)
    thresholds = tune_thresholds(dev_logits, dev, support=fit_support, minimum=MINIMUM_LABEL_EXAMPLES)

    # ---- 2) 전체 train으로 refit
    by_recipe: dict[str, dict[int, list[str]]] = {}
    for task_name, value in selection.items():
        by_recipe.setdefault(value["recipe"], {}).setdefault(value["target_epoch"], []).append(task_name)
    full_support = label_support(train_cache)
    refit_heads = build_attribute_heads(int(train_cache["features"].shape[1]), hidden, 0.4).to(device)
    print()
    for recipe_name, targets_by_epoch in by_recipe.items():
        recipe = recipes[recipe_name]
        started = time.time()
        snapshots = train_full(recipe, train_cache, targets_by_epoch, device=device)
        print(
            f"  refit {recipe_name:<18} tasks={sum(len(v) for v in targets_by_epoch.values()):<3}"
            f" max_epoch={max(targets_by_epoch):<4} {time.time() - started:.0f}s",
            flush=True,
        )
        for task_name, state in snapshots.items():
            refit_heads.heads[task_name].load_state_dict(state)
    refit_heads.eval()

    # ---- 3) 평가
    baseline_heads, baseline_payload = load_attribute_heads(BASELINE_CHECKPOINT, device)
    baseline_val = evaluate_logits(
        head_logits(baseline_heads, val_cache, device), val_cache,
        baseline_payload.get("thresholds", {}),
        label_support=baseline_payload.get("label_support", {}),
        minimum_label_examples=int(baseline_payload.get("minimum_label_examples", 5)),
    )
    fit_val = evaluate_logits(
        head_logits(fit_heads, val_cache, device), val_cache, thresholds,
        label_support=fit_support, minimum_label_examples=MINIMUM_LABEL_EXAMPLES,
    )
    refit_val = evaluate_logits(
        head_logits(refit_heads, val_cache, device), val_cache, thresholds,
        label_support=full_support, minimum_label_examples=MINIMUM_LABEL_EXAMPLES,
    )

    header = (
        f"{'task':<16}{'kind':<7}{'base':>8}{'fit4071':>9}{'refit':>8}{'Δrefit':>8}"
        f"{'bMacro':>8}{'nMacro':>8}{'Δ':>8}{'cov':>7}{'Δcov':>8}{'n':>7}"
    )
    print()
    print("=" * len(header))
    print("val 1,195 crop — 기준선 vs 미세조정 (Δrefit = refit − 기준선, 개선 큰 순)")
    print("=" * len(header))
    print(header)
    print("-" * len(header))
    rows = [
        (name, baseline_val[name], fit_val[name], refit_val[name])
        for name in ATTRIBUTE_TASKS
        if baseline_val.get(name, {}).get("score") is not None
    ]
    rows.sort(key=lambda row: -(row[3]["score"] - row[1]["score"]))
    for name, base, fitm, refit in rows:
        kind = "multi" if refit.get("multi_label") else "single"
        delta = refit["score"] - base["score"]
        bm, nm = base.get("macro_f1"), refit.get("macro_f1")
        dm = (nm - bm) if (bm is not None and nm is not None) else None
        bc, nc = base.get("accepted_coverage"), refit.get("accepted_coverage")
        dc = (nc - bc) if (bc is not None and nc is not None) else None
        print(
            f"{name:<16}{kind:<7}{base['score']:>8.3f}{fitm['score']:>9.3f}{refit['score']:>8.3f}{delta:>+8.3f}"
            f"{(f'{bm:.3f}' if bm is not None else '-'):>8}{(f'{nm:.3f}' if nm is not None else '-'):>8}"
            f"{(f'{dm:+.3f}' if dm is not None else '-'):>8}"
            f"{(f'{nc:.3f}' if nc is not None else '-'):>7}{(f'{dc:+.3f}' if dc is not None else '-'):>8}"
            f"{refit['samples']:>7}"
        )

    base_overall = overall_summary(baseline_val)
    fit_overall = overall_summary(fit_val)
    refit_overall = overall_summary(refit_val)
    print()
    print(f"{'baseline (4789, val로 early stop)':<40}", base_overall)
    print(f"{'finetuned A (4071, dev로 선택)':<40}", fit_overall)
    print(f"{'finetuned B refit (4789, dev로 선택)':<40}", refit_overall)

    print()
    print("붕괴 라벨 (val support >= 3, 기준선 F1 < 0.40) — 기준선 → refit")
    recovered = []
    for name, base, _, refit in rows:
        for label, stats in (base.get("per_label") or {}).items():
            if stats["support"] < 3:
                continue
            base_f1 = stats.get("f1")
            if base_f1 is not None and base_f1 >= 0.40:
                continue
            new_stats = (refit.get("per_label") or {}).get(label, {})
            new_f1 = new_stats.get("f1")
            recovered.append((name, label, stats["support"], base_f1, new_f1, new_stats.get("blocked")))
            print(
                f"  {name:<16}{label:<14} n={stats['support']:>3}"
                f"  {(f'{base_f1:.3f}' if base_f1 is not None else 'None'):>6}"
                f" -> {(f'{new_f1:.3f}' if new_f1 is not None else 'None'):>6}"
                f"{'  [blocked]' if new_stats.get('blocked') else ''}"
            )

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (REPORT_DIR / "05_refit_comparison.json").write_text(
        json.dumps(
            {
                "hidden_dim": hidden,
                "selection": selection,
                "thresholds": thresholds,
                "overall": {
                    "baseline_val": base_overall,
                    "finetuned_train_fit_val": fit_overall,
                    "finetuned_refit_val": refit_overall,
                },
                "baseline_val": baseline_val,
                "finetuned_train_fit_val": fit_val,
                "finetuned_refit_val": refit_val,
                "recipes": configs,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )
    save_attribute_checkpoint(
        OUTPUT_CHECKPOINT,
        refit_heads,
        backbone_model_id=train_cache["backbone_model_id"],
        thresholds=thresholds,
        training_summary={
            "note": (
                "설정(레시피·hidden_dim·epoch·임계값)은 train_dev(718)에서만 선택하고 "
                "전체 train(4789)으로 refit. val(1195)은 평가 전용."
            ),
            "hidden_dim": hidden,
            "selection": selection,
            "recipes": configs,
            "metrics": refit_val,
            "baseline_metrics": baseline_val,
            "overall": {
                "baseline_val": base_overall,
                "finetuned_refit_val": refit_overall,
            },
        },
        label_support=full_support,
        minimum_label_examples=MINIMUM_LABEL_EXAMPLES,
    )
    print(f"\nsaved: {OUTPUT_CHECKPOINT}")
    print(f"saved: {REPORT_DIR / '05_refit_comparison.json'}")


if __name__ == "__main__":
    main()
