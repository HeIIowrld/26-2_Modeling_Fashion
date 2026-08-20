"""5단계 평가 — 재라벨링이 실제로 도움이 되는지 두 가지 기준으로 나눠 본다.

표 A (엄격 비교): 원본 val 라벨로만 평가한다.
  detail은 원래 유효했던 262행, material은 156행. 정답이 변하지 않으므로 BASE/D/K/DK를
  같은 자로 비교할 수 있다. "단추 같은 실제 디테일을 더 잘 찾는가"를 측정한다.

표 B (새 능력): 재라벨링한 val 라벨로 평가한다.
  detail 436행(디테일 없음 174행 포함). BASE는 `디테일 없음`의 train support가 0이라
  게이팅으로 차단되어 구조적으로 이 행들을 맞출 수 없다. D/DK가 얻는 새 능력을 본다.

임계값은 전부 train_dev에서 고른다. val은 평가에만 쓴다.
"""

from __future__ import annotations

import json

import torch

from eval_lib import (
    PROJECT_DIR,
    REPORT_DIR,
    TRAIN_CACHE,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    split_train_dev,
    subset,
)
from fashion_attribute_model import build_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS
from finetune import HIDDEN_DIM, MULTI_THRESHOLD_GRID, RUN_DIR
from run_relabel_experiment import CACHE_DIR, RELABEL_RUN_DIR

TASKS = ("detail", "material")


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


def tune_multi_threshold(logits, cache, task_name, support, minimum):
    """기존 코드와 같은 목표(micro-F1 최대)로 dev에서 임계값을 고른다."""
    task = ATTRIBUTE_TASKS[task_name]
    mask = cache["valid"][task_name]
    if not bool(mask.any()):
        return task.minimum_confidence
    probabilities = logits[task_name][mask].float().sigmoid()
    blocked = [
        i for i, label in enumerate(task.labels)
        if int(support.get(task_name, {}).get(label, 0)) < minimum
    ]
    if blocked:
        probabilities[:, blocked] = 0.0
    expected = cache["targets"][task_name][mask].bool()
    best, best_f1 = task.minimum_confidence, -1.0
    for threshold in MULTI_THRESHOLD_GRID:
        predicted = probabilities >= threshold
        tp = int((predicted & expected).sum())
        fp = int((predicted & ~expected).sum())
        fn = int((~predicted & expected).sum())
        denominator = 2 * tp + fp + fn
        f1 = 2 * tp / denominator if denominator else 0.0
        if f1 > best_f1:
            best, best_f1 = threshold, f1
    return best


def assemble(runs, fit):
    """태스크별로 dev 점수 최고인 시드를 골라 헤드를 만든다."""
    heads = build_attribute_heads(int(fit["features"].shape[1]), HIDDEN_DIM, 0.4)
    picked = {}
    for task_name in ATTRIBUTE_TASKS:
        candidates = [run for run in runs if task_name in run["state"]]
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


def main() -> None:
    torch.set_num_threads(4)
    device = "cpu"

    variants: dict[str, dict] = {}
    base = torch.load(RUN_DIR / "R2_mixup.pt", map_location="cpu", weights_only=False)
    variants["BASE"] = {"runs": base["runs"], "cache": None}
    for name in ("D", "D25", "D50", "K", "DK"):
        path = RELABEL_RUN_DIR / f"{name}.pt"
        if not path.is_file():
            print(f"skip {name}: {path} 없음")
            continue
        payload = torch.load(path, map_location="cpu", weights_only=False)
        variants[name] = {"runs": payload["runs"], "cache": name}

    original_train = load_cache(TRAIN_CACHE)
    original_val = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(original_train)

    report: dict = {"strict": {}, "relabeled": {}, "picked": {}, "thresholds": {}}

    for name, entry in variants.items():
        suffix = entry["cache"]
        train_cache = (
            original_train if suffix is None
            else load_cache(CACHE_DIR / f"fashion_attributes_train_{suffix}.pt")
        )
        val_variant = (
            original_val if suffix is None
            else load_cache(CACHE_DIR / f"fashion_attributes_val_{suffix}.pt")
        )
        fit = subset(train_cache, fit_indices)
        dev = subset(train_cache, dev_indices)
        heads, picked = assemble(entry["runs"], fit)
        report["picked"][name] = picked

        support = label_support(fit)
        minimum = 5
        dev_logits = head_logits(heads, dev, device)
        thresholds = {
            task_name: tune_multi_threshold(dev_logits, dev, task_name, support, minimum)
            for task_name in TASKS
        }
        report["thresholds"][name] = thresholds

        # 표 A: 원본 val 라벨 (모든 변형 공통 자)
        report["strict"][name] = evaluate_logits(
            head_logits(heads, original_val, device), original_val, thresholds,
            label_support=support, minimum_label_examples=minimum,
        )
        # 표 B: 각 변형이 학습한 라벨 정의로 평가
        report["relabeled"][name] = evaluate_logits(
            head_logits(heads, val_variant, device), val_variant, thresholds,
            label_support=support, minimum_label_examples=minimum,
        )

    order = [name for name in ("BASE", "D25", "D50", "D", "K", "DK") if name in variants]

    print()
    print("=" * 84)
    print("표 A — 원본 val 라벨로 엄격 비교 (detail 262행 / material 156행, 정답 불변)")
    print("=" * 84)
    print(f"{'변형':<6}{'detail micro':>14}{'detail macro':>14}{'material micro':>16}{'material macro':>16}")
    print("-" * 84)
    for name in order:
        d = report["strict"][name]["detail"]
        m = report["strict"][name]["material"]
        print(
            f"{name:<6}{d['micro_f1']:>14.4f}{d['macro_f1']:>14.4f}"
            f"{m['micro_f1']:>16.4f}{m['macro_f1']:>16.4f}"
        )

    print()
    print("detail 라벨별 F1 (원본 val 라벨)")
    labels = ATTRIBUTE_TASKS["detail"].labels
    head = f"{'label':<12}{'val n':>7}" + "".join(f"{name:>9}" for name in order)
    print(head)
    print("-" * len(head))
    for label in labels:
        stats = {name: report["strict"][name]["detail"]["per_label"][label] for name in order}
        support_count = stats[order[0]]["support"]
        if support_count == 0:
            continue
        row = f"{label:<12}{support_count:>7}"
        for name in order:
            f1 = stats[name]["f1"]
            row += f"{(f'{f1:.3f}' if f1 is not None else '-'):>9}"
        print(row)

    print()
    print("material 라벨별 F1 (원본 val 라벨)")
    for label in ATTRIBUTE_TASKS["material"].labels:
        stats = {name: report["strict"][name]["material"]["per_label"][label] for name in order}
        support_count = stats[order[0]]["support"]
        if support_count == 0:
            continue
        row = f"{label:<12}{support_count:>7}"
        for name in order:
            f1 = stats[name]["f1"]
            row += f"{(f'{f1:.3f}' if f1 is not None else '-'):>9}"
        print(row)

    print()
    print("=" * 84)
    print("표 B — 각 변형이 학습한 라벨 정의로 평가 (D/DK는 detail 436행, 디테일 없음 174행 포함)")
    print("=" * 84)
    print(f"{'변형':<6}{'detail n':>10}{'micro':>9}{'macro':>9}{'디테일없음 F1':>16}{'material n':>12}{'micro':>9}")
    print("-" * 84)
    for name in order:
        d = report["relabeled"][name]["detail"]
        m = report["relabeled"][name]["material"]
        none_stats = d["per_label"]["디테일 없음"]
        none_f1 = none_stats["f1"]
        print(
            f"{name:<6}{d['samples']:>10}{d['micro_f1']:>9.4f}{d['macro_f1']:>9.4f}"
            f"{(f'{none_f1:.3f}' if none_f1 is not None else ('blocked' if none_stats['blocked'] else '-')):>16}"
            f"{m['samples']:>12}{m['micro_f1']:>9.4f}"
        )

    print()
    print("dev에서 고른 임계값:", {k: v for k, v in report["thresholds"].items()})

    output = REPORT_DIR / "09_relabel_results.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
