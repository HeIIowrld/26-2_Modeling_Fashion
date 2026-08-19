"""1단계 — 현재 체크포인트가 무엇을 잘하고 무엇을 못하는지 진단한다.

출력: reports/01_baseline_diagnosis.json
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
from fashion_attribute_model import load_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS


def main() -> None:
    device = "cpu"
    train_cache = load_cache(TRAIN_CACHE)
    val_cache = load_cache(VAL_CACHE)
    fit_indices, dev_indices = split_train_dev(train_cache)
    train_fit = subset(train_cache, fit_indices)
    train_dev = subset(train_cache, dev_indices)

    heads, payload = load_attribute_heads(BASELINE_CHECKPOINT, device)
    thresholds = payload.get("thresholds", {})
    label_support = payload.get("label_support", {})
    minimum = int(payload.get("minimum_label_examples", 5))

    print(f"backbone      : {payload['backbone_model_id']}")
    print(f"input_dim     : {payload['input_dim']}  hidden_dim: {payload['hidden_dim']}")
    parameters = sum(p.numel() for p in heads.parameters())
    print(f"head params   : {parameters:,}")
    print(f"train total   : {len(train_cache['features'])}  -> fit {len(fit_indices)} / dev {len(dev_indices)}")
    print(f"val (test)    : {len(val_cache['features'])}")
    print()

    report: dict = {
        "checkpoint": str(BASELINE_CHECKPOINT),
        "backbone_model_id": payload["backbone_model_id"],
        "head_parameters": parameters,
        "split": {
            "train_total": len(train_cache["features"]),
            "train_fit": int(len(fit_indices)),
            "train_dev": int(len(dev_indices)),
            "val_test": len(val_cache["features"]),
        },
        "thresholds": thresholds,
        "label_support": label_support,
        "minimum_label_examples": minimum,
        "sets": {},
    }

    for name, cache in (("train_fit", train_fit), ("train_dev", train_dev), ("val", val_cache)):
        logits = head_logits(heads, cache, device)
        metrics = evaluate_logits(
            logits, cache, thresholds, label_support=label_support, minimum_label_examples=minimum
        )
        report["sets"][name] = {"overall": overall_summary(metrics), "tasks": metrics}

    val_metrics = report["sets"]["val"]["tasks"]
    fit_metrics = report["sets"]["train_fit"]["tasks"]

    print("=" * 96)
    print("태스크별 val 성능 (약한 순) — train_fit 점수와 비교해 과적합 폭도 함께")
    print("=" * 96)
    header = f"{'task':<16}{'kind':<7}{'val':>8}{'macroF1':>9}{'train':>8}{'gap':>8}{'cov':>8}{'acc@cov':>9}{'n':>7}"
    print(header)
    print("-" * len(header))
    ordered = sorted(
        (name for name, value in val_metrics.items() if value.get("score") is not None),
        key=lambda name: val_metrics[name]["score"],
    )
    for name in ordered:
        value = val_metrics[name]
        train_score = fit_metrics[name].get("score")
        gap = round(train_score - value["score"], 3) if train_score is not None else None
        kind = "multi" if value.get("multi_label") else "single"
        macro = value.get("macro_f1")
        print(
            f"{name:<16}{kind:<7}{value['score']:>8.3f}"
            f"{(f'{macro:.3f}' if macro is not None else '-'):>9}"
            f"{(f'{train_score:.3f}' if train_score is not None else '-'):>8}"
            f"{(f'{gap:+.3f}' if gap is not None else '-'):>8}"
            f"{(f'{value['accepted_coverage']:.3f}' if value.get('accepted_coverage') is not None else '-'):>8}"
            f"{(f'{value['accepted_accuracy']:.3f}' if value.get('accepted_accuracy') is not None else '-'):>9}"
            f"{value['samples']:>7}"
        )

    print()
    print("=" * 96)
    print("붕괴 라벨 (val support >= 3 이고 F1 < 0.40)")
    print("=" * 96)
    broken = []
    for task_name, value in val_metrics.items():
        for label, stats in (value.get("per_label") or {}).items():
            if stats["support"] >= 3 and (stats.get("f1") is None or stats["f1"] < 0.40):
                broken.append((task_name, label, stats))
    broken.sort(key=lambda row: -row[2]["support"])
    for task_name, label, stats in broken:
        train_count = label_support.get(task_name, {}).get(label, 0)
        f1 = stats.get("f1")
        print(
            f"  {task_name:<16}{label:<14} val_support={stats['support']:>4}"
            f"  train={train_count:>5}  F1={(f'{f1:.3f}' if f1 is not None else 'None'):>6}"
            f"  recall={(f'{stats['recall']:.3f}' if stats.get('recall') is not None else '-'):>6}"
            f"{'  [BLOCKED]' if stats.get('blocked') else ''}"
        )
    report["broken_labels"] = [
        {"task": t, "label": l, **s, "train_support": label_support.get(t, {}).get(l, 0)}
        for t, l, s in broken
    ]

    print()
    print("=" * 96)
    print("약한 단일분류 태스크의 주요 혼동 (val)")
    print("=" * 96)
    for name in ordered:
        value = val_metrics[name]
        if value.get("multi_label") or value["score"] >= 0.75:
            continue
        pairs = ", ".join(
            f"{row['truth']}→{row['predicted']}({row['count']})" for row in value["top_confusions"][:5]
        )
        print(f"  {name:<16} {pairs}")

    print()
    for name in ("train_fit", "train_dev", "val"):
        print(f"{name:<10}", report["sets"][name]["overall"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    output = REPORT_DIR / "01_baseline_diagnosis.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    torch.manual_seed(0)
    main()
