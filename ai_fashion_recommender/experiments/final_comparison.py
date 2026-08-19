"""최종 비교 — 배포 모델 vs 2차 보강 모델.

평가셋: 기존 val 1,195 crop (두 모델 모두 학습에 쓰지 않음)
  주의: 배포 모델은 이 셋으로 early stopping·임계값을 골랐다(원본 학습 코드가 그렇게 되어 있다).
        따라서 이 비교는 배포 모델에 유리한 쪽으로 기울어 있다. 그럼에도 보강 모델이 이긴다.

출력: reports/14_final_comparison.json
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
    REPORT_DIR,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
)
from fashion_attribute_model import load_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS

AUGMENTED_CHECKPOINT = BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_augmented.pt"
FOCUS = [("category", "니트"), ("material", "니트"), ("detail", "단추")]


def short_path(path: str) -> str:
    parts = Path(str(path).replace("\\", "/")).parts
    return "/".join(parts[-3:])


def load(checkpoint, cache, device="cpu"):
    heads, payload = load_attribute_heads(checkpoint, device)
    thresholds = payload.get("thresholds", {})
    support = payload.get("label_support", {})
    minimum = int(payload.get("minimum_label_examples", 5))
    logits = head_logits(heads, cache, device)
    metrics = evaluate_logits(
        logits, cache, thresholds, label_support=support, minimum_label_examples=minimum
    )
    return {
        "logits": logits, "metrics": metrics, "thresholds": thresholds,
        "support": support, "minimum": minimum,
    }


def label_pr(model, cache, task_name, label):
    """단일·다중 분류 모두에서 특정 라벨의 precision/recall/F1 을 계산한다."""
    task = ATTRIBUTE_TASKS[task_name]
    index = task.labels.index(label)
    mask = cache["valid"][task_name]
    if not bool(mask.any()):
        return None
    expected = cache["targets"][task_name][mask]
    selected = model["logits"][task_name][mask].float()
    allowed = [
        i for i, name in enumerate(task.labels)
        if int(model["support"].get(task_name, {}).get(name, 0)) >= model["minimum"]
    ]
    threshold = float(model["thresholds"].get(task_name, task.minimum_confidence))
    if task.multi_label:
        probabilities = selected.sigmoid()
        blocked = sorted(set(range(len(task.labels))) - set(allowed))
        if blocked:
            probabilities[:, blocked] = 0.0
        predicted = probabilities[:, index] >= threshold
        truth = expected[:, index].bool()
    else:
        masked = selected.clone()
        blocked = sorted(set(range(len(task.labels))) - set(allowed))
        if blocked:
            masked[:, blocked] = float("-inf")
        predicted = masked.argmax(dim=-1) == index
        truth = expected == index
    tp = int((predicted & truth).sum())
    fp = int((predicted & ~truth).sum())
    fn = int((~predicted & truth).sum())
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision and recall else (0.0 if tp + fn else None)
    )
    return {
        "support": int(truth.sum()), "tp": tp, "fp": fp, "fn": fn,
        "precision": round(precision, 4) if precision is not None else None,
        "recall": round(recall, 4) if recall is not None else None,
        "f1": round(f1, 4) if f1 is not None else None,
        "blocked": index not in allowed,
        "threshold": threshold,
    }


def single_label_examples(baseline, augmented, cache, limit=3):
    """개선/실패 사례를 실제 val crop에서 뽑는다."""
    improved, regressed, still_wrong = [], [], []
    for task_name, task in ATTRIBUTE_TASKS.items():
        if task.multi_label:
            continue
        mask = cache["valid"][task_name]
        if not bool(mask.any()):
            continue
        rows = mask.nonzero(as_tuple=True)[0].tolist()
        expected = cache["targets"][task_name][mask].tolist()

        def predict(model):
            selected = model["logits"][task_name][mask].float()
            allowed = [
                i for i, name in enumerate(task.labels)
                if int(model["support"].get(task_name, {}).get(name, 0)) >= model["minimum"]
            ]
            masked = selected.clone()
            blocked = sorted(set(range(len(task.labels))) - set(allowed))
            if blocked:
                masked[:, blocked] = float("-inf")
            probabilities = masked.softmax(dim=-1)
            confidence, predicted = probabilities.max(dim=-1)
            return predicted.tolist(), confidence.tolist()

        base_pred, base_conf = predict(baseline)
        aug_pred, aug_conf = predict(augmented)
        for position, row in enumerate(rows):
            truth = int(expected[position])
            entry = {
                "task": task_name,
                "image": short_path(cache["image_paths"][row]),
                "truth": task.labels[truth],
                "baseline": task.labels[base_pred[position]],
                "baseline_conf": round(base_conf[position], 3),
                "augmented": task.labels[aug_pred[position]],
                "augmented_conf": round(aug_conf[position], 3),
            }
            if base_pred[position] != truth and aug_pred[position] == truth:
                improved.append(entry)
            elif base_pred[position] == truth and aug_pred[position] != truth:
                regressed.append(entry)
            elif base_pred[position] != truth and aug_pred[position] != truth:
                still_wrong.append(entry)
    improved.sort(key=lambda e: -e["augmented_conf"])
    regressed.sort(key=lambda e: -e["augmented_conf"])
    still_wrong.sort(key=lambda e: -e["augmented_conf"])
    return improved, regressed, still_wrong


def main() -> None:
    torch.set_num_threads(4)
    cache = load_cache(VAL_CACHE)
    baseline = load(BASELINE_CHECKPOINT, cache)
    augmented = load(AUGMENTED_CHECKPOINT, cache)

    base_overall = overall_summary(baseline["metrics"])
    aug_overall = overall_summary(augmented["metrics"])

    print("=" * 78)
    print("최종 비교 — 기존 배포 모델 vs 2차 보강 모델 (기존 val 1,195 crop)")
    print("=" * 78)
    print(f"{'지표':<26}{'배포 모델':>12}{'2차 보강':>12}{'차이':>10}")
    print("-" * 78)
    for key, label in (
        ("mean_score", "mean score (17 태스크 평균)"),
        ("sample_weighted_score", "표본가중 score"),
        ("mean_macro_f1", "mean macro-F1"),
    ):
        b, a = base_overall[key], aug_overall[key]
        print(f"{label:<26}{b:>12.4f}{a:>12.4f}{a - b:>+10.4f}")

    print()
    print("=" * 78)
    print("태스크별 (개선 폭 순)")
    print("=" * 78)
    header = f"{'task':<16}{'kind':<7}{'배포':>8}{'보강':>8}{'Δ':>8}{'bMacro':>8}{'aMacro':>8}{'Δ':>8}{'n':>7}"
    print(header)
    print("-" * len(header))
    rows = []
    for task_name in ATTRIBUTE_TASKS:
        b = baseline["metrics"].get(task_name, {})
        a = augmented["metrics"].get(task_name, {})
        if b.get("score") is None or a.get("score") is None:
            continue
        rows.append((task_name, b, a))
    rows.sort(key=lambda r: -(r[2]["score"] - r[1]["score"]))
    improved_tasks = 0
    for task_name, b, a in rows:
        kind = "multi" if a.get("multi_label") else "single"
        bm, am = b.get("macro_f1"), a.get("macro_f1")
        dm = (am - bm) if (bm is not None and am is not None) else None
        if a["score"] > b["score"]:
            improved_tasks += 1
        print(
            f"{task_name:<16}{kind:<7}{b['score']:>8.3f}{a['score']:>8.3f}"
            f"{a['score'] - b['score']:>+8.3f}"
            f"{(f'{bm:.3f}' if bm is not None else '-'):>8}"
            f"{(f'{am:.3f}' if am is not None else '-'):>8}"
            f"{(f'{dm:+.3f}' if dm is not None else '-'):>8}{a['samples']:>7}"
        )
    print(f"\n  상승 {improved_tasks} / {len(rows)} 태스크")

    print()
    print("=" * 78)
    print("니트 · 단추 정밀 비교 (precision / recall / F1)")
    print("=" * 78)
    focus_results = {}
    for task_name, label in FOCUS:
        b = label_pr(baseline, cache, task_name, label)
        a = label_pr(augmented, cache, task_name, label)
        focus_results[f"{task_name}|{label}"] = {"baseline": b, "augmented": a}
        if b is None or a is None:
            continue
        print(f"\n  [{task_name} | {label}]  val 표본 {b['support']}장")
        print(f"    {'':<12}{'precision':>11}{'recall':>10}{'F1':>9}{'TP':>6}{'FP':>6}{'FN':>6}")
        for name, stats in (("배포 모델", b), ("2차 보강", a)):
            print(
                f"    {name:<12}"
                f"{(f'{stats[chr(112)+chr(114)+chr(101)+chr(99)+chr(105)+chr(115)+chr(105)+chr(111)+chr(110)]:.3f}' if stats['precision'] is not None else '-'):>11}"
                f"{(f'{stats[chr(114)+chr(101)+chr(99)+chr(97)+chr(108)+chr(108)]:.3f}' if stats['recall'] is not None else '-'):>10}"
                f"{(f'{stats[chr(102)+chr(49)]:.3f}' if stats['f1'] is not None else '-'):>9}"
                f"{stats['tp']:>6}{stats['fp']:>6}{stats['fn']:>6}"
                + ("  [차단됨]" if stats["blocked"] else "")
            )

    improved, regressed, still_wrong = single_label_examples(baseline, augmented, cache)
    print()
    print("=" * 78)
    print(f"개선 사례 — 배포 모델이 틀리고 보강 모델이 맞춘 crop ({len(improved)}건 중 상위 12)")
    print("=" * 78)
    for entry in improved[:12]:
        print(f"  {entry['task']:<15} 정답 {entry['truth']:<12} "
              f"배포 {entry['baseline']:<12}({entry['baseline_conf']:.2f}) → "
              f"보강 {entry['augmented']:<12}({entry['augmented_conf']:.2f})")
        print(f"      {entry['image']}")

    print()
    print("=" * 78)
    print(f"실패 사례 ① 보강 모델이 새로 틀린 crop ({len(regressed)}건 중 상위 8)")
    print("=" * 78)
    for entry in regressed[:8]:
        print(f"  {entry['task']:<15} 정답 {entry['truth']:<12} "
              f"배포 {entry['baseline']:<12}({entry['baseline_conf']:.2f}) → "
              f"보강 {entry['augmented']:<12}({entry['augmented_conf']:.2f})")
        print(f"      {entry['image']}")

    print()
    print("=" * 78)
    print(f"실패 사례 ② 두 모델 모두 틀린 crop ({len(still_wrong)}건 중 상위 8)")
    print("=" * 78)
    for entry in still_wrong[:8]:
        print(f"  {entry['task']:<15} 정답 {entry['truth']:<12} "
              f"배포 {entry['baseline']:<12} / 보강 {entry['augmented']:<12}({entry['augmented_conf']:.2f})")
        print(f"      {entry['image']}")

    payload = {
        "eval_set": {
            "name": "기존 val",
            "crops": len(cache["features"]),
            "note": "두 모델 모두 학습에 사용하지 않음. 단 배포 모델은 이 셋으로 early stopping·임계값을 선택했으므로 배포 모델에 유리한 비교다.",
        },
        "overall": {"baseline": base_overall, "augmented": aug_overall},
        "per_task": {
            name: {"baseline": b, "augmented": a} for name, b, a in rows
        },
        "focus_labels": focus_results,
        "examples": {
            "improved": improved[:60],
            "regressed": regressed[:40],
            "still_wrong": still_wrong[:40],
            "counts": {
                "improved": len(improved),
                "regressed": len(regressed),
                "still_wrong": len(still_wrong),
            },
        },
    }
    output = REPORT_DIR / "14_final_comparison.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n단일분류 전체: 개선 {len(improved)}건 / 악화 {len(regressed)}건 / 둘 다 오답 {len(still_wrong)}건")
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
