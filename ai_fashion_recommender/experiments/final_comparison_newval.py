"""최종 비교 보충 — 새 validation set(1,716 crop)에서 배포 모델 vs 2차 보강 모델.

기존 val은 니트가 4장뿐이라 니트 지표가 표본 1~2장에 좌우된다.
새 validation set은 니트 54장이라 여기서 다시 잰다.
두 모델 모두 이 셋을 학습에 쓰지 않았다.
"""

from __future__ import annotations

import json

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
    PROJECT_DIR,
    REPORT_DIR,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
)
from fashion_attribute_schema import ATTRIBUTE_TASKS
from final_comparison import AUGMENTED_CHECKPOINT, FOCUS, label_pr, load

NEW_VAL = PROJECT_DIR / "data" / "cache" / "fashion_attributes_fp_val.pt"


def main() -> None:
    torch.set_num_threads(4)
    cache = load_cache(NEW_VAL)
    baseline = load(BASELINE_CHECKPOINT, cache)
    augmented = load(AUGMENTED_CHECKPOINT, cache)

    base_overall = overall_summary(baseline["metrics"])
    aug_overall = overall_summary(augmented["metrics"])

    print("=" * 78)
    print(f"새 validation set {len(cache['features']):,} crop — 배포 모델 vs 2차 보강 모델")
    print("=" * 78)
    print(f"{'지표':<26}{'배포 모델':>12}{'2차 보강':>12}{'차이':>10}")
    print("-" * 78)
    for key, label in (
        ("mean_score", "mean score"),
        ("sample_weighted_score", "표본가중 score"),
        ("mean_macro_f1", "mean macro-F1"),
    ):
        b, a = base_overall[key], aug_overall[key]
        print(f"{label:<26}{b:>12.4f}{a:>12.4f}{a - b:>+10.4f}")

    print()
    print("=" * 78)
    print("태스크별")
    print("=" * 78)
    header = f"{'task':<16}{'배포':>8}{'보강':>8}{'Δ':>8}{'bMacro':>8}{'aMacro':>8}{'Δ':>8}{'n':>7}"
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
    for task_name, b, a in rows:
        bm, am = b.get("macro_f1"), a.get("macro_f1")
        dm = (am - bm) if (bm is not None and am is not None) else None
        print(
            f"{task_name:<16}{b['score']:>8.3f}{a['score']:>8.3f}{a['score'] - b['score']:>+8.3f}"
            f"{(f'{bm:.3f}' if bm is not None else '-'):>8}"
            f"{(f'{am:.3f}' if am is not None else '-'):>8}"
            f"{(f'{dm:+.3f}' if dm is not None else '-'):>8}{a['samples']:>7}"
        )

    print()
    print("=" * 78)
    print("니트 · 단추 (표본이 충분한 새 validation set 기준)")
    print("=" * 78)
    focus = {}
    for task_name, label in FOCUS:
        b = label_pr(baseline, cache, task_name, label)
        a = label_pr(augmented, cache, task_name, label)
        focus[f"{task_name}|{label}"] = {"baseline": b, "augmented": a}
        if b is None or a is None:
            print(f"\n  [{task_name} | {label}] 이 평가셋에 라벨 없음")
            continue
        print(f"\n  [{task_name} | {label}]  val 표본 {b['support']}장")
        print(f"    {'':<12}{'precision':>11}{'recall':>10}{'F1':>9}{'TP':>6}{'FP':>6}{'FN':>6}")
        for name, stats in (("배포 모델", b), ("2차 보강", a)):
            precision = stats["precision"]
            recall = stats["recall"]
            f1 = stats["f1"]
            print(
                f"    {name:<12}"
                f"{(f'{precision:.3f}' if precision is not None else '-'):>11}"
                f"{(f'{recall:.3f}' if recall is not None else '-'):>10}"
                f"{(f'{f1:.3f}' if f1 is not None else '-'):>9}"
                f"{stats['tp']:>6}{stats['fp']:>6}{stats['fn']:>6}"
                + ("  [차단됨]" if stats["blocked"] else "")
            )

    payload = {
        "eval_set": {
            "name": "새 validation set (Fashionpedia train split의 20%)",
            "crops": len(cache["features"]),
            "note": "두 모델 모두 학습에 사용하지 않음. 다만 2차 보강 설계 시 1차 결과를 참고했으므로 완전 독립 test set이 아니라 validation set으로 표기한다.",
        },
        "overall": {"baseline": base_overall, "augmented": aug_overall},
        "per_task": {name: {"baseline": b, "augmented": a} for name, b, a in rows},
        "focus_labels": focus,
    }
    output = REPORT_DIR / "15_final_comparison_newval.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nsaved: {output}")


if __name__ == "__main__":
    main()
