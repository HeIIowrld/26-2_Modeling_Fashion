"""3차 모델 평가 — 2차 모델과 3차 seed 3개를 완전히 동일한 코드로 평가한다.

평가셋은 old validation 1,195 / new validation 1,716 두 개다.
둘 다 학습·early stopping·임계값 선택에 쓰이지 않았다.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

import torch

from eval_lib import (
    BASELINE_CHECKPOINT,
    PROJECT_DIR,
    REPORT_DIR,
    VAL_CACHE,
    evaluate_logits,
    head_logits,
    load_cache,
    overall_summary,
)
from fashion_attribute_model import load_attribute_heads
from fashion_attribute_schema import ATTRIBUTE_TASKS
from final_comparison import label_pr

MODELS = PROJECT_DIR / "models"
NEW_VAL = PROJECT_DIR / "data" / "cache" / "fashion_attributes_fp_val.pt"

FOCUS_TASKS = ("category", "neckline", "collar", "upper_fit", "lower_fit",
               "silhouette", "pant_leg_shape")
FOCUS_LABELS = (("category", "니트"), ("category", "가디건"), ("category", "후드티"),
                ("detail", "단추"), ("sleeve_shape", "퍼프 소매"),
                ("upper_fit", "오버핏"), ("collar", "피터팬 칼라"))
# Fashionpedia 에 라벨이 없어 3차에서도 새 표본을 받지 못한 태스크
NO_NEW_DATA_TASKS = ("lower_subtype", "pant_leg_shape", "pant_length", "lower_detail")


def load_model(path: Path, cache: dict, device: str = "cpu") -> dict:
    heads, payload = load_attribute_heads(path, device)
    logits = head_logits(heads, cache, device)
    thresholds = payload.get("thresholds", {})
    support = payload.get("label_support", {})
    minimum = int(payload.get("minimum_label_examples", 5))
    return {
        "logits": logits, "thresholds": thresholds, "support": support, "minimum": minimum,
        "metrics": evaluate_logits(logits, cache, thresholds,
                                   label_support=support, minimum_label_examples=minimum),
    }


def evaluate_set(name: str, cache: Path, models: dict[str, Path]) -> dict:
    data = load_cache(cache)
    loaded = {label: load_model(path, data) for label, path in models.items()}
    result = {
        "crops": len(data["features"]),
        "overall": {label: overall_summary(m["metrics"]) for label, m in loaded.items()},
        "per_task": {},
        "focus_labels": {},
    }

    seeds = [k for k in loaded if k.startswith("r3_seed")]
    for key in ("mean_score", "sample_weighted_score", "mean_macro_f1"):
        values = [result["overall"][s][key] for s in seeds]
        result.setdefault("seed_stats", {})[key] = {
            "mean": round(statistics.mean(values), 4),
            "std": round(statistics.pstdev(values), 4),
            "values": values,
        }

    for task_name in ATTRIBUTE_TASKS:
        entry = {}
        for label, model in loaded.items():
            metrics = model["metrics"].get(task_name, {})
            if metrics.get("score") is None:
                continue
            entry[label] = {
                "score": metrics["score"],
                "macro_f1": metrics.get("macro_f1"),
                "accepted_coverage": metrics.get("accepted_coverage"),
                "accepted_accuracy": metrics.get("accepted_accuracy"),
                "samples": metrics["samples"],
                "per_label": metrics.get("per_label"),
            }
        if entry:
            entry["no_new_data_in_round3"] = task_name in NO_NEW_DATA_TASKS
            result["per_task"][task_name] = entry

    for task_name, label in FOCUS_LABELS:
        result["focus_labels"][f"{task_name}|{label}"] = {
            model_label: label_pr(model, data, task_name, label)
            for model_label, model in loaded.items()
        }
    return result


def print_set(title: str, result: dict) -> None:
    print()
    print("=" * 96)
    print(f"{title} — {result['crops']:,} crop")
    print("=" * 96)
    print(f"{'모델':<16}{'mean score':>13}{'표본가중':>12}{'mean macro-F1':>16}")
    print("-" * 96)
    for label, summary in result["overall"].items():
        print(f"{label:<16}{summary['mean_score']:>13.4f}"
              f"{summary['sample_weighted_score']:>12.4f}{summary['mean_macro_f1']:>16.4f}")
    stats = result["seed_stats"]
    print(f"\n  3 seed 평균±표준편차  mean {stats['mean_score']['mean']:.4f}"
          f"±{stats['mean_score']['std']:.4f}   "
          f"macro-F1 {stats['mean_macro_f1']['mean']:.4f}±{stats['mean_macro_f1']['std']:.4f}")

    r2 = result["overall"].get("r2_final")
    final = result["overall"].get("r3_final")
    if r2 and final:
        print(f"\n  최종 3차 − 2차:  mean {final['mean_score'] - r2['mean_score']:+.4f}   "
              f"표본가중 {final['sample_weighted_score'] - r2['sample_weighted_score']:+.4f}   "
              f"macro-F1 {final['mean_macro_f1'] - r2['mean_macro_f1']:+.4f}")

    print()
    header = (f"{'task':<16}{'2차':>8}{'3차':>8}{'Δ':>8}"
              f"{'2차macro':>10}{'3차macro':>10}{'Δ':>8}{'cov Δ':>9}{'n':>7}")
    print(header)
    print("-" * len(header))
    rows = []
    for task_name, entry in result["per_task"].items():
        if "r2_final" not in entry or "r3_final" not in entry:
            continue
        rows.append((task_name, entry))
    rows.sort(key=lambda r: -(r[1]["r3_final"]["score"] - r[1]["r2_final"]["score"]))
    for task_name, entry in rows:
        before, after = entry["r2_final"], entry["r3_final"]
        macro_delta = ((after["macro_f1"] - before["macro_f1"])
                       if before["macro_f1"] is not None and after["macro_f1"] is not None else None)
        cov_delta = ((after["accepted_coverage"] - before["accepted_coverage"])
                     if before["accepted_coverage"] is not None
                     and after["accepted_coverage"] is not None else None)
        mark = " *" if entry["no_new_data_in_round3"] else ""
        print(f"{task_name:<16}{before['score']:>8.3f}{after['score']:>8.3f}"
              f"{after['score'] - before['score']:>+8.3f}"
              f"{(f'{before[chr(109)+chr(97)+chr(99)+chr(114)+chr(111)+chr(95)+chr(102)+chr(49)]:.3f}' if before['macro_f1'] is not None else '-'):>10}"
              f"{(f'{after[chr(109)+chr(97)+chr(99)+chr(114)+chr(111)+chr(95)+chr(102)+chr(49)]:.3f}' if after['macro_f1'] is not None else '-'):>10}"
              f"{(f'{macro_delta:+.3f}' if macro_delta is not None else '-'):>8}"
              f"{(f'{cov_delta:+.3f}' if cov_delta is not None else '-'):>9}"
              f"{after['samples']:>7}{mark}")
    print("  * = Fashionpedia에 라벨이 없어 3차에서 새 표본을 받지 못한 태스크")
    print("      → 변화는 데이터 보강 효과가 아니라 학습 변동으로 해석해야 함")

    print(f"\n  주요 라벨 (precision / recall / F1)")
    print(f"  {'라벨':<24}{'n':>5}   {'2차':<22}{'3차':<22}")
    for key, entry in result["focus_labels"].items():
        before, after = entry.get("r2_final"), entry.get("r3_final")
        if not before or not after:
            continue
        def fmt(s):
            if s is None:
                return "—"
            p, r, f = s.get("precision"), s.get("recall"), s.get("f1")
            if s.get("blocked"):
                return "차단됨"
            return (f"{p:.3f} / {r:.3f} / {f:.3f}" if None not in (p, r, f) else "—")
        print(f"  {key:<24}{before['support']:>5}   {fmt(before):<22}{fmt(after):<22}")


def main() -> None:
    torch.set_num_threads(6)
    models = {
        "r2_final": BASELINE_CHECKPOINT.parent / "fashion_attribute_heads_augmented.pt",
        "r3_seed0": MODELS / "fashion_attribute_heads_augmented_r3_seed0.pt",
        "r3_seed1": MODELS / "fashion_attribute_heads_augmented_r3_seed1.pt",
        "r3_seed2": MODELS / "fashion_attribute_heads_augmented_r3_seed2.pt",
        "r3_final": MODELS / "fashion_attribute_heads_augmented_r3.pt",
    }
    missing = [k for k, v in models.items() if not v.is_file()]
    if missing:
        raise SystemExit(f"체크포인트 없음: {missing}")

    old = evaluate_set("old validation", VAL_CACHE, models)
    print_set("old validation (기존 1,195)", old)
    (REPORT_DIR / "18_round3_comparison_oldval.json").write_text(
        json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")

    new = evaluate_set("new validation", NEW_VAL, models)
    print_set("new validation (1,716, 희소 라벨 편향 선별)", new)
    (REPORT_DIR / "19_round3_comparison_newval.json").write_text(
        json.dumps(new, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n저장: reports/18_round3_comparison_oldval.json, reports/19_round3_comparison_newval.json")


if __name__ == "__main__":
    main()
