"""동일 쌍의 native/wide 결과를 재사용한 선택 정책 사후 평가 (새 합성 검증 아님)."""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np

from vton_eval_utils import cluster_ci, is_short_bottom, is_wide_bottom, load_dedup


def within_gap(rows, key):
    groups = defaultdict(list)
    for row in rows:
        groups[row["person"]].append(row)
    diffs = []
    for group in groups.values():
        wide = [r[key] for r in group if r["wide"]]
        other = [r[key] for r in group if not r["wide"]]
        if wide and other:
            diffs.append(np.mean(wide) - np.mean(other))
    return float(np.mean(diffs)) if diffs else float("nan")


def compare(root):
    native, errors = load_dedup(root / "results.jsonl")
    wide, wide_errors = load_dedup(root / "results_wide.jsonl")
    wide = {r["pair"]: r for r in wide}
    refs = {r["product_id"]: r for r in json.loads((root / "refs.json").read_text(encoding="utf-8"))}
    pairs = []
    missing = []
    for row in native:
        if row["category"] != "bottom" or is_short_bottom(refs[row["product_id"]]):
            continue
        other = wide.get(row["pair"])
        band = "shin" if row["width_orig"].get("shin") is not None else "thigh"
        if other is None or row["width_res"].get(band) is None or other["width_res"].get(band) is None:
            missing.append(row["pair"])
            continue
        selected = other if is_wide_bottom(refs[row["product_id"]]) else row
        pairs.append({"person": row["person"], "wide": selected is other,
                      "seller_fit": row["seller_fit"], "native": row["width_res"][band],
                      "all_wide": other["width_res"][band], "selected": selected["width_res"][band],
                      "rank_native": row.get("rank_res"), "rank_selected": selected.get("rank_res")})
    slim = [r for r in pairs if r["seller_fit"] in {"슬림핏", "레귤러핏"}]
    ranked = [r for r in pairs if r["rank_native"] is not None and r["rank_selected"] is not None]
    return {
        "kind": "posthoc_policy_replay_not_new_generation",
        "pairs": len(pairs), "people": len({r["person"] for r in pairs}),
        "unmatched": missing, "errors_native": len(errors), "errors_wide": len(wide_errors),
        "native_gap_ci": cluster_ci(pairs, lambda s: within_gap(s, "native")),
        "all_wide_gap_ci": cluster_ci(pairs, lambda s: within_gap(s, "all_wide")),
        "selected_gap_ci": cluster_ci(pairs, lambda s: within_gap(s, "selected")),
        "slim_regular_width_increase_ci": cluster_ci(slim, lambda s: float(np.mean([r["selected"] - r["native"] for r in s]))),
        "slim_regular_widened_pairs": sum(r["wide"] for r in slim),
        "top1_native": float(np.mean([r["rank_native"] == 1 for r in ranked])),
        "top1_selected": float(np.mean([r["rank_selected"] == 1 for r in ranked])),
        "top1_delta_ci": cluster_ci(ranked, lambda s: float(np.mean([int(r["rank_selected"] == 1) - int(r["rank_native"] == 1) for r in s]))),
        "outside_psnr_comparable": False,
        "outside_psnr_reason": "기존 결과의 측정 마스크가 다름. audit_vton_gates.py로 같은 영역에서 재측정 필요.",
    }


def compare_torso(root, variant_path):
    baseline, _ = load_dedup(root / "results.jsonl")
    variants, errors = load_dedup(variant_path)
    baseline = {r["pair"]: r for r in baseline}
    pairs = [(baseline[r["pair"]], r) for r in variants if r["pair"] in baseline]
    result = {"kind": "paired_generation", "pairs": len(pairs), "errors": len(errors),
              "people": len({v["person"] for _, v in pairs}),
              "note": "한 사람 실험에는 사람 단위 모집단 신뢰구간을 제시하지 않는다."}
    for side, label in enumerate(("native", "torso")):
        rows = [pair[side] for pair in pairs]
        result[label] = {
            "skin_at_least_10_percent": sum(r["skin_in_orig"] >= 0.1 for r in rows),
            "skin_mean": float(np.mean([r["skin_in_orig"] for r in rows])),
            "top1_count": sum(r.get("rank_res") == 1 for r in rows),
            "similarity_mean": float(np.mean([r["sim_res_ref"] for r in rows])),
            "color_de_median": float(np.median([r["color_de_res"] for r in rows])),
            "native_outside_psnr_median": float(np.median([
                r.get("native_outside_psnr", r["outside_psnr"]) for r in rows])),
        }
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--torso-results", type=Path)
    args = ap.parse_args()
    result = compare_torso(args.root, args.torso_results) if args.torso_results else compare(args.root)
    print(json.dumps(result, ensure_ascii=False, indent=2))
