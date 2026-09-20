"""2026-09-15 후속 실험의 사전 등록 기준 비교.

합성 결과를 보기 전에 채택 기준을 코드로 고정한다. 기준을 결과에 맞춰 바꾸지 않는다.
모든 신뢰구간은 사람 단위 클러스터 부트스트랩(vton_eval_utils.cluster_ci, 2000회)이다.

입력은 eval_vton_fit.py 결과(JSONL)와 같은 이미지에 대한 replay_tryon_quality.py 재생 결과다.

    python compare_vton_variants.py agnostic  --base <base.jsonl> --base-q <base_q.jsonl> --var <var.jsonl> --var-q <var_q.jsonl> --refs refs.json
    python compare_vton_variants.py lower-shape ... (같은 인자)
    python compare_vton_variants.py seeds --base ... --base-q ... --var s43.jsonl,s44.jsonl --var-q q43.jsonl,q44.jsonl
    python compare_vton_variants.py flags --var-q <q.jsonl>
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vton_eval_utils import cluster_ci, is_short_bottom, is_wide_bottom, load_dedup

CROP_CHECKS = {"torso_skin_inside_mask", "torso_skin_outside_mask", "garment_coverage"}
FIDELITY_CHECKS = {"reference_fidelity", "color_fidelity", "garment_leak"}
BOTTOM_CHECKS = {"bottom_length_fidelity", "color_fidelity", "reference_fidelity"}


def load_quality(path: Path, *, allow_unassessed=False) -> dict[str, dict]:
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["assessed"] = not row.get("skipped") and not row.get("error") and bool(row.get("checks"))
        if not row["assessed"] and not allow_unassessed:
            raise ValueError(f"미검사 결과는 통과로 집계할 수 없습니다: {path}: {row.get('pair')}")
        row.setdefault("checks", [])
        row["failed"] = {c["name"] for c in row["checks"] if not c["passed"]}
        row["retry"] = any(not c["passed"] and c["retryable"] for c in row["checks"])
        row["values"] = {c["name"]: c["value"] for c in row["checks"]}
        rows[row["pair"]] = row
    return rows


def paired(base: Path, base_q: Path, var: Path, var_q: Path, category: str) -> list[dict]:
    base_rows = {r["pair"]: r for r in load_dedup(base)[0] if r["category"] == category}
    successful, errors = load_dedup(var)
    if any(r.get("category", category) == category for r in errors):
        raise ValueError("변형 실험에 생성 오류가 있습니다. 오류를 제외하고 채택할 수 없습니다.")
    var_rows = {r["pair"]: r for r in successful if r["category"] == category}
    bq, vq = load_quality(base_q), load_quality(var_q)
    if not var_rows or set(var_rows) - (set(base_rows) & set(bq) & set(vq)):
        raise ValueError("비교 대상 또는 대응하는 기준/품질 결과가 누락됐습니다.")
    items = []
    for pair, v in var_rows.items():
        if pair in base_rows and pair in bq and pair in vq:
            items.append({"pair": pair, "person": v["person"], "product_id": v["product_id"],
                          "seller_fit": v["seller_fit"], "base": base_rows[pair], "var": v,
                          "bq": bq[pair], "vq": vq[pair]})
    return items


def rate_diff(items, predicate):
    return cluster_ci(items, lambda s: float(np.mean([predicate(i["vq"], i["var"]) for i in s]) -
                                         np.mean([predicate(i["bq"], i["base"]) for i in s])))


def verdict(name: str, ok: bool, detail) -> dict:
    return {"criterion": name, "passed": bool(ok), "detail": detail}


def compare_agnostic(items: list[dict]) -> dict:
    crop = rate_diff(items, lambda q, _r: bool(q["failed"] & CROP_CHECKS))
    top1 = rate_diff(items, lambda _q, r: r.get("rank_res") == 1)
    fidelity = rate_diff(items, lambda q, _r: bool(q["failed"] & FIDELITY_CHECKS))
    any_fail = rate_diff(items, lambda q, _r: bool(q["failed"]))
    color = [(i["var"].get("color_de_res"), i["base"].get("color_de_res")) for i in items]
    color = [(a, b) for a, b in color if a is not None and b is not None]
    color_shift = float(np.median([a for a, _ in color]) - np.median([b for _, b in color]))
    face = [i["var"].get("face_psnr") for i in items if i["var"].get("face_psnr") is not None]
    criteria = [
        verdict("A1 크롭·배 노출·덜 덮음 실패율 감소(CI 상한<0)", crop[2] < 0, crop),
        verdict("A2a 자기 상품 top-1 비열등(CI 하한≥-0.05)", top1[1] >= -0.05, top1),
        verdict("A2b 색차 중앙값 증가 ≤1.0", color_shift <= 1.0, color_shift),
        verdict("A3 충실도·잔류 실패율 비열등(허용 증가 5%p, CI 상한≤0.05)", fidelity[2] <= 0.05, fidelity),
        verdict("A4 얼굴 PSNR 최소 ≥30dB", min(face) >= 30.0, min(face)),
    ]
    return {"pairs": len(items), "people": len({i["person"] for i in items}), "criteria": criteria,
            "adopt": all(c["passed"] for c in criteria), "any_failure_rate_diff": any_fail,
            "top_fit_gap": top_fit_gap(items), "per_person_crop": per_person(items, CROP_CHECKS)}


def top_fit_gap(items: list[dict]) -> dict:
    """정보용(항목 1): 같은 사람 안에서 오버핏−슬림핏 판매자 태그의 가슴 폭 차이."""
    def gap(sample, side):
        groups = defaultdict(lambda: defaultdict(list))
        for i in sample:
            width = i[side]["width_res"].get("chest")
            if width is not None:
                groups[i["person"]][i["seller_fit"]].append(width)
        diffs = [np.mean(g["오버핏"]) - np.mean(g["슬림핏"]) for g in groups.values() if g["오버핏"] and g["슬림핏"]]
        return float(np.mean(diffs)) if diffs else float("nan")
    return {"native": cluster_ci(items, lambda s: gap(s, "base")),
            "variant": cluster_ci(items, lambda s: gap(s, "var")),
            "difference": cluster_ci(items, lambda s: gap(s, "var") - gap(s, "base"))}


def per_person(items, checks):
    table = defaultdict(lambda: [0, 0, 0])
    for i in items:
        row = table[i["person"]]
        row[0] += 1
        row[1] += bool(i["bq"]["failed"] & checks)
        row[2] += bool(i["vq"]["failed"] & checks)
    return {person: {"pairs": n, "native": b, "variant": v} for person, (n, b, v) in sorted(table.items())}


def compare_lower_shape(items: list[dict], refs: dict) -> dict:
    items = [i for i in items if not is_short_bottom(refs[i["product_id"]])]
    for i in items:
        i["wide"] = is_wide_bottom(refs[i["product_id"]])
        i["shaped"] = any(note.startswith("lower:shaped") for note in i["var"].get("mask_notes", []))
    # 원본 사진으로 부위를 고정한다. 결과마다 종아리/허벅지를 바꾸면 폭 차이가 왜곡된다.
    for i in items:
        i["width_band"] = "shin" if i["base"]["width_orig"].get("shin") is not None else "thigh"
    band = lambda i, side: i[side]["width_res"].get(i["width_band"])  # noqa: E731
    total = len(items)
    items = [i for i in items if band(i, "base") is not None and band(i, "var") is not None]

    def within_gap(sample, side):
        groups = defaultdict(lambda: {True: [], False: []})
        for i in sample:
            groups[i["person"]][i["wide"]].append(band(i, side))
        diffs = [np.mean(g[True]) - np.mean(g[False]) for g in groups.values() if g[True] and g[False]]
        return float(np.mean(diffs)) if diffs else float("nan")

    gap_diff = cluster_ci(items, lambda s: within_gap(s, "var") - within_gap(s, "base"))
    slim = [i for i in items if i["seller_fit"] in {"슬림핏", "레귤러핏"} and not i["wide"]]
    slim_increase = cluster_ci(slim, lambda s: float(np.mean([band(i, "var") - band(i, "base") for i in s])))
    top1 = rate_diff(items, lambda _q, r: r.get("rank_res") == 1)
    bottom_fail = rate_diff(items, lambda q, _r: bool(q["failed"] & BOTTOM_CHECKS))
    criteria = [
        verdict("L0 고정한 부위의 폭 측정 누락 없음", len(items) == total, {"expected": total, "measured": len(items)}),
        verdict("L1 같은 사람 안 와이드−나머지 폭 차이 증가(CI 하한>0)", gap_diff[1] > 0, gap_diff),
        verdict("L2 슬림·레귤러(와이드 이름 제외) 폭 증가 없음(CI 상한≤0.02)", slim_increase[2] <= 0.02, slim_increase),
        verdict("L3 자기 상품 top-1 비열등(CI 하한≥-0.05)", top1[1] >= -0.05, top1),
        verdict("L4 하의 기장·색·충실도 실패율 증가 없음(CI 상한≤0.05)", bottom_fail[2] <= 0.05, bottom_fail),
    ]
    return {"pairs": len(items), "people": len({i["person"] for i in items}),
            "shaped_pairs": sum(i["shaped"] for i in items),
            "shaped_by_seller_fit": dict(Counter(i["seller_fit"] for i in items if i["shaped"])),
            "shaped_wide_named": sum(i["shaped"] and i["wide"] for i in items),
            "native_gap": cluster_ci(items, lambda s: within_gap(s, "base")),
            "variant_gap": cluster_ci(items, lambda s: within_gap(s, "var")),
            "criteria": criteria, "adopt": all(c["passed"] for c in criteria)}


def compare_seeds(base_q: Path, var_qs: list[Path]) -> dict:
    """시드 42(기존)와 추가 시드의 검사 결과로 재생성 효과와 판정 안정성을 잰다.

    저장된 이미지 재검사 점수로 선택 정책을 재현한다. 운영 중 선택·육안 검증을 대체하지 않는다.
    """
    seeds = [load_quality(base_q)] + [load_quality(p) for p in var_qs]
    if len(seeds) < 2 or not seeds[1] or any(set(s) != set(seeds[1]) for s in seeds[2:]) \
            or set(seeds[1]) - set(seeds[0]):
        raise ValueError("추가 시드의 평가 대상이 비었거나 시드별 결과가 누락됐습니다.")
    pairs = sorted(set.intersection(*(set(s) for s in seeds)))
    first, second = seeds[0], seeds[1]
    retry_pairs = [p for p in pairs if first[p]["retry"]]
    def key(row):
        penalty = sum(1.0 if c["retryable"] else 0.25 for c in row["checks"] if not c["passed"])
        return penalty, -(row["values"].get("sharpness") or 0.0)
    selected = {p: (second[p] if first[p]["retry"] and key(second[p]) < key(first[p]) else first[p])
                for p in pairs}
    structural = [p for p in pairs if first[p]["failed"] and not first[p]["retry"]]
    agreement = float(np.mean([len({s[p]["retry"] for s in seeds}) == 1 for p in pairs]))
    spread = {}
    for name in ("torso_skin_inside_mask", "garment_coverage", "color_fidelity", "reference_fidelity"):
        stds = [np.std([s[p]["values"][name] for s in seeds]) for p in pairs
                if all(name in s[p]["values"] for s in seeds)]
        if stds:
            spread[name] = {"pairs": len(stds), "median_std": float(np.median(stds)), "p90_std": float(np.percentile(stds, 90))}
    return {
        "pairs": len(pairs), "seeds": len(seeds),
        "retry_decision_agreement_all_seeds": agreement,
        "seed42_retry_pairs": len(retry_pairs),
        "retry_flags_cleared_in_second_attempt": sum(not second[p]["retry"] for p in retry_pairs),
        "selected_all_checks_passed_after_retry": sum(not selected[p]["failed"] for p in retry_pairs),
        "improved_penalty": sum(second[p]["penalty"] < first[p]["penalty"] for p in retry_pairs),
        "selected_retry_rate": float(np.mean([selected[p]["retry"] for p in pairs])),
        "selected_any_failure_rate": float(np.mean([bool(selected[p]["failed"]) for p in pairs])),
        "seed42_retry_rate": float(np.mean([first[p]["retry"] for p in pairs])),
        "structural_pairs": len(structural),
        "structural_persisting_in_second_seed": sum(any(not c["passed"] and not c["retryable"]
                                                      for c in second[p]["checks"]) for p in structural),
        "metric_spread": spread,
    }


def flag_summary(var_q: Path) -> dict:
    rows = load_quality(var_q, allow_unassessed=True)
    by_category = defaultdict(Counter)
    for row in rows.values():
        by_category[row["category"]]["pairs"] += 1
        if not row["assessed"]:
            by_category[row["category"]]["unassessed"] += 1
            continue
        by_category[row["category"]]["assessed"] += 1
        by_category[row["category"]]["any"] += bool(row["failed"])
        by_category[row["category"]]["retry"] += row["retry"]
        for name in row["failed"]:
            by_category[row["category"]][name] += 1
    return {category: dict(counter) for category, counter in by_category.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("agnostic", "lower-shape", "seeds", "flags"))
    ap.add_argument("--base", type=Path)
    ap.add_argument("--base-q", type=Path)
    ap.add_argument("--var")
    ap.add_argument("--var-q")
    ap.add_argument("--refs", type=Path)
    opts = ap.parse_args()
    if opts.mode == "agnostic":
        result = compare_agnostic(paired(opts.base, opts.base_q, Path(opts.var), Path(opts.var_q), "top"))
    elif opts.mode == "lower-shape":
        refs = {r["product_id"]: r for r in json.loads(opts.refs.read_text(encoding="utf-8"))}
        result = compare_lower_shape(paired(opts.base, opts.base_q, Path(opts.var), Path(opts.var_q), "bottom"), refs)
    elif opts.mode == "seeds":
        result = compare_seeds(opts.base_q, [Path(p) for p in opts.var_q.split(",")])
    else:
        result = flag_summary(Path(opts.var_q))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
