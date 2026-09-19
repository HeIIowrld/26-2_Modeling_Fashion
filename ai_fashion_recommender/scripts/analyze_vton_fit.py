"""eval_vton_fit.py 결과를 집계한다 (reports/vton_quality/fit_preservation_2026-09-14.md). 신뢰구간은 사람 단위 클러스터 부트스트랩(2000회)."""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from vton_eval_utils import cluster_ci, is_short_bottom, load_dedup

HERE = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
RESULTS = HERE / (sys.argv[2] if len(sys.argv) > 2 else "results.jsonl")
REFS = HERE / (sys.argv[3] if len(sys.argv) > 3 else "refs.json")

SELLER = {"슬림핏": 0, "레귤러핏": 1, "여유핏": 2, "오버핏": 3}
UPPER_W = {"슬림핏": 0, "레귤러핏": 1, "여유핏": 2, "오버핏": 3}
LOWER_W = {"슬림핏": 0, "테이퍼드핏": 1, "스트레이트핏": 1, "플레어핏": 2, "와이드핏": 3}
FIT_TASK = {"top": ("upper_fit", UPPER_W), "bottom": ("lower_fit", LOWER_W)}
BAND = {"top": "chest", "bottom": "shin"}

rows, errors = load_dedup(RESULTS)
refs = {r["product_id"]: r for r in json.loads(REFS.read_text(encoding="utf-8"))}


def looseness(judge: dict, category: str) -> float | None:
    task, weights = FIT_TASK[category]
    scores = (judge or {}).get(task)
    if not scores:
        return None
    total = sum(scores.values())
    return sum(weights[k] * v for k, v in scores.items()) / total


def argmax(judge: dict, task: str) -> str | None:
    scores = (judge or {}).get(task)
    return max(scores, key=scores.get) if scores else None


def rank(values: np.ndarray) -> np.ndarray:
    order = values.argsort(kind="mergesort")
    ranks = np.empty(len(values))
    ranks[order] = np.arange(len(values))
    # 동점은 평균 순위
    for v in np.unique(values):
        idx = values == v
        ranks[idx] = ranks[idx].mean()
    return ranks


def spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0:
        return float("nan")
    return float(np.corrcoef(rank(x), rank(y))[0, 1])


def fmt(triple, digits=2):
    p, lo, hi = triple
    return f"{p:.{digits}f} [{lo:.{digits}f}, {hi:.{digits}f}]"


def within(items, xkey, ykey):
    """사람마다 Spearman을 구해 평균한다. 사람 간 체격·측정 구간 차이를 없앤다."""
    groups = defaultdict(list)
    for item in items:
        if item.get(xkey) is not None and item.get(ykey) is not None:
            groups[item["person"]].append(item)
    values = [spearman([g[xkey] for g in grp], [g[ykey] for g in grp]) for grp in groups.values()]
    values = [v for v in values if v == v]
    return float(np.mean(values)) if values else float("nan")


def ols(y, *xs):
    """표준화 회귀계수."""
    X = np.column_stack([(np.asarray(x, float) - np.mean(x)) / (np.std(x) or 1) for x in xs])
    Y = (np.asarray(y, float) - np.mean(y)) / (np.std(y) or 1)
    X = np.column_stack([np.ones(len(Y)), X])
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return beta[1:]


print(f"# 결과 {len(rows)}쌍 · 오류 {len(errors)}쌍 · 사람 {len({r['person'] for r in rows})}명 · "
      f"상품 {len({r['product_id'] for r in rows})}개")
for e in errors[:5]:
    print("  오류:", e["pair"], e["error"])

report: dict = {}
for category in ("top", "bottom"):
    items = [r for r in rows if r["category"] == category]
    if not items:
        continue
    task, _ = FIT_TASK[category]
    for r in items:
        r["r"] = SELLER[r["seller_fit"]]
        r["E_ref"] = looseness(r["judge_ref"], category)
        r["E_orig"] = looseness(r["judge_orig"], category)
        r["E_res"] = looseness(r["judge_res"], category)
        band = BAND[category]
        # 사진이 무릎 아래에서 잘리면 정강이 폭이 없다 → 허벅지 폭으로 대체
        if category == "bottom" and r["width_orig"].get("shin") is None:
            band = "thigh"
        r["band"] = band
        wo, wr = r["width_orig"].get(band), r["width_res"].get(band)
        wm = r["width_res"].get(band + "_mask")
        r["W_res"], r["W_orig"] = wr, wo
        r["dW"] = None if wo is None or wr is None else wr - wo
        r["fill"] = None if wr is None or not wm else wr / wm
    # 판매자 상품명·카테고리의 반바지는 기장까지 바뀌어 핏 판단을 흐린다 → 핏 지표에서만 뺀다.
    shorts = {pid for pid, v in refs.items() if v["category"] == category and is_short_bottom(v)}
    for r in items:
        r["is_shorts"] = r["product_id"] in shorts
    fit_items = [r for r in items if None not in (r["E_ref"], r["E_orig"], r["E_res"]) and not r["is_shorts"]]
    if shorts:
        print(f"(핏 지표에서 반바지 {len(shorts)}개 제외: {sorted(shorts)})")

    print(f"\n## {category} — {len(items)}쌍 (핏 판정 가능 {len(fit_items)})")

    # 1) 채점자(속성 헤드)가 레퍼런스 사진에서 판매자 태그를 얼마나 읽나 — 천장
    ref_rows = [v for v in refs.values() if v["category"] == category and not is_short_bottom(v)]
    ref_E = [(SELLER[v["seller_fit"]], looseness(v["judge"], category)) for v in ref_rows if v["judge"]]
    print(f"[레퍼런스 상관 기준: 수학적 천장 아님] 레퍼런스 {len(ref_E)}개: Spearman(판매자 태그, 레퍼런스 판정) = "
          f"{spearman(*zip(*ref_E)):.2f}")
    if category == "top":
        acc = np.mean([argmax(v["judge"], task) == v["seller_fit"] for v in ref_rows if v["judge"]])
        print(f"            레퍼런스 argmax 정확도 = {acc:.2f} (4클래스 우연 0.25)")

    # 2) 결과 핏이 레퍼런스를 따르나, 원래 옷을 따르나
    s_res = cluster_ci(fit_items, lambda s: spearman([x["r"] for x in s], [x["E_res"] for x in s]))
    s_ref = cluster_ci(fit_items, lambda s: spearman([x["r"] for x in s], [x["E_ref"] for x in s]))
    s_orig = cluster_ci(fit_items, lambda s: spearman([x["E_orig"] for x in s], [x["E_res"] for x in s]))
    s_refres = cluster_ci(fit_items, lambda s: spearman([x["E_ref"] for x in s], [x["E_res"] for x in s]))
    print(f"[전이] Spearman(판매자 태그, 결과 판정)   = {fmt(s_res)}")
    print(f"       Spearman(판매자 태그, 레퍼런스 판정) = {fmt(s_ref)}  ← 같은 채점자의 천장")
    print(f"       Spearman(레퍼런스 판정, 결과 판정)   = {fmt(s_refres)}")
    print(f"[누수] Spearman(원래 옷 판정, 결과 판정)   = {fmt(s_orig)}")
    w_res = cluster_ci(fit_items, lambda s: within(s, "r", "E_res"))
    w_ref = cluster_ci(fit_items, lambda s: within(s, "r", "E_ref"))
    print(f"[사람 안] 평균 Spearman(판매자 태그, 결과 판정) = {fmt(w_res)} · (태그, 레퍼런스 판정) = {fmt(w_ref)}")
    b_ref, b_orig = ols([x["E_res"] for x in fit_items], [x["E_ref"] for x in fit_items],
                        [x["E_orig"] for x in fit_items])
    print(f"[회귀] 결과 ~ 레퍼런스 + 원래 옷 (표준화 계수): 레퍼런스 {b_ref:.2f} / 원래 옷 {b_orig:.2f}")
    retained = cluster_ci(fit_items, lambda s: (
        spearman([x["r"] for x in s], [x["E_res"] for x in s])
        / spearman([x["r"] for x in s], [x["E_ref"] for x in s])))
    print(f"[보존율] 결과 상관 / 천장 상관 = {fmt(retained)}")

    # 차이가 큰 쌍만: 결과가 어느 쪽에 더 가깝나
    contrast = [x for x in fit_items if abs(x["E_ref"] - x["E_orig"]) >= 0.75]
    if contrast:
        follow = cluster_ci(contrast, lambda s: float(np.mean(
            [abs(x["E_res"] - x["E_ref"]) < abs(x["E_res"] - x["E_orig"]) for x in s])))
        print(f"[대조쌍] |레퍼런스−원래| ≥ 0.75인 {len(contrast)}쌍 중 결과가 레퍼런스 쪽에 더 가까운 비율 = "
              f"{fmt(follow)}")
    if category == "top":
        agree = cluster_ci(fit_items, lambda s: float(np.mean(
            [argmax(x["judge_res"], task) == x["seller_fit"] for x in s])))
        agree_orig = cluster_ci(fit_items, lambda s: float(np.mean(
            [argmax(x["judge_res"], task) == argmax(x["judge_orig"], task) for x in s])))
        print(f"[일치] 결과 argmax = 판매자 태그 {fmt(agree)} / 결과 argmax = 원래 옷 argmax {fmt(agree_orig)}")

    # 판매자 태그별 평균
    print("[태그별] 판매자 태그 | n | 레퍼런스 판정 | 결과 판정 | 원래 옷 판정 | 폭 변화 | 마스크 채움")
    for fit, code in SELLER.items():
        group = [x for x in items if x["r"] == code and not x["is_shorts"]]
        if not group:
            continue
        mean = lambda k: np.nanmean([x[k] for x in group if x[k] is not None]) if any(x[k] is not None for x in group) else float("nan")  # noqa: E731
        print(f"   {fit:6s} | {len(group):3d} | {mean('E_ref'):.2f} | {mean('E_res'):.2f} | "
              f"{mean('E_orig'):.2f} | {mean('dW'):+.3f} | {mean('fill'):.2f}")

    # 3) 분류기 없이: 폭
    geo = [x for x in items if x["dW"] is not None and x["fill"] is not None and not x["is_shorts"]]
    s_geo = cluster_ci(geo, lambda s: spearman([x["r"] for x in s], [x["dW"] for x in s]))
    s_geo_abs = cluster_ci(geo, lambda s: spearman([x["r"] for x in s],
                                                   [x["W_res"] for x in s]))
    w_geo = cluster_ci(geo, lambda s: within(s, "r", "W_res"))
    print(f"[기하·사람 안] 평균 Spearman(판매자 태그, 결과 폭) = {fmt(w_geo)}")
    capped = cluster_ci(geo, lambda s: float(np.mean([x["fill"] >= 0.9 for x in s])))
    print(f"[기하] {'가슴(소매가 섞임 · 보조 지표)' if category == 'top' else '정강이(없으면 허벅지)'} 폭: Spearman(판매자 태그, 결과 폭) = {fmt(s_geo_abs)} · "
          f"Spearman(판매자 태그, 폭 변화) = {fmt(s_geo)}")
    print(f"       결과 옷 폭이 마스크 폭의 90% 이상(마스크에 막힘) = {fmt(capped)}")
    loose = [x for x in geo if x["r"] >= 2]
    if loose:
        cap_loose = cluster_ci(loose, lambda s: float(np.mean([x["fill"] >= 0.9 for x in s])))
        print(f"       └ 여유·오버 태그만: {fmt(cap_loose)}")
    # 원래 옷 폭 vs 결과 폭 (사람 효과)
    s_w_orig = cluster_ci(geo, lambda s: spearman([x["W_orig"] for x in s],
                                                  [x["W_res"] for x in s]))
    print(f"       Spearman(원래 옷 폭, 결과 폭) = {fmt(s_w_orig)}")

    # 4) 품질
    def dist(key, digits=3, subset=None):
        vals = np.array([x[key] for x in (subset or items) if x.get(key) is not None], float)
        if len(vals) == 0:
            return "없음"
        return (f"중앙 {np.median(vals):.{digits}f} · p10 {np.percentile(vals, 10):.{digits}f} · "
                f"p90 {np.percentile(vals, 90):.{digits}f} · 최소 {vals.min():.{digits}f} (n={len(vals)})")

    print("[보존] 마스크 밖 PSNR:", dist("outside_psnr", 1))
    print("       마스크 밖 SSIM:", dist("outside_ssim"))
    print("       마스크 밖 변화 픽셀 비율:", dist("changed_outside_ratio", 4))
    print("       얼굴·머리 PSNR:", dist("face_psnr", 1))
    sim_gain = [x for x in items if x.get("sim_res_ref") is not None and x.get("sim_orig_ref") is not None]
    gain = cluster_ci(sim_gain, lambda s: float(np.mean([x["sim_res_ref"] - x["sim_orig_ref"] for x in s]))) if sim_gain else None
    print("[충실] 결과↔레퍼런스 코사인:", dist("sim_res_ref"))
    print("       원래 옷↔레퍼런스 코사인:", dist("sim_orig_ref"))
    if gain:
        print(f"       합성으로 오른 코사인 평균 = {fmt(gain, 3)}")
    ranked = [x for x in items if x.get("rank_res")]
    if ranked:
        pool = ranked[0]["pool_size"]
        t1 = cluster_ci(ranked, lambda s: float(np.mean([x["rank_res"] == 1 for x in s])))
        t3 = cluster_ci(ranked, lambda s: float(np.mean([x["rank_res"] <= 3 for x in s])))
        print(f"       {pool}개 중 자기 레퍼런스 식별: top-1 {fmt(t1)} (우연 {1/pool:.2f}) · "
              f"top-3 {fmt(t3)} (우연 {3/pool:.2f}) · 평균 순위 {np.mean([x['rank_res'] for x in ranked]):.1f}")
    print("[색]   ΔE00 결과↔레퍼런스:", dist("color_de_res", 1), "| 원래 옷↔레퍼런스:", dist("color_de_orig", 1))
    de_bad = cluster_ci([x for x in items if x.get("color_de_res") is not None],
                        lambda s: float(np.mean([x["color_de_res"] > 20 for x in s])))
    print(f"       ΔE00 > 20 (다른 색으로 보임) 비율 = {fmt(de_bad)}")
    print("[구조] 마스크 안 옷 채움:", dist("target_fill"))
    print("       옷이 마스크 밖으로 나간 비율:", dist("garment_outside_edit", 4))
    print("       원래 옷 자리에 드러난 피부:", dist("skin_in_orig"))
    print("       옷 조각 수(1%↑):", Counter(x["garment_components"] for x in items).most_common())
    print("       선명도 결과/원래 (참고: 원래 옷이 무지면 무의미):", dist("sharp_ratio"))
    print("[시간] 합성 초:", dist("synth_sec", 1), "| 쌍당 전체 초:", dist("total_sec", 1))

    # 5) 품질 게이트 경고
    kinds = Counter()
    for x in items:
        for w in x["warnings"]:
            kinds[re.sub(r"\(.*?\)|[0-9.]+", "", w)[:48]] += 1
    warned = [x for x in items if x["warnings"]]
    print(f"[경고] 경고가 난 쌍 {len(warned)}/{len(items)}")
    for k, v in kinds.most_common(8):
        print(f"       {v:4d}× {k}")
    clean = [x for x in items if not x["warnings"]]
    if warned and clean:
        for key in ("sim_res_ref", "color_de_res", "skin_in_orig", "target_fill"):
            a = [x[key] for x in warned if x.get(key) is not None]
            b = [x[key] for x in clean if x.get(key) is not None]
            if a and b:
                print(f"       {key}: 경고 {np.median(a):.3f} vs 무경고 {np.median(b):.3f}")
    report[category] = items

# 사람별 요약 (누수 확인용)
print("\n## 사람별 결과 판정 평균 (핏 헤드 looseness)")
for person in sorted({r["person"] for r in rows}):
    for category in ("top", "bottom"):
        group = [r for r in rows if r["person"] == person and r["category"] == category and r.get("E_res") is not None]
        if group:
            print(f"  {person:14s} {category:6s} 원래 {group[0]['E_orig']:.2f} → 결과 평균 "
                  f"{np.mean([g['E_res'] for g in group]):.2f} (sd {np.std([g['E_res'] for g in group]):.2f}) · "
                  f"Spearman(태그, 결과) {spearman([g['r'] for g in group], [g['E_res'] for g in group]):.2f}")


# ───────────── 마스크 대조 실험 (하의) ─────────────
WIDE = HERE / "results_wide.jsonl"
if WIDE.is_file():
    wide_rows, wide_errors = load_dedup(WIDE)
    if wide_errors:
        print(f"  (대조 실험 오류 {len(wide_errors)}쌍)")
    native = {r["pair"]: r for r in report.get("bottom", [])}
    matched = []
    for w in wide_rows:
        n = native.get(w["pair"])
        if not n or n["is_shorts"]:
            continue
        band = n["band"]
        if w["width_res"].get(band) is None:
            continue
        matched.append({"person": w["person"], "r": n["r"], "seller_fit": n["seller_fit"],
                        "W_native": n["W_res"], "W_wide": w["width_res"][band],
                        "M_native": n["width_res"].get(band + "_mask"), "M_wide": w["width_res"].get(band + "_mask"),
                        "W_orig": n["W_orig"],
                        "E_native": n["E_res"], "E_wide": looseness(w["judge_res"], "bottom"),
                        "psnr_wide": w["outside_psnr"], "sim_wide": w.get("sim_res_ref"), "sim_native": n.get("sim_res_ref"),
                        "de_wide": w.get("color_de_res"), "de_native": n.get("color_de_res"),
                        "rank_wide": w.get("rank_res"), "rank_native": n.get("rank_res")})
    print(f"\n## 마스크 대조 실험 (하의, 같은 쌍·같은 시드) — {len(matched)}쌍")
    print("   판매자 태그 | n | 결과 폭(운영 마스크) | 결과 폭(넓힌 마스크) | 마스크 폭 운영→넓힘 | 핏 판정 운영→넓힘")
    for fit, code in SELLER.items():
        g = [m for m in matched if m["r"] == code]
        if g:
            avg = lambda k: np.mean([m[k] for m in g if m[k] is not None])  # noqa: E731
            print(f"   {fit:6s} | {len(g):3d} | {avg('W_native'):.3f} | {avg('W_wide'):.3f} | "
                  f"{avg('M_native'):.2f}→{avg('M_wide'):.2f} | {avg('E_native'):.2f}→{avg('E_wide'):.2f}")
    for label, key in (("운영 마스크", "W_native"), ("넓힌 마스크", "W_wide")):
        print(f"   Spearman(판매자 태그, 결과 폭) {label}: "
              f"{fmt(cluster_ci(matched, lambda s: spearman([m['r'] for m in s], [m[key] for m in s])))}")
    for label, key in (("운영 마스크", "W_native"), ("넓힌 마스크", "W_wide")):
        print(f"   사람 안 평균 Spearman(판매자 태그, 결과 폭) {label}: "
              f"{fmt(cluster_ci(matched, lambda s: within(s, 'r', key)))}")
    for label, key in (("운영 마스크", "E_native"), ("넓힌 마스크", "E_wide")):
        ok = [m for m in matched if m[key] is not None]
        print(f"   Spearman(판매자 태그, 결과 핏 판정) {label}: "
              f"{fmt(cluster_ci(ok, lambda s: spearman([m['r'] for m in s], [m[key] for m in s])))}")
    print(f"   Spearman(원래 옷 폭, 결과 폭) 운영 {fmt(cluster_ci(matched, lambda s: spearman([m['W_orig'] for m in s], [m['W_native'] for m in s])))}"
          f" / 넓힘 {fmt(cluster_ci(matched, lambda s: spearman([m['W_orig'] for m in s], [m['W_wide'] for m in s])))}")
    loose_gain = cluster_ci([m for m in matched if m["r"] >= 2], lambda s: float(np.mean([m["W_wide"] - m["W_native"] for m in s])))
    slim_gain = cluster_ci([m for m in matched if m["r"] <= 1], lambda s: float(np.mean([m["W_wide"] - m["W_native"] for m in s])))
    print(f"   넓힌 마스크로 늘어난 폭: 여유·오버 {fmt(loose_gain, 3)} / 슬림·레귤러 {fmt(slim_gain, 3)}")
    print(f"   대가 — 결과↔레퍼런스 코사인 중앙 운영 {np.median([m['sim_native'] for m in matched if m['sim_native']]):.3f} → 넓힘 {np.median([m['sim_wide'] for m in matched if m['sim_wide']]):.3f} · "
          f"ΔE00 중앙 {np.median([m['de_native'] for m in matched if m['de_native'] is not None]):.1f} → {np.median([m['de_wide'] for m in matched if m['de_wide'] is not None]):.1f} · "
          f"top-1 식별 {np.mean([m['rank_native']==1 for m in matched]):.2f} → {np.mean([m['rank_wide']==1 for m in matched]):.2f} · "
          f"마스크 밖 PSNR 중앙 {np.median([m['psnr_wide'] for m in matched]):.1f}")


# ───────────── 분산 분해 · 경고 분포 · 최악 사례 ─────────────
def eta2(items, key, factor):
    """균형 설계에서 한 요인이 설명하는 분산 비율."""
    vals = [x for x in items if x.get(key) is not None]
    y = np.array([x[key] for x in vals], float)
    if len(y) < 4:
        return float("nan")
    groups = defaultdict(list)
    for x in vals:
        groups[x[factor]].append(x[key])
    ss_between = sum(len(g) * (np.mean(g) - y.mean()) ** 2 for g in groups.values())
    return float(ss_between / ((y - y.mean()) ** 2).sum())


print("\n## 결과를 결정하는 것은 사람인가 상품인가 (분산 설명 비율 η²)")
for category, items in report.items():
    items = [x for x in items if not x["is_shorts"]]
    shin = [x for x in items if category == "top" or x["band"] == "shin"]
    for key, label, pool in (("E_res", "결과 핏 판정", items), ("W_res", "결과 폭", shin)):
        print(f"  {category:6s} {label}: 사람 {eta2(pool, key, 'person'):.2f} · 상품 {eta2(pool, key, 'product_id'):.2f}"
              f"  (n={len(pool)}{', 정강이 구간만' if key == 'W_res' and category == 'bottom' else ''})")
    print(f"  {category:6s} 참고 — 레퍼런스 판정: 상품 {eta2(items, 'E_ref', 'product_id'):.2f}")

print("\n## 경고가 누구에게 났나")
for category, items in report.items():
    c = Counter()
    for x in items:
        for w in x["warnings"]:
            c[(x["person"], w.split(":")[0][:20])] += 1
    for (person, kind), v in sorted(c.items()):
        print(f"  {category:6s} {person:14s} {kind:22s} {v}")

print("\n## 최악 사례")
for category, items in report.items():
    for key, reverse, label in (("sim_res_ref", False, "레퍼런스와 가장 안 닮음"),
                                ("color_de_res", True, "색이 가장 다름"),
                                ("skin_in_orig", True, "원래 옷 자리 피부 노출 최대"),
                                ("target_fill", False, "마스크 안 옷 채움 최소")):
        vals = sorted([x for x in items if x.get(key) is not None], key=lambda x: x[key], reverse=reverse)[:3]
        print(f"  {category:6s} {label}: " + " | ".join(f"{x['pair']}({x['seller_fit']},{x['item_type']}) {x[key]}" for x in vals))


# ───────────── 이름 기반 와이드 라벨로 본 마스크 효과 (가장 직접적인 검증) ─────────────
# 판매자 태그는 서열이 거칠고 하의 채점자 천장이 낮다. 상품명에 '와이드·벌룬·배기·커브드'가 있는지는
# 폭에 대한 독립적이고 명확한 라벨이다. 같은 사람 안에서 두 묶음의 결과 폭 차이를 본다.
if WIDE.is_file():
    WIDE_WORDS = ("와이드", "벌룬", "배기", "커브드")
    wide_by_pair = {r["pair"]: r for r in wide_rows}
    named = []
    for n in report.get("bottom", []):
        if n["is_shorts"] or n["W_res"] is None:
            continue
        w = wide_by_pair.get(n["pair"])
        named.append({"person": n["person"], "wide": any(t in refs[n["product_id"]]["name"] for t in WIDE_WORDS),
                      "native": n["W_res"], "ablation": w["width_res"].get(n["band"]) if w else None})

    def gap(sample, key):
        groups = defaultdict(list)
        for x in sample:
            if x[key] is not None:
                groups[x["person"]].append(x)
        diffs = []
        for g in groups.values():
            a = [x[key] for x in g if x["wide"]]
            b = [x[key] for x in g if not x["wide"]]
            if a and b:
                diffs.append(np.mean(a) - np.mean(b))
        return float(np.mean(diffs)) if diffs else float("nan")

    print(f"\n## 이름 기반 라벨: 같은 사람 안에서 (와이드 계열 − 나머지 긴바지) 결과 폭 차이")
    print(f"   운영 마스크: {fmt(cluster_ci(named, lambda s: gap(s, 'native')), 3)}")
    print(f"   넓힌 마스크: {fmt(cluster_ci(named, lambda s: gap(s, 'ablation')), 3)}")
