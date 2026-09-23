"""Aggregate an eval_body_visibility.py run: per-signal cost, policy variants, sensitivity.

Recomputes policy variants from the recorded evidence, so comparing candidate
rules needs no second GPU run. Nothing here re-labels a photo.

usage: report_body_visibility.py RUN_DIR [--categories lower_categories.json]
  RUN_DIR: eval_body_visibility.py --output directory (predictions.jsonl)
  --categories: {"<id>": ["pants", "skirt", ...]} to split skirts out of a
                truth-labelled sample. Without it, 'truth' alone is used, and a
                tight skirt then counts as a body-revealing photo.
"""
import argparse
import collections
import json
import statistics
import sys
from pathlib import Path

LOOSE = ("오버", "여유", "루즈", "와이드", "배기", "벌룬", "플레어")
OUTER = ("코트", "패딩", "다운", "판초")
SKIRT_KINDS = ("skirt", "dress", "jumpsuit")
REVEALING, OCCLUDING = "몸선 드러나는 바지·상의", "헐렁한 바지·상의"
SKIRTY = "치마·원피스"


def signal(row, name, skirt_threshold=0.01):
    """The four candidate blocking signals, read from what the run recorded."""
    if name == "skirt":
        return (row.get("skirt_fraction") or 0) >= skirt_threshold
    if name in ("trained_loose", "mask_loose"):
        trained = name == "trained_loose"
        return any(any(word in (row.get(key) or "") for word in LOOSE)
                   and ((row.get(f"{key}_source") in {"trained_head", "fused_agreement"}) == trained)
                   for key in ("fit", "lower_fit"))
    if name == "outer":
        return any(word in f"{row.get('outer_category', '')} {row.get('upper_type', '')}" for word in OUTER)
    raise ValueError(name)


def load(run_dir: Path, categories: dict):
    rows = [json.loads(line) for line in (run_dir/"predictions.jsonl").read_text(encoding="utf-8").splitlines()]
    measured = [row for row in rows if "error" not in row and row.get("status")]
    failures = collections.Counter(row.get("error", "")[:12] for row in rows if "error" in row)
    print(f"측정 {len(measured)} / 표본 {len(rows)} (실패 {len(rows) - len(measured)}: {failures.most_common(3)})")
    for row in measured:
        row["case"] = case_of(row, categories.get(row["id"], []))
    unclassified = sum(row["case"] is None for row in measured)
    if unclassified:
        # 사람 핏 라벨도, 수집 그룹도 없는 사진은 오차단·탐지 집계에서 빼야 한다.
        # 넣으면 실제로 가려진 사진이 '정상 오차단'으로 세어진다.
        print(f"  (핏 정답이 없어 신호 집계에서 제외: {unclassified}장)")
    return measured


def case_of(row, kinds):
    """가림 여부의 정답. 사람 핏 라벨이 있으면 그것을, 없으면 수집 그룹을 쓴다."""
    if kinds:
        return SKIRTY if any(kind in SKIRT_KINDS for kind in kinds) else (
            OCCLUDING if row.get("truth") == "loose" else REVEALING)
    truth = row.get("truth")
    if truth:
        return SKIRTY if truth == "dress" else OCCLUDING if truth == "loose" else REVEALING
    group = row.get("group")
    if group in ("skirt", "dress"):
        return SKIRTY
    # 'outer' 는 몸선이 드러나야 하는지 자체가 애매하므로 정답으로 쓰지 않는다.
    return REVEALING if group in ("top", "knit", "hoodie", "pants", "shorts") else None


def premise(measured):
    """Does clothing actually move the numbers the body-shape classifier reads?"""
    print("\n== 옷이 체형 측정을 바꾸는가 ==")
    for case in (REVEALING, OCCLUDING, SKIRTY):
        # 측정값이 없는 예전 실행 결과에서는 이 표를 건너뛴다.
        rows = [row for row in measured if row["case"] == case and row.get("shape")]
        if not rows:
            continue
        shapes = collections.Counter(row["shape"] for row in rows).most_common(3)
        print(f"{case:16s} n={len(rows):3d} 허리/가슴 {statistics.median(r['waist_chest'] for r in rows):.3f}"
              f" 골반/가슴 {statistics.median(r['hip_chest'] for r in rows):.3f} 체형 {shapes}")


def signals(measured):
    groups = {case: [row for row in measured if row["case"] == case] for case in (REVEALING, OCCLUDING, SKIRTY)}
    print("\n== 신호별 (몸선 드러나는 사진 오차단 / 헐렁 탐지 / 치마 탐지) ==")
    for name in ("skirt", "trained_loose", "mask_loose", "outer"):
        print(f"  {name:14s} " + "  ".join(
            f"{label} {sum(signal(r, name) for r in rows) / max(1, len(rows)):5.0%}"
            for label, rows in (("오차단", groups[REVEALING]), ("헐렁", groups[OCCLUDING]), ("치마", groups[SKIRTY]))))

    print("\n== 치마 면적 임계값 ==")
    for threshold in (0.01, 0.10, 0.20, 0.30, 0.40, 0.50):
        print(f"  {threshold:.2f} 치마·원피스 {sum(signal(r, 'skirt', threshold) for r in groups[SKIRTY]) / max(1, len(groups[SKIRTY])):5.0%}"
              f"  정상 오차단 {sum(signal(r, 'skirt', threshold) for r in groups[REVEALING]) / max(1, len(groups[REVEALING])):5.1%}")

    print("\n== 정책 조합 ==")
    variants = (("초기 PR #48", dict(skirt=0.01, trained=True, mask=True, outer=True)),
                ("보류 통과(채택)", dict(skirt=0.01, trained=True, mask=False, outer=True)),
                ("보류 통과+치마30%", dict(skirt=0.30, trained=True, mask=False, outer=True)),
                ("치마30%+외투만", dict(skirt=0.30, trained=False, mask=False, outer=True)))
    for name, options in variants:
        def blocked(row, options=options):
            return (signal(row, "skirt", options["skirt"])
                    or (options["trained"] and signal(row, "trained_loose"))
                    or (options["mask"] and signal(row, "mask_loose"))
                    or (options["outer"] and signal(row, "outer")))
        print(f"  {name:18s} " + "  ".join(
            f"{label} {sum(blocked(r) for r in rows) / max(1, len(rows)):5.0%}"
            for label, rows in (("정상 오차단", groups[REVEALING]), ("헐렁 차단", groups[OCCLUDING]),
                                ("치마 차단", groups[SKIRTY]))))
    return groups


def sensitivity(revealing, classify):
    """Does the distortion the gate is meant to avoid change the answer at all?"""
    print("\n== 관찰된 왜곡을 입혔을 때 체형이 바뀌는 비율 (몸선 드러나는 사진) ==")
    for label, waist_gain, hip_gain in (("헐렁한 옷 수준(허리 +6%)", 0.06, 0.0),
                                        ("치마 수준(골반 +5%)", 0.0, 0.05),
                                        ("둘 다", 0.06, 0.05),
                                        ("측정 잡음 수준(+2%)", 0.02, 0.0)):
        flips = sum(classify(r["chest"], r["waist"], r["hip"])[0]
                    != classify(r["chest"], r["waist"] * (1 + waist_gain), r["hip"] * (1 + hip_gain))[0]
                    for r in revealing)
        print(f"  {label:22s} {flips:3d}/{len(revealing)} ({flips / max(1, len(revealing)):.0%})")


def blocked_examples(revealing):
    print("\n== 오차단의 정체 ==")
    outer = [r for r in revealing if signal(r, "outer")]
    print("  외투 신호:", collections.Counter(
        f"{r.get('outer_category', '')}|{r.get('upper_type', '')}" for r in outer).most_common(5))
    trained = [r for r in revealing if signal(r, "trained_loose")]
    print("  학습 핏 신호:", collections.Counter(
        f"{r.get('fit')}|{r.get('lower_fit')}" for r in trained).most_common(5))


def groups_table(measured):
    """Per-clothing-group verdicts, for samples collected by garment type."""
    if not any(row.get("group") for row in measured):
        return
    print("\n== 수집 그룹별 판정 ==")
    print(f"{'그룹':10s} {'수':>4s} {'통과':>8s} {'보류':>8s} {'차단':>8s}")
    for group in dict.fromkeys(row.get("group") for row in measured if row.get("group")):
        rows = [row for row in measured if row.get("group") == group]
        counts = collections.Counter(row["status"] for row in rows)
        print(f"{group:10s} {len(rows):4d} " + " ".join(
            f"{counts[status]:3d}({counts[status] / len(rows):4.0%})"
            for status in ("no_obvious_occlusion", "uncertain", "occluded")))


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("run_dir", type=Path)
    cli.add_argument("--categories", type=Path)
    cli.add_argument("--src", type=Path, help="ai_fashion_recommender/src, for the sensitivity check")
    args = cli.parse_args()
    categories = json.loads(args.categories.read_text(encoding="utf-8")) if args.categories else {}
    measured = load(args.run_dir, categories)
    if not measured:
        return
    groups_table(measured)
    premise(measured)
    groups = signals(measured)
    blocked_examples(groups[REVEALING])
    measurable = [row for row in groups[REVEALING] if row.get("shape")]
    if args.src and measurable:
        sys.path.insert(0, str(args.src))
        from body_shape import classify_from_circumferences
        sensitivity(measurable, classify_from_circumferences)


if __name__ == "__main__":
    main()
