#!/usr/bin/env python
"""체형 분류 경계값(`data/body_shape_reference.json`)을 사진 표본에서 다시 만든다.

경계는 절대값이 아니라 **기준 분포의 33·67 백분위**다. 그래서 표본을 바꾸면
코드를 고치지 않고 이 스크립트만 다시 돌리면 된다.

측정값은 `shoulder_hip_ratio` — MediaPipe `pose_world_landmarks`(3D 미터 좌표)의
어깨 관절 간격 ÷ 골반 관절 간격이다. **신체 표면 치수가 아니다.** Size Korea 의
어깨사이길이·엉덩이너비를 그대로 옮겨 쓸 수 없는 이유가 이것이다.

표본 필터는 기존 기준표와 동일하게 맞춘다(그래야 이전 값과 비교할 수 있다):
  · 전신 인식 성공 (`valid`)
  · 정면에 가까운 자세 (`posture == "정면에 가까움"`)
  · `full_body_score >= 0.85`

사용법
------
    # 어떤 값이 나오는지만 보기 (파일 안 건드림)
    python scripts/calibrate_body_shape.py --images data/fashionpedia_seed/images --dry-run

    # 여러 폴더를 합쳐서 기준표 갱신
    python scripts/calibrate_body_shape.py \
        --images datasets/people data/fashionpedia_seed/images \
        --source "한국인 전신사진 (자체 촬영)" --write

한국인 표본으로 갈아끼울 때
--------------------------
`--images` 에 한국인 전신사진 폴더를 주고 `--source` 를 그에 맞게 적은 뒤
`--write` 하면 끝이다. 코드 수정은 필요 없다.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import DATA_DIR  # noqa: E402
from pose_analyzer import PoseAnalyzer  # noqa: E402

REFERENCE_PATH = DATA_DIR / "body_shape_reference.json"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
# 기존 기준표와 같은 백분위 지점을 낸다. 33·67 만 경계로 쓰이고 나머지는 분포 확인용이다.
PERCENTILES = (5, 10, 25, 33, 50, 67, 75, 90, 95)
MIN_SAMPLE = 200


def collect_images(roots: list[Path]) -> list[Path]:
    found: list[Path] = []
    for root in roots:
        if root.is_file():
            found.append(root)
            continue
        if not root.is_dir():
            print(f"  건너뜀 — 폴더가 없다: {root}")
            continue
        found.extend(
            p for p in sorted(root.rglob("*")) if p.suffix.lower() in IMAGE_SUFFIXES
        )
    # 같은 파일이 여러 경로로 잡히지 않게 한다
    seen: set[Path] = set()
    unique = []
    for p in found:
        resolved = p.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique.append(p)
    return unique


def measure(images: list[Path], min_full_body: float, progress_every: int) -> dict:
    """필터를 통과한 사진의 shoulder_hip_ratio 를 모은다."""
    ratios: list[float] = []
    reject = {"분석 실패": 0, "전신 아님": 0, "정면 아님": 0, "점수 미달": 0}
    used: list[str] = []

    with PoseAnalyzer() as analyzer:
        for i, path in enumerate(images, 1):
            if progress_every and i % progress_every == 0:
                print(f"  {i}/{len(images)} 처리 · 채택 {len(ratios)}", flush=True)
            try:
                pose = analyzer.analyze(path)
            except Exception:
                reject["분석 실패"] += 1
                continue
            if not pose.valid:
                reject["전신 아님"] += 1
                continue
            if pose.posture != "정면에 가까움":
                reject["정면 아님"] += 1
                continue
            if pose.full_body_score < min_full_body:
                reject["점수 미달"] += 1
                continue
            ratios.append(float(pose.shoulder_hip_ratio))
            used.append(path.name)

    return {"ratios": ratios, "reject": reject, "used": used}


def summarize(ratios: list[float]) -> dict:
    arr = np.asarray(ratios, dtype=float)
    return {
        "percentiles": {str(p): round(float(np.percentile(arr, p)), 4) for p in PERCENTILES},
        "mean": round(float(arr.mean()), 4),
        "std": round(float(arr.std(ddof=1)), 4) if arr.size > 1 else 0.0,
        "min": round(float(arr.min()), 4),
        "max": round(float(arr.max()), 4),
    }


def bootstrap_ci(ratios: list[float], pct: int, rounds: int = 2000, seed: int = 0) -> tuple[float, float]:
    """경계값이 표본에 얼마나 흔들리는지 본다. 좁으면 표본이 충분하다는 뜻이다."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(ratios, dtype=float)
    draws = np.percentile(rng.choice(arr, size=(rounds, arr.size), replace=True), pct, axis=1)
    return round(float(np.percentile(draws, 2.5)), 4), round(float(np.percentile(draws, 97.5)), 4)


def main() -> int:
    ap = argparse.ArgumentParser(description="체형 분류 경계값 재보정")
    ap.add_argument("--images", nargs="+", required=True, help="사진 폴더 또는 파일")
    ap.add_argument("--source", default="", help="기준표에 기록할 표본 출처 설명")
    ap.add_argument("--min-full-body", type=float, default=0.85, help="기본 0.85 — 기존 기준표와 동일")
    ap.add_argument("--limit", type=int, default=0, help="앞에서 N장만 (시험용)")
    ap.add_argument("--progress-every", type=int, default=250)
    ap.add_argument("--write", action="store_true", help="기준표 파일을 실제로 갱신")
    ap.add_argument("--dry-run", action="store_true", help="계산만 하고 파일은 그대로 둔다")
    args = ap.parse_args()

    if args.write and args.dry_run:
        print("--write 와 --dry-run 을 같이 줄 수 없다.")
        return 2
    if not args.write and not args.dry_run:
        print("--write 또는 --dry-run 중 하나를 지정해야 한다.")
        return 2

    roots = [Path(a) if Path(a).is_absolute() else (PROJECT_DIR / a) for a in args.images]
    images = collect_images(roots)
    if args.limit:
        images = images[: args.limit]
    print(f"이미지 {len(images)}장에서 측정을 시작한다.")
    if not images:
        print("측정할 이미지가 없다.")
        return 1

    result = measure(images, args.min_full_body, args.progress_every)
    ratios = result["ratios"]

    print()
    print(f"채택 {len(ratios)}장 / 전체 {len(images)}장 ({len(ratios) / len(images) * 100:.1f}%)")
    for reason, count in result["reject"].items():
        print(f"  제외 · {reason}: {count}")

    if not ratios:
        print("\n필터를 통과한 사진이 없다. --min-full-body 를 낮추거나 사진을 확인할 것.")
        return 1

    stats = summarize(ratios)
    print()
    print("분포")
    print(f"  평균 {stats['mean']}  표준편차 {stats['std']}  범위 {stats['min']} ~ {stats['max']}")
    for p in PERCENTILES:
        mark = "  ← 경계" if p in (33, 67) else ""
        print(f"  {p:>2}% : {stats['percentiles'][str(p)]}{mark}")

    if len(ratios) >= 30:
        lo33, hi33 = bootstrap_ci(ratios, 33)
        lo67, hi67 = bootstrap_ci(ratios, 67)
        print()
        print("경계값의 95% 부트스트랩 구간 (좁을수록 표본이 충분하다)")
        print(f"  33% : {stats['percentiles']['33']}  [{lo33} ~ {hi33}]  폭 {round(hi33 - lo33, 4)}")
        print(f"  67% : {stats['percentiles']['67']}  [{lo67} ~ {hi67}]  폭 {round(hi67 - lo67, 4)}")

    if len(ratios) < MIN_SAMPLE:
        print()
        print(f"주의 — 표본이 {len(ratios)}장으로 {MIN_SAMPLE}장 미만이다. 경계값이 흔들릴 수 있다.")

    if REFERENCE_PATH.is_file():
        old = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
        old_p = old.get("percentiles", {})
        print()
        print("기존 기준표와 비교")
        print(f"  표본  {old.get('sample', {}).get('n', '?')} → {len(ratios)}")
        for p in ("33", "67"):
            if p in old_p:
                delta = stats["percentiles"][p] - float(old_p[p])
                print(f"  {p}%   {old_p[p]} → {stats['percentiles'][p]}  ({delta:+.4f})")
    else:
        old = {}

    if args.dry_run:
        print()
        print("--dry-run 이라 파일은 건드리지 않았다.")
        return 0

    updated = dict(old)
    updated["percentiles"] = stats["percentiles"]
    updated["distribution"] = {k: stats[k] for k in ("mean", "std", "min", "max")}
    updated["sample"] = {
        "n": len(ratios),
        "source": args.source or "미기재 — --source 로 출처를 적을 것",
        "filter": f"전신 인식 + 정면 + full_body_score>={args.min_full_body}",
        "measured_on": date.today().isoformat(),
        "screened": len(images),
        "rejected": result["reject"],
    }
    REFERENCE_PATH.write_text(
        json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print()
    print(f"기준표를 갱신했다: {REFERENCE_PATH}")
    print("코드 수정은 필요 없다 — pose_analyzer 가 이 파일에서 경계를 읽는다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
