"""8단계 A — 2차 보강 선택. 1차에서 쓰지 않은 shard와 이미지 중에서 고른다.

1차(shard 0-1, 4,500장)로 붕괴 라벨은 살아났지만 대부분 200장대다.
2차는 목표를 800장으로 올리되, 이미 충분한 라벨만 든 이미지는 여전히 배제한다.
예산은 CPU 임베딩 속도(2.2 crop/s)에서 역산한다.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)
    parser.add_argument("--previous-selection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--target", type=int, default=800)
    parser.add_argument("--budget-images", type=int, default=6000)
    parser.add_argument("--min-image-id", type=int, default=15131,
                        help="shard 0-1 이후 구간만 사용 (1차와 겹치지 않게)")
    args = parser.parse_args()

    scan = json.loads(Path(args.scan).read_text(encoding="utf-8"))
    previous = set(json.loads(Path(args.previous_selection).read_text())["image_ids"])
    image_labels = {int(k): set(v) for k, v in scan["image_labels"].items()}
    annotation_count = {int(k): v for k, v in scan["image_annotation_count"].items()}

    # 1차 보강까지 반영된 현재 학습 표본 수
    results = json.loads(
        (PROJECT_DIR / "reports" / "11_train_split_results.json").read_text(encoding="utf-8")
    )
    current = {
        f"{task}|{label}": count
        for task, labels in results["label_support_fit"].items()
        for label, count in labels.items()
    }
    remaining = {
        key: max(0, args.target - count)
        for key, count in current.items()
        if count < args.target
    }
    print(f"목표 {args.target}장 미달 라벨 {len(remaining)}개")

    pool = [
        image_id for image_id in image_labels
        if image_id >= args.min_image_id and image_id not in previous and image_labels[image_id]
    ]
    print(f"후보 이미지 {len(pool):,}개 (shard 2~6 구간)")

    selected: list[int] = []
    gained: Counter[str] = Counter()
    rng = random.Random(23)
    rng.shuffle(pool)
    while len(selected) < args.budget_images and pool:
        best = None
        best_score = 0
        sample = rng.sample(pool, min(500, len(pool)))
        for image_id in sample:
            score = sum(1 for label in image_labels[image_id] if remaining.get(label, 0) > 0)
            if score > best_score:
                best, best_score = image_id, score
        if best is None or best_score == 0:
            print("  더 채울 라벨이 없어 조기 종료")
            break
        selected.append(best)
        pool.remove(best)
        for label in image_labels[best]:
            gained[label] += 1
            if label in remaining:
                remaining[label] = max(0, remaining[label] - 1)

    annotations = sum(annotation_count[i] for i in selected)
    crops = annotations * 8513 / 36999  # 1차 실측 변환율
    print(f"\n선택 {len(selected):,}장  주석 {annotations:,}  예상 crop {crops:,.0f}")
    print(f"예상 임베딩 시간 @2.2 crop/s: {crops / 2.2 / 60:.0f}분")

    print(f"\n{'라벨':<28}{'1차후':>8}{'2차추가':>9}{'합계':>8}")
    print("-" * 55)
    rows = sorted(
        ((current[key], key, gained.get(key, 0)) for key in current if current[key] < args.target),
        key=lambda row: row[0],
    )
    for count, key, add in rows:
        if add:
            print(f"{key:<28}{count:>8}{add:>9}{count + add:>8}")

    Path(args.output).write_text(
        json.dumps({
            "image_ids": sorted(selected),
            "target": args.target,
            "min_image_id": args.min_image_id,
            "gained": dict(gained),
        }),
        encoding="utf-8",
    )
    print(f"\nsaved: {args.output}")


if __name__ == "__main__":
    main()
