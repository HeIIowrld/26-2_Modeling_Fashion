from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fit_vision_dataset import load_fit_vision_csv
from fit_vision_schema import FIT_VISION_TASKS


def main() -> None:
    parser = argparse.ArgumentParser(description="핏 라벨 CSV의 수량·누수·파일을 검사합니다.")
    parser.add_argument("--annotations-csv", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--allow-missing-masks", action="store_true")
    args = parser.parse_args()
    records = load_fit_vision_csv(args.annotations_csv, args.dataset_root)
    report = {
        "records": len(records),
        "split": dict(Counter(record.split for record in records)),
        "domain": dict(Counter(record.source_domain for record in records)),
        "category": dict(Counter(record.category for record in records)),
        "labels": {
            task: dict(Counter(record.labels.get(task) for record in records if record.labels.get(task)))
            for task in FIT_VISION_TASKS
        },
        "missing_masks": sum(record.mask_path is None for record in records),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if report["missing_masks"] and not args.allow_missing_masks:
        raise SystemExit("마스크가 없는 레코드가 있습니다. build_fit_masks.py를 먼저 실행하세요.")


if __name__ == "__main__":
    main()
