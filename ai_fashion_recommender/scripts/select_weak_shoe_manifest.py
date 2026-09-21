"""Merge conservative pseudo labels into a balanced research-training manifest."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--trusted-inputs", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-train-per-label", type=int, default=2000)
    parser.add_argument("--max-eval-per-label", type=int, default=300)
    args = parser.parse_args()
    rows = []
    for path in args.inputs:
        with path.open(encoding="utf-8-sig", newline="") as source:
            rows.extend(row for row in csv.DictReader(source) if row["quality"] == "pseudo_ok")
    for path in args.trusted_inputs:
        with path.open(encoding="utf-8-sig", newline="") as source:
            for row in csv.DictReader(source):
                if row["quality"] == "ok":
                    row.setdefault("pseudo_confidence", "1")
                    row.setdefault("pseudo_margin", "1")
                    rows.append(row)
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["shoe_label"])].append(row)
    selected = []
    for (split, _label), part in grouped.items():
        part.sort(key=lambda row: (float(row["pseudo_margin"]), float(row["pseudo_confidence"])), reverse=True)
        limit = args.max_train_per_label if split == "train" else args.max_eval_per_label
        selected.extend(part[:limit])
    selected.sort(key=lambda row: (row["split"], row["shoe_label"], row["source_id"]))
    if not selected:
        raise ValueError("No pseudo_ok rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in selected for key in row})
    with args.output.open("w", encoding="utf-8-sig", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields); writer.writeheader(); writer.writerows(selected)
    summary = {
        "input_rows": len(rows), "selected_rows": len(selected),
        "counts": {f"{split}/{label}": count for (split, label), count in sorted(Counter((r['split'], r['shoe_label']) for r in selected).items())},
        "warning": "Pseudo labels are suitable for research bootstrapping, not final threshold calibration.",
    }
    args.output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
