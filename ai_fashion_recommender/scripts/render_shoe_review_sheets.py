"""Render labelled contact sheets from a weak-label CSV for quick human review."""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

SHOE_LABELS = ("스니커즈", "러닝화", "로퍼", "더비슈즈", "메리제인", "펌프스",
               "샌들", "슬리퍼", "부츠", "워커", "기타 신발", "맨발")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--per-label", type=int, default=20)
    args = parser.parse_args()
    with args.manifest.open(encoding="utf-8-sig", newline="") as source:
        rows = list(csv.DictReader(source))
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["shoe_label"]].append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    font = ImageFont.load_default()
    for candidate in (Path("C:/Windows/Fonts/malgun.ttf"), Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")):
        if candidate.is_file():
            font = ImageFont.truetype(str(candidate), 13)
            break
    for label, label_rows in grouped.items():
        label_rows.sort(key=lambda row: float(row["pseudo_margin"]), reverse=True)
        chosen = label_rows[:args.per_label]
        cell_w, cell_h, columns = 240, 240, 5
        rows_count = (len(chosen) + columns - 1) // columns
        sheet = Image.new("RGB", (cell_w * columns, cell_h * rows_count), "white")
        draw = ImageDraw.Draw(sheet)
        for index, row in enumerate(chosen):
            image_path = row.get("crop_path") or row["image_path"]
            with Image.open(image_path) as source:
                image = source.convert("RGB")
            image.thumbnail((cell_w - 10, cell_h - 45))
            x = (index % columns) * cell_w
            y = (index // columns) * cell_h
            sheet.paste(image, (x + (cell_w - image.width) // 2, y + 5))
            text = f"{row['source_id']} p={float(row['pseudo_confidence']):.2f} m={float(row['pseudo_margin']):.2f}"
            draw.text((x + 5, y + cell_h - 34), text, fill="black", font=font)
            draw.text((x + 5, y + cell_h - 18), f"2nd: {row['second_label']}", fill="black", font=font)
        sheet.save(args.output / f"{SHOE_LABELS.index(label):02d}_{label}.jpg", quality=92)
    print(f"Rendered {len(grouped)} review sheets in {args.output}")


if __name__ == "__main__":
    main()
