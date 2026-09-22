"""사람마다 한 장: 행 = 상품, 열 = 상품 | 원본 | native | hull | hull_shoe (하체만 크게).

usage: bottom_shape_sheets.py OUT PEOPLE_ROOT   (PEOPLE_ROOT 아래에서 사람 파일을 이름으로 찾는다)
"""
import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

OUT = Path(sys.argv[1])
PEOPLE = Path(sys.argv[2])
VARIANTS = ("native", "hull", "hull_shoe")
rows = [json.loads(l) for l in (OUT / "results.jsonl").read_text(encoding="utf-8").splitlines()]
ok = [r for r in rows if "error" not in r and r["variant"] in VARIANTS]
people = sorted({(r["group"], r["person"]) for r in ok})
pids = sorted({r["pid"] for r in ok})
index = {path.name: path for path in PEOPLE.rglob("*.jpg")}
dest = OUT / "zoom"
dest.mkdir(exist_ok=True)
H = 300


def lower(image):
    w, h = image.size
    crop = image.crop((0, int(h * 0.4), w, h))
    return crop.resize((max(1, int(crop.width * H / crop.height)), H))


for group, person in people:
    stem = Path(person).stem
    original = lower(Image.open(index[person]).convert("RGB"))
    lines = []
    for pid in pids:
        cells = [Image.open(OUT / "refs" / f"{pid}.jpg").convert("RGB")]
        cells[0] = cells[0].resize((int(cells[0].width * H / cells[0].height), H))
        cells.append(original)
        for v in VARIANTS:
            path = OUT / "renders" / f"{stem}__{pid}__{v}.jpg"
            cells.append(lower(Image.open(path).convert("RGB")) if path.is_file() else Image.new("RGB", (10, H), "white"))
        line = Image.new("RGB", (sum(c.width for c in cells) + 4 * len(cells), H + 16), "white")
        draw = ImageDraw.Draw(line)
        x = 0
        for label, cell in zip(("product", "original") + VARIANTS, cells):
            line.paste(cell, (x, 16))
            draw.text((x + 2, 2), f"{pid} {label}" if label == "product" else label, fill="red")
            x += cell.width + 4
        lines.append(line)
    sheet = Image.new("RGB", (max(l.width for l in lines), sum(l.height for l in lines)), "white")
    y = 0
    for line in lines:
        sheet.paste(line, (0, y))
        y += line.height
    sheet.save(dest / f"{group}__{stem}.jpg", quality=85)
print(f"{len(people)}장 -> {dest}")
