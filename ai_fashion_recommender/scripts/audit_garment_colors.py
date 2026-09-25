"""상품명에도 컬러칩에도 색이 없는 상품의 색을, 의류 영역에서 정한다.

왜 따로 만들었나. `audit_product_colors.py` 는 모서리색 대비로 전경을 잡고 그 **평균**을
쓴다. 모델 착용컷이면 피부·머리·다른 옷·배경까지 섞여 회색으로 수렴한다 — 사람이 라벨한
192장에서 43%였다. 여기서는 의류 분할(FASHN)로 그 옷 픽셀만 남기고 명도·채도를 먼저
가르는 규칙으로 색을 정한다(같은 표본 68%, 확신도 0.8 이상만 보면 87%).

결과는 `product_photo_colors.csv` 로 남기고, ProductCatalog 이 **비어 있는 색만** 채운다.
이 경로로 정한 색은 회피 색 필터에 쓰지 않는다(catalog_derivation.json 의 _error_note).

usage:
    python scripts/audit_garment_colors.py                       # 색이 빈 상품만
    python scripts/audit_garment_colors.py --all --limit 50
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from config import DATA_DIR, garment_image_path  # noqa: E402
from product_colors import color_from_pixels, load_photo_rules  # noqa: E402
from schemas import PoseAnalysis  # noqa: E402

# 3 상의, 4 원피스, 5 치마, 6 바지. 원피스는 상·하의 어느 쪽 상품으로도 올라온다.
GARMENT_LABELS = {"top": (3, 4), "bottom": (5, 6, 4)}
FIELDS = ("product_id", "photo_color", "agreement", "garment_ratio", "reason")
SAMPLE = 400
EMPTY_POSE = PoseAnalysis(False, 0, "", 0, 0, 0, "", 0)


def garment_pixels(image: Image.Image, segmentation, category: str):
    seg = np.asarray(segmentation)
    rgb = np.asarray(image.resize((seg.shape[1], seg.shape[0])), dtype=np.uint8)
    mask = np.isin(seg, GARMENT_LABELS.get(category, GARMENT_LABELS["top"]))
    ratio = float(mask.mean())
    pixels = rgb[mask]
    if len(pixels) < 400:
        return None, ratio
    if len(pixels) > SAMPLE:
        picked = np.random.default_rng(0).choice(len(pixels), SAMPLE, replace=False)
        pixels = pixels[picked]
    return [tuple(int(value) for value in pixel) for pixel in pixels], ratio


def main() -> int:
    cli = argparse.ArgumentParser()
    cli.add_argument("--catalog", type=Path, default=DATA_DIR / "products_musinsa_enriched.csv")
    cli.add_argument("--output", type=Path, default=DATA_DIR / "product_photo_colors.csv")
    cli.add_argument("--limit", type=int)
    cli.add_argument("--all", action="store_true", help="색이 이미 있는 상품도 판정한다(비교용)")
    args = cli.parse_args()

    with args.catalog.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    key = next(k for k in rows[0] if k.endswith("product_id"))
    targets = [row for row in rows if args.all or not (row.get("color") or "").strip()]
    if args.limit:
        targets = targets[: args.limit]
    print(f"카탈로그 {len(rows)} / 판정 대상 {len(targets)}")

    from clothing_parser import ClothingParser

    parser = ClothingParser(use_fashn=True)
    if parser.backend != "fashn-human-parser":
        print(f"의류 분할을 쓸 수 없습니다(backend={parser.backend}). 중단합니다.")
        return 1
    rules = load_photo_rules()
    records, counts = [], {"ok": 0, "low_agreement": 0, "no_garment": 0, "no_image": 0}
    for index, row in enumerate(targets):
        product_id = row[key]
        path = garment_image_path(row.get("image_path") or "")
        if path is None or not path.is_file():
            counts["no_image"] += 1
            continue
        try:
            image = Image.open(path).convert("RGB")
            parsed = parser.parse(image, EMPTY_POSE)
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            counts["no_image"] += 1
            print(f"  {product_id}: {type(exc).__name__}")
            continue
        pixels, ratio = garment_pixels(image, parsed["segmentation"], row.get("category", "top"))
        if pixels is None or ratio < rules["min_garment_ratio"]:
            counts["no_garment"] += 1
            records.append({"product_id": product_id, "photo_color": "", "agreement": "0.000",
                            "garment_ratio": f"{ratio:.4f}", "reason": "garment_too_small"})
            continue
        color, agreement = color_from_pixels(pixels, rules)
        decided = agreement >= rules["min_agreement"]
        counts["ok" if decided else "low_agreement"] += 1
        records.append({"product_id": product_id, "photo_color": color if decided else "",
                        "agreement": f"{agreement:.3f}", "garment_ratio": f"{ratio:.4f}",
                        "reason": "garment_vote" if decided else "low_agreement"})
        if index % 100 == 99:
            print(f"  {index + 1}/{len(targets)}", flush=True)

    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    print(f"저장: {args.output}  기록 {len(records)}")
    print(f"  색을 정함 {counts['ok']}  확신 부족 {counts['low_agreement']}  "
          f"의류 영역 부족 {counts['no_garment']}  이미지 없음 {counts['no_image']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
