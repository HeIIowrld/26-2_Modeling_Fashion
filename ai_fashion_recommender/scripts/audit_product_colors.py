"""상품 대표 이미지와 카탈로그 색상을 대조해 작은 sidecar CSV를 만든다.

원본 쇼핑몰 CSV와 이미지는 저장소에 넣지 않는다. 이 스크립트가 만드는
``product_image_colors.csv``만 ProductCatalog이 선택적으로 읽어, 충분히 단색이고
신뢰도가 높은 불일치만 이미지 색상으로 교정한다.
"""

from __future__ import annotations

import argparse
import csv
import colorsys
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from outfit_analyzer import COLOR_PALETTE, NEUTRALS, _nearest_color  # noqa: E402


OUTPUT_FIELDS = (
    "product_id",
    "catalog_color",
    "image_color",
    "confidence",
    "foreground_ratio",
    "override",
    "reason",
)


def _hue_distance(first: str, second: str) -> float:
    """팔레트 두 색상의 원형 hue 거리(0~180도)를 반환한다."""
    first_hue = colorsys.rgb_to_hsv(*(value / 255 for value in COLOR_PALETTE[first]))[0]
    second_hue = colorsys.rgb_to_hsv(*(value / 255 for value in COLOR_PALETTE[second]))[0]
    distance = abs(first_hue - second_hue)
    return min(distance, 1.0 - distance) * 360.0


def _safe_to_override(catalog_color: str, image_color: str, confidence: float) -> bool:
    """오탐 비용이 큰 만큼 명백하게 다른 유채색만 자동 교정한다.

    모델 착용컷에서는 피부·배경·그림자가 상품 영역에 섞이기 쉽다. 따라서
    무채색 계열 변경과 유사색 변경은 점검 대상으로만 남기고, 색상 집중도가
    충분하면서 hue가 90도 이상 다른 경우에만 카탈로그 값을 덮는다.
    """
    if confidence < 0.60 or catalog_color == image_color:
        return False
    if catalog_color not in COLOR_PALETTE or image_color not in COLOR_PALETTE:
        return False
    if catalog_color in NEUTRALS or image_color in NEUTRALS:
        return False
    return _hue_distance(catalog_color, image_color) >= 90.0


def _foreground_mask(rgb: np.ndarray) -> np.ndarray:
    """대표 이미지의 모서리색을 배경으로 보고 가장 큰 상품 영역만 남긴다."""
    height, width = rgb.shape[:2]
    patch = max(3, min(height, width) // 20)
    corners = np.concatenate(
        (
            rgb[:patch, :patch].reshape(-1, 3),
            rgb[:patch, -patch:].reshape(-1, 3),
            rgb[-patch:, :patch].reshape(-1, 3),
            rgb[-patch:, -patch:].reshape(-1, 3),
        )
    )
    background = np.median(corners, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    mask = (distance > 24) & ((hsv[:, :, 1] > 18) | (hsv[:, :, 2] < 235))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count <= 1:
        return mask
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    mask = (labels == largest).astype(np.uint8)
    kernel_size = max(3, (min(height, width) // 100) | 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel).astype(bool)


def analyze_image_color(
    image_path: str | Path,
    catalog_color: str,
    *,
    override_threshold: float = 0.60,
) -> dict[str, str]:
    """한 상품 이미지의 대표색과 교정 가능 여부를 계산한다.

    여러 색이 비슷한 비율로 섞인 그래픽·체크 상품은 색상 집중도가 낮아 자동
    교정하지 않는다. 단색 상품처럼 색상 집중도가 높은 경우에만 원문을 덮는다.
    """
    path = Path(image_path)
    try:
        image = Image.open(path).convert("RGB")
    except (FileNotFoundError, OSError, UnidentifiedImageError):
        return {
            "catalog_color": catalog_color,
            "image_color": "",
            "confidence": "0.000",
            "foreground_ratio": "0.000",
            "override": "false",
            "reason": "image_unavailable",
        }
    image.thumbnail((512, 512))
    rgb = np.asarray(image, dtype=np.uint8)
    mask = _foreground_mask(rgb)
    foreground_ratio = float(mask.mean())
    if not 0.03 <= foreground_ratio <= 0.85 or int(mask.sum()) < 400:
        return {
            "catalog_color": catalog_color,
            "image_color": "",
            "confidence": "0.000",
            "foreground_ratio": f"{foreground_ratio:.3f}",
            "override": "false",
            "reason": "foreground_uncertain",
        }

    pixels = rgb[mask]
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)[mask]
    chromatic = hsv[:, 1] > 35
    chromatic_ratio = float(chromatic.mean())
    if chromatic_ratio >= 0.55:
        sample = pixels[chromatic]
        angles = hsv[chromatic, 0].astype(np.float64) / 180.0 * 2 * math.pi
        hue_concentration = abs(np.mean(np.exp(1j * angles)))
        confidence = float(hue_concentration * min(1.0, chromatic_ratio / 0.80))
        reason = "chromatic_consensus"
    else:
        sample = pixels
        neutral_ratio = 1.0 - chromatic_ratio
        value_spread = float(np.std(hsv[:, 2]))
        confidence = neutral_ratio * max(0.60, 1.0 - value_spread / 180.0)
        reason = "neutral_consensus"

    representative = tuple(np.rint(sample.mean(axis=0)).astype(int))
    image_color = _nearest_color(representative)
    mismatch = image_color != catalog_color
    override = mismatch and confidence >= override_threshold and _safe_to_override(
        catalog_color,
        image_color,
        confidence,
    )
    if mismatch and not override:
        reason = f"manual_review_{reason}"
    return {
        "catalog_color": catalog_color,
        "image_color": image_color,
        "confidence": f"{confidence:.3f}",
        "foreground_ratio": f"{foreground_ratio:.3f}",
        "override": "true" if override else "false",
        "reason": reason if mismatch else "catalog_agrees",
    }


def _resolve_image(catalog_path: Path, image_path: str) -> Path:
    path = Path(image_path)
    if path.is_absolute():
        return path
    # .../<project>/ai_fashion_recommender/data/catalog.csv 기준으로 프로젝트 루트.
    project_root = catalog_path.resolve().parents[2]
    return project_root / path


def audit_catalog(catalog_path: Path, *, workers: int = 8, limit: int = 0) -> list[dict[str, str]]:
    with catalog_path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if limit:
        rows = rows[:limit]

    def analyze(row: dict[str, str]) -> dict[str, str]:
        result = analyze_image_color(
            _resolve_image(catalog_path, row.get("image_path") or ""),
            (row.get("color") or "").strip(),
        )
        return {"product_id": row["product_id"], **result}

    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(analyze, rows))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output = args.output or args.catalog.with_name("product_image_colors.csv")
    audited = audit_catalog(args.catalog, workers=args.workers, limit=args.limit)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(audited)

    overrides = [row for row in audited if row["override"] == "true"]
    unavailable = [row for row in audited if not row["image_color"]]
    print(f"상품 {len(audited)}개 점검 / 자동 교정 {len(overrides)}개 / 판정 보류 {len(unavailable)}개")
    for row in overrides[:20]:
        print(
            f"  {row['product_id']}: {row['catalog_color']} → {row['image_color']} "
            f"({float(row['confidence']) * 100:.1f}%)"
        )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
