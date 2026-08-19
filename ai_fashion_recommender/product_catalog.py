from __future__ import annotations

import csv
from pathlib import Path

from schemas import Product


class ProductCatalog:
    """현재는 로컬 샘플 CSV를 사용하며, 이후 공식 쇼핑몰 API 어댑터로 교체한다."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.products = self._load()

    def _load(self) -> list[Product]:
        def split_values(value: str | None) -> list[str]:
            return [item.strip() for item in (value or "").split("|") if item.strip()]

        def integer(row: dict[str, str], key: str, default: int) -> int:
            value = (row.get(key) or "").strip()
            return int(value) if value else default

        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            return [
                Product(
                    product_id=row["product_id"],
                    name=row["name"],
                    category=row["category"],
                    color=row["color"],
                    style=row["style"],
                    purposes=split_values(row["purposes"]),
                    body_shapes=split_values(row["body_shapes"]),
                    price=int(row["price"]),
                    season=row["season"],
                    stock=row["stock"].lower() == "true",
                    url=row.get("url", ""),
                    item_type=row.get("item_type", ""),
                    fit=row.get("fit", ""),
                    length=row.get("length", ""),
                    pattern=row.get("pattern", "무지") or "무지",
                    material=row.get("material", ""),
                    neckline=row.get("neckline", ""),
                    formality=integer(row, "formality", 3),
                    activity_tags=split_values(row.get("activity_tags")),
                    warmth=integer(row, "warmth", 3),
                    breathability=integer(row, "breathability", 3),
                    water_resistant=(row.get("water_resistant") or "false").lower() == "true",
                    visual_weight=integer(row, "visual_weight", 3),
                    detail_level=integer(row, "detail_level", 1),
                    waistline=row.get("waistline", ""),
                    pattern_scale=row.get("pattern_scale", ""),
                    pattern_contrast=integer(row, "pattern_contrast", 0),
                )
                for row in rows
            ]

    def available(self, category: str | None = None) -> list[Product]:
        return [
            product
            for product in self.products
            if product.stock and (category is None or product.category == category)
        ]
