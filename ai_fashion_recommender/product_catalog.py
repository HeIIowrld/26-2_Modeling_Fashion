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
        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            return [
                Product(
                    product_id=row["product_id"],
                    name=row["name"],
                    category=row["category"],
                    color=row["color"],
                    style=row["style"],
                    purposes=row["purposes"].split("|"),
                    body_shapes=row["body_shapes"].split("|"),
                    price=int(row["price"]),
                    season=row["season"],
                    stock=row["stock"].lower() == "true",
                    url=row.get("url") or "",
                    brand=row.get("brand") or "",
                    gender=row.get("gender") or "",
                    image_url=row.get("image_url") or "",
                    image_path=row.get("image_path") or "",
                )
                for row in rows
            ]

    def available(self, category: str | None = None, gender: str = "") -> list[Product]:
        """재고가 있는 상품을 카테고리·성별로 거른다. gender가 빈 값이면 성별 무관,
        지정하면 해당 성별과 공용(성별 미표기 포함) 상품만 남긴다."""
        return [
            product
            for product in self.products
            if product.stock
            and (category is None or product.category == category)
            and (not gender or product.gender in ("", "공용", gender))
        ]
