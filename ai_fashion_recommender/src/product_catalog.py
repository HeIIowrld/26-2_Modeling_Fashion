from __future__ import annotations

import csv
import os
from pathlib import Path

from schemas import Product


class ProductCatalog:
    """현재는 로컬 샘플 CSV를 사용하며, 이후 공식 쇼핑몰 API 어댑터로 교체한다."""

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        configured_audit = os.environ.get("FASHION_PRODUCT_COLOR_AUDIT", "").strip()
        self.color_audit_path = (
            Path(configured_audit).expanduser()
            if configured_audit
            else self.csv_path.with_name("product_image_colors.csv")
        )
        self.color_audits = self._load_color_audits()
        self.products = self._load()
        self.color_override_count = sum(
            product.color_source == "image" for product in self.products
        )
        self.color_mismatch_count = sum(
            bool(product.image_color) and product.image_color != product.catalog_color
            for product in self.products
        )

    def _load_color_audits(self) -> dict[str, dict[str, str]]:
        if not self.color_audit_path.is_file():
            return {}
        with self.color_audit_path.open(encoding="utf-8-sig", newline="") as handle:
            return {
                row["product_id"]: row
                for row in csv.DictReader(handle)
                if (row.get("product_id") or "").strip()
            }

    def _load(self) -> list[Product]:
        def split_values(value: str | None) -> list[str]:
            return [item.strip() for item in (value or "").split("|") if item.strip()]

        def integer(row: dict[str, str], key: str, default: int) -> int:
            value = (row.get(key) or "").strip()
            return int(value) if value else default

        def product_from_row(row: dict[str, str]) -> Product:
            catalog_color = row["color"]
            audit = self.color_audits.get(row["product_id"], {})
            image_color = (audit.get("image_color") or "").strip()
            try:
                image_confidence = float(audit.get("confidence") or 0.0)
            except ValueError:
                image_confidence = 0.0
            use_image_color = (
                (audit.get("override") or "").strip().lower() == "true"
                and image_confidence >= 0.60
                and bool(image_color)
            )
            return Product(
                product_id=row["product_id"],
                name=row["name"],
                category=row["category"],
                color=image_color if use_image_color else catalog_color,
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
                brand=row.get("brand") or "",
                gender=row.get("gender") or "",
                image_url=row.get("image_url") or "",
                image_path=row.get("image_path") or "",
                catalog_color=catalog_color,
                image_color=image_color,
                image_color_confidence=round(image_confidence, 3),
                color_source="image" if use_image_color else "catalog",
            )

        with self.csv_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = csv.DictReader(handle)
            return [product_from_row(row) for row in rows]

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
