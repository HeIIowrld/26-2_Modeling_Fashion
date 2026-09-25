"""사진에서 정한 색은 '빈 칸만' 채우고, 회피 색 필터에는 쓰지 않는다.

의류 영역 판정은 내가 라벨한 192장에서 68%(확신도 0.8 이상만 보면 87%)다. 상품명 색
(78%)보다 낮으므로 상품명·컬러칩이 있으면 그쪽을 쓴다. 그리고 오차가 어두운 색을 블랙으로
뭉개는 쪽이라, 회피 색 필터에 쓰면 멀쩡한 상품이 통째로 사라진다.
"""
import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from product_catalog import ProductCatalog  # noqa: E402

COLUMNS = ["product_id", "name", "category", "color", "style", "purposes", "body_shapes",
           "price", "season", "stock", "url", "item_type", "fit", "length", "pattern",
           "material", "neckline", "formality", "activity_tags", "warmth", "breathability",
           "water_resistant", "visual_weight", "detail_level", "waistline", "pattern_scale",
           "pattern_contrast", "brand", "gender", "image_url", "image_path", "color_options"]


def row(product_id, color):
    values = dict.fromkeys(COLUMNS, "")
    values.update({"product_id": product_id, "name": f"{product_id} 반팔 티셔츠", "category": "top",
                   "color": color, "style": "캐주얼", "purposes": "데일리", "body_shapes": "",
                   "price": "39000", "season": "사계절", "stock": "true", "pattern": "무지"})
    return values


class CatalogBuilder:
    """임시 카탈로그와 사진 색 sidecar 를 만든다."""

    def build(self, rows, photo_rows):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        catalog_path = folder / "products.csv"
        with catalog_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        with (folder / "product_photo_colors.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["product_id", "photo_color", "agreement",
                                                        "garment_ratio", "reason"])
            writer.writeheader()
            writer.writerows(photo_rows)
        return ProductCatalog(catalog_path)


class PhotoColorFillTests(CatalogBuilder, unittest.TestCase):
    def test_empty_colour_is_filled_from_the_photo(self):
        catalog = self.build([row("MS1", "")],
                             [{"product_id": "MS1", "photo_color": "네이비", "agreement": "0.85",
                               "garment_ratio": "0.30", "reason": "garment_vote"}])
        product = catalog.products[0]
        self.assertEqual(product.color, "네이비")
        self.assertEqual(product.color_source, "photo")
        self.assertEqual(catalog.photo_color_count, 1)

    def test_an_existing_colour_is_never_replaced(self):
        catalog = self.build([row("MS1", "블랙")],
                             [{"product_id": "MS1", "photo_color": "네이비", "agreement": "0.95",
                               "garment_ratio": "0.30", "reason": "garment_vote"}])
        self.assertEqual(catalog.products[0].color, "블랙")
        self.assertEqual(catalog.products[0].color_source, "catalog")

    def test_products_the_photo_could_not_decide_stay_empty(self):
        catalog = self.build([row("MS1", "")],
                             [{"product_id": "MS1", "photo_color": "", "agreement": "0.40",
                               "garment_ratio": "0.30", "reason": "low_agreement"}])
        self.assertEqual(catalog.products[0].color, "")
        self.assertEqual(catalog.products[0].color_source, "none")

    def test_missing_sidecar_is_not_an_error(self):
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = folder / "products.csv"
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerow(row("MS1", ""))
        catalog = ProductCatalog(path)
        self.assertEqual(catalog.products[0].color, "")
        self.assertEqual(catalog.photo_color_count, 0)


class AvoidedColourTests(CatalogBuilder, unittest.TestCase):
    def engine(self, catalog):
        from recommendation_engine import RecommendationEngine

        return RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", catalog)

    def test_photo_colour_never_removes_a_product_from_the_avoided_filter(self):
        """사진 판정은 그레이·카키·브라운을 블랙으로 뭉갠다. 그 오차로 상품을 지우지 않는다."""
        from schemas import UserProfile

        catalog = self.build([row("MS1", ""), row("MS2", "블랙")],
                             [{"product_id": "MS1", "photo_color": "블랙", "agreement": "0.90",
                               "garment_ratio": "0.30", "reason": "garment_vote"}])
        profile = UserProfile(avoided_colors=["블랙"], provided_fields=["avoided_colors"])

        kept = self.engine(catalog)._available_for_profile("top", profile)

        self.assertEqual([product.product_id for product in kept], ["MS1"])
        self.assertEqual(kept[0].color_source, "photo")


if __name__ == "__main__":
    unittest.main()
