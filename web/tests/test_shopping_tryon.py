import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from web import pipeline
from musinsa_live_search import ShoppingProduct
from schemas import Product


def catalog_product(product_id: str, category: str = "top") -> Product:
    return Product(
        product_id,
        "카탈로그 상품",
        category,
        "블랙",
        "캐주얼",
        ["데일리"],
        [],
        59_000,
        "사계절",
        True,
        url="https://www.musinsa.com/products/1",
        image_url="https://image.msscdn.net/1.jpg",
        image_path="1.jpg",
    )


def shopping_product(product_id: str, category: str = "top") -> ShoppingProduct:
    return ShoppingProduct(
        product_id,
        "검색 상품",
        "브랜드",
        59_000,
        "https://image.msscdn.net/1.jpg",
        "https://www.musinsa.com/products/1",
        category,
    )


class ShoppingTryOnResolutionTests(unittest.TestCase):
    def test_catalog_search_result_exposes_tryon_without_leaking_local_path(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "1.jpg"
            image.write_bytes(b"garment")
            with patch("web.pipeline.garment_image_path", return_value=image):
                payloads, resolved = pipeline._shopping_tryon_payloads(
                    [shopping_product("MS1")],
                    [catalog_product("MS1")],
                    Path(directory),
                    adapter_available=True,
                )

        self.assertTrue(payloads[0]["tryon_available"])
        self.assertEqual(payloads[0]["tryon_reason"], "")
        self.assertNotIn("image_path", payloads[0])
        self.assertEqual(list(resolved), ["MS1"])

    def test_shoes_are_disclosed_as_unsupported_instead_of_using_lower_mask(self):
        payloads, resolved = pipeline._shopping_tryon_payloads(
            [shopping_product("SHOE1", "shoes")],
            [catalog_product("SHOE1", "shoes")],
            Path("."),
            adapter_available=True,
        )

        self.assertFalse(payloads[0]["tryon_available"])
        self.assertIn("상의와 하의만", payloads[0]["tryon_reason"])
        self.assertEqual(resolved, {})

    def test_only_musinsa_cdn_urls_can_be_downloaded(self):
        self.assertTrue(pipeline._shopping_image_host_allowed("https://image.msscdn.net/a.jpg"))
        self.assertFalse(pipeline._shopping_image_host_allowed("http://image.msscdn.net/a.jpg"))
        self.assertFalse(pipeline._shopping_image_host_allowed("https://msscdn.net.evil.example/a.jpg"))


if __name__ == "__main__":
    unittest.main()
