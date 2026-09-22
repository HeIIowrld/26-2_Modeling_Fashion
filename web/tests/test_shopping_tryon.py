import tempfile
import io
import unittest
from pathlib import Path
from unittest.mock import patch

from web import pipeline
from musinsa_live_search import ShoppingProduct
from schemas import Product
from PIL import Image


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
    def test_photo_cache_preserves_pixels_and_keys_by_image_url(self):
        data = io.BytesIO()
        Image.new("RGB", (16, 16), (123, 45, 67)).save(data, "JPEG")
        raw = data.getvalue()
        class Response(io.BytesIO):
            headers = {"Content-Length": str(len(raw))}
            def geturl(self):
                return "https://image.msscdn.net/example.jpg"
        product = shopping_product("MS1")
        with tempfile.TemporaryDirectory() as directory, patch(
            "web.pipeline.urllib.request.urlopen", side_effect=lambda *a, **kw: Response(raw)
        ) as download:
            first = pipeline._cache_live_shopping_image(product, Path(directory), timeout=.1)
            cached = pipeline._cache_live_shopping_image(product, Path(directory), timeout=.1)
            self.assertEqual(first, cached)
            self.assertEqual(download.call_count, 1)
            with Image.open(first) as saved, Image.open(io.BytesIO(raw)) as original:
                self.assertEqual(saved.tobytes(), original.convert("RGB").tobytes())
            product.image_url = "https://image.msscdn.net/replaced.jpg"
            replaced = pipeline._cache_live_shopping_image(product, Path(directory), timeout=.1)
            self.assertNotEqual(first, replaced)
            self.assertFalse(list(Path(directory).glob("*.tmp")))
            self.assertEqual(download.call_args.kwargs["timeout"], .1)

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

    def test_shoes_require_both_model_capability_and_visible_feet(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Path(directory) / "1.jpg"
            image.write_bytes(b"shoe")
            for reason in ("", "양쪽 발이 보이지 않습니다."):
                with patch("web.pipeline.garment_image_path", return_value=image):
                    payloads, resolved = pipeline._shopping_tryon_payloads(
                        [shopping_product("S", "shoes")], [catalog_product("S", "shoes")], Path(directory),
                        adapter_available=True, supported_categories={"top", "bottom", "shoes"},
                        shoe_unavailable_reason=reason,
                    )
                self.assertEqual(payloads[0]["tryon_available"], not bool(reason))
                self.assertEqual(bool(resolved), not bool(reason))
                self.assertEqual(payloads[0]["tryon_reason"], reason)


if __name__ == "__main__":
    unittest.main()
