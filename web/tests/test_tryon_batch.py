import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import web.app as web_app
from schemas import Product

def shopping_product(product_id: str, category: str) -> Product:
    return Product(
        product_id,
        f"{category} product",
        category,
        "블랙",
        "캐주얼",
        ["데일리"],
        [],
        50_000,
        "사계절",
        True,
        image_path=f"{product_id}.jpg",
    )


class TryOnBatchTests(unittest.TestCase):
    def setUp(self):
        self.job_id = "b" * 32
        self.tempdir = tempfile.TemporaryDirectory()
        self.session = Path(self.tempdir.name)
        (self.session / "person.jpg").write_bytes(b"person")
        self.job = {
            "id": self.job_id,
            "status": "done",
            "result": {
                "tryon": {
                    "available": True,
                    "reason": "",
                    "warnings": [],
                },
                "shopping_results": [],
            },
            "person_image": self.session / "person.jpg",
            "tryon_context": {"mask": object()},
            "work_lock": threading.Lock(),
            "cancelled": False,
        }
        with web_app._jobs_lock:
            web_app._jobs[self.job_id] = self.job

    def tearDown(self):
        with web_app._jobs_lock:
            web_app._jobs.pop(self.job_id, None)
        self.tempdir.cleanup()

    def test_selected_musinsa_top_and_bottom_are_composited_and_cached(self):
        self.job["shopping_tryon_products"] = {
            "MS_TOP": shopping_product("MS_TOP", "top"),
            "MS_BOTTOM": shopping_product("MS_BOTTOM", "bottom"),
        }
        calls = []

        def fake_generate(_person, reco, output, *, context):
            calls.append([product.category for product in reco.products])
            output.write_bytes(b"selected look")
            return output, ["quality warning"]

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            first = web_app.create_product_tryon(
                self.job_id, {"product_ids": ["MS_BOTTOM", "MS_TOP"]}
            )
            second = web_app.create_product_tryon(
                self.job_id, {"product_ids": ["MS_TOP", "MS_BOTTOM"]}
            )

        self.assertEqual(calls, [["top", "bottom"]])
        self.assertFalse(first["cached"])
        self.assertTrue(second["cached"])
        self.assertEqual(first["image"], second["image"])
        self.assertEqual(first["product_ids"], ["MS_TOP", "MS_BOTTOM"])
        self.assertEqual(first["warnings"], ["quality warning"])

    def test_same_category_musinsa_products_are_rejected(self):
        self.job["shopping_tryon_products"] = {
            "MS_TOP_1": shopping_product("MS_TOP_1", "top"),
            "MS_TOP_2": shopping_product("MS_TOP_2", "top"),
        }
        with self.assertRaisesRegex(Exception, "카테고리별로 한 개"):
            web_app.create_product_tryon(
                self.job_id, {"product_ids": ["MS_TOP_1", "MS_TOP_2"]}
            )

    def test_uncached_live_musinsa_product_is_not_faked(self):
        self.job["shopping_tryon_products"] = {}
        with self.assertRaisesRegex(Exception, "준비하지 못했습니다"):
            web_app.create_product_tryon(self.job_id, {"product_ids": ["MS404"]})

    def test_all_musinsa_top_bottom_combinations_are_queued_and_failures_are_isolated(self):
        products = {
            "TOP1": shopping_product("TOP1", "top"),
            "TOP2": shopping_product("TOP2", "top"),
            "BOTTOM1": shopping_product("BOTTOM1", "bottom"),
            "BOTTOM2": shopping_product("BOTTOM2", "bottom"),
        }
        self.job["shopping_tryon_products"] = products
        self.job["result"]["shopping_results"] = [
            {"product_id": product_id} for product_id in products
        ]
        calls = []

        def fake_generate(_person, reco, output, *, context):
            product_ids = [product.product_id for product in reco.products]
            calls.append(product_ids)
            if product_ids == ["TOP2", "BOTTOM1"]:
                raise RuntimeError("one combination failed")
            output.write_bytes("+".join(product_ids).encode())
            return output, []

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            initialized = web_app._initialize_shopping_tryon_batch(self.job_id)
            self.assertEqual(
                [item["product_ids"] for item in initialized["items"]],
                [
                    ["TOP1", "BOTTOM1"],
                    ["TOP1", "BOTTOM2"],
                    ["TOP2", "BOTTOM1"],
                    ["TOP2", "BOTTOM2"],
                ],
            )
            web_app._start_shopping_tryon_batch(self.job_id)
            for _ in range(300):
                snapshot = web_app._read_shopping_tryon_batch(self.job_id)
                if snapshot["status"] == "partial":
                    break
                time.sleep(0.005)
            else:
                self.fail("shopping try-on batch did not finish")

        self.assertEqual(len(calls), 4)
        self.assertEqual(snapshot["total"], 4)
        self.assertEqual(snapshot["ready"], 3)
        self.assertEqual(snapshot["finished"], 4)
        self.assertEqual(snapshot["items"][2]["status"], "failed")
        self.assertEqual(snapshot["items"][3]["status"], "done")

    def test_single_category_musinsa_results_are_all_rendered_individually(self):
        self.job["shopping_tryon_products"] = {
            "TOP1": shopping_product("TOP1", "top"),
            "TOP2": shopping_product("TOP2", "top"),
        }
        self.job["result"]["shopping_results"] = [
            {"product_id": "TOP1"},
            {"product_id": "TOP2"},
        ]

        initialized = web_app._initialize_shopping_tryon_batch(self.job_id)

        self.assertEqual(initialized["total"], 2)
        self.assertEqual(
            [item["product_ids"] for item in initialized["items"]],
            [["TOP1"], ["TOP2"]],
        )


if __name__ == "__main__":
    unittest.main()
