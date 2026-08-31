import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import web.app as web_app
from schemas import Product


def recommendation(rank: int):
    return SimpleNamespace(rank=rank, products=[SimpleNamespace(product_id=f"P{rank}")])


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
        (self.session / "preview.jpg").write_bytes(b"preview")
        self.job = {
            "id": self.job_id,
            "status": "done",
            "recommendations": [recommendation(rank) for rank in (1, 2, 3)],
            "result": {
                "tryon": {
                    "available": True,
                    "reason": "",
                    "warnings": ["rank one warning"],
                    "preview_kind": "tryon",
                },
                "images": {"preview": "preview.jpg"},
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

    def test_batch_reuses_rank_one_and_generates_remaining_ranks(self):
        calls = []

        def fake_generate(_person, reco, output, *, context):
            self.assertIs(context, self.job["tryon_context"])
            calls.append(reco.rank)
            output.write_bytes(f"rank-{reco.rank}".encode())
            return output, [f"warning-{reco.rank}"]

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            initialized = web_app._initialize_tryon_batch(self.job_id)
            self.assertEqual(initialized["ready"], 1)
            self.assertEqual(initialized["items"][0]["image"], "preview.jpg")
            web_app._start_tryon_batch(self.job_id)
            for _ in range(200):
                snapshot = web_app._read_tryon_batch(self.job_id)
                if snapshot["status"] == "done":
                    break
                time.sleep(0.005)
            else:
                self.fail("try-on batch did not finish")

        self.assertEqual(calls, [2, 3])
        self.assertEqual(snapshot["ready"], 3)
        self.assertEqual(
            [item["image"] for item in snapshot["items"]],
            ["preview.jpg", "tryon_2.jpg", "tryon_3.jpg"],
        )

    def test_preview_board_is_not_reused_as_a_real_tryon(self):
        self.job["recommendations"] = [recommendation(1)]
        self.job["result"]["tryon"]["preview_kind"] = "preview"

        def fake_generate(_person, _reco, output, *, context):
            self.assertTrue(context is self.job["tryon_context"])
            output.write_bytes(b"generated")
            return output, []

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            response = web_app.create_tryon(self.job_id, 1)

        self.assertEqual(response["image"], "tryon_1.jpg")
        self.assertFalse(response["cached"])

    def test_unavailable_adapter_does_not_start_a_worker(self):
        self.job["result"]["tryon"] = {
            "available": False,
            "reason": "GPU unavailable",
            "warnings": [],
            "preview_kind": "preview",
        }
        initialized = web_app._initialize_tryon_batch(self.job_id)
        started = web_app._start_tryon_batch(self.job_id)
        self.assertEqual(initialized["status"], "unavailable")
        self.assertEqual(started["status"], "unavailable")
        self.assertEqual(started["ready"], 0)

    def test_one_failed_rank_does_not_stop_the_remaining_batch(self):
        calls = []

        def fake_generate(_person, reco, output, *, context):
            calls.append(reco.rank)
            if reco.rank == 2:
                raise RuntimeError("rank two failed")
            output.write_bytes(f"rank-{reco.rank}".encode())
            return output, []

        with (
            patch.object(web_app, "_session_dir", return_value=self.session),
            patch.object(web_app, "generate_tryon_with_warnings", side_effect=fake_generate),
        ):
            web_app._initialize_tryon_batch(self.job_id)
            web_app._start_tryon_batch(self.job_id)
            for _ in range(200):
                snapshot = web_app._read_tryon_batch(self.job_id)
                if snapshot["status"] == "partial":
                    break
                time.sleep(0.005)
            else:
                self.fail("try-on batch did not finish after a partial failure")

        self.assertEqual(calls, [2, 3])
        self.assertEqual(snapshot["ready"], 2)
        self.assertEqual(snapshot["finished"], 3)
        self.assertEqual(snapshot["items"][1]["status"], "failed")
        self.assertEqual(snapshot["items"][2]["status"], "done")

    def test_manual_cached_result_does_not_finish_an_active_batch_early(self):
        web_app._initialize_tryon_batch(self.job_id)
        web_app._set_tryon_batch_item(self.job_id, 1, status="done", image="preview.jpg")
        web_app._set_tryon_batch_item(self.job_id, 2, status="done", image="tryon_2.jpg")
        web_app._set_tryon_batch_item(self.job_id, 3, status="running")

        web_app._finish_tryon_batch(self.job_id)

        snapshot = web_app._read_tryon_batch(self.job_id)
        self.assertEqual(snapshot["status"], "running")
        self.assertEqual(snapshot["ready"], 2)
        self.assertEqual(snapshot["finished"], 2)

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
