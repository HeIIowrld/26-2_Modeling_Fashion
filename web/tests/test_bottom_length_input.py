"""사진 밖으로 잘린 하의 기장을 사용자 입력으로 보완하는 흐름.

2026-09-15 게이트 재평가에서 하의 192조합 중 72조합이 밑단 미관측으로 기장 비교를
보류했다. 추측해서 통과시키지 않고 사용자에게 묻는다.
"""
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException

import web.app as web_app
import web.pipeline as web_pipeline
from schemas import Product

HOLD = web_pipeline.bottom_length_warnings("", "", "쇼츠·미니 기장")[0]


def product(product_id: str, category: str) -> Product:
    return Product(product_id, f"{category} product", category, "블랙", "캐주얼",
                   ["데일리"], [], 50_000, "사계절", True, image_path=f"{product_id}.jpg")


class LengthCheckStatusTests(unittest.TestCase):
    def test_cut_hem_asks_for_input_and_measured_photo_does_not(self):
        cut = web_pipeline.length_check_status(SimpleNamespace(bottom_length="분석 불가"), has_bottom=True)
        self.assertEqual(cut["status"], "needs_input")
        self.assertIn("밑단", cut["message"])
        self.assertEqual(cut["options"], ["반바지", "무릎 기장", "7부 기장", "긴바지"])
        measured = web_pipeline.length_check_status(SimpleNamespace(bottom_length="긴바지"), has_bottom=True)
        self.assertEqual(measured["status"], "measured")
        none = web_pipeline.length_check_status(SimpleNamespace(bottom_length="분석 불가"), has_bottom=False)
        self.assertEqual(none["status"], "no_bottom")


class CurrentBottomLengthEndpointTests(unittest.TestCase):
    def setUp(self):
        self.job_id = "c" * 32
        self.job = {
            "id": self.job_id,
            "status": "done",
            "result": {"length_check": {"status": "needs_input"}},
            "tryon_context": {"outfit": SimpleNamespace(bottom_length="분석 불가")},
            "shopping_tryon_products": {
                "TOP1": product("TOP1", "top"),
                "SHORTS1": product("SHORTS1", "bottom"),
                "PANTS1": product("PANTS1", "bottom"),
            },
            "shopping_tryon_batch": {"status": "done", "reason": "", "items": {
                1: {"product_ids": ["TOP1", "SHORTS1"], "categories": ["top", "bottom"],
                    "status": "done", "image": "a.jpg", "warnings": ["합성 품질 점검: 흐릿함", HOLD]},
                2: {"product_ids": ["TOP1", "PANTS1"], "categories": ["top", "bottom"],
                    "status": "done", "image": "b.jpg", "warnings": [HOLD]},
            }},
            "product_tryon_warnings": {"TOP1|SHORTS1": [HOLD]},
            "work_lock": threading.Lock(),
            "cancelled": False,
        }
        with web_app._jobs_lock:
            web_app._jobs[self.job_id] = self.job
        self.lengths = {"SHORTS1": "쇼츠·미니 기장", "PANTS1": "롱·긴바지 기장"}

    def tearDown(self):
        with web_app._jobs_lock:
            web_app._jobs.pop(self.job_id, None)

    def _post(self, length):
        with patch.object(web_app, "reference_bottom_lengths", return_value=self.lengths):
            return web_app.set_current_bottom_length(self.job_id, {"length": length})

    def test_input_refreshes_finished_warnings_without_regeneration(self):
        response = self._post("긴바지")
        self.assertEqual(response["length_check"]["status"], "user_input")
        self.assertEqual(self.job["tryon_context"]["user_bottom_length"], "긴바지")
        items = {item["index"]: item for item in response["shopping_tryon_batch"]["items"]}
        # 긴바지 → 쇼츠는 큰 기장 차이, 다른 품질 경고는 그대로 남는다.
        self.assertEqual(items[1]["warnings"][0], "합성 품질 점검: 흐릿함")
        self.assertTrue(items[1]["warnings"][1].startswith("하의 기장 차이가 큽니다"))
        self.assertEqual(items[2]["warnings"], [])  # 긴바지 → 긴바지는 경고 없음
        self.assertTrue(self.job["product_tryon_warnings"]["TOP1|SHORTS1"][0].startswith("하의 기장 차이"))

    def test_unknown_product_length_keeps_existing_warnings(self):
        self.lengths = {}
        response = self._post("긴바지")
        items = {item["index"]: item for item in response["shopping_tryon_batch"]["items"]}
        self.assertEqual(items[2]["warnings"], [HOLD])

    def test_invalid_values_and_jobs_are_rejected(self):
        with self.assertRaises(HTTPException) as caught:
            self._post("발목까지")
        self.assertEqual(caught.exception.status_code, 400)
        with self.assertRaises(HTTPException) as caught:
            web_app.set_current_bottom_length("../etc", {"length": "긴바지"})
        self.assertEqual(caught.exception.status_code, 400)
        with self.assertRaises(HTTPException) as caught:
            with patch.object(web_app, "reference_bottom_lengths", return_value={}):
                web_app.set_current_bottom_length("d" * 32, {"length": "긴바지"})
        self.assertEqual(caught.exception.status_code, 404)

    def test_ui_offers_length_choices(self):
        from pathlib import Path

        javascript = (Path(web_app.WEB_DIR) / "static" / "app.js").read_text(encoding="utf-8")
        self.assertIn("/current-bottom-length", javascript)
        self.assertIn("data-bottom-length", javascript)


if __name__ == "__main__":
    unittest.main()
