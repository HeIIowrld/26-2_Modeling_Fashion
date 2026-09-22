"""배포 스모크 테스트의 무신사 조합 기대값이 앱 규칙과 같은지 지킨다.

2026-09-22까지 스모크는 합성 가능한 상의×하의 곱집합(9조합)을 기대했다. 앱은 룩 탭
이후 추천 코디마다 한 조합(3조합)을 만들어, 정상 서버에서도 스모크가 항상 실패했다.
기대값을 앱과 같은 입력(분석 응답)으로 만들고, 두 쪽을 같은 작업으로 비교한다.
"""

import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path

import web.app as web_app
from schemas import Product

SCRIPT = Path(__file__).resolve().parents[2] / "gpu_server" / "scripts" / "smoke_web_api.py"
_spec = importlib.util.spec_from_file_location("smoke_web_api", SCRIPT)
smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(smoke)


def _product(product_id: str, category: str) -> Product:
    return Product(product_id, f"{category} product", category, "블랙", "캐주얼", ["데일리"], [],
                   50_000, "사계절", True, image_path=f"{product_id}.jpg")


class SmokeCombinationTests(unittest.TestCase):
    def setUp(self):
        self.job_id = "c" * 32
        self.tempdir = tempfile.TemporaryDirectory()
        person = Path(self.tempdir.name) / "person.jpg"
        person.write_bytes(b"person")
        self.job = {
            "id": self.job_id, "status": "done",
            "result": {"tryon": {"available": True, "reason": "", "warnings": []}},
            "person_image": person, "tryon_context": {"mask": object()},
            "work_lock": threading.Lock(), "cancelled": False,
        }
        with web_app._jobs_lock:
            web_app._jobs[self.job_id] = self.job

    def tearDown(self):
        with web_app._jobs_lock:
            web_app._jobs.pop(self.job_id, None)
        self.tempdir.cleanup()

    def _both(self, products: dict[str, str], prepared: set[str], outfits: list[list[str]] | None):
        """같은 분석 결과로 앱이 만든 조합과 스모크가 기대한 조합을 돌려준다."""
        result = self.job["result"]
        result["shopping_results"] = [
            {"product_id": pid, "category": category, "tryon_available": pid in prepared, "tryon_reason": ""}
            for pid, category in products.items()
        ]
        if outfits is not None:
            result["shopping_outfits"] = [{"product_ids": ids} for ids in outfits]
        self.job["shopping_tryon_products"] = {
            pid: _product(pid, category) for pid, category in products.items() if pid in prepared
        }
        batch = web_app._initialize_shopping_tryon_batch(self.job_id)
        app = [item["product_ids"] for item in batch["items"]]
        return app, smoke.expected_shopping_combinations(result)

    def test_recommended_outfits_define_one_combination_each(self):
        # 2026-09-22 운영 서버에서 받은 실제 응답(상의 3·하의 3, 추천 코디 3개).
        products = {"MS6013026": "top", "MS6829670": "top", "MS3835929": "top",
                    "MS6270643": "bottom", "MS6981068": "bottom", "MS7066724": "bottom"}
        outfits = [["MS6013026", "MS6270643"], ["MS6829670", "MS6981068"], ["MS3835929", "MS7066724"]]
        app, expected = self._both(products, set(products), outfits)
        self.assertEqual(expected, outfits)
        self.assertEqual(app, expected)

    def test_unprepared_products_and_duplicate_outfits_follow_the_app(self):
        products = {"T1": "top", "B1": "bottom", "B2": "bottom", "S1": "shoes"}
        outfits = [["T1", "B1", "S1"], ["T1", "B2", "S1"], ["T1", "B1", "S1"]]
        app, expected = self._both(products, {"T1", "B1", "S1"}, outfits)
        self.assertEqual(expected, [["T1", "B1", "S1"], ["T1", "S1"]])
        self.assertEqual(app, expected)

    def test_results_without_outfits_fall_back_to_the_category_product(self):
        products = {"T1": "top", "T2": "top", "B1": "bottom", "B2": "bottom"}
        app, expected = self._both(products, set(products), None)
        self.assertEqual(expected, [["T1", "B1"], ["T1", "B2"], ["T2", "B1"], ["T2", "B2"]])
        self.assertEqual(app, expected)


if __name__ == "__main__":
    unittest.main()
