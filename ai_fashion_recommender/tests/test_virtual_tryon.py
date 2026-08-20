"""예상 착장샷 자리와 생성 모델이 붙을 이음새를 지킨다.

생성 모델을 연결할 때 이 테스트가 통과하면 웹 화면도 그대로 동작한다.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent / "web"))

import app as web_app
from schemas import Product, Recommendation
from virtual_tryon import TryOnNotReady, VirtualTryOnAdapter


def sample_recommendation(rank: int = 1, with_products: bool = True) -> Recommendation:
    products = (
        [
            Product("TOP001", "화이트 셔츠", "top", "화이트", "미니멀", ["데일리"], ["균형형"], 59000, "사계절", True),
            Product("BOT001", "블랙 슬랙스", "bottom", "블랙", "포멀", ["데일리"], ["균형형"], 69000, "사계절", True),
        ]
        if with_products
        else []
    )
    return Recommendation(rank=rank, products=products, total_score=95.0, score_breakdown={}, reasons=[])


class AdapterAvailabilityTests(unittest.TestCase):
    def test_disabled_adapter_reports_unavailable_with_a_reason(self):
        adapter = VirtualTryOnAdapter(enabled=False)
        self.assertFalse(adapter.available)
        self.assertIn("생성 모델", adapter.NOT_READY_REASON)

    def test_synthesize_refuses_instead_of_silently_returning_a_board(self):
        """추천 보드를 착장샷인 척 돌려주면 사용자가 오해한다."""
        adapter = VirtualTryOnAdapter(enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TryOnNotReady):
                adapter.synthesize(
                    person_image=_person(directory),
                    recommendation=sample_recommendation(),
                    output_path=Path(directory) / "out.jpg",
                )

    def test_generate_still_produces_the_recommendation_board(self):
        """Notebook 경로는 예전처럼 추천 보드를 만들어야 한다."""
        adapter = VirtualTryOnAdapter(enabled=False)
        with tempfile.TemporaryDirectory() as directory:
            output = adapter.generate(
                person_image=_person(directory),
                recommendation=sample_recommendation(),
                output_path=Path(directory) / "board.jpg",
            )
            self.assertTrue(output.is_file())

    def test_enabled_adapter_delegates_to_the_generative_model(self):
        adapter = VirtualTryOnAdapter(enabled=True)
        self.assertTrue(adapter.available)
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "tryon_1.jpg"

            def fake_model(person_image, recommendation, output_path):
                Image.new("RGB", (64, 96), "white").save(output_path)
                return Path(output_path)

            with mock.patch.object(VirtualTryOnAdapter, "_synthesize", staticmethod(fake_model)):
                result = adapter.synthesize(
                    person_image=_person(directory),
                    recommendation=sample_recommendation(),
                    output_path=target,
                )
            self.assertEqual(result, target)
            self.assertTrue(target.is_file())

    def test_unimplemented_model_raises_a_typed_error(self):
        """구현 전에는 NotImplementedError가 아니라 화면에 띄울 수 있는 오류여야 한다."""
        adapter = VirtualTryOnAdapter(enabled=True)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TryOnNotReady):
                adapter.synthesize(
                    person_image=_person(directory),
                    recommendation=sample_recommendation(),
                    output_path=Path(directory) / "out.jpg",
                )


class TryOnEndpointTests(unittest.TestCase):
    def test_generated_filenames_are_servable(self):
        """tryon_1.jpg 같은 이름이 이미지 경로 검사를 통과해야 한다."""
        for name in ("tryon_1.jpg", "tryon_2.jpg", "original.jpg", "pose_landmarks.jpg"):
            with self.subTest(name=name):
                self.assertTrue(web_app.IMAGE_NAME_PATTERN.match(name))

    def test_path_traversal_names_are_still_rejected(self):
        for name in ("../secret.jpg", "a/b.jpg", "tryon_1.png"):
            with self.subTest(name=name):
                self.assertIsNone(web_app.IMAGE_NAME_PATTERN.match(name))

    def test_unknown_job_is_not_found(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            web_app.create_tryon("f" * 32, 1)
        self.assertEqual(caught.exception.status_code, 404)

    def test_malformed_job_id_is_rejected(self):
        from fastapi import HTTPException

        with self.assertRaises(HTTPException) as caught:
            web_app.create_tryon("../../etc", 1)
        self.assertEqual(caught.exception.status_code, 400)


def _person(directory: str) -> Path:
    path = Path(directory) / "person.jpg"
    Image.new("RGB", (400, 600), "gray").save(path)
    return path


if __name__ == "__main__":
    unittest.main()
