"""옷 전체를 보여 주지 않는 상품 사진을 합성에서 거절하는지 검사한다.

2026-09-21 무신사 96개 조사에서, 옷이 사진의 두 변 이상에 닿은 사진(뒷모습 확대컷,
여러 색을 쌓은 묶음 사진)은 합성 결과가 상품과 전혀 다른 옷이 됐다. 해상도를 4~9배
올려도 충실도는 그대로였다 — 원인은 해상도가 아니라 사진 구성이다.
"""

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catvton_tryon import (  # noqa: E402
    CatVTONTryOn,
    MAX_EDGE_CONTACT,
    reference_edge_contact,
)
from virtual_tryon import TryOnNotReady  # noqa: E402


def _garment_on_white(box, size=(200, 240)):
    """흰 배경 위, 주어진 상자 영역에만 옷 픽셀이 있는 정제본."""
    image = np.full((size[1], size[0], 3), 255, np.uint8)
    x1, y1, x2, y2 = box
    image[y1:y2, x1:x2] = (160, 40, 40)
    return Image.fromarray(image)


class EdgeContactTests(unittest.TestCase):
    def test_flat_lay_with_margin_touches_no_edge(self):
        self.assertEqual(reference_edge_contact(_garment_on_white((40, 40, 160, 200))), 0)

    def test_model_shot_cut_at_the_bottom_touches_one_edge_and_is_allowed(self):
        # 모델 착용컷에서 하의가 발목 아래로 잘린 흔한 경우. 막으면 멀쩡한 상품을 잃는다.
        contact = reference_edge_contact(_garment_on_white((50, 60, 150, 240)))
        self.assertEqual(contact, 1)
        self.assertLessEqual(contact, MAX_EDGE_CONTACT)

    def test_close_up_crop_running_off_the_frame_is_flagged(self):
        # MS7009192: 셔츠 뒷모습을 확대해 옷이 아래·양옆 테두리를 채운다.
        self.assertGreater(reference_edge_contact(_garment_on_white((0, 70, 200, 240))), MAX_EDGE_CONTACT)


class RejectUnusableReferenceTests(unittest.TestCase):
    def _tryon(self, cache_dir, segmentation):
        tryon = CatVTONTryOn.fast(garment_cache_dir=cache_dir)
        parser = MagicMock()
        parser.backend = "fashn-human-parser"
        parser.parse.return_value = {"segmentation": segmentation}
        tryon._garment_parser = parser
        return tryon

    def test_cropped_reference_is_refused_instead_of_drawing_another_garment(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "shopping_crop.jpg"
            Image.new("RGB", (200, 240), (160, 40, 40)).save(source)
            seg = np.zeros((240, 200), np.uint8)
            seg[70:, :] = 3  # 상의가 아래·양옆으로 사진 밖까지 이어진다
            with self.assertRaises(TryOnNotReady) as caught:
                self._tryon(root / "cache", seg)._prepare_garment_reference(source, "top")
        self.assertIn("옷 전체를 보여 주지 않습니다", str(caught.exception))

    def test_garment_too_small_to_isolate_is_refused(self):
        # MS6704434: 전신 모델컷에서 반바지가 1% 남짓. 예전에는 사람·배경이 든 원본을 넘겼다.
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "shopping_tiny.jpg"
            Image.new("RGB", (200, 240), "gray").save(source)
            seg = np.zeros((240, 200), np.uint8)
            seg[150:160, 90:110] = 6
            with self.assertRaises(TryOnNotReady) as caught:
                self._tryon(root / "cache", seg)._prepare_garment_reference(source, "bottom")
        self.assertIn("너무 작게", str(caught.exception))

    def test_clean_reference_with_margin_still_renders(self):
        with TemporaryDirectory() as raw:
            root = Path(raw)
            source = root / "shopping_ok.jpg"
            Image.new("RGB", (200, 240), (160, 40, 40)).save(source)
            seg = np.zeros((240, 200), np.uint8)
            seg[40:200, 40:160] = 3
            cleaned = self._tryon(root / "cache", seg)._prepare_garment_reference(source, "top")
        # 정제본은 옷 둘레에 여백을 두고 잘린다. 거절되지 않고, 테두리에 닿지 않아야 한다.
        self.assertEqual(reference_edge_contact(cleaned), 0)

    def test_previously_cached_crop_is_refused_on_reuse(self):
        # 예전 코드로 정제해 둔 캐시(운영에 61개)도 다시 판별해야 한다.
        with TemporaryDirectory() as raw:
            root = Path(raw)
            cache = root / "cache"
            cache.mkdir()
            source = root / "shopping_old.jpg"
            Image.new("RGB", (200, 240), "white").save(source)
            _garment_on_white((0, 70, 200, 240)).save(cache / source.name)
            with self.assertRaises(TryOnNotReady):
                self._tryon(cache, np.zeros((240, 200), np.uint8))._prepare_garment_reference(source, "top")


if __name__ == "__main__":
    unittest.main()
