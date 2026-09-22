"""긴 바지 상품의 하의 마스크(long-hull) 회귀 테스트.

2026-09-21 A/B(reports/vton_quality/bottom_shape_2026-09-21.md)에서 확인한 실패를 고정한다.
- 원래 하의 윤곽을 담은 마스크는 7부 조거 밑단의 계단을 새 바지에 옮긴다 → 볼록 껍질로 지운다.
- 드러난 신발은 보호 영역이라 긴 바짓단이 발목에서 모인다 → 신발 윗부분만 보호를 푼다.
- 반바지·치마·기장 판정 보류 상품은 검증하지 않았으므로 기존 마스크를 쓴다.
- 스키니·레깅스 상품은 껍질이 바지를 넓혀(2026-09-22 확인) 기존 마스크를 쓴다.
"""

import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from catvton_tryon import (  # noqa: E402
    CatVTONTryOn,
    long_bottom_mask,
    shoe_top_region,
)
from schemas import Product, Recommendation  # noqa: E402


def _jogger_scene(height=200, width=120, scale=1):
    """7부 조거를 입은 사람: 넓은 바지(6) 아래로 가는 종아리(14), 그 아래 신발(15)."""
    s = scale
    seg = np.zeros((height * s, width * s), np.uint8)
    seg[20 * s:60 * s, 30 * s:90 * s] = 3            # 상의
    seg[60 * s:100 * s, 22 * s:28 * s] = 12          # 팔
    seg[100 * s:110 * s, 22 * s:28 * s] = 13         # 손
    seg[60 * s:140 * s, 30 * s:58 * s] = 6           # 왼 다리 조거
    seg[60 * s:140 * s, 62 * s:90 * s] = 6           # 오른 다리 조거
    seg[60 * s:80 * s, 58 * s:62 * s] = 6            # 가랑이 위
    seg[140 * s:175 * s, 38 * s:50 * s] = 14         # 왼 종아리
    seg[140 * s:175 * s, 70 * s:82 * s] = 14         # 오른 종아리
    seg[175 * s:190 * s, 32 * s:54 * s] = 15         # 왼 신발
    seg[175 * s:190 * s, 66 * s:88 * s] = 15         # 오른 신발
    lower = np.isin(seg, (4, 5, 6, 7, 14))           # clothing_parser의 lower_style_mask
    return seg, lower


class LongBottomMaskTests(unittest.TestCase):
    def test_original_hem_step_is_removed(self):
        seg, lower = _jogger_scene()
        mask, _ = long_bottom_mask(lower, seg)
        # 원래 마스크는 조거 밑단 아래(160행)에서 종아리 폭으로 좁아진다. 긴 바지 마스크는 좁아지지 않는다.
        self.assertFalse(lower[160, 32])
        self.assertTrue(mask[160, 32])
        self.assertTrue(mask[160, 60])   # 다리 사이(와이드 바지가 채울 자리)
        self.assertTrue(mask[lower].all())  # 원래 마스크 픽셀은 그대로 남는다

    def test_only_the_top_of_each_shoe_is_released(self):
        seg, lower = _jogger_scene()
        mask, released = long_bottom_mask(lower, seg)
        self.assertTrue(released[176, 40] and mask[176, 40])   # 신발 등: 바짓단이 덮을 자리
        self.assertFalse(released[188, 40] or mask[188, 40])   # 신발 바닥 쪽은 계속 보호
        self.assertTrue(np.array_equal(released, shoe_top_region(seg)))

    def test_top_arms_and_hands_are_never_added(self):
        seg, lower = _jogger_scene()
        mask, _ = long_bottom_mask(lower, seg)
        for label in (3, 12, 13):
            self.assertFalse(mask[seg == label].any(), label)

    def test_a_second_person_is_not_bridged_or_released(self):
        # 오른쪽에 떨어져 선 행인: 두 사람 사이 배경을 껍질로 메우거나 행인 신발을 풀면 안 된다.
        seg = np.zeros((200, 260), np.uint8)
        for offset in (0, 150):
            seg[60:140, 30 + offset:90 + offset] = 6
            seg[140:175, 40 + offset:80 + offset] = 14
            seg[175:190, 35 + offset:85 + offset] = 15
        lower = np.isin(seg, (4, 5, 6, 7, 14))
        landmarks = {"left_hip": (80.0, 62.0), "right_hip": (40.0, 62.0),
                     "left_ankle": (70.0, 172.0), "right_ankle": (50.0, 172.0)}
        mask, released = long_bottom_mask(lower, seg, landmarks)
        self.assertFalse(mask[100, 120])                 # 두 사람 사이 배경
        self.assertTrue(released[176, 60])               # 포즈가 잡은 사람의 신발 등
        self.assertFalse(released[176, 210] or mask[176, 210])  # 행인의 신발
        self.assertTrue(mask[lower].all())               # 행인 하의는 원래 마스크 그대로

    def test_tiny_mislabeled_shoe_fragments_are_not_released(self):
        seg, lower = _jogger_scene()
        seg[100:105, 100:105] = 15  # 25픽셀짜리 오라벨 조각
        self.assertFalse(shoe_top_region(seg)[100:105, 100:105].any())


class LongHullPolicyTests(unittest.TestCase):
    def _apply(self, reference_length, *, name="와이드 슬랙스", policy="long-hull"):
        adapter = CatVTONTryOn(lower_mask_policy=policy)
        seg, lower = _jogger_scene()
        product = SimpleNamespace(name=name, product_id="MS1")
        return lower, adapter._apply_mask_policy(
            lower, "bottom", Path("MS1.jpg"), None, product, seg, {}, {},
            reference_length=reference_length,
        )

    def test_long_pants_get_the_long_hull_mask(self):
        lower, (mask, released) = self._apply("롱·긴바지 기장")
        self.assertIsNotNone(released)
        self.assertGreater(mask.sum(), lower.sum())

    def test_other_bottoms_keep_the_native_mask(self):
        # 반바지·기장 판정 보류·치마·native 정책은 2026-09-21 A/B 범위 밖이다.
        for reference, name, policy in (("쇼츠·미니 기장", "와이드 슬랙스", "long-hull"),
                                        ("", "와이드 슬랙스", "long-hull"),
                                        ("롱·긴바지 기장", "플리츠 롱 스커트", "long-hull"),
                                        # 스키니·레깅스는 껍질이 원래 하의 폭만큼 넓혀 일자·와이드로 그려진다.
                                        ("롱·긴바지 기장", "하이웨이스트 스키니 진", "long-hull"),
                                        ("롱·긴바지 기장", "CS CURVE SKINNY LEGGINGS (BLACK)", "long-hull"),
                                        ("롱·긴바지 기장", "와이드 슬랙스", "native")):
            lower, (mask, released) = self._apply(reference, name=name, policy=policy)
            self.assertIsNone(released, (reference, name, policy))
            self.assertIs(mask, lower)

    def test_long_hull_is_the_default(self):
        self.assertEqual(CatVTONTryOn().lower_mask_policy, "long-hull")
        self.assertEqual(CatVTONTryOn.fast().lower_mask_policy, "long-hull")


class LongHullGenerateTests(unittest.TestCase):
    def test_released_shoe_top_is_painted_and_not_restored(self):
        seg, lower = _jogger_scene(scale=3)  # 600 x 360
        adapter = CatVTONTryOn(width=72, height=120, post_quality_gate=False)
        seen = {}

        def paint_red(current, _garment, mask, **_kwargs):
            seen["mask"] = np.asarray(mask)
            return Image.new("RGB", current.size, (220, 20, 20))

        with tempfile.TemporaryDirectory() as directory:
            person = Path(directory) / "person.png"
            Image.new("RGB", (360, 600), (128, 128, 128)).save(person)
            garment = Path(directory) / "MS1.jpg"
            Image.new("RGB", (48, 64), "black").save(garment)
            product = Product("MS1", "와이드 슬랙스", "bottom", "블랙", "", [], [], 0, "", True,
                              image_path=str(garment))
            recommendation = Recommendation(rank=1, products=[product], total_score=0.0,
                                            score_breakdown={}, reasons=[])
            fake_utils = SimpleNamespace(resize_and_crop=lambda image, size: image.resize(size))
            with (
                mock.patch.dict(sys.modules, {"utils": fake_utils}),
                mock.patch.object(adapter, "_load_pipeline"),
                mock.patch.object(adapter, "_prepare_garment_reference",
                                  return_value=Image.new("RGB", (48, 64), "black")),
                mock.patch.object(adapter, "_check_length_gap", return_value="롱·긴바지 기장"),
                mock.patch.object(adapter, "_tryon_once", side_effect=paint_red),
            ):
                output = adapter.generate(person, recommendation, Path(directory) / "out.png",
                                          context={"lower_style_mask": lower, "segmentation": seg})
            result = np.asarray(Image.open(output).convert("RGB")).astype(int)

        self.assertIn("lower:long-hull", adapter.last_mask_notes)
        # 모델 좌표(72x120, 여백 없음)에서 신발 등(원본 529행)은 마스크 안, 신발 바닥(565행)은 밖이다.
        self.assertGreater(seen["mask"][int(529 / 5), int(120 / 5)], 127)
        self.assertLess(seen["mask"][int(565 / 5), int(120 / 5)], 128)
        # 풀어 준 신발 등은 새로 칠한 색이 남고, 신발 바닥은 원본(회색)으로 복원된다.
        self.assertGreater(result[529, 120, 0], 180)
        self.assertLess(abs(result[565, 120] - 128).max(), 12)


if __name__ == "__main__":
    unittest.main()
