"""무신사가 파는 색 전부를 쓰는지 지킨다.

대표 색 하나만 저장하던 때: 색 옵션이 있는 상품의 68%가 2색 이상인데 나머지를 잃었고,
'그레이'로 저장된 상품이 실제로는 화이트·블랙·베이지로도 팔렸다(2026-09-25, 표본 60개).
색은 체형 규칙(preferred_top_colors)과 회피 색 조건이 직접 보는 값이다.
"""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from product_colors import palette_of, palettes_for  # noqa: E402
from product_measurements import color_options_from, normalize_size_table  # noqa: E402

TABLE = {"블랙": ("블랙", "차콜", "BLACK"), "화이트": ("화이트", "아이보리", "WHITE", "IVORY"),
         "그레이": ("그레이", "멜란지", "GRAY"), "베이지": ("베이지", "BEIGE"),
         "브라운": ("브라운", "진한갈색", "황토", "BROWN")}


def options_payload(colors, sizes=("M", "L")):
    return {"data": {"basic": [
        {"displayType": "COLOR_CHIP", "optionValues": [{"name": name} for name in colors]},
        {"displayType": "DROPDOWN", "optionValues": [{"name": name} for name in sizes]},
    ]}}


class ColorExtractionTests(unittest.TestCase):
    def test_every_color_chip_is_read(self):
        colors = color_options_from(options_payload(["화이트", "블랙", "멜란지", "베이지"]))
        self.assertEqual(colors, ["화이트", "블랙", "멜란지", "베이지"])

    def test_size_dropdown_is_not_mistaken_for_a_color(self):
        self.assertEqual(color_options_from(options_payload([], sizes=("S", "M"))), [])

    def test_seller_named_option_groups_do_not_break_it(self):
        """optionItems 의 optionName 은 판매자가 정한다('컬러' 대신 'C' 인 상품이 있다).

        그래서 색은 optionItems 가 아니라 basic 의 COLOR_CHIP 에서만 읽는다.
        """
        payload = options_payload(["(19)BLACK", "(39)IVORY"])
        payload["data"]["optionItems"] = [
            {"no": 1, "optionValues": [{"optionName": "C", "name": "(19)BLACK"},
                                       {"optionName": "S", "name": "S_085"}]}]
        self.assertEqual(color_options_from(payload), ["(19)BLACK", "(39)IVORY"])

    def test_missing_or_broken_payloads_return_nothing(self):
        for payload in (None, {}, {"data": None}, {"data": {"basic": None}}, {"data": []}):
            with self.subTest(payload=payload):
                self.assertEqual(color_options_from(payload), [])

    def test_duplicates_are_dropped_and_order_kept(self):
        self.assertEqual(color_options_from(options_payload(["블랙", "블랙", "화이트"])),
                         ["블랙", "화이트"])


class PaletteTests(unittest.TestCase):
    def test_seller_spellings_map_to_our_palette(self):
        for name, expected in (("(19)BLACK", "블랙"), ("코튼아이보리", "화이트"),
                               ("MELANGE GRAY (기모)", "그레이"), ("진한갈색", "브라운"),
                               ("황토색", "브라운")):
            with self.subTest(name=name):
                self.assertEqual(palette_of(name, TABLE), expected)

    def test_longest_match_wins_so_compound_names_are_not_misread(self):
        # '모카 그레이'는 그레이다. 짧은 단어가 먼저 걸리면 브라운으로 새어 나간다.
        table = dict(TABLE, 브라운=TABLE["브라운"] + ("모카",))
        self.assertEqual(palette_of("모카 그레이", table), "그레이")

    def test_unknown_name_maps_to_nothing_rather_than_a_guess(self):
        self.assertEqual(palette_of("형광 무지개", TABLE), "")
        self.assertEqual(palettes_for(["형광 무지개"], TABLE), [])

    def test_palettes_keep_order_and_drop_repeats(self):
        names = ["화이트", "아이보리", "(19)BLACK", "멜란지"]
        self.assertEqual(palettes_for(names, TABLE), ["화이트", "블랙", "그레이"])


class SizeTableTests(unittest.TestCase):
    def actual(self):
        return {"data": {"unit": "cm", "sizes": [
            {"name": "M", "items": [{"name": "총장", "value": 70}]}]}}

    def test_colors_ride_along_with_the_size_table(self):
        record = normalize_size_table("MS1", self.actual(), options_payload(["블랙", "아이보리"]))
        self.assertEqual(record["color_options"], ["블랙", "아이보리"])
        self.assertEqual([size["label"] for size in record["sizes"]], ["M"])

    def test_no_options_means_no_colors_not_an_error(self):
        record = normalize_size_table("MS1", self.actual(), None)
        self.assertEqual(record["color_options"], [])

    def test_schema_version_moved_so_old_caches_are_refetched(self):
        # 옛 캐시 파일에는 색이 없다. 판이 그대로면 한 시간 동안 색 없이 돈다.
        record = normalize_size_table("MS1", self.actual(), None)
        self.assertGreaterEqual(record["schema_version"], 2)


if __name__ == "__main__":
    unittest.main()
