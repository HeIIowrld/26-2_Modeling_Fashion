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

from product_colors import (  # noqa: E402
    color_from_pixels, palette_from_rgb, palette_of, palettes_for, title_palettes)
from product_measurements import color_options_from, normalize_size_table  # noqa: E402

TABLE = {"블랙": ("블랙", "차콜", "BLACK"), "화이트": ("화이트", "아이보리", "WHITE", "IVORY"),
         "그레이": ("그레이", "멜란지", "GRAY", "GREY"), "베이지": ("베이지", "BEIGE"),
         "브라운": ("브라운", "진한갈색", "황토", "BROWN"), "블루": ("블루", "BLUE"),
         "레드": ("레드", "RED")}


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


VOCABULARY = {"non_color_terms": ["블루종", "데님", "인디고", "화이트라벨"],
              "english_needs_left_boundary": True, "min_korean_term_length": 2}


class TitleColorTests(unittest.TestCase):
    """상품명에서 색을 읽는 규격. 색 단어를 품은 다른 말에 걸리면 안 된다.

    2026-09-25 카탈로그 2224개 전수 확인: '블루종'이 블루로 42건, 영어 어미(LAYERED·
    TEXTURED·COVERED·EMBROIDERED·Flared·Tailored)의 RED 가 21건 잡혔다.
    """

    def read(self, name):
        return title_palettes(name, TABLE, VOCABULARY)

    def test_garment_names_that_contain_a_color_word_are_not_colors(self):
        for name in ("어반스퀘어 소프트쉘 3L 블루종 점퍼 SILVER BIRCH",
                     "오버핏 데님 미니 로고 셔츠 [인디고]"):
            with self.subTest(name=name):
                self.assertEqual(self.read(name), [])

    def test_english_color_glued_to_another_word_is_not_a_color(self):
        for name in ("V-cut LAYERED Short Sleeve", "TEXTURED Knit", "Tailored Jacket"):
            with self.subTest(name=name):
                self.assertEqual(self.read(name), [])

    def test_english_suffix_still_counts_as_the_color(self):
        # YELLOWISH·GREYISH 는 진짜 색이다. 뒤에 붙는 접미는 막지 않는다.
        self.assertEqual(self.read("GREYISH 맨투맨"), ["그레이"])

    def test_modifiers_in_front_keep_the_color(self):
        self.assertEqual(self.read("다크그레이 후드"), ["그레이"])
        self.assertEqual(self.read("포미 다크블루 스판 부츠컷 팬츠"), ["블루"])

    def test_brand_line_is_masked_but_a_real_color_survives(self):
        self.assertEqual(self.read("NM5PS01J 화이트라벨 웰트 후디 BLACK"), ["블랙"])

    def test_two_colors_are_both_reported_so_the_caller_can_skip(self):
        self.assertEqual(self.read("블랙:화이트 트랙 재킷"), ["블랙", "화이트"])

    def test_no_color_in_the_name_returns_nothing_not_a_default(self):
        # 예전 크롤러는 여기서 '그레이'를 넣었고, 근거 없는 그레이가 329개 쌓였다.
        self.assertEqual(self.read("짱구 닭살커플 반팔 티셔츠 2팩"), [])

    def test_one_letter_color_words_are_ignored(self):
        self.assertEqual(title_palettes("스탠다드 핏 셔츠", dict(TABLE, 브라운=("탄",)), VOCABULARY), [])


HSV = {"hsv": {"dark": 0.16, "gray_low": 0.22, "neutral_sat": 0.12,
                "navy_value": 0.35, "beige_sat": 0.30}}


class GarmentPixelColorTests(unittest.TestCase):
    """상품명·컬러칩에 색이 없을 때 의류 영역에서 색을 정한다.

    RGB 최근접은 어두운 옷을 전부 검정으로 보낸다. 명도·채도를 먼저 가르면 내가 사진을
    보고 라벨한 192장에서 58% → 69%(평가 절반)였다. 임계값은 나머지 절반에서 골랐다.
    """

    def test_dark_colours_are_not_all_black(self):
        for rgb, expected in (((30, 40, 80), "네이비"), ((90, 60, 35), "브라운"),
                              ((110, 105, 60), "카키"), ((20, 20, 22), "블랙")):
            with self.subTest(rgb=rgb):
                self.assertEqual(palette_from_rgb(rgb, HSV), expected)

    def test_neutrals_split_by_brightness(self):
        self.assertEqual(palette_from_rgb((128, 128, 130), HSV), "그레이")
        self.assertEqual(palette_from_rgb((240, 240, 240), HSV), "화이트")

    def test_agreement_is_the_share_of_the_winning_colour(self):
        color, agreement = color_from_pixels([(20, 20, 22)] * 8 + [(90, 60, 35)] * 2, HSV)
        self.assertEqual(color, "블랙")
        self.assertEqual(agreement, 0.8)

    def test_no_pixels_means_no_answer(self):
        self.assertEqual(color_from_pixels([], HSV), ("", 0.0))

    def test_a_mixed_garment_reports_low_agreement(self):
        # 프린트가 많은 옷은 일치율이 낮게 나오고, 그 값으로 걸러진다.
        pixels = [(20, 20, 22)] * 5 + [(240, 240, 240)] * 5
        _color, agreement = color_from_pixels(pixels, HSV)
        self.assertLessEqual(agreement, 0.5)


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
