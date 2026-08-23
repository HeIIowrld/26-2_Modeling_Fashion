"""크롤링 카탈로그를 추천 엔진에 이어붙이는 경로를 지킨다.

크롤링 원본(products_musinsa.csv)은 15개 칼럼뿐이라 그대로 쓰면 규칙이 잠든다.
그래서 (1) 카탈로그를 고르는 규칙이 한 곳에만 있고, (2) 크롤링 원본이 런타임
카탈로그로 뽑히지 않으며, (3) 속성을 채운 결과가 규칙을 발동시킬 수 있는지 검사한다.
"""

import csv
import importlib
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import config  # noqa: E402
from product_catalog import ProductCatalog  # noqa: E402

import enrich_catalog  # noqa: E402


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class CatalogResolutionTests(unittest.TestCase):
    """어떤 CSV를 쓸지는 config.resolve_catalog 한 곳에서만 정한다."""

    def setUp(self):
        self._saved = os.environ.pop("FASHION_PRODUCTS_CSV", None)

    def tearDown(self):
        os.environ.pop("FASHION_PRODUCTS_CSV", None)
        if self._saved is not None:
            os.environ["FASHION_PRODUCTS_CSV"] = self._saved

    def test_falls_back_to_handmade_catalog(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "products.csv").write_text("product_id\n", encoding="utf-8")
            self.assertEqual(config.resolve_catalog(base).name, "products.csv")

    def test_enriched_catalog_wins_over_handmade(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "products.csv").write_text("product_id\n", encoding="utf-8")
            (base / "products_musinsa_enriched.csv").write_text("product_id\n", encoding="utf-8")
            self.assertEqual(
                config.resolve_catalog(base).name, "products_musinsa_enriched.csv"
            )

    def test_raw_crawl_is_never_selected(self):
        """속성 16개가 비어 있어 규칙이 조용히 잠들기 때문이다."""
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "products.csv").write_text("product_id\n", encoding="utf-8")
            (base / "products_musinsa.csv").write_text("product_id\n", encoding="utf-8")
            self.assertEqual(config.resolve_catalog(base).name, "products.csv")

    def test_environment_override_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / "products.csv").write_text("product_id\n", encoding="utf-8")
            chosen = base / "mine.csv"
            chosen.write_text("product_id\n", encoding="utf-8")
            os.environ["FASHION_PRODUCTS_CSV"] = str(chosen)
            self.assertEqual(config.resolve_catalog(base), chosen.resolve())


class DerivationTableTests(unittest.TestCase):
    """유도 표가 규칙이 읽는 범위를 벗어나지 않아야 한다."""

    @classmethod
    def setUpClass(cls):
        cls.table = json.loads(
            (ROOT / "data" / "catalog_derivation.json").read_text(encoding="utf-8")
        )

    def test_marked_as_provisional(self):
        """근거 없는 수치를 확정값처럼 두지 않는다."""
        self.assertIn("잠정값", self.table["_status"])

    def test_material_scores_stay_in_range(self):
        for material, values in self.table["material"].items():
            with self.subTest(material=material):
                for key in ("warmth", "breathability", "formality"):
                    self.assertIn(values[key], range(1, 6))
                self.assertIsInstance(values["water_resistant"], bool)

    def test_every_known_material_is_covered(self):
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        labels = ATTRIBUTE_TASKS["material"]
        labels = labels if isinstance(labels, (list, tuple)) else labels.labels
        self.assertEqual(set(labels) - set(self.table["material"]), set())

    def test_every_known_pattern_is_covered(self):
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        labels = ATTRIBUTE_TASKS["pattern"]
        labels = labels if isinstance(labels, (list, tuple)) else labels.labels
        self.assertEqual(set(labels) - set(self.table["pattern"]), set())

    def test_plain_pattern_has_no_scale(self):
        """무지인데 pattern_scale이 차 있으면 카탈로그 검사가 깨진다."""
        self.assertEqual(self.table["pattern"]["무지"]["scale"], "")
        self.assertEqual(self.table["pattern"]["무지"]["contrast"], 0)


class DerivedRowTests(unittest.TestCase):
    """사진에서 읽은 라벨이 규칙이 쓸 수 있는 값으로 바뀌는지 본다."""

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def derive(self, category, predicted):
        return enrich_catalog.derive({"category": category}, predicted, self.table)

    def test_blazer_is_formal_even_in_cotton(self):
        out = self.derive("top", {"item_type": "블레이저", "material": "코튼", "pattern": "무지"})
        self.assertGreaterEqual(out["formality"], 5)

    def test_leather_is_water_resistant(self):
        out = self.derive("top", {"item_type": "재킷", "material": "가죽", "pattern": "무지"})
        self.assertEqual(out["water_resistant"], "true")

    def test_sleeveless_is_cooler_than_long_sleeve(self):
        base = {"item_type": "티셔츠", "material": "코튼", "pattern": "무지"}
        sleeveless = self.derive("top", {**base, "_sleeve": "민소매"})
        long_sleeve = self.derive("top", {**base, "_sleeve": "긴팔"})
        self.assertLess(sleeveless["warmth"], long_sleeve["warmth"])
        self.assertGreater(sleeveless["breathability"], long_sleeve["breathability"])

    def test_track_pants_carry_an_exercise_tag(self):
        out = self.derive("bottom", {"item_type": "트랙팬츠", "material": "코튼", "pattern": "무지"})
        self.assertIn("운동", out["activity_tags"])

    def test_activity_tags_are_never_empty(self):
        out = self.derive("top", {"item_type": "베스트", "material": "울", "pattern": "무지"})
        self.assertTrue(out["activity_tags"])

    def test_numeric_columns_stay_in_catalog_range(self):
        out = self.derive("top", {"item_type": "블레이저", "material": "스팽글에 없는 소재",
                                  "pattern": "그래픽", "_detail": "스팽글"})
        for name in ("formality", "warmth", "breathability", "visual_weight", "detail_level"):
            with self.subTest(column=name):
                self.assertIn(out[name], range(1, 6))

    def test_tops_get_no_waistline(self):
        out = self.derive("top", {"item_type": "셔츠", "material": "코튼", "pattern": "무지"})
        self.assertEqual(out["waistline"], "")


class NormalizationTests(unittest.TestCase):
    """모델 라벨을 규칙 엔진이 비교하는 어휘로 옮겼는지 본다.

    recommendation_engine.py 는 손으로 만든 카탈로그 표기로 비교하므로, 모델의
    스키마 라벨을 그대로 넣으면 값이 차 있는데도 규칙이 조용히 빗나간다.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def norm(self, **predicted):
        return enrich_catalog.normalize(dict(predicted), self.table)

    def test_long_bottom_length_matches_the_rule_vocabulary(self):
        """R-SIL: bottom["length"] in {"긴바지", "롱·맥시 기장"} 와 맞아야 한다."""
        self.assertEqual(self.norm(length="롱·긴바지 기장")["length"], "긴바지")

    def test_short_bottom_length_is_normalized(self):
        self.assertEqual(self.norm(length="쇼츠·미니 기장")["length"], "반바지")

    def test_top_lengths_pass_through(self):
        for label in ("크롭 기장", "기본 기장", "롱 기장"):
            with self.subTest(label=label):
                self.assertEqual(self.norm(length=label)["length"], label)

    def test_coat_length_comes_from_the_item_type(self):
        """평면 상품컷에서는 upper_length 헤드가 "롱 기장"을 내지 못한다(실측 214개 중 0개).

        스키마가 "재킷"과 "코트"를 나눠 두므로 코트만 종류에서 기장을 유도한다.
        """
        self.assertEqual(self.norm(item_type="코트", length="기본 기장")["length"], "롱 기장")

    def test_jacket_length_is_left_alone(self):
        """재킷은 실제로 짧으므로 덮어쓰지 않는다."""
        self.assertEqual(self.norm(item_type="재킷", length="기본 기장")["length"], "기본 기장")

    def test_no_collar_is_not_a_neckline(self):
        self.assertEqual(self.norm(neckline="칼라 없음")["neckline"], "")

    def test_real_necklines_survive(self):
        self.assertEqual(self.norm(neckline="V넥")["neckline"], "V넥")

    def test_every_model_length_label_is_mapped(self):
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        for task in ("upper_length", "lower_length"):
            labels = ATTRIBUTE_TASKS[task]
            labels = labels if isinstance(labels, (list, tuple)) else labels.labels
            with self.subTest(task=task):
                self.assertEqual(set(labels) - set(self.table["normalize_length"]), set())

    def test_normalized_lengths_are_values_the_rules_compare_against(self):
        """규칙이 실제로 쓰는 표기 밖으로 새 값을 만들어내지 않는지 확인한다."""
        known = {"크롭 기장", "기본 기장", "롱 기장", "긴바지", "반바지",
                 "미디 기장", "무릎 기장", "롱·맥시 기장"}
        self.assertEqual(set(self.table["normalize_length"].values()) - known, set())


class MusinsaFactTests(unittest.TestCase):
    """무신사가 직접 표기한 값이 모델 추측을 이겨야 한다.

    상품명 키워드 추측은 실측에서 색상 38%·계절 85%가 기본값으로 떨어졌다.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def apply(self, row, out=None):
        out = dict(out or {})
        used = enrich_catalog.apply_musinsa_facts(row, out, self.table)
        return out, used

    def test_color_overrides_the_name_guess(self):
        out, used = self.apply({"detail_color": "화이트"}, {"color": "그레이"})
        self.assertEqual(out["color"], "화이트")
        self.assertIn("color", used)

    def test_color_synonyms_map_into_the_palette(self):
        from outfit_analyzer import COLOR_PALETTE

        palette = set(COLOR_PALETTE)
        for musinsa, expected in (("아이보리", "화이트"), ("차콜", "블랙"),
                                  ("올리브", "그린"), ("와인", "버건디")):
            with self.subTest(musinsa=musinsa):
                out, _ = self.apply({"detail_color": musinsa})
                self.assertEqual(out["color"], expected)
                self.assertIn(out["color"], palette)

    def test_unknown_color_leaves_the_guess_alone(self):
        out, used = self.apply({"detail_color": "형광 무지개"}, {"color": "그레이"})
        self.assertEqual(out["color"], "그레이")
        self.assertNotIn("color", used)

    def test_season_comes_from_musinsa(self):
        out, used = self.apply({"detail_season": "겨울"}, {"season": "사계절"})
        self.assertEqual(out["season"], "겨울")
        self.assertIn("season", used)

    def test_fit_overrides_the_model(self):
        out, _ = self.apply({"detail_fit": "오버핏"}, {"fit": "레귤러핏"})
        self.assertEqual(out["fit"], "오버핏")

    def test_thickness_sets_warmth(self):
        thick, _ = self.apply({"detail_thickness": "두꺼움"})
        thin, _ = self.apply({"detail_thickness": "얇음"})
        self.assertGreater(thick["warmth"], thin["warmth"])
        self.assertIn(thick["warmth"], range(1, 6))

    def test_sheer_sets_breathability(self):
        sheer, _ = self.apply({"detail_sheer": "있음"})
        opaque, _ = self.apply({"detail_sheer": "없음"})
        self.assertGreater(sheer["breathability"], opaque["breathability"])

    def test_missing_detail_changes_nothing(self):
        """상세를 못 받은 상품은 모델 추정을 그대로 쓴다."""
        before = {"color": "그레이", "season": "사계절", "fit": "레귤러핏"}
        out, used = self.apply({}, before)
        self.assertEqual(out, before)
        self.assertEqual(used, [])

    def test_material_is_never_filled_from_musinsa(self):
        """소재는 상세 이미지 안에 있어 API로 못 가져온다. 모델 추정으로 남는다."""
        out, used = self.apply({"detail_color": "블랙", "detail_season": "겨울"},
                               {"material": "코튼"})
        self.assertEqual(out["material"], "코튼")
        self.assertNotIn("material", used)


class MusinsaItemTypeTests(unittest.TestCase):
    """item_type 은 무신사 카테고리 경로에서 뽑는다.

    모델 category 는 상의 248개 실측에서 72.2%였고 재킷을 코트로 43회 오판했다.
    그 오판이 "코트면 롱 기장" 규칙까지 오염시켜, 짧은 재킷이 롱 기장이 된다.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def item_type(self, path):
        return enrich_catalog.musinsa_item_type({"detail_category": path}, self.table)

    def test_jacket_is_not_read_as_a_coat(self):
        """실측에서 가장 많았던 오판 방향이다."""
        self.assertEqual(self.item_type("Clothing > 재킷 > 블루종/MA-1"), "재킷")
        self.assertEqual(self.item_type("Clothing > 아우터 > 레더 재킷"), "재킷")

    def test_real_coats_still_map_to_coat(self):
        self.assertEqual(self.item_type("Clothing > 코트 > 싱글 코트"), "코트")
        self.assertEqual(self.item_type("Clothing > 코트 > 트렌치 코트"), "코트")

    def test_specific_keywords_win_over_general_ones(self):
        """'후드 집업'이 '후드'보다 먼저 검사돼야 한다."""
        self.assertEqual(self.item_type("Clothing > 후드 집업"), "후드티")
        self.assertEqual(self.item_type("Clothing > 티셔츠 > 반소매 티셔츠"), "티셔츠")

    def test_bottoms_use_the_lower_subtype_vocabulary(self):
        self.assertEqual(self.item_type("Clothing > 바지 > 데님 팬츠"), "청바지")
        self.assertEqual(self.item_type("Clothing > 바지 > 슬랙스"), "슬랙스")
        self.assertEqual(self.item_type("Clothing > 스커트 > 미니 스커트"), "스커트")

    def test_unmapped_path_falls_back_to_the_model(self):
        self.assertEqual(self.item_type("Clothing > 듣도보도 못한 분류"), "")

    def test_missing_path_falls_back_to_the_model(self):
        self.assertEqual(self.item_type(""), "")

    def test_every_mapped_label_is_known_to_the_schema(self):
        """규칙 엔진이 모르는 라벨을 만들어내면 격식·활동 태그가 기본값으로 뭉개진다."""
        from fashion_attribute_schema import ATTRIBUTE_TASKS

        known = set()
        for task in ("category", "lower_subtype"):
            labels = ATTRIBUTE_TASKS[task]
            known |= set(labels if isinstance(labels, (list, tuple)) else labels.labels)
        produced = {label for _, label in self.table["musinsa_item_type"]}
        self.assertEqual(produced - known, set())

    def test_coat_rule_uses_the_corrected_item_type(self):
        """무신사가 재킷이라 하면 롱 기장으로 늘어나면 안 된다."""
        jacket = enrich_catalog.normalize(
            {"item_type": "재킷", "length": "기본 기장"}, self.table)
        coat = enrich_catalog.normalize(
            {"item_type": "코트", "length": "기본 기장"}, self.table)
        self.assertEqual(jacket["length"], "기본 기장")
        self.assertEqual(coat["length"], "롱 기장")


class PurposeAndStyleTests(unittest.TestCase):
    """목적·스타일을 종류와 격식에서 유도한다.

    크롤러의 상품명 키워드 추측은 데일리·출근·데이트·여행만 낼 수 있어 면접·결혼식
    상품이 0개가 되고, 그 목적을 고르면 추천 엔진이 ValueError 를 낸다(재현 확인).
    스타일도 480개 중 410개(85%)가 캐주얼 기본값으로 떨어져 포멀 후보가 사라졌다.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def purposes(self, item_type, formality=3):
        return enrich_catalog.derive_purposes(item_type, formality, self.table).split("|")

    def test_formal_items_carry_the_interview_purpose(self):
        self.assertIn("면접", self.purposes("블레이저", 5))
        self.assertIn("면접", self.purposes("슬랙스", 4))

    def test_formal_items_carry_the_wedding_purpose(self):
        self.assertIn("결혼식", self.purposes("블레이저", 5))

    def test_casual_items_do_not_claim_formal_purposes(self):
        casual = self.purposes("티셔츠", 2)
        self.assertNotIn("면접", casual)
        self.assertNotIn("결혼식", casual)

    def test_every_item_gets_at_least_one_purpose(self):
        """목적이 비면 그 상품은 어떤 조건에도 안 걸려 후보에서 사라진다."""
        for item_type in ("", "듣도보도 못한 종류", "티셔츠"):
            with self.subTest(item_type=item_type):
                self.assertTrue(self.purposes(item_type, 3)[0])

    def test_style_comes_from_the_item_type(self):
        self.assertEqual(enrich_catalog.derive_style("블레이저", 5, "캐주얼", self.table), "포멀")
        self.assertEqual(enrich_catalog.derive_style("트랙팬츠", 1, "캐주얼", self.table), "스포티")

    def test_unknown_item_type_keeps_the_crawler_guess(self):
        self.assertEqual(
            enrich_catalog.derive_style("듣도보도 못한 종류", 2, "스트리트", self.table), "스트리트")

    def test_style_stays_inside_the_engine_vocabulary(self):
        """규칙 엔진이 모르는 스타일은 목적별 필터에서 통째로 빠진다."""
        from recommendation_engine import STYLE_FORMALITY

        produced = set(self.table["style_by_item_type"].values()) | {"포멀"}
        self.assertEqual(produced - set(STYLE_FORMALITY), set())


class BodyShapeColumnTests(unittest.TestCase):
    """상품이 어느 실루엣 목표에 도움이 되는지 채운다.

    크롤러는 전 체형 허용으로 뭉뚱그려 넣어 변별력이 없었다.
    """

    @classmethod
    def setUpClass(cls):
        cls.table = enrich_catalog.load_derivation()

    def shapes(self, is_top, **out):
        return enrich_catalog.derive_body_shapes(is_top, out, self.table).split("|")

    def test_eye_catching_top_draws_the_upper_focus(self):
        self.assertIn("상체 강조형", self.shapes(True, pattern="플로럴", visual_weight="4"))

    def test_eye_catching_bottom_draws_the_lower_focus(self):
        self.assertIn("하체 강조형", self.shapes(False, pattern="체크", visual_weight="4"))

    def test_a_quiet_item_is_only_balanced(self):
        self.assertEqual(self.shapes(True, pattern="무지", visual_weight="2", detail_level="1"),
                         ["균형형"])

    def test_balanced_is_always_present(self):
        for is_top in (True, False):
            with self.subTest(is_top=is_top):
                self.assertIn("균형형", self.shapes(is_top, pattern="그래픽", visual_weight="5"))

    def test_labels_match_the_engine_vocabulary(self):
        """엔진이 비교하는 어휘와 어긋나면 이 칼럼이 다시 죽는다."""
        from schemas import CATALOG_FOCUS_LABELS

        produced = set(self.shapes(True, pattern="체크", visual_weight="5"))
        produced |= set(self.shapes(False, pattern="무지", visual_weight="1"))
        self.assertEqual(produced - set(CATALOG_FOCUS_LABELS), set())


class EnrichedCatalogLoadsTests(unittest.TestCase):
    """채운 결과가 ProductCatalog으로 그대로 읽혀야 이어붙인 의미가 있다."""

    def test_enriched_rows_load_and_keep_the_image_path(self):
        row = {field: "" for field in enrich_catalog.OUTPUT_FIELDS}
        row.update({
            "product_id": "M001", "name": "테스트 셔츠", "category": "top",
            "color": "화이트", "style": "미니멀", "purposes": "데일리|출근",
            "body_shapes": "균형형", "price": "39000", "season": "사계절",
            "stock": "true", "item_type": "셔츠", "fit": "레귤러핏",
            "length": "기본 기장", "pattern": "무지", "material": "코튼",
            "neckline": "셔츠 칼라", "formality": "4", "activity_tags": "업무",
            "warmth": "3", "breathability": "4", "water_resistant": "false",
            "visual_weight": "2", "detail_level": "2", "pattern_contrast": "0",
            "image_path": "garments/raw/M001.jpg",
        })
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "products_musinsa_enriched.csv"
            write_csv(path, enrich_catalog.OUTPUT_FIELDS, [row])
            product = ProductCatalog(path).products[0]

        self.assertEqual(product.product_id, "M001")
        self.assertEqual(product.image_path, "garments/raw/M001.jpg")
        self.assertEqual(product.formality, 4)
        self.assertTrue(product.stock)
        self.assertEqual(product.purposes, ["데일리", "출근"])

    def test_output_schema_covers_everything_products_csv_has(self):
        """칼럼이 빠지면 그 속성을 보는 규칙이 조용히 잠든다."""
        with (ROOT / "data" / "products.csv").open(encoding="utf-8-sig") as handle:
            existing = next(csv.reader(handle))
        self.assertEqual(set(existing) - set(enrich_catalog.OUTPUT_FIELDS), set())


class CoverageReportTests(unittest.TestCase):
    """채운 카탈로그가 규칙을 발동시킬 수 있는지 스스로 점검할 수 있어야 한다."""

    def test_handmade_catalog_passes_its_own_report(self):
        with (ROOT / "data" / "products.csv").open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        failures = [label for ok, label in enrich_catalog.coverage_report(rows) if not ok]
        self.assertEqual(failures, [])

    def test_bare_crawl_is_reported_as_insufficient(self):
        """속성이 빈 크롤링 원본은 '부족'으로 잡혀야 한다. 조용히 통과하면 안 된다."""
        rows = [
            {"product_id": f"P{i}", "category": "top" if i % 2 else "bottom",
             "pattern": "", "activity_tags": "", "waistline": "", "length": "",
             "water_resistant": "", "warmth": "", "breathability": "", "neckline": ""}
            for i in range(10)
        ]
        failures = [label for ok, label in enrich_catalog.coverage_report(rows) if not ok]
        self.assertTrue(failures)


if __name__ == "__main__":
    unittest.main()
