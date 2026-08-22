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
