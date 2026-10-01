from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from product_measurements import ProductMeasurementClient, normalize_size_table
from size_fit import size_table


def actual(sizes=None, unit="cm"):
    return {"data": {"unit": unit, "description": "측정 방법에 따라 오차가 있을 수 있습니다.", "sizes": sizes or [
        {"name": "S", "items": [{"name": "가슴단면", "value": 50}, {"name": "총장", "value": 68}]},
        {"name": "M", "items": [{"name": "가슴단면", "value": 54}, {"name": "총장", "value": 70}]},
        {"name": "L", "items": [{"name": "가슴단면", "value": 58}, {"name": "총장", "value": 72}]},
    ]}}


def option(label, **kwargs):
    return {"no": label, "activated": True, "optionValues": [{"optionName": "사이즈", "name": label}], **kwargs}


class MeasurementTests(unittest.TestCase):
    def test_zero_range_nan_and_circumference_are_not_flat_measurements(self):
        row = {"name": "FREE", "items": [
            {"name": "허리단면", "value": 0}, {"name": "총장", "value": "90~100"},
            {"name": "가슴단면", "value": float("nan")}, {"name": "가슴둘레", "value": 100},
            {"name": "어깨너비", "value": 42.5},
        ]}
        record = normalize_size_table("MS1", actual([row]))
        self.assertEqual(record["sizes"][0]["measurements"], {"shoulder_cm": 42.5})

    def test_other_units_are_not_silently_treated_as_cm(self):
        record = normalize_size_table("MS1", actual(unit="inch"))
        self.assertEqual(record["sizes"], [])
        self.assertIn("unsupported_unit", record["issues"])

    def test_conflicting_aliases_remain_unknown_even_with_a_third_value(self):
        row = {"name": "M", "items": [{"name": "총장", "value": 70}, {"name": "총기장", "value": 75},
                                      {"name": "총장", "value": 70}, {"name": "가슴단면", "value": 54}]}
        record = normalize_size_table("MS1", actual([row]))
        self.assertNotIn("length_cm", record["sizes"][0]["measurements"])
        self.assertIn("conflicting_measurements", record["issues"])

    def test_activated_option_does_not_establish_stock(self):
        record = normalize_size_table("MS1", actual(), {"data": {"optionItems": [option("M")]}})
        self.assertIsNone(record["sizes"][1]["available"])
        self.assertEqual(record["sizes"][1]["option_ids"], ["M"])

    def test_size_matching_is_exact_and_color_variants_are_preserved(self):
        options = {"data": {"optionItems": [option("M", isSoldOut=True), option("M", no="blue-M"), option("S / 기모", isSoldOut=True)]}}
        record = normalize_size_table("MS1", actual(), options)
        self.assertIsNone(record["sizes"][1]["available"])
        self.assertEqual(len(record["sizes"][1]["option_ids"]), 2)
        self.assertEqual(record["sizes"][0]["option_ids"], [])

    def test_persistent_cache_keeps_raw_table_and_avoids_repeat_network(self):
        with tempfile.TemporaryDirectory() as directory:
            client = ProductMeasurementClient(Path(directory))
            with patch.object(client, "_fetch", side_effect=[actual(), {"data": {}}]) as fetch:
                first = client.get("MS1")
            self.assertEqual(fetch.call_count, 2)
            saved = json.loads((Path(directory) / "MS1.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["raw"]["actual_size"]["sizes"][1]["name"], "M")
            other = ProductMeasurementClient(Path(directory))
            with patch.object(other, "_fetch", side_effect=AssertionError("cache missed")):
                self.assertEqual(other.get("MS1"), first)

    def test_failure_does_not_fake_missing_table(self):
        client = ProductMeasurementClient()
        with patch.object(client, "_fetch", side_effect=OSError("offline")):
            self.assertEqual(client.get("MS1")["status"], "unavailable")

    def test_expired_persistent_cache_is_refreshed(self):
        with tempfile.TemporaryDirectory() as directory:
            first = ProductMeasurementClient(Path(directory))
            with patch.object(first, "_fetch", side_effect=[actual(), {"data": {}}]):
                first.get("MS1")
            second = ProductMeasurementClient(Path(directory), cache_ttl=0)
            with patch.object(second, "_fetch", side_effect=[actual(), {"data": {}}]) as fetch:
                second.get("MS1")
            self.assertEqual(fetch.call_count, 2)

    def test_options_failure_preserves_measurements(self):
        client = ProductMeasurementClient()
        with patch.object(client, "_fetch", side_effect=[actual(), OSError("offline")]):
            result = client.get("MS1")
        self.assertEqual(result["status"], "ready")
        self.assertIn("options_unavailable", result["issues"])


class SizeTableTests(unittest.TestCase):
    """기준 옷 비교를 없앤 뒤(2026-10-01) 남은 것은 상품 실측표뿐이다.

    표를 못 만드는 경우에도 '안 맞는다'로 단정하지 않는다 —
    모르는 것과 안 맞는 것은 다르다.
    """

    def setUp(self):
        self.record = normalize_size_table("MS1", actual())

    def test_table_keeps_every_size_row_and_its_columns(self):
        result = size_table(self.record, "top")
        self.assertEqual(result["status"], "available")
        self.assertEqual([row["size"] for row in result["size_options"]], ["S", "M", "L"])
        self.assertEqual(result["columns"], {"chest_width_cm": "가슴단면", "length_cm": "총장"})
        self.assertEqual(result["size_options"][1]["measurements"]["chest_width_cm"], 54)

    def test_bottom_uses_waist_not_chest(self):
        record = normalize_size_table("MS2", actual([{"name": "28", "items": [
            {"name": "허리단면", "value": 38.5}, {"name": "총장", "value": 103}]}]))
        result = size_table(record, "bottom")
        self.assertEqual(result["columns"], {"waist_width_cm": "허리단면", "length_cm": "총장"})
        self.assertEqual(result["size_options"][0]["measurements"]["waist_width_cm"], 38.5)

    def test_missing_or_unavailable_record_is_not_a_misfit_verdict(self):
        for record, status in (({}, "missing_measurements"),
                               ({"status": "unavailable"}, "unavailable"),
                               (None, "missing_measurements")):
            with self.subTest(status=status):
                result = size_table(record, "top")
                self.assertEqual(result["status"], status)
                self.assertEqual(result["size_options"], [])

    def test_unknown_category_yields_no_columns(self):
        self.assertEqual(size_table(self.record, "shoes")["columns"], {})

    def test_measurement_note_rides_along_for_the_caption(self):
        self.assertIn("오차", size_table(self.record, "top")["measurement_note"])


if __name__ == "__main__":
    unittest.main()
