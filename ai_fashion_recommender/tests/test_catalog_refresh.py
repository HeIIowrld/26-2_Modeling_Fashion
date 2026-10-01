import csv
import json
import sys
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import backfill_product_colors as colors
import musinsa_crawler as crawler
from enrich_catalog import OUTPUT_FIELDS


def test_multiple_options_do_not_invent_representative_color():
    table = {"블랙": ["black"], "화이트": ["white"]}
    assert colors.representative_color("Basic shirt", ["BLACK", "WHITE"], table)[0] == ""
    assert colors.representative_color("Basic shirt", ["BLACK", "unknown code"], table)[0] == ""
    assert colors.representative_color("Basic shirt", ["BLACK", "black (cotton)"], table) == ("블랙", "single_option_palette")
    assert colors.representative_color("WHITE shirt", ["BLACK", "WHITE"], table) == ("화이트", "product_title")
    assert colors.representative_color("BLACK WHITE pack", ["BLACK"], table)[0] == ""


def test_measurement_backfill_retries_rate_limit_without_discarding_cache(tmp_path):
    import urllib.error
    import backfill_product_measurements as measurements
    from product_measurements import ProductMeasurementClient
    client = measurements.BackfillClient(tmp_path)
    limited = urllib.error.HTTPError("https://example.com", 429, "rate limited", {}, None)
    with patch.object(ProductMeasurementClient, "_fetch", side_effect=[limited, {"data": {}}]) as fetch:
        with patch.object(measurements.time, "sleep") as sleep:
            assert client._fetch("1", "actual-size") == {"data": {}}
    assert fetch.call_count == 2
    sleep.assert_called_once_with(30)
    assert not client.rate_limited


def test_failed_color_fetch_preserves_original_even_with_force(tmp_path):
    path = tmp_path / "catalog.csv"
    fields = ["product_id", "name", "color", "color_options", "detail_colors"]
    rows = [dict(zip(fields, ["MS1", "기본 셔츠", "블랙", "블랙", "BLACK"])),
            dict(zip(fields, ["MS2", "기본 바지", "", "블루", "BLUE"]))]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with patch.object(sys, "argv", ["backfill", "--input", str(path), "--fetch", "--missing-only", "--force"]):
        with patch.object(colors, "fetch_colors", side_effect=OSError("offline")) as fetch:
            with patch.object(colors.time, "sleep"):
                assert colors.main() == 0
    assert fetch.call_count == 1
    with path.open(encoding="utf-8", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert saved[0]["color"] == "블랙"
    assert saved[1]["detail_colors"] == "BLUE"
    assert saved[1]["color_options"] == "블루"


def test_batch_only_allows_a_larger_but_bounded_options_response(tmp_path):
    import backfill_product_measurements as measurements
    from product_measurements import ProductMeasurementClient
    client = measurements.BackfillClient(tmp_path)
    with patch.object(ProductMeasurementClient, "_fetch", side_effect=ValueError("상품 응답 크기 제한 초과")):
        with patch.object(measurements, "fetch_json", return_value={"data": {}}) as fetch:
            assert client._fetch("1", "options") == {"data": {}}
    assert fetch.call_args.kwargs["max_bytes"] == 8_000_000


def test_partial_crawl_only_requests_bottom_categories():
    with patch.object(crawler, "fetch_page", return_value=[]) as fetch:
        crawler.crawl(1, 0, only_category="bottom")
    assert fetch.call_count > 0
    assert all(crawler.CATEGORY_MAP[call.args[0]] == "bottom" for call in fetch.call_args_list)


def test_cached_dropdown_colors_enrich_options_without_overwriting_existing_color(tmp_path):
    path = tmp_path / "catalog.csv"
    cache = tmp_path / "cache"
    cache.mkdir()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["product_id", "name", "color"])
        writer.writeheader()
        writer.writerows([{"product_id": "MS1", "name": "shirt", "color": "그린"},
                          {"product_id": "MS2", "name": "shirt", "color": ""}])
    for product_id in ("MS1", "MS2"):
        (cache / f"{product_id}.json").write_text(json.dumps({
            "record": {"product_id": product_id, "color_options": []},
            "raw": {"options": {"basic": [{"displayType": "DROPDOWN", "optionValues": [
                {"name": "BLACK", "color": {"colorCode": "1"}}]}]}}
        }), encoding="utf-8")
    with patch.object(sys, "argv", ["backfill", "--input", str(path),
                                  "--from-measurement-cache", str(cache), "--preserve-existing-colors"]):
        assert colors.main() == 0
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["color"] == "그린"
    assert rows[0]["color_options"] == "블랙"
    assert rows[1]["color"] == "블랙"
    assert rows[1]["color_evidence"] == "single_option_palette"


def test_stock_timestamp_is_collected_saved_and_preserved_by_enrichment(tmp_path):
    product = crawler.parse_item({"goodsNo": 123, "goodsName": "기본 바지", "price": 10000,
                                  "normalPrice": 10000, "salePrice": 10000}, "bottom")
    assert product is not None
    assert datetime.fromisoformat(product.stock_checked_at).tzinfo is not None
    path = tmp_path / "collected.csv"
    crawler.save_csv([product], path)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        assert next(csv.DictReader(handle))["stock_checked_at"] == product.stock_checked_at
    assert "stock_checked_at" in OUTPUT_FIELDS


def test_offline_candidates_reject_confirmed_measurement_category_conflict():
    from product_catalog import ProductCatalog
    catalog = ProductCatalog(ROOT / "data" / "products.csv")
    top = next(product for product in catalog.products if product.category == "top")
    top.measurement_record = {"type_name": "바지"}
    assert top not in catalog.available("top")
    assert top not in catalog.available()
    top.measurement_record = {"type_name": ""}
    assert top in catalog.available("top")
