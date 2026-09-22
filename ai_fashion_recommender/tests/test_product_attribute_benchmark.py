from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import benchmark_product_attributes as benchmark


def labeled_sample():
    rows = []
    for category, count in benchmark.QUOTAS.items():
        for i in range(count):
            row = dict.fromkeys(benchmark.FIELDS, "")
            row.update(product_id=f"{category}{i}", category=category, name="상품",
                       image_sha256="abc", cut_type="worn" if i < count // 2 else "flat",
                       cut_reviewer="human", cut_reviewed_at="2026-09-21", human_reviewer="human",
                       reviewed_at="2026-09-21", label_evidence="독립적으로 확인한 판매자 정보")
            for axis in benchmark.AXES:
                vocab = benchmark.vocabulary(category, axis)
                row[f"gold_{axis}"] = vocab[0][0] if vocab else "not_applicable"
            rows.append(row)
    return rows


def predictions(rows):
    results = []
    for row in rows:
        r = {key: row[key] for key in ("product_id", "category", "name", "brand_name", "brand", "image_url", "image_sha256")}
        r["title"] = benchmark.title_prediction(row)
        r["photo"] = {axis: {"labels": row[f"gold_{axis}"].split("|"), "status": "accepted"}
                      for axis in benchmark.AXES}
        results.append(r)
    return {"results": results, "checkpoint_sha256": "test"}


class AttributeBenchmarkTests(unittest.TestCase):
    def test_seed_and_strata_are_order_independent(self):
        rows = labeled_sample()
        self.assertEqual(benchmark.choose_sample(rows), benchmark.choose_sample(list(reversed(rows))))
        rows[0]["cut_type"] = "flat"
        with self.assertRaisesRegex(ValueError, "Insufficient"):
            benchmark.choose_sample(rows)

    def test_no_keyword_uses_title_only_but_score_keeps_brand(self):
        row = labeled_sample()[0]
        row.update(name="상품123", brand_name="코튼")
        result = benchmark.title_prediction(row)["material"]
        self.assertTrue(result["no_title_keyword"])
        self.assertEqual(result["matched"], ["코튼"])
        self.assertEqual(result["score"], 3.0)

    def test_real_aliases_and_first_match_are_preserved(self):
        row = labeled_sample()[0]
        row["name"] = "세미 오버핏 워싱 데님 셔츠 블루"
        result = benchmark.title_prediction(row)
        self.assertEqual(result["fit"]["labels"], ["오버핏"])
        self.assertEqual(result["material"]["labels"], ["데님"])
        self.assertEqual(result["item_type"]["matched"], ["셔츠"])

    def test_abstentions_count_as_errors_and_unknowns_excluded(self):
        rows = labeled_sample()[:3]
        rows[2]["gold_material"] = "unknown"
        payload = predictions(rows)
        payload["results"][1]["photo"]["material"] = {"labels": [], "status": "abstained"}
        records = {r["product_id"]: r for r in payload["results"]}
        self.assertEqual(benchmark.metric(rows, records, "material", "photo"), (1, 2, 1))

    def test_material_uses_exact_sets_and_keeps_cut_denominators(self):
        rows = labeled_sample()[:2]
        rows[0]["gold_material"] = "코튼|데님"
        rows[1]["cut_type"] = "flat"
        payload = predictions(rows)
        payload["results"][0]["photo"]["material"]["labels"] = ["데님", "코튼"]
        records = {r["product_id"]: r for r in payload["results"]}
        self.assertEqual(benchmark.metric(rows, records, "material", "photo", "worn"), (1, 1, 1))
        records[rows[0]["product_id"]]["photo"]["material"]["labels"] = ["데님"]
        self.assertEqual(benchmark.metric(rows, records, "material", "photo", "worn"), (0, 1, 1))

    def test_report_refuses_missing_human_review_and_changed_images(self):
        rows = labeled_sample()
        payload = predictions(rows)
        rows[0]["human_reviewer"] = ""
        with self.assertRaisesRegex(ValueError, "human_reviewer"):
            benchmark.render_report(rows, payload, "a", "b")
        rows[0]["human_reviewer"] = "human"
        rows[0]["image_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "input changed"):
            benchmark.render_report(rows, payload, "a", "b")

    def test_prediction_errors_and_duplicates_do_not_silently_disappear(self):
        rows = labeled_sample()
        payload = predictions(rows)
        payload["results"][0]["error"] = "bad image"
        with self.assertRaisesRegex(ValueError, "Inference failed"):
            benchmark.render_report(rows, payload, "a", "b")
        payload = predictions(rows)
        payload["results"].append(copy.deepcopy(payload["results"][0]))
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            benchmark.render_report(rows, payload, "a", "b")

    def test_report_replays_identically_and_covers_missing_keyword_subset(self):
        rows = labeled_sample()
        payload = predictions(rows)
        a = benchmark.render_report(rows, payload, "a", "b")
        self.assertEqual(a, benchmark.render_report(rows, payload, "a", "b"))
        self.assertIn("해당 축의 상품명 키워드가 없는 상품", a)
        self.assertIn("100.0% (34/34)", a)
        self.assertEqual(benchmark.cell((0, 0, 0)), "N/A (0개)")

    def test_shoes_are_not_passed_to_clothing_type_fit_or_length_heads(self):
        self.assertEqual(benchmark.TASKS["shoes"], {"item_type": None, "fit": None, "length": None, "material": "material"})


if __name__ == "__main__":
    unittest.main()
