from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_product_attributes as benchmark
import evaluate_attribute_c1_ai_review as evaluation


class AIReferenceEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = ROOT / "reports/attribute_c1"
        cls.rows = benchmark.read_csv(cls.directory / "labels_ai_reviewed.csv")
        cls.payload = json.loads((cls.directory / "predictions.json").read_text(encoding="utf-8"))

    def test_every_reference_is_explicitly_ai_and_human_report_still_rejects_it(self):
        evaluation.validate_references(self.rows)
        self.assertEqual(len(self.rows), 100)
        self.assertTrue(all(not row["human_reviewer"] for row in self.rows))
        with self.assertRaises(ValueError):
            benchmark.validate_labels(self.rows)
        bad = copy.deepcopy(self.rows)
        bad[0]["human_reviewer"] = "pretend human"
        with self.assertRaisesRegex(ValueError, "impersonate"):
            evaluation.validate_references(bad)

    def test_changed_images_and_duplicate_or_failed_predictions_are_rejected(self):
        rows = copy.deepcopy(self.rows)
        rows[0]["image_sha256"] = "different image"
        with self.assertRaisesRegex(ValueError, "input changed"):
            evaluation.pair_predictions(rows, self.payload)
        payload = copy.deepcopy(self.payload)
        payload["results"].append(payload["results"][0])
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            evaluation.pair_predictions(self.rows, payload)
        payload = copy.deepcopy(self.payload)
        pid = self.rows[0]["product_id"]
        next(r for r in payload["results"] if r["product_id"] == pid)["error"] = "missing image"
        with self.assertRaisesRegex(ValueError, "failed inference"):
            evaluation.pair_predictions(self.rows, payload)

    def test_fallback_keeps_title_labels_and_counts_its_wrong_additions(self):
        _, pairs = evaluation.pair_predictions(self.rows, self.payload)
        for pair in pairs:
            if pair["title"]:
                self.assertEqual(pair["hybrid_simulation"], pair["title"])
                self.assertEqual(pair["hybrid_correct"], pair["title_correct"])
        summary = evaluation.summarize(pairs)
        self.assertEqual(summary["material"]["fallback_count"], 30)
        self.assertEqual(summary["material"]["fallback_correct"], 3)
        self.assertEqual(summary["material"]["fallback_wrong"], 27)
        self.assertEqual(sum(p["evaluable"] for p in pairs), 281)
        self.assertEqual(sum(p["hybrid_correct"] for p in pairs), 156)

    def test_reference_provenance_and_original_sample_ids_match(self):
        original = benchmark.read_csv(self.directory / "labels_provisional.csv")
        self.assertEqual([r["product_id"] for r in self.rows], [r["product_id"] for r in original])
        expected = benchmark.digest(self.directory / "ai_visual_review.json")
        self.assertTrue(all(r["review_artifact_sha256"] == expected for r in self.rows))
        self.assertEqual(sum(r["cut_type"] == "other" for r in self.rows), 1)

    def test_report_replays_without_network_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(labels=self.directory / "labels_ai_reviewed.csv",
                                   predictions=self.directory / "predictions.json",
                                   output=Path(directory) / "report.md")
            evaluation.report(args)
            first = args.output.read_bytes()
            metrics = args.output.with_suffix(".metrics.json").read_bytes()
            evaluation.report(args)
            self.assertEqual(first, args.output.read_bytes())
            self.assertEqual(metrics, args.output.with_suffix(".metrics.json").read_bytes())
            self.assertIn("사람 정답 기반 정확도가 아니다", first.decode())


if __name__ == "__main__":
    unittest.main()
