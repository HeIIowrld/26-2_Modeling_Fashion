import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import evaluate_attribute_c2 as evaluation


class C2EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        folder = ROOT / "reports/attribute_c1"
        self.args = SimpleNamespace(labels=folder / "labels_ai_reviewed.csv",
            baseline=folder / "predictions.json", predictions=folder / "c2_predictions.json",
            output=Path(self.directory.name) / "result.md")

    def run_report(self):
        with contextlib.redirect_stdout(io.StringIO()):
            evaluation.report(self.args)
        return json.loads(self.args.output.with_suffix(".json").read_text())

    def test_replay_is_deterministic_and_counts_abstentions_and_unchanged_axes(self):
        first = self.run_report()
        first_bytes = self.args.output.with_suffix(".json").read_bytes()
        second = self.run_report()
        self.assertEqual(first_bytes, self.args.output.with_suffix(".json").read_bytes())
        pairs = first["pairs"]
        self.assertEqual(len(pairs), 400)
        self.assertEqual(sum(p["evaluable"] for p in pairs), 281)
        self.assertEqual(sum(p["before_correct"] for p in pairs), 60)
        self.assertEqual(sum(p["after_correct"] for p in pairs), 76)
        self.assertEqual(sum(p["supplemented"] and p["evaluable"] for p in pairs), 16)
        self.assertTrue(all(p["before"] == p["after"] for p in pairs
                            if p["axis"] in {"item_type", "material"} or p["cut_type"] == "flat"))

    def altered_payload(self, change):
        payload = json.loads(self.args.predictions.read_text())
        change(payload)
        self.args.predictions = Path(self.directory.name) / "changed.json"
        self.args.predictions.write_text(json.dumps(payload))

    def test_rejects_mismatched_model_or_image(self):
        self.altered_payload(lambda p: p.update(checkpoint_sha256="wrong"))
        with self.assertRaisesRegex(ValueError, "Checkpoints differ"):
            self.run_report()

    def test_rejects_duplicates_even_when_there_are_one_hundred_rows(self):
        self.altered_payload(lambda p: p["results"].__setitem__(1, p["results"][0]))
        with self.assertRaisesRegex(ValueError, "distinct"):
            self.run_report()

    def test_rejects_edited_decision_that_does_not_follow_frozen_model_outputs(self):
        self.altered_payload(lambda p: p["results"][0].update(attributes={"fit": {"label": "와이드핏"}}))
        with self.assertRaisesRegex(ValueError, "runtime policy"):
            self.run_report()


if __name__ == "__main__":
    unittest.main()
