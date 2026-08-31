import json
import unittest
from pathlib import Path


STATIC = Path(__file__).resolve().parents[1] / "static"


class UIContractTests(unittest.TestCase):
    def test_budget_uses_two_range_handles(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        active_css = (STATIC / "lookbook.css").read_text(encoding="utf-8")
        self.assertIn('href="lookbook.css"', html)
        self.assertIn('type="range" name="min_budget"', html)
        self.assertIn('type="range" name="max_budget"', html)
        self.assertNotIn('type="text" name="min_budget"', html)
        self.assertIn('.budget-range input[type="range"]', active_css)

    def test_fallback_exposes_detailed_progress_stages(self):
        options = json.loads((STATIC / "fallback-options.json").read_text(encoding="utf-8"))
        self.assertEqual(
            [stage["key"] for stage in options["stages"]],
            [
                "prepare",
                "wardrobe",
                "pose",
                "quality",
                "body",
                "segment",
                "attributes",
                "candidates",
                "scoring",
                "preview",
                "finalize",
            ],
        )

    def test_result_shows_the_submitted_conditions(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn('id="request-summary"', html)
        self.assertIn("renderRequestSummary(result?.request)", javascript)
        self.assertIn("reco.ranking_tied", javascript)
        self.assertIn("상품 이미지 확인", javascript)


if __name__ == "__main__":
    unittest.main()
