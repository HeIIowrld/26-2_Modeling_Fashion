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
        self.assertIn("생성 품질 확인 필요", javascript)
        self.assertIn("payload.warnings", javascript)

    def test_multiple_tryon_renders_are_queued_and_switchable(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        active_css = (STATIC / "lookbook.css").read_text(encoding="utf-8")
        self.assertIn('id="tryon-batch-status"', html)
        self.assertIn("/tryon-batch", javascript)
        self.assertIn("moveToReadyRender", javascript)
        self.assertIn("결과 사진 다운로드", javascript)
        self.assertIn(".tryon-batch-status", active_css)
        self.assertIn(".tryon-switcher", active_css)

    def test_musinsa_products_can_be_selected_for_real_tryon(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        active_css = (STATIC / "lookbook.css").read_text(encoding="utf-8")
        self.assertIn('id="shopping-tryon-panel"', html)
        self.assertIn("/tryon-products", javascript)
        self.assertIn("toggleShoppingSelection", javascript)
        self.assertIn("선택 조합 렌더링", javascript)
        self.assertIn("신발은 전용 마스크와 모델이 없어", javascript)
        self.assertIn(".shopping-tryon-panel", active_css)


if __name__ == "__main__":
    unittest.main()
