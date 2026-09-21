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
        self.assertIn("무신사 상품 추천", html)
        self.assertIn("renderShoppingProducts(result.shopping_results || [], result.shopping_outfits || [])", javascript)
        self.assertIn("왜 이 조합인가요?", javascript)
        # 룩 번호는 탭이 들고, 탭 하나에 상품 카드와 합성 사진이 함께 있어야 한다.
        self.assertIn('role="tablist" aria-label="추천 코디 조합"', javascript)
        self.assertIn("LOOK ${index + 1}", javascript)
        self.assertIn('class="look-render"', javascript)
        self.assertIn('class="look-products"', javascript)
        self.assertIn('id="current-score-matrix"', html)
        self.assertIn('id="current-outfit-points"', html)
        self.assertIn("renderCurrentOutfitEvaluation(result.current_outfit_evaluation)", javascript)
        self.assertIn("product.recommendation_reason", javascript)
        self.assertIn("왜 추천했나요?", javascript)
        self.assertIn("product.fit_evidence", javascript)
        self.assertIn("product.reason_rule_ids", javascript)
        self.assertNotIn('id="reco-picker"', html)
        self.assertNotIn('id="reco-detail"', html)
        self.assertNotIn("renderRecommendations", javascript)
        self.assertNotIn('data-image="preview"', html)
        self.assertIn("예상 착장샷은 실제 핏을 보장하지", html)
        self.assertIn("예상 착장샷은 실제 핏을 보장하지", javascript)

    def test_full_body_upload_guide_is_rendered(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        guide = STATIC / "assets" / "full-body-guide.png"
        self.assertTrue(guide.is_file())
        self.assertIn('src="assets/full-body-guide.png"', html)
        self.assertIn('alt="정면을 향해 팔을 자연스럽게 내린 전신사진 촬영 예시"', html)
        for text in (
            "정면 전신사진을 올려주세요",
            "머리부터 발끝까지 모두 나오게 촬영",
            "몸과 얼굴은 정면을 향하기",
            "팔은 몸 옆에 자연스럽게 내리기",
            "몸을 가리는 물건 없이 한 명만 촬영",
        ):
            self.assertIn(text, html)

    def test_photo_preflight_blocks_step_two_until_server_validation(self):
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        self.assertIn('/api/validate-photo', javascript)
        self.assertIn('사진을 확인하고 있어요', javascript)
        self.assertIn('note.dataset.tone = "bad"', javascript)
        self.assertIn('unlock(2);', javascript)
        self.assertIn('goto(2);', javascript)
        self.assertIn('state.photoValidation', javascript)

    def test_musinsa_products_can_be_selected_for_real_tryon(self):
        html = (STATIC / "index.html").read_text(encoding="utf-8")
        javascript = (STATIC / "app.js").read_text(encoding="utf-8")
        active_css = (STATIC / "lookbook.css").read_text(encoding="utf-8")
        self.assertIn('id="shopping-tryon-panel"', html)
        self.assertIn("/tryon-products", javascript)
        self.assertIn("/shopping-tryon-batch", javascript)
        self.assertIn("toggleShoppingSelection", javascript)
        self.assertIn("추천 코디 3가지 입어보기", javascript)
        self.assertNotIn("신발은 전용 마스크와 모델이 없어", javascript)
        self.assertIn('["top", "bottom", "shoes"]', javascript)
        self.assertIn(".shopping-tryon-panel", active_css)
        self.assertIn(".shopping-batch", active_css)


if __name__ == "__main__":
    unittest.main()
