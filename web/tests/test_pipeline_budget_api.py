import tempfile
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from types import SimpleNamespace
from PIL import Image

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline import run_pipeline, PipelineResult, UserProfile

class FakePoseAnalyzer:
    def analyze(self, image_path):
        from ai_fashion_recommender.src.schemas import PoseAnalysis
        return PoseAnalysis(True, 0.9, "사각체형", 1.0, 0.5, 0.5, "정면")
    def draw_landmarks(self, image_path, analysis=None):
        return Image.new("RGB", (10,10), (255,255,255))

class FakeQualityChecker:
    def check_input(self, image_path, pose=None):
        return {"passed": True, "issues": []}

class FakeOutfitAnalyzer:
    def __init__(self):
        from types import SimpleNamespace
        self.parser = SimpleNamespace(colorize=lambda segmentation: Image.new("RGB", (10,10), (240,240,240)))

    def analyze(self, image_path, pose_result):
        from ai_fashion_recommender.src.schemas import OutfitAnalysis
        import numpy as np
        parsed = {"segmentation": np.full((10, 10), 6), "backend": "fashn-human-parser"}
        return OutfitAnalysis("test", "화이트", "블랙", "안정적인 무채색 조합", ["top","pants"], "캐주얼",
                              fit="레귤러핏", lower_fit="스트레이트핏"), parsed

class FakeRecommender:
    # Minimal attributes referenced by pipeline payload construction
    active_rule_ids = []
    documented_rule_ids = []
    scoring_rule_ids = []
    unsupported_rule_ids = []
    UNSUPPORTED_RULE_REASONS = {}

    def __init__(self):
        self.catalog = SimpleNamespace(products=[])

    def recommend(self, profile, pose_result, outfit_result, top_k=3):
        raise AssertionError("CSV 카탈로그 추천은 웹 파이프라인에서 호출하면 안 됩니다.")

    def generate_target_keywords(self, profile, pose_result, outfit_result):
        return SimpleNamespace(targets={"top": {}, "bottom": {}})

    def evaluate_current_outfit(self, profile, pose_result, outfit_result):
        from ai_fashion_recommender.src.schemas import CurrentOutfitEvaluation
        matrix = {
            "top": {"body_fit": 86.0, "situation_fit": 87.0, "style_fit": 88.0},
            "bottom": {"body_fit": 84.0, "situation_fit": 89.0, "style_fit": 90.0},
        }
        return CurrentOutfitEvaluation(
            total_score=86.7, score_breakdown={}, reasons=[], applied_rules=[],
            score_coverage=100.0, analysis_confidence=0.8, reliable=True,
            keep_threshold=85.0, should_keep=False, verdict="추천 코디로 보완할 수 있어요",
            diagnostic_matrix=matrix,
            pass_matrix={
                "top": {"body_fit": True, "situation_fit": True, "style_fit": True},
                "bottom": {"body_fit": False, "situation_fit": True, "style_fit": True},
            },
            harmony_score=88.0,
        )


class FakeProductSearch:
    def __init__(self):
        self.called = False

    def search(self, targets, profile, limit=3, fallback_products=(), photo_loader=None):
        self.called = True
        if not callable(photo_loader):
            raise AssertionError("실시간 사진 로더가 검색에 연결되어야 합니다.")
        if list(fallback_products):
            raise AssertionError("CSV fallback 상품을 실시간 검색에 넘기면 안 됩니다.")
        return []

class FakeEngine:
    def __init__(self):
        self.pose_analyzer = FakePoseAnalyzer()
        self.quality_checker = FakeQualityChecker()
        self.outfit_analyzer = FakeOutfitAnalyzer()
        self.recommender = FakeRecommender()
        self.product_search = FakeProductSearch()
        self.tryon = SimpleNamespace(enabled=False, available=False, NOT_READY_REASON="")
        self.device = "cpu"
        self.trained_heads = False
        self.parser_backend = "fashn"

class PipelineBudgetAPITests(unittest.TestCase):
    def test_hidden_body_skips_photo_body_shape_but_keeps_outfit_search(self):
        import pipeline
        engine = FakeEngine()
        outfit, parsed = engine.outfit_analyzer.analyze(None, None)
        parsed['segmentation'][:] = 5
        engine.outfit_analyzer.analyze = Mock(return_value=(outfit, parsed))
        with tempfile.TemporaryDirectory() as temporary, patch.object(pipeline, 'get_engine', return_value=engine), patch.object(pipeline, 'classify') as classify:
            result = run_pipeline(
                Path(temporary) / 'person.jpg', pipeline.build_profile({}),
                Path(temporary), lambda stage: None,
            )
            classify.assert_not_called()
            self.assertTrue(engine.product_search.called)
            self.assertEqual(result.payload['pose']['body_shape'], '분석 불확실')
            self.assertTrue(any('치마' in warning for warning in result.payload['input_quality']['warnings']))

    def test_separate_body_photo_is_checked_with_its_own_pose(self):
        import pipeline
        engine = FakeEngine()
        outfit, parsed = engine.outfit_analyzer.analyze(None, None)
        hidden = {**parsed, 'segmentation': parsed['segmentation'] * 0 + 5}
        engine.outfit_analyzer.analyze = Mock(side_effect=[(outfit, parsed), (outfit, hidden)])
        main_pose = engine.pose_analyzer.analyze(None)
        body_pose = engine.pose_analyzer.analyze(None)
        engine.pose_analyzer.analyze = Mock(side_effect=[main_pose, body_pose])
        with tempfile.TemporaryDirectory() as temporary, patch.object(pipeline, 'get_engine', return_value=engine), patch.object(pipeline, 'classify') as classify:
            body_path = Path(temporary) / 'body.jpg'
            result = run_pipeline(
                Path(temporary) / 'person.jpg', pipeline.build_profile({}),
                Path(temporary), lambda stage: None, body_path,
            )
            self.assertEqual(engine.pose_analyzer.analyze.call_args.args, (body_path,))
            classify.assert_not_called()
            self.assertEqual(result.payload['pose']['body_shape'], '분석 불확실')

    def test_valid_separate_body_photo_supplies_its_pose_to_body_classifier(self):
        import pipeline
        engine = FakeEngine()
        main_pose = engine.pose_analyzer.analyze(None)
        body_pose = engine.pose_analyzer.analyze(None)
        engine.pose_analyzer.analyze = Mock(side_effect=[main_pose, body_pose])
        with tempfile.TemporaryDirectory() as temporary, patch.object(pipeline, 'get_engine', return_value=engine), patch.object(pipeline, 'classify', return_value=('사각체형', '사진 추정')) as classify:
            body_path = Path(temporary) / 'body.jpg'
            run_pipeline(Path(temporary) / 'person.jpg', pipeline.build_profile({}), Path(temporary), lambda stage: None, body_path)
            self.assertIs(classify.call_args.args[1], body_pose)
            self.assertEqual(classify.call_args.kwargs['person_image'], body_path)

    def test_live_size_comparison_reaches_web_payload_before_final_selection(self):
        import pipeline
        from musinsa_live_search import MusinsaLiveSearch
        from product_measurements import normalize_size_table

        client = Mock()
        def table(product_id):
            return normalize_size_table(product_id, {"data": {"sizes": [{"name": "M", "items": [
                {"name": "가슴단면", "value": 54 if product_id == "MS2" else 65},
                {"name": "총장", "value": 70}]}]}})
        client.get.side_effect = table
        search = MusinsaLiveSearch(measurements=client)
        self.addCleanup(search.close)
        fake_engine = FakeEngine()
        fake_engine.product_search = search
        profile = pipeline.build_profile({"reference_measurements": {"top": {"chest_width_cm": 54, "length_cm": 70}}})
        rows = [{"goodsNo": number, "goodsName": "상의", "finalPrice": 50000, "reviewCount": 100 if number == 1 else 1}
                for number in (1, 2)]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            image = root / "person.jpg"
            Image.new("RGB", (64, 128)).save(image)
            with patch.object(pipeline, "get_engine", return_value=fake_engine), patch.object(
                search, "_fetch", side_effect=lambda category, *a, **kw: rows if category == "top" else []
            ):
                result = pipeline.run_pipeline(image, profile, root / "out", lambda stage: None)
        products = result.payload["shopping_results"]
        self.assertEqual(products[0]["product_id"], "MS2")
        self.assertEqual(products[0]["size_fit"]["closest_size"], "M")
        self.assertEqual(products[0]["size_fit"]["differences"][0]["delta_cm"], 0)
        self.assertNotIn("ranking_bonus", products[0]["size_fit"])
        self.assertEqual(result.recommendations, [])

    def test_run_pipeline_uses_only_live_product_search(self):
        # Create a tiny image file
        tmpdir = Path(tempfile.mkdtemp())
        img = Image.new("RGB", (64, 128), (200,200,200))
        img_path = tmpdir / "person.jpg"
        img.save(img_path, format="JPEG")

        # Monkeypatch get_engine inside pipeline
        import pipeline
        original_get_engine = pipeline.get_engine
        fake_engine = FakeEngine()
        pipeline.get_engine = lambda: fake_engine

        profile = UserProfile(
            purpose="데일리",
            desired_style="캐주얼",
            change_scope="전체 변경",
            min_budget=1000,
            max_budget=2000,
            preferred_colors=["블루"],
        )

        stages = []
        def on_stage(stage):
            stages.append(stage)

        try:
            result = run_pipeline(img_path, profile, tmpdir / "out", on_stage)
            self.assertIsInstance(result, PipelineResult)
            payload = result.payload
            self.assertNotIn("target_keywords", payload)
            self.assertNotIn("recommendations", payload)
            self.assertEqual(payload["shopping_results"], [])
            self.assertEqual(payload["current_outfit_evaluation"]["total_score"], 86.7)
            self.assertEqual(len(payload["current_outfit_evaluation"]["summary_points"]), 3)
            self.assertFalse(
                payload["current_outfit_evaluation"]["pass_matrix"]["bottom"]["body_fit"]
            )
            self.assertTrue(fake_engine.product_search.called)
            self.assertEqual(
                stages,
                ["pose", "quality", "segment", "attributes", "body", "candidates", "scoring", "preview", "finalize"],
            )
            self.assertEqual(payload["request"]["desired_style"], "캐주얼")
            self.assertEqual(payload["request"]["min_budget"], 1000)
            self.assertEqual(payload["request"]["preferred_colors"], ["블루"])
        finally:
            # restore
            pipeline.get_engine = original_get_engine

if __name__ == "__main__":
    unittest.main()
