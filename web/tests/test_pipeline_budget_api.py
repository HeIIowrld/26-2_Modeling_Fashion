import tempfile
import unittest
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
        # parsed segmentation can be any object understood by parser.colorize
        parsed = {"segmentation": None}
        return OutfitAnalysis("test", "화이트", "블랙", "안정적인 무채색 조합", ["top","pants"], "캐주얼"), parsed

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


class FakeProductSearch:
    def __init__(self):
        self.called = False

    def search(self, targets, profile, limit=3, fallback_products=()):
        self.called = True
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
            self.assertTrue(fake_engine.product_search.called)
            self.assertEqual(
                stages,
                ["pose", "quality", "body", "segment", "attributes", "candidates", "scoring", "preview", "finalize"],
            )
            self.assertEqual(payload["request"]["desired_style"], "캐주얼")
            self.assertEqual(payload["request"]["min_budget"], 1000)
            self.assertEqual(payload["request"]["preferred_colors"], ["블루"])
        finally:
            # restore
            pipeline.get_engine = original_get_engine

if __name__ == "__main__":
    unittest.main()
