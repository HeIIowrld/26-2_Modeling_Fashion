from __future__ import annotations

import gc
import json
import hashlib
import sys
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from offline_outfit_library import (OutfitLibrary, atomic_json, build_library, default_scenarios,
                                   digest, score_triple)
from product_catalog import ProductCatalog
from recommendation_engine import RecommendationEngine
from schemas import Product, UserProfile, PoseAnalysis
from offline_render_quality import QUALITY_VERSION, quality_fingerprint


def product(pid, category, **kwargs):
    return Product(pid, pid, category, kwargs.pop("color", "화이트"), "캐주얼", ["데일리"], [],
                   kwargs.pop("price", 10000), "사계절", kwargs.pop("stock", True),
                   item_type=kwargs.pop("item_type", "티셔츠" if category == "top" else "스니커즈" if category == "shoes" else "팬츠"),
                   fit="레귤러핏" if category == "top" else "와이드핏", length="기본 기장", **kwargs)


class OfflineLibraryTests(unittest.TestCase):
    def verified_record(self, ids, score):
        cid = digest(ids)[:20]
        path = self.path / "renders" / f"{cid}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.with_suffix(".png").write_bytes(b"test image content")
        image_sha = hashlib.sha256(b"test image content").hexdigest()
        atomic_json(path, {"combination_id": cid, "product_ids": ids, "status": "complete",
            "library_identity": self.library.metadata["identity"], "scores": {self.scenario: score},
            "preservation": {"version": QUALITY_VERSION, "fingerprint": quality_fingerprint(), "status": "auto_passed"},
            "visual_review": {"decision": "accepted", "output_sha256": image_sha},
            "output_sha256": image_sha, "render_fingerprint": "fixture_renderer"})
        return path

    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)
        cls.engine = RecommendationEngine(ROOT / "FASHION_RULES_MASTER.md", ProductCatalog(ROOT / "data/products.csv"))
        cls.products = [product("T1", "top", stock=False, price=999999), product("T2", "top", color="블랙"),
                        product("B1", "bottom"), product("B2", "bottom", price=30000),
                        product("S1", "shoes"), product("S2", "shoes", stock=False)]
        cls.catalog = {"products": [{"product": asdict(p), "attributes": {"sleeve_length": "반팔"}}
                                    for p in cls.products]}
        cls.scenarios = default_scenarios()[:1]
        build_library(cls.catalog, cls.scenarios, cls.path, cls.engine)
        cls.library = OutfitLibrary(cls.path)
        cls.scenario = cls.scenarios[0]["id"]

    @classmethod
    def tearDownClass(cls):
        # scores 는 mmap 으로 열려 있다. 참조를 놓기 전에 임시 폴더를 지우면 Windows 에서
        # "다른 프로세스가 사용 중"(WinError 32)으로 정리가 실패한다.
        del cls.library
        gc.collect()
        cls.temp.cleanup()

    def test_index_matches_direct_three_item_scoring(self):
        scenario = self.scenarios[0]
        for ti, top in enumerate(self.products[:2]):
            for bi, bottom in enumerate(self.products[2:4]):
                for hi, shoe in enumerate(self.products[4:]):
                    direct, _ = score_triple(top, bottom, shoe, UserProfile(**scenario["profile"]),
                                            PoseAnalysis(**scenario["pose"]), self.engine)
                    self.assertAlmostEqual(float(self.library.scores[0, ti, bi, hi]), direct, places=4)

    def test_keep_white_tee_replace_only_bottom_and_shoes(self):
        result = self.library.query(self.scenario, {"top": {"color": "화이트", "sleeve_length": "반팔"}},
                                    max_budget=25000)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["replacement_ids"], ["B1", "S1"])
        self.assertEqual(result[0]["product_ids"][0], "T1")
        self.assertEqual(result[0]["replacement_price"], 20000)
        self.assertIsNone(result[0]["render"])

    def test_unseen_anchor_or_scenario_does_not_silently_match(self):
        self.assertEqual(self.library.query(self.scenario, {"top": {"color": "핑크"}}), [])
        with self.assertRaises(ValueError):
            self.library.query("uncomputed", {})

    def test_can_keep_two_slots_and_only_change_shoes(self):
        result = self.library.query(self.scenario, {"top": {"product_id": "T2"}, "bottom": {"product_id": "B1"}})
        self.assertEqual(result[0]["replacement_ids"], ["S1"])

    def test_failed_and_stale_renders_never_qualify(self):
        cid = digest(["T1", "B1", "S1"])[:20]
        path = self.path / "renders" / f"{cid}.json"
        atomic_json(path, {"combination_id": cid, "status": "failed", "library_identity": self.library.metadata["identity"]})
        self.assertEqual(self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True), [])
        atomic_json(path, {"combination_id": cid, "status": "complete", "library_identity": "old"})
        self.assertEqual(self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True), [])
        path.unlink()

    def test_verified_lookup_uses_rendered_score_instead_of_prediction(self):
        paths = []
        for bottom, score in (("B1", 10), ("B2", 90)):
            ids = ["T1", bottom, "S1"]
            path = self.verified_record(ids, score)
            paths.append(path)
        try:
            result = self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True)
            self.assertEqual(result[0]["replacement_ids"], ["B2", "S1"])
            self.assertEqual(result[0]["score"], 90)
            self.assertEqual(result[0]["ranking_source"], "rendered")
        finally:
            for path in paths:
                path.unlink()
                path.with_suffix(".png").unlink()

    def test_old_automatic_pass_and_pending_review_never_qualify(self):
        path = self.verified_record(["T1", "B1", "S1"], 95)
        original = json.loads(path.read_text(encoding="utf-8"))
        try:
            for field, value in (("preservation", {}), ("visual_review", "pending")):
                atomic_json(path, {**original, field: value})
                self.assertEqual(self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True), [])
        finally:
            path.unlink()
            path.with_suffix(".png").unlink()

    def test_changed_image_invalidates_review_without_restarting_query(self):
        path = self.verified_record(["T1", "B1", "S1"], 95)
        try:
            self.assertEqual(len(self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True)), 1)
            path.with_suffix(".png").write_bytes(b"a different generated image")
            self.assertEqual(self.library.query(self.scenario, {"top": {"product_id": "T1"}}, verified_only=True), [])
        finally:
            path.unlink()
            path.with_suffix(".png").unlink()

    def test_changed_scoring_code_requires_rebuild(self):
        from unittest.mock import patch
        with patch("offline_outfit_library.scoring_fingerprint", return_value="changed"):
            with self.assertRaises(ValueError):
                OutfitLibrary(self.path)

    def test_unassessed_or_failed_images_do_not_become_verified(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from precompute_outfits import render_quality_status
        passing = [{"category": c, "checks": [{"passed": True}]} for c in ("top", "bottom")]
        passing.append({"backend": "test_shoes", "outside_mask_preserved": True})
        self.assertEqual(render_quality_status(passing, True), "auto_passed")
        self.assertEqual(render_quality_status(passing, False), "quality_failed")
        self.assertEqual(render_quality_status([], True), "unassessed")
        self.assertEqual(render_quality_status(passing[:1], True), "unassessed")
        passing[0]["checks"][0]["passed"] = False
        self.assertEqual(render_quality_status(passing, True), "quality_failed")


if __name__ == "__main__":
    unittest.main()
