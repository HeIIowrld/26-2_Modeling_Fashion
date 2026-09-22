from __future__ import annotations

import json
import os
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from live_product_attributes import (LiveProductAttributes, POLICY_VERSION, accepted_attributes,
                                     photo_matches, title_has_axis)
from musinsa_live_search import MusinsaLiveSearch, ShoppingProduct
from recommendation_keywords import TargetKeywordResult
from recommendation_explanations import add_product_recommendation_reasons
from schemas import UserProfile, PoseAnalysis


def evidence(label="와이드핏", confidence=0.96):
    return {"fit": {"label": label, "confidence": confidence, "source": "product_photo", "policy": POLICY_VERSION}}


def row(number, name="팬츠", reviews=1, brand=""):
    return {"goodsNo": number, "goodsName": name, "finalPrice": 50000, "brandName": brand,
            "reviewCount": reviews, "thumbnail": f"https://image.msscdn.net/{number}.jpg"}


class PhotoSearchTests(unittest.TestCase):
    def setUp(self):
        self.provider = Mock(last_stats={})
        self.provider.get_many.return_value = {"MS2": evidence()}
        self.search = MusinsaLiveSearch(photo_provider=self.provider)
        self.addCleanup(self.search.close)
        self.targets = TargetKeywordResult("mixed", {"bottom": {"fit": ["와이드"]}})

    def test_unmatched_product_improves_without_claiming_title_match(self):
        with patch.object(self.search, "_fetch", return_value=[row(1, reviews=100), row(2)]):
            result = self.search.search(self.targets, UserProfile(), limit=1)
        self.assertEqual(result[0].product_id, "MS2")
        self.assertEqual(result[0].retrieval_score, 4)
        self.assertEqual(result[0].matched_keywords, [])
        add_product_recommendation_reasons(result, UserProfile(), PoseAnalysis(False, 0, "", 0, 0, 0, ""),
                                           self.targets, use_llm=False)
        self.assertIn("상품 사진", result[0].recommendation_reason)
        self.assertIn("추정", result[0].recommendation_reason)
        self.assertNotIn("상품명", result[0].recommendation_reason)
        self.assertNotIn("photo_attributes", result[0].public_dict())

    def test_contradicting_name_and_unsupported_semantics_never_overridden(self):
        for name in ("슬림 팬츠", "테이퍼드 팬츠", "regular pants", "스키니 팬츠"):
            score, matched = self.search._score(dict(row(2, name), category="bottom"),
                self.targets.targets["bottom"], 0, photo_attributes=evidence())
            self.assertEqual((score, matched), (0, []))
        self.assertEqual(photo_matches("bottom", "팬츠", {"fit": ["세미와이드"]}, evidence()), {})
        self.assertEqual(photo_matches("shoes", "슈즈", {"fit": ["와이드"]}, evidence()), {})
        self.assertTrue(title_has_axis("부츠컷 팬츠", "fit"))
        self.assertFalse(title_has_axis("부츠컷 팬츠", "length"))

    def test_photos_cannot_change_matched_display_positions_with_diversity_and_measurements(self):
        rows = [row(1, "와이드", 100, "A"), row(3, "와이드", 99, "A"), row(4, "와이드", 98, "A"),
                row(5, "와이드", 97, "B"), row(6, reviews=100), row(2)]
        with patch.object(self.search, "_fetch", return_value=rows):
            self.search.photo_budget = 0
            before = self.search.search(self.targets, UserProfile(), limit=6)
            self.search.photo_budget = 0.5
            after = self.search.search(self.targets, UserProfile(), limit=6)
        for index, product in enumerate(before):
            if product.matched_keywords:
                self.assertEqual(after[index].product_id, product.product_id)
        self.assertEqual([p.product_id for p in after][-2:], ["MS2", "MS6"])

    def test_cap_is_top_eight_not_first_eight_eligible_out_of_three_hundred(self):
        rows = [row(i, "와이드 롱 팬츠", 400 - i) for i in range(1, 9)] + [row(i) for i in range(9, 301)]
        with patch.object(self.search, "_fetch", return_value=rows):
            self.search.search(self.targets, UserProfile())
        self.provider.get_many.assert_not_called()
        with patch.object(self.search, "_fetch", return_value=[row(i) for i in range(1, 301)]):
            self.search.search(self.targets, UserProfile())
        self.assertEqual(len(self.provider.get_many.call_args.args[0]), 8)

    def test_photo_failure_preserves_baseline(self):
        self.provider.get_many.side_effect = RuntimeError("GPU unavailable")
        with patch.object(self.search, "_fetch", return_value=[row(1), row(2)]):
            result = self.search.search(self.targets, UserProfile())
        self.assertEqual([p.product_id for p in result], ["MS1", "MS2"])
        self.assertTrue(self.search.last_search_stats["photo"]["failed"])

    def test_source_and_confidence_guard(self):
        raw = {"lower_fit": {"labels": ["와이드핏"], "confidence": .96, "accepted": True}}
        self.assertEqual(accepted_attributes("bottom", {"worn": False}, raw), {})
        raw["lower_fit"]["confidence"] = .89
        self.assertEqual(accepted_attributes("bottom", {"worn": True}, raw), {})
        ev = evidence()
        ev["fit"]["source"] = "title"
        self.assertEqual(photo_matches("bottom", "팬츠", {"fit": ["와이드"]}, ev), {})

    def test_llm_cannot_describe_photo_evidence_as_a_title_match(self):
        import io
        product = ShoppingProduct("MS2", "팬츠", "", 100, "https://image/2", "", "bottom",
                                  photo_attributes={"fit": dict(evidence()["fit"], keyword="와이드")})
        pose = PoseAnalysis(False, 0, "", 0, 0, 0, "")
        for reason, accepted in (("와이드 핏이 상품명과 일치합니다.", False),
                                  ("상품 사진에서 와이드 핏으로 추정됩니다.", True)):
            generated = json.dumps({"items": [{"product_id": "MS2", "summary": reason,
                                                "evidence_ids": ["MS2-E1"]}]})
            response = {"candidates": [{"content": {"parts": [{"text": generated}]}}]}
            with patch.dict(os.environ, {"FASHION_LLM_REASONS": "1", "FASHION_LLM_PROVIDER": "gemini", "GEMINI_API_KEY": "test"}), patch(
                "recommendation_explanations.urllib.request.urlopen",
                return_value=io.BytesIO(json.dumps(response).encode())
            ):
                add_product_recommendation_reasons([product], UserProfile(), pose, self.targets, use_llm=True)
            self.assertEqual(product.recommendation_reason_source, "llm" if accepted else "rules")
            self.assertIn("상품 사진", product.recommendation_reason)
            self.assertNotIn("상품명", product.recommendation_reason)


class PhotoCacheTests(unittest.TestCase):
    def setUp(self):
        self.provider = LiveProductAttributes(None, None, cache_size=2)
        self.provider.predict_image = Mock(return_value={"attributes": evidence()})
        self.addCleanup(self.provider.close)
        self.product = ShoppingProduct("MS1", "팬츠", "", 100, "https://image/1", "", "bottom")

    def test_cache_reuses_negative_results_and_invalidates_url_and_ttl(self):
        self.provider.predict_image.return_value = {"attributes": {}}
        loader = Mock(return_value="image")
        self.provider.get_many([self.product], loader, .2)
        self.provider.get_many([self.product], loader, .2)
        self.assertEqual(loader.call_count, 1)
        self.product.image_url += "new"
        self.provider.get_many([self.product], loader, .2)
        self.assertEqual(loader.call_count, 2)
        self.provider.cache_ttl = 0
        self.provider.get_many([self.product], loader, .2)
        self.assertEqual(loader.call_count, 3)

    def test_slow_download_never_triggers_late_inference(self):
        def slow(product, timeout):
            time.sleep(.10)
            return "image"
        started = time.monotonic()
        result = self.provider.get_many([self.product], slow, .02)
        self.assertLess(time.monotonic() - started, .08)
        self.assertEqual(result, {})
        time.sleep(.12)
        self.provider.predict_image.assert_not_called()

    def test_deadline_finishes_current_forward_and_starts_no_more(self):
        def slow(*args):
            time.sleep(.04)
            return {"attributes": evidence()}
        self.provider.predict_image.side_effect = slow
        other = ShoppingProduct("MS2", "팬츠", "", 100, "https://image/2", "", "bottom")
        result = self.provider.get_many([self.product, other], Mock(return_value="image"), .02)
        self.assertEqual(list(result), ["MS1"])
        self.assertEqual(self.provider.predict_image.call_count, 1)

    def test_download_overlaps_size_lookup_and_inference_stays_on_caller_thread(self):
        import threading
        inference_threads = []
        def predict(*args):
            inference_threads.append(threading.get_ident())
            return {"attributes": evidence()}
        self.provider.predict_image.side_effect = predict
        def loader(product, timeout):
            time.sleep(.04)
            return "image"
        def sizes(product_id):
            time.sleep(.08)
            return {"status": "unavailable"}
        search = MusinsaLiveSearch(photo_provider=self.provider, measurements=SimpleNamespace(get=sizes), photo_budget=.01)
        self.addCleanup(search.close)
        targets = TargetKeywordResult("mixed", {"bottom": {"fit": ["와이드"]}})
        with patch.object(search, "_fetch", return_value=[row(1)]):
            results = search.search(targets, UserProfile(), photo_loader=loader)
        self.assertEqual(results[0].photo_attributes["fit"]["label"], "와이드핏")
        self.assertEqual(inference_threads, [threading.get_ident()])


if __name__ == "__main__":
    unittest.main()
