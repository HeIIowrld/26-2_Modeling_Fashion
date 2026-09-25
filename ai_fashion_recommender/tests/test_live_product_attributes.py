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
from live_product_attributes import (LiveProductAttributes, POLICY_VERSION,
                                     SHARED_FIT_POLICY_VERSION, accepted_attributes,
                                     accepted_shared_fit_attributes,
                                     accepted_design_attributes, flat_product_design_metrics,
                                     photo_matches, photo_validation_axes,
                                     rule_backed_photo_attributes, title_has_axis)
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

    def test_default_photo_budget_is_one_second(self):
        self.assertEqual(self.search.photo_budget, 1.0)

    def test_unmatched_product_improves_without_claiming_title_match(self):
        with patch.object(self.search, "_fetch", return_value=[row(1, reviews=100), row(2)]):
            result = self.search.search(self.targets, UserProfile(), limit=1)
        self.assertEqual(result[0].product_id, "MS2")
        self.assertEqual(result[0].retrieval_score, 11)  # photo fit 8 + current wide policy 3
        self.assertEqual(result[0].matched_keywords, [])
        add_product_recommendation_reasons(result, UserProfile(), PoseAnalysis(False, 0, "", 0, 0, 0, ""),
                                           self.targets, use_llm=False)
        self.assertIn("상품 사진", result[0].recommendation_reason)
        self.assertIn("추정", result[0].recommendation_reason)
        self.assertNotIn("상품명", result[0].recommendation_reason)
        self.assertNotIn("photo_attributes", result[0].public_dict())
        self.assertEqual(result[0].public_dict()["ranking_evidence_source"], "title+photo")
        self.assertEqual(self.search.last_search_stats["photo"]["ranking_mode"], "photo_assisted")

    def test_contradicting_name_and_unsupported_semantics_never_overridden(self):
        for name in ("슬림 팬츠", "테이퍼드 팬츠", "regular pants", "스키니 팬츠"):
            score, matched = self.search._score(dict(row(2, name), category="bottom"),
                self.targets.targets["bottom"], 0, photo_attributes=evidence())
            self.assertEqual((score, matched), (0, []))
        self.assertEqual(photo_matches("bottom", "팬츠", {"fit": ["세미와이드"]}, evidence()), {})
        self.assertEqual(photo_matches("shoes", "슈즈", {"fit": ["와이드"]}, evidence()), {})
        self.assertTrue(title_has_axis("부츠컷 팬츠", "fit"))
        self.assertFalse(title_has_axis("부츠컷 팬츠", "length"))

    def test_visual_rule_match_competes_with_title_match(self):
        rows = [row(1, "와이드", 1, "A"), row(2, reviews=100, brand="B")]
        with patch.object(self.search, "_fetch", return_value=rows):
            result = self.search.search(self.targets, UserProfile(), limit=2)
        self.assertEqual([product.product_id for product in result], ["MS2", "MS1"])
        self.assertEqual(result[0].photo_attributes["fit"]["label"], "와이드핏")

    def test_sample_is_eight_eligible_products_even_below_title_matches(self):
        rows = [row(i, "와이드 롱 팬츠", 400 - i) for i in range(1, 9)] + [row(i) for i in range(9, 301)]
        with patch.object(self.search, "_fetch", return_value=rows):
            self.search.search(self.targets, UserProfile())
        first_sample = self.provider.get_many.call_args.args[0]
        self.assertEqual(len(first_sample), 8)
        self.assertTrue(all(product.name == "팬츠" for product in first_sample))
        self.provider.get_many.reset_mock()
        with patch.object(self.search, "_fetch", return_value=[row(i) for i in range(1, 301)]):
            self.search.search(self.targets, UserProfile())
        self.assertEqual(len(self.provider.get_many.call_args.args[0]), 8)

    def test_photo_sample_is_diverse_by_item_type_fit_and_brand(self):
        search = MusinsaLiveSearch(photo_provider=self.provider, photo_candidates=4)
        self.addCleanup(search.close)
        products = [
            ShoppingProduct(f"T{i}", "베이직 레귤러 티셔츠", "브랜드A", 1, "", "", "top",
                            retrieval_score=20 - i)
            for i in range(5)
        ] + [
            ShoppingProduct("S", "오버핏 옥스포드 셔츠", "브랜드B", 1, "", "", "top", retrieval_score=10),
            ShoppingProduct("N", "여유핏 니트", "브랜드C", 1, "", "", "top", retrieval_score=9),
            ShoppingProduct("J", "테일러드 블레이저", "브랜드D", 1, "", "", "top", retrieval_score=8),
        ]
        sample = search._photo_shortlist(
            {"top": products}, TargetKeywordResult("user_input", {"top": {}})
        )
        dimensions = {search._photo_sample_dimensions(product)[:2] for product in sample}
        self.assertEqual(len(sample), 4)
        self.assertGreaterEqual(len(dimensions), 3)
        self.assertGreaterEqual(len({product.brand for product in sample}), 3)

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

    def test_bottom_fit_uses_existing_shape_head_and_abstains_on_conflict(self):
        context = {"worn": True}
        shape = {"pant_leg_shape": {"labels": ["스트레이트"], "confidence": .96, "accepted": True}}
        result = accepted_attributes("bottom", context, shape)
        self.assertEqual(result["fit"]["label"], "스트레이트핏")

        conflicting = {
            "lower_fit": {"labels": ["와이드핏"], "confidence": .97, "accepted": True},
            "pant_leg_shape": {"labels": ["스트레이트"], "confidence": .96, "accepted": True},
        }
        self.assertNotIn("fit", accepted_attributes("bottom", context, conflicting))

    def test_shared_predictor_keeps_semiwide_distinct_from_straight(self):
        predictions = {
            "bottom_silhouette": {
                "labels": ["세미와이드"], "confidence": .82, "accepted": True,
            },
            "bottom_length": {
                "labels": ["풀렝스"], "confidence": .79, "accepted": True,
            },
        }
        attributes = accepted_shared_fit_attributes("bottom", {"worn": True}, predictions)
        self.assertEqual(attributes["fit"]["label"], "세미와이드")
        self.assertEqual(attributes["fit"]["policy"], SHARED_FIT_POLICY_VERSION)
        self.assertEqual(attributes["length"]["label"], "롱·긴바지 기장")
        matched = photo_matches("bottom", "팬츠", {"fit": ["세미와이드"]}, attributes)
        self.assertEqual(matched["fit"]["keyword"], "세미와이드")

    def test_shared_predictor_rechecks_fit_written_in_product_title(self):
        visual = {
            "fit": {
                "label": "스트레이트", "confidence": .84,
                "source": "product_photo", "policy": SHARED_FIT_POLICY_VERSION,
            }
        }
        attributes = {"fit": ["세미와이드"]}
        self.assertEqual(
            photo_validation_axes("bottom", "세미와이드 팬츠", attributes, visual),
            ["fit"],
        )
        self.assertEqual(photo_matches("bottom", "세미와이드 팬츠", attributes, visual), {})

    def test_design_head_and_point_area_distinguish_small_logo_from_large_graphic(self):
        predictions = {
            "category": {"labels": ["티셔츠"], "confidence": .94, "accepted": True},
            # Even when a tiny logo activates '그래픽', its occupied area keeps
            # it in the basic/small-logo bucket.
            "pattern": {"labels": ["그래픽"], "confidence": .91, "accepted": True},
            "detail": {"labels": ["디테일 없음"], "confidence": .88, "accepted": True},
        }
        small = accepted_design_attributes(
            "top", predictions, {"flat_product": True, "point_area_ratio": .008}
        )["design"]
        large = accepted_design_attributes(
            "top", predictions, {"flat_product": True, "point_area_ratio": .09}
        )["design"]
        self.assertTrue(small["plain_basic"])
        self.assertFalse(large["plain_basic"])

    def test_flat_product_metric_measures_visible_point_area(self):
        from PIL import Image, ImageDraw
        small = Image.new("RGB", (400, 500), "white")
        large = Image.new("RGB", (400, 500), "white")
        for image in (small, large):
            ImageDraw.Draw(image).rectangle((80, 70, 320, 440), fill="black")
        ImageDraw.Draw(small).rectangle((235, 170, 250, 178), fill="white")
        ImageDraw.Draw(large).rectangle((130, 160, 270, 230), fill="white")
        small_metric = flat_product_design_metrics(small)
        large_metric = flat_product_design_metrics(large)
        self.assertTrue(small_metric["flat_product"])
        self.assertLess(small_metric["point_area_ratio"], .035)
        self.assertGreater(large_metric["point_area_ratio"], .035)

    def test_saturated_plain_tee_counts_as_a_color_point(self):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (400, 500), "white")
        ImageDraw.Draw(image).rectangle((80, 70, 320, 440), fill=(35, 105, 220))
        metric = flat_product_design_metrics(image)
        predictions = {
            "category": {"labels": ["티셔츠"], "confidence": .94, "accepted": True},
            "pattern": {"labels": ["무지"], "confidence": .91, "accepted": True},
        }
        design = accepted_design_attributes("top", predictions, metric)["design"]
        self.assertTrue(metric["color_point"])
        self.assertFalse(design["plain_basic"])

    def test_large_wordmark_is_still_a_logo_only_design(self):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (400, 500), "white")
        draw = ImageDraw.Draw(image)
        draw.rectangle((80, 70, 320, 440), fill="black")
        for x in (125, 165, 205, 245):
            draw.rectangle((x, 165, x + 22, 225), fill="white")
        metric = flat_product_design_metrics(image)
        predictions = {
            "category": {"labels": ["티셔츠"], "confidence": .94, "accepted": True},
            "pattern": {"labels": ["그래픽"], "confidence": .91, "accepted": True},
        }
        design = accepted_design_attributes("top", predictions, metric)["design"]
        self.assertTrue(metric["wordmark_like"])
        self.assertTrue(design["plain_basic"])

    def test_photo_ranking_uses_only_keywords_from_active_fashion_rules(self):
        targets = TargetKeywordResult(
            "mixed", {"bottom": {"fit": ["와이드", "스트레이트"]}},
            applied_rules=["R-SIL-01"],
            keyword_rules={"bottom": {
                "와이드": ["R-SIL-01"], "스트레이트": ["R-NOT-ACTIVE"],
            }},
        )
        self.assertEqual(rule_backed_photo_attributes(targets, "bottom")["fit"], ["와이드"])

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
