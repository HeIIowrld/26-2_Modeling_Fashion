from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from body_shape_scoring import active_r_bod_keywords, semantic_body_shape_score
from musinsa_live_search import ATTRIBUTE_WEIGHTS, MusinsaLiveSearch
from recommendation_keywords import TargetKeywordResult


class BodyShapeScoringTests(unittest.TestCase):
    def test_synonyms_in_one_group_count_once(self):
        score, groups = semantic_body_shape_score(
            ["레귤러", "정돈된 핏", "스트레이트"],
            ["레귤러", "정돈된 핏", "스트레이트"],
        )

        self.assertEqual(score, 4.0)
        self.assertEqual(groups, ["fit_silhouette"])

    def test_two_distinct_groups_increase_coverage(self):
        one_group, _ = semantic_body_shape_score(
            ["레귤러", "V넥"], ["레귤러"]
        )
        two_groups, groups = semantic_body_shape_score(
            ["레귤러", "V넥"], ["레귤러", "V넥"]
        )

        self.assertEqual(one_group, 2.0)
        self.assertEqual(two_groups, 4.0)
        self.assertEqual(groups, ["fit_silhouette", "neckline_structure"])

    def test_keyword_without_r_bod_provenance_is_excluded(self):
        self.assertEqual(active_r_bod_keywords({"레귤러": ["R-SIL-01"]}), [])
        targets = TargetKeywordResult(
            mode="mixed",
            targets={"top": {"fit": ["레귤러"]}},
            keyword_rules={"top": {"레귤러": ["R-SIL-01"]}},
        )
        result = MusinsaLiveSearch().score_with_body_shape(
            {"goodsName": "레귤러 셔츠", "brandName": "brand"},
            targets.targets["top"], 0, "top", targets, 1.0,
            body_shape_mode="semantic_group",
        )
        self.assertEqual(result[1], 0.0)

    def test_no_active_group_scores_zero(self):
        self.assertEqual(semantic_body_shape_score([], ["레귤러"]), (0.0, []))

    def test_binary_is_default_and_fit_weight_is_unchanged(self):
        targets = TargetKeywordResult(
            mode="mixed",
            targets={"top": {"fit": ["레귤러"], "structure": ["V넥"]}},
            keyword_rules={"top": {"레귤러": ["R-BOD-07"], "V넥": ["R-BOD-07"]}},
        )
        searcher = MusinsaLiveSearch()
        item = {
            "goodsName": "레귤러 V넥 셔츠",
            "brandName": "brand",
            "reviewCount": 0,
            "reviewScore": 0,
        }

        binary = searcher.score_with_body_shape(
            item, targets.targets["top"], 0, "top", targets, 1.0
        )
        semantic = searcher.score_with_body_shape(
            item, targets.targets["top"], 0, "top", targets, 1.0,
            body_shape_mode="semantic_group",
        )

        self.assertEqual(ATTRIBUTE_WEIGHTS["fit"], 4.0)
        self.assertEqual(binary[0], semantic[0])
        self.assertEqual(binary[2], binary[0])
        self.assertEqual(semantic[2], semantic[0])
        self.assertEqual(binary[1], 5.5)
        self.assertEqual(semantic[1], 4.0)

    def test_multiplier_one_has_same_final_score_in_both_modes(self):
        targets = TargetKeywordResult(
            mode="mixed",
            targets={"top": {"fit": ["레귤러"]}},
            keyword_rules={"top": {"레귤러": ["R-BOD-07"]}},
        )
        searcher = MusinsaLiveSearch()
        item = {"goodsName": "레귤러 셔츠", "brandName": "brand"}

        binary = searcher.score_with_body_shape(
            item, targets.targets["top"], 0, "top", targets, 1.0
        )
        semantic = searcher.score_with_body_shape(
            item, targets.targets["top"], 0, "top", targets, 1.0,
            body_shape_mode="semantic_group",
        )

        self.assertEqual(binary[2], semantic[2])
        self.assertEqual(binary[0], semantic[0])


if __name__ == "__main__":
    unittest.main()
