from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_body_shape_weight import rank_fixed_candidates, union_candidate_pools
from musinsa_live_search import MusinsaLiveSearch, ShoppingProduct
from recommendation_keywords import TargetKeywordResult


def product(product_id: str, name: str) -> ShoppingProduct:
    return ShoppingProduct(
        product_id, name, "brand", 50_000, "image", "url", "top",
        review_count=10, review_score=90,
    )


class BodyShapeWeightEvalTests(unittest.TestCase):
    def setUp(self):
        self.searcher = MusinsaLiveSearch()
        self.targets = TargetKeywordResult(
            mode="mixed",
            targets={"top": {"fit": ["여유핏"]}},
            keyword_rules={"top": {"여유핏": ["R-BOD-07"]}},
        )

    def test_union_candidate_pool_deduplicates_product_ids(self):
        pool = union_candidate_pools([
            {"top": [product("a", "A"), product("b", "B")]},
            {"top": [product("b", "B duplicate"), product("c", "C")]},
        ])

        self.assertEqual([item.product_id for item in pool["top"]], ["a", "b", "c"])

    def test_every_multiplier_uses_same_candidate_ids(self):
        pool = {"top": [product("a", "여유핏 재킷"), product("b", "기본 재킷")]}
        rankings = [
            rank_fixed_candidates(pool, self.targets, self.searcher, multiplier)
            for multiplier in (0.0, 1.0, 1.5, 2.0, 3.0)
        ]

        self.assertTrue(all(
            {item.product_id for item in ranking} == {"a", "b"}
            for ranking in rankings
        ))

    def test_rank_at_one_matches_base_order_and_scores(self):
        pool = {"top": [product("a", "여유핏 재킷"), product("b", "기본 재킷")]}
        ranking = rank_fixed_candidates(pool, self.targets, self.searcher, 1.0)

        self.assertEqual(ranking[0].final_score, ranking[0].base_score)


if __name__ == "__main__":
    unittest.main()