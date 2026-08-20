from __future__ import annotations

import itertools
import json
from pathlib import Path

from outfit_analyzer import NEUTRALS, color_harmony
from product_catalog import ProductCatalog
from schemas import OutfitAnalysis, PoseAnalysis, Product, Recommendation, UserProfile


class RecommendationEngine:
    """전문가 규칙과 상품 조건을 합산하는 설명 가능한 추천 엔진."""

    WEIGHTS = {
        "body_fit": 0.30,
        "color_harmony": 0.20,
        "purpose_fit": 0.20,
        "preference_fit": 0.15,
        "product_condition": 0.15,
    }

    def __init__(self, rules_path: str | Path, catalog: ProductCatalog) -> None:
        self.rules = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        self.catalog = catalog

    @staticmethod
    def _pair_harmony(first: Product | None, second: Product | None, outfit: OutfitAnalysis) -> float:
        first_color = first.color if first else outfit.upper_color
        second_color = second.color if second else outfit.lower_color
        harmony = color_harmony(first_color, second_color)
        return {
            "안정적인 무채색 조합": 1.0,
            "톤온톤": 0.95,
            "유사색 조합": 0.85,
            "대비색 조합": 0.75,
            "보통 조합": 0.65,
        }[harmony]

    def _score_candidate(
        self,
        top: Product | None,
        bottom: Product | None,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
    ) -> tuple[float, dict[str, float], list[str]]:
        products = [product for product in (top, bottom) if product]
        if not products:
            return 0.0, {}, ["현재 코디를 유지합니다."]

        body_fit = sum(pose.body_shape in product.body_shapes for product in products) / len(products)
        purpose_fit = sum(profile.purpose in product.purposes for product in products) / len(products)
        preference_fit = sum(profile.desired_style == product.style for product in products) / len(products)
        budget_fit = 1.0 if sum(product.price for product in products) <= profile.budget else 0.0
        stock_fit = sum(product.stock for product in products) / len(products)
        product_condition = 0.75 * budget_fit + 0.25 * stock_fit
        harmony = self._pair_harmony(top, bottom, outfit)

        breakdown = {
            "body_fit": body_fit,
            "color_harmony": harmony,
            "purpose_fit": purpose_fit,
            "preference_fit": preference_fit,
            "product_condition": product_condition,
        }
        total = sum(breakdown[name] * self.WEIGHTS[name] for name in self.WEIGHTS) * 100

        body_rule = self.rules["body_shape_rules"].get(pose.body_shape, {})
        reasons = [
            body_rule.get("tip", "체형 비율과 전체 실루엣을 고려했습니다."),
            f"{profile.purpose} 목적과 {profile.desired_style} 취향을 기준으로 골랐습니다.",
            f"상의와 하의의 색상 관계는 '{color_harmony(top.color if top else outfit.upper_color, bottom.color if bottom else outfit.lower_color)}'입니다.",
        ]
        return round(total, 2), {key: round(value * 100, 1) for key, value in breakdown.items()}, reasons

    def recommend(
        self,
        profile: UserProfile,
        pose: PoseAnalysis,
        outfit: OutfitAnalysis,
        top_k: int = 3,
    ) -> list[Recommendation]:
        scopes = self.rules["change_scope_map"].get(profile.change_scope, ["top", "bottom"])
        tops = self.catalog.available("top", profile.gender) if "top" in scopes else [None]
        bottoms = self.catalog.available("bottom", profile.gender) if "bottom" in scopes else [None]

        candidates = []
        for top, bottom in itertools.product(tops, bottoms):
            products = [product for product in (top, bottom) if product]
            if not products:
                continue
            score, breakdown, reasons = self._score_candidate(top, bottom, profile, pose, outfit)
            candidates.append((score, products, breakdown, reasons))

        candidates.sort(key=lambda item: item[0], reverse=True)
        return [
            Recommendation(rank=index, products=products, total_score=score, score_breakdown=breakdown, reasons=reasons)
            for index, (score, products, breakdown, reasons) in enumerate(candidates[:top_k], start=1)
        ]
