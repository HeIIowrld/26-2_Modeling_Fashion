"""추천 키워드로 무신사 상품을 실시간 검색하는 작은 어댑터."""

from __future__ import annotations

import json
import math
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Iterable

from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile


API_URL = "https://api.musinsa.com/api2/dp/v2/plp/goods"
CATEGORY_CODES = {"top": "001", "bottom": "003"}
CATEGORY_FALLBACK_QUERY = {"top": "베이직 상의", "bottom": "팬츠"}
ATTRIBUTE_WEIGHTS = {
    "fit": 4.0,
    "length": 3.0,
    "waistline": 3.0,
    "material": 3.0,
    "color": 2.0,
    "style": 1.5,
    "structure": 1.5,
    "silhouette": 1.5,
    "function": 1.0,
}
KEYWORD_ALIASES = {
    "세미와이드": ("세미와이드", "세미 와이드"),
    "와이드": ("와이드", "wide"),
    "스트레이트": ("스트레이트", "straight", "일자"),
    "레귤러": ("레귤러", "regular", "스탠다드"),
    "정돈된 핏": ("레귤러", "스탠다드", "슬림"),
    "여유핏": ("여유", "오버핏", "오버사이즈", "루즈"),
    "풀렝스": ("풀렝스", "풀 렝스", "롱"),
    "허리선": ("크롭", "세미크롭", "숏"),
    "미드라이즈": ("미드라이즈", "미드 라이즈", "중고층"),
    "하이라이즈": ("하이라이즈", "하이 라이즈", "고층"),
    "코튼": ("코튼", "면"),
    "니트": ("니트", "knit"),
    "데님": ("데님", "denim", "진"),
    "가죽": ("레더", "가죽", "leather"),
    "린넨": ("린넨", "리넨", "linen"),
}


@dataclass
class ShoppingProduct:
    product_id: str
    name: str
    brand: str
    price: int
    image_url: str
    url: str
    category: str
    gender: str = "공용"
    review_count: int = 0
    review_score: float = 0.0
    source: str = "musinsa_live"
    search_keywords: list[str] = field(default_factory=list)
    matched_keywords: list[str] = field(default_factory=list)
    retrieval_score: float = 0.0

    def public_dict(self) -> dict:
        """내부 키워드와 점수는 웹 UI에 보내지 않는다."""
        data = asdict(self)
        data.pop("matched_keywords", None)
        data.pop("retrieval_score", None)
        return data


class MusinsaLiveSearch:
    """무신사 검색을 짧게 캐시하고 실패를 FITTA 분석과 격리한다."""

    def __init__(self, timeout: float = 6.0, cache_ttl: float = 300.0) -> None:
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: dict[tuple[str, str], tuple[float, list[dict]]] = {}

    @staticmethod
    def _normalized(value: str) -> str:
        return "".join(value.lower().split())

    @staticmethod
    def _aliases(keyword: str) -> tuple[str, ...]:
        return KEYWORD_ALIASES.get(keyword, (keyword,))

    @staticmethod
    def _preferred_term(attributes: dict[str, list[str]], attribute: str) -> str:
        values = attributes.get(attribute, [])
        if not values:
            return ""
        value = values[0]
        return MusinsaLiveSearch._aliases(value)[0]

    def _queries(self, category: str, attributes: dict[str, list[str]]) -> list[str]:
        fit = self._preferred_term(attributes, "fit")
        material = self._preferred_term(attributes, "material")
        style = self._preferred_term(attributes, "style")
        color = self._preferred_term(attributes, "color")
        primary = " ".join(value for value in (fit, material) if value)
        candidates = [primary, material, fit, " ".join(value for value in (style, material) if value), color]
        candidates.append(CATEGORY_FALLBACK_QUERY[category])
        return list(dict.fromkeys(value.strip() for value in candidates if value.strip()))[:3]

    def _fetch(self, category: str, query: str, size: int = 40) -> list[dict]:
        key = (category, query)
        cached = self._cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]
        params = urllib.parse.urlencode({
            "gf": "A",
            "category": CATEGORY_CODES[category],
            "keyword": query,
            "page": 1,
            "size": size,
            "sortCode": "POPULAR",
            "caller": "SEARCH",
        })
        request = urllib.request.Request(
            f"{API_URL}?{params}",
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; FITTA/1.0; product-search)",
                "Accept": "application/json",
                "Referer": "https://www.musinsa.com/",
                "Accept-Language": "ko-KR,ko;q=0.9",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        items = payload.get("data", {}).get("list", [])
        self._cache[key] = (now, items)
        return items

    def _allowed(self, item: dict, profile: UserProfile) -> bool:
        if item.get("isSoldOut"):
            return False
        price = int(item.get("finalPrice") or item.get("price") or 0)
        if price <= 0:
            return False
        if profile.min_budget is not None and price < profile.min_budget:
            return False
        upper = profile.max_budget if profile.max_budget is not None else profile.budget
        if upper and price > upper:
            return False
        gender = str(item.get("displayGenderText") or "공용")
        if profile.gender and gender not in ("", "공용", profile.gender):
            return False
        text = self._normalized(
            " ".join(str(item.get(key) or "") for key in ("goodsName", "brandName", "brand"))
        )
        blocked = [*profile.avoided_colors, *profile.avoided_materials, *profile.excluded_item_types]
        return not any(self._normalized(value) in text for value in blocked if value)

    def _score(self, item: dict, attributes: dict[str, list[str]], rank: int) -> tuple[float, list[str]]:
        text = self._normalized(
            " ".join(str(item.get(key) or "") for key in ("goodsName", "brandName", "brand"))
        )
        score = max(0.0, 1.0 - rank * 0.02)
        matched: list[str] = []
        for attribute, weight in ATTRIBUTE_WEIGHTS.items():
            for keyword in attributes.get(attribute, []):
                if any(self._normalized(alias) in text for alias in self._aliases(keyword)):
                    score += weight
                    matched.append(keyword)
                    break
        review_score = float(item.get("reviewScore") or 0) / 100
        review_count = int(item.get("reviewCount") or 0)
        score += review_score * 0.5 + math.log1p(review_count) * 0.04
        return score, matched

    @staticmethod
    def _representative_keywords(
        matched: list[str],
        attributes: dict[str, list[str]],
        limit: int = 3,
    ) -> list[str]:
        """내부 전체 조건 대신 검색을 대표하는 키워드만 고른다."""
        candidates = list(matched)
        for attribute in ("fit", "material", "length", "style", "color", "silhouette", "function"):
            candidates.extend(attributes.get(attribute, []))
        return list(dict.fromkeys(keyword for keyword in candidates if keyword))[:limit]

    def _live_category(
        self,
        category: str,
        attributes: dict[str, list[str]],
        profile: UserProfile,
    ) -> list[ShoppingProduct]:
        found: dict[str, ShoppingProduct] = {}
        for query in self._queries(category, attributes):
            try:
                items = self._fetch(category, query)
            except (OSError, TimeoutError, ValueError, json.JSONDecodeError):
                continue
            for rank, item in enumerate(items):
                if not self._allowed(item, profile):
                    continue
                goods_no = str(item.get("goodsNo") or "")
                if not goods_no:
                    continue
                score, matched = self._score(item, attributes, rank)
                product = ShoppingProduct(
                    product_id=f"MS{goods_no}",
                    name=str(item.get("goodsName") or "상품명 없음"),
                    brand=str(item.get("brandName") or item.get("brand") or ""),
                    price=int(item.get("finalPrice") or item.get("price") or 0),
                    image_url=str(item.get("thumbnail") or ""),
                    # 외부 응답의 임의 링크 대신 검증된 상품 번호로 무신사 주소를 직접 만든다.
                    url=f"https://www.musinsa.com/products/{urllib.parse.quote(goods_no, safe='')}",
                    category=category,
                    gender=str(item.get("displayGenderText") or "공용"),
                    review_count=int(item.get("reviewCount") or 0),
                    review_score=float(item.get("reviewScore") or 0),
                    search_keywords=self._representative_keywords(matched, attributes),
                    matched_keywords=list(dict.fromkeys(matched)),
                    retrieval_score=round(score, 4),
                )
                previous = found.get(goods_no)
                if previous is None or product.retrieval_score > previous.retrieval_score:
                    found[goods_no] = product
            if len(found) >= 12:
                break
        return sorted(found.values(), key=lambda product: (-product.retrieval_score, -product.review_count))

    def _local_fallback(
        self,
        targets: TargetKeywordResult,
        profile: UserProfile,
        products: Iterable[Product],
    ) -> dict[str, list[ShoppingProduct]]:
        grouped = {category: [] for category in targets.targets}
        for product in products:
            if product.category not in grouped or not product.stock or not product.url or not product.image_url:
                continue
            item = {
                "goodsName": product.name,
                "brandName": product.brand,
                "finalPrice": product.price,
                "displayGenderText": product.gender,
            }
            if not self._allowed(item, profile):
                continue
            score, matched = self._score(item, targets.targets[product.category], 0)
            grouped[product.category].append(ShoppingProduct(
                product_id=product.product_id,
                name=product.name,
                brand=product.brand,
                price=product.price,
                image_url=product.image_url,
                url=product.url,
                category=product.category,
                gender=product.gender or "공용",
                source="musinsa_catalog_fallback",
                search_keywords=self._representative_keywords(
                    matched, targets.targets[product.category]
                ),
                matched_keywords=matched,
                retrieval_score=score,
            ))
        for category in grouped:
            grouped[category].sort(key=lambda product: -product.retrieval_score)
        return grouped

    def search(
        self,
        targets: TargetKeywordResult,
        profile: UserProfile,
        limit: int = 3,
        fallback_products: Iterable[Product] = (),
    ) -> list[ShoppingProduct]:
        grouped = {
            category: self._live_category(category, attributes, profile)
            for category, attributes in targets.targets.items()
            if category in CATEGORY_CODES
        }
        if not any(grouped.values()):
            grouped = self._local_fallback(targets, profile, fallback_products)

        # 상·하의를 모두 찾을 때 한 카테고리만 세 장을 독점하지 않도록 교차 선택한다.
        selected: list[ShoppingProduct] = []
        depth = 0
        categories = list(targets.targets)
        while len(selected) < limit and any(depth < len(grouped.get(category, [])) for category in categories):
            for category in categories:
                products = grouped.get(category, [])
                if depth < len(products):
                    selected.append(products[depth])
                    if len(selected) >= limit:
                        break
            depth += 1
        return selected
