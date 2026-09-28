"""추천 키워드로 무신사 상품을 실시간 검색하는 작은 어댑터."""

from __future__ import annotations

import time
import urllib.parse
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from functools import partial
from typing import Iterable

from product_colors import palettes_for
from product_measurements import ProductMeasurementClient, category_from_type_name
from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile
from shopping_http import bounded_results, fetch_json
from size_fit import compare_sizes
from live_product_attributes import (
    confident_fit_evidence,
    missing_photo_axes,
    photo_matches,
    photo_validation_axes,
    rule_backed_photo_attributes,
)
from fashion_ranking_policy import (
    basic_logo_tee_adjustment,
    canonical_bottom_fit,
    formal_context_adjustment,
    sporty_context_adjustment,
    trend_fit_adjustment,
)


API_URL = "https://api.musinsa.com/api2/dp/v2/plp/goods"
CATEGORY_CODES = {"top": "001", "bottom": "003", "shoes": "103"}
CATEGORY_FALLBACK_QUERY = {"top": "베이직 상의", "bottom": "팬츠", "shoes": "신발"}
SORT_CODES = ("POPULAR", "NEW")
PAGE_SIZE = 100
MAX_MATERIAL_QUERY_TERMS = 3
CASUAL_SHIRT_LIMIT = 1
SHIRT_DIVERSITY_EXEMPT_PURPOSES = {"데이트", "소개팅", "출근", "면접"}
SHIRT_DIVERSITY_EXEMPT_STYLES = {"포멀", "클래식"}
SHIRT_DIVERSITY_EXEMPT_DRESS_CODES = {"비즈니스 캐주얼", "포멀", "비즈니스 포멀", "클래식"}
# 입력 속성과 관련 있는 세부 분류만 추가한다. 무관한 품목은 검색하지 않는다.
SUBCATEGORIES = {
    "top": {"셔츠": "001002", "니트": "001006", "후드": "001008", "맨투맨": "001004"},
    "bottom": {"데님": "003002", "슬랙스": "003008", "쇼츠": "003009"},
}
ATTRIBUTE_WEIGHTS = {
    "item_type": 5.0,
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
PHOTO_SAMPLE_ITEM_TERMS = {
    "top": ("티셔츠", "후드", "맨투맨", "셔츠", "블라우스", "니트", "가디건", "블레이저", "재킷", "자켓", "코트", "베스트"),
    "bottom": ("슬랙스", "데님", "청바지", "카고", "조거", "트레이닝", "쇼츠", "반바지", "스커트", "팬츠"),
}
PHOTO_SAMPLE_TOP_FIT_TERMS = (
    ("오버핏", ("오버핏", "오버사이즈", "루즈")),
    ("여유핏", ("여유", "릴랙스")),
    ("슬림핏", ("슬림", "스키니")),
    ("레귤러핏", ("레귤러", "스탠다드")),
)
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
    photo_attributes: dict = field(default_factory=dict)
    # 무신사가 파는 색 이름 전부. 회피 색 검사에만 쓴다(대표 사진이 어느 색인지 모른다).
    color_options: list[str] = field(default_factory=list)
    # 실측표가 말하는 옷 종류('바지'·'셔츠'…). 무신사 카테고리 오류를 잡는 데 쓴다.
    measurement_type: str = ""
    retrieval_score: float = 0.0
    recommendation_reason: str = ""
    recommendation_reason_source: str = "rules"
    size_fit: dict = field(default_factory=dict)
    fit_evidence: list[str] = field(default_factory=list)
    fit_evidence_labels: list[str] = field(default_factory=list)
    reason_rule_ids: list[str] = field(default_factory=list)
    ranking_adjustments: dict = field(default_factory=dict)
    ranking_evidence_source: str = "title"

    def public_dict(self) -> dict:
        """내부 키워드와 점수는 웹 UI에 보내지 않는다."""
        data = asdict(self)
        data.pop("matched_keywords", None)
        data.pop("retrieval_score", None)
        data.pop("photo_attributes", None)
        data.pop("ranking_adjustments", None)
        data["size_fit"].pop("ranking_bonus", None)
        return data


class MusinsaLiveSearch:
    """무신사 검색을 짧게 캐시하고 실패를 FITTA 분석과 격리한다."""

    def __init__(
        self, timeout: float = 3.0, cache_ttl: float = 300.0, *,
        search_budget: float = 8.0, measurement_budget: float = 5.0,
        measurements: ProductMeasurementClient | None = None,
        candidates_per_category: int = 300, measurement_candidates: int = 8,
        photo_provider=None, photo_candidates: int = 8, photo_budget: float = 1.0,
    ) -> None:
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self.search_budget = search_budget
        self.measurement_budget = measurement_budget
        self.measurements = measurements
        self.candidates_per_category = max(1, candidates_per_category)
        self.measurement_candidates = max(1, measurement_candidates)
        self.photo_provider = photo_provider
        self.photo_candidates = min(8, max(0, photo_candidates))
        self.photo_budget = max(0, photo_budget)
        self._cache: dict[tuple, tuple[float, list[dict]]] = {}
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fitta-products")
        self.last_search_stats: dict = {}

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.photo_provider is not None:
            self.photo_provider.close()

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

    @classmethod
    def _keyword_matches_text(cls, attribute: str, keyword: str, text: str) -> bool:
        # Korean "티셔츠" contains the substring "셔츠". Treating a tee as a
        # button shirt erased the intended daily/date distinction.
        normalized_keyword = cls._normalized(keyword)
        if attribute == "item_type" and normalized_keyword in {"셔츠", "shirt"}:
            non_button = (
                "티셔츠", "t-shirt", "tshirt", "스웨트셔츠", "sweatshirt",
                "폴로셔츠", "피케셔츠", "럭비셔츠",
            )
            return (
                any(cls._normalized(alias) in text for alias in cls._aliases(keyword))
                and not any(cls._normalized(term) in text for term in non_button)
            )
        return any(cls._normalized(alias) in text for alias in cls._aliases(keyword))

    def _queries(
        self,
        category: str,
        attributes: dict[str, list[str]],
    ) -> list[str]:
        if category == "shoes":
            item_type = self._preferred_term(attributes, "item_type") or "스니커즈"
            color = self._preferred_term(attributes, "color")
            candidates = list(dict.fromkeys(" ".join(filter(None, terms)) for terms in (
                (color, item_type), (item_type,),
            )))
            return list(dict.fromkeys(candidates))
        item_types = list(dict.fromkeys(
            self._aliases(value)[0]
            for value in attributes.get("item_type", [])
            if value
        ))[:3]
        item_type = item_types[0] if item_types else ""
        fit = self._preferred_term(attributes, "fit")
        style = self._preferred_term(attributes, "style")
        color = self._preferred_term(attributes, "color")
        fallback_noun = CATEGORY_FALLBACK_QUERY[category]
        noun = item_type or fallback_noun
        # Fit is deliberately not required by the primary queries.  We first
        # collect visually plausible items and let the trained fit heads apply
        # Fashion Rules to the candidate images.  A fit-text query remains as a
        # fallback for resilience when photo inference is unavailable.
        materials = list(dict.fromkeys(
            self._aliases(value)[0]
            for value in attributes.get("material", [])
            if value
        ))[:MAX_MATERIAL_QUERY_TERMS]
        material_queries = [f"{material} {noun}" for material in materials]
        item_type_queries = [
            " ".join(value for value in (color or style, item) if value)
            for item in item_types
        ]
        fits = attributes.get("fit", [])
        alternative_fit = fits[1] if len(fits) > 1 else (self._aliases(fits[0])[-1] if fits else "")
        context_material = materials[0] if materials else noun
        candidates = [
            *material_queries,
            *item_type_queries,
            " ".join(value for value in (color or style, context_material) if value)
            if not item_type_queries else "",
            f"{alternative_fit or fit} {noun}" if fit else "",
        ]
        specific = list(dict.fromkeys(value.strip() for value in candidates if value.strip()))
        # Keep one broad query for resilience, but spend the other slots on
        # separate item types so the candidate set cannot collapse to shirts.
        return list(dict.fromkeys([*specific[:4], fallback_noun]))[:5]

    def _fetch(
        self, category: str, query: str, size: int = PAGE_SIZE, *,
        sort_code: str = "POPULAR", page: int = 1, category_code: str | None = None,
    ) -> list[dict]:
        code = category_code or CATEGORY_CODES[category]
        key = (code, query, sort_code, page, size)
        with self._cache_lock:
            cached = self._cache.get(key)
        now = time.monotonic()
        if cached and now - cached[0] < self.cache_ttl:
            return cached[1]
        params = urllib.parse.urlencode({
            "gf": "A",
            "category": code,
            "keyword": query,
            "page": page,
            "size": size,
            "sortCode": sort_code,
            "caller": "SEARCH",
        })
        payload = fetch_json(f"{API_URL}?{params}", self.timeout)
        data = payload.get("data") or {}
        if not isinstance(data, dict):
            raise ValueError("상품 목록 형식 오류")
        items = data.get("list", [])
        if not isinstance(items, list):
            raise ValueError("상품 목록 형식 오류")
        with self._cache_lock:
            if len(self._cache) >= 256:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = (time.monotonic(), items)
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

    def _score(self, item: dict, attributes: dict[str, list[str]], rank: int,
               photo_attributes: dict | None = None) -> tuple[float, list[str]]:
        text = self._normalized(
            " ".join(str(item.get(key) or "") for key in ("goodsName", "brandName", "brand"))
        )
        score = 0.0
        matched: list[str] = []
        matched_axes = set()
        for attribute, weight in ATTRIBUTE_WEIGHTS.items():
            for keyword in attributes.get(attribute, []):
                if self._keyword_matches_text(attribute, keyword, text):
                    score += weight
                    matched.append(keyword)
                    matched_axes.add(attribute)
                    break
        if photo_attributes:
            matches = photo_matches(item.get("category", ""), str(item.get("goodsName") or ""),
                                    attributes, photo_attributes)
            # A verified visual match must be able to compete with the same
            # keyword in a title (title matches currently receive 2x weight).
            score += sum(2 * ATTRIBUTE_WEIGHTS[axis] for axis in matches if axis not in matched_axes)
        return score, matched

    @staticmethod
    def _representative_keywords(
        matched: list[str],
        attributes: dict[str, list[str]],
        limit: int = 3,
    ) -> list[str]:
        """내부 전체 조건 대신 검색을 대표하는 키워드만 고른다."""
        candidates = list(matched)
        for attribute in ("item_type", "fit", "material", "length", "style", "color", "silhouette", "function"):
            candidates.extend(attributes.get(attribute, []))
        return list(dict.fromkeys(keyword for keyword in candidates if keyword))[:limit]

    @staticmethod
    def _sort_key(product: ShoppingProduct) -> tuple:
        bonus = product.size_fit.get("ranking_bonus", 0.0)
        return (-(product.retrieval_score + bonus), -product.review_score,
                -product.review_count, product.product_id)

    def _search_plan(self, targets: TargetKeywordResult) -> list[tuple[str, str, str, str]]:
        plans = {}
        for category, attributes in targets.targets.items():
            if category not in CATEGORY_CODES:
                continue
            words = " ".join(word for values in attributes.values() for word in values)
            subcategory = next((code for term, code in SUBCATEGORIES.get(category, {}).items() if term in words), None)
            plans[category] = [
                (category, query, sort_code, subcategory if index == 1 and subcategory else CATEGORY_CODES[category])
                for index, query in enumerate(self._queries(category, attributes))
                for sort_code in SORT_CODES
            ]
        # 상의가 느려도 하의 검색에 기회가 돌아가도록 요청을 교차 배치한다.
        return [plan[index] for index in range(max(map(len, plans.values()), default=0))
                for plan in plans.values() if index < len(plan)]

    def collect_candidates(self, targets: TargetKeywordResult, profile: UserProfile) -> dict[str, list[ShoppingProduct]]:
        started = time.monotonic()
        plan = self._search_plan(targets)
        batches = bounded_results(self._executor, [
            partial(self._fetch, category, query, sort_code=sort_code, category_code=code)
            for category, query, sort_code, code in plan
        ], started + self.search_budget)
        grouped: dict[str, dict[str, ShoppingProduct]] = {c: {} for c in targets.targets if c in CATEGORY_CODES}
        raw_count = 0
        for (category, query, sort_code, code), items in zip(plan, batches):
            attributes = targets.targets[category]
            for rank, item in enumerate(items or []):
                raw_count += 1
                try:
                    if not isinstance(item, dict) or not self._allowed(item, profile):
                        continue
                    goods_no = str(item.get("goodsNo") or "")
                    if not goods_no.isascii() or not goods_no.isdigit() or len(goods_no) > 12:
                        continue
                    score, matched = self._score(item, attributes, rank)
                    product = ShoppingProduct(
                        product_id=f"MS{goods_no}", name=str(item.get("goodsName") or "상품명 없음"),
                        brand=str(item.get("brandName") or item.get("brand") or ""),
                        price=int(item.get("finalPrice") or item.get("price") or 0),
                        image_url=("https:" + str(item["thumbnail"]) if str(item.get("thumbnail") or "").startswith("//")
                                   else str(item.get("thumbnail") or "")),
                        url=f"https://www.musinsa.com/products/{goods_no}", category=category,
                        gender=str(item.get("displayGenderText") or "공용"),
                        review_count=int(item.get("reviewCount") or 0), review_score=float(item.get("reviewScore") or 0),
                        search_keywords=self._representative_keywords(matched, attributes),
                        matched_keywords=list(dict.fromkeys(matched)), retrieval_score=round(score, 4),
                    )
                except (ValueError, TypeError, OverflowError):
                    continue  # 잘못된 한 상품이 전체 후보 생성을 막지 않는다.
                previous = grouped[category].get(goods_no)
                if previous is None or self._sort_key(product) < self._sort_key(previous):
                    grouped[category][goods_no] = product
        self.last_search_stats = {
            "requests": len(plan), "completed_requests": sum(batch is not None for batch in batches),
            "raw_count": raw_count, "unique_candidates": {c: len(p) for c, p in grouped.items()},
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        candidates = {category: sorted(products.values(), key=self._sort_key)[:self.candidates_per_category]
                      for category, products in grouped.items()}
        self.last_search_stats["retained_candidates"] = {c: len(p) for c, p in candidates.items()}
        return candidates

    def _compare_shortlist(self, grouped: dict[str, list[ShoppingProduct]], profile: UserProfile) -> None:
        if self.measurements is None:
            return
        # 최종 3개를 고르기 전에 카테고리별 상위 후보의 실측을 비교한다.
        shortlist = [products[index] for index in range(self.measurement_candidates)
                     for category, products in grouped.items() if category in {"top", "bottom"} and index < len(products)]
        records = bounded_results(self._executor, [partial(self.measurements.get, p.product_id) for p in shortlist],
                                  time.monotonic() + self.measurement_budget)
        for index, product in enumerate(shortlist):
            record = records[index] if index < len(records) else None
            product.size_fit = compare_sizes(record or {"status": "unavailable"}, product.category,
                                            profile.reference_measurements.get(product.category))
            # 같은 응답에 색 옵션과 실측표 종류가 들어 있다. 추가 요청은 없다.
            product.color_options = list((record or {}).get("color_options") or [])
            product.measurement_type = str((record or {}).get("type_name") or "")
        self.last_search_stats["measurement_candidates"] = len(shortlist)
        self.last_search_stats["measurement_tables"] = sum(bool(r and r.get("sizes")) for r in records)
        for products in grouped.values():
            products.sort(key=self._sort_key)

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
        photo_loader=None,
    ) -> list[ShoppingProduct]:
        if limit <= 0:
            return []
        started = time.monotonic()
        grouped = self.collect_candidates(targets, profile)
        if any(not products for products in grouped.values()):
            fallback = self._local_fallback(targets, profile, fallback_products)
            for category in grouped:
                if not grouped[category]:
                    grouped[category] = fallback.get(category, [])
        prefetched = None
        if self.photo_provider is not None and self.photo_budget > 0:
            try:
                prefetched = self.photo_provider.prefetch(self._photo_shortlist(grouped, targets), photo_loader)
            except Exception:
                pass  # Prefetch is optional; it must not block normal size ranking.
        self._compare_shortlist(grouped, profile)
        self._drop_wrong_category(grouped)
        self._drop_fully_avoided_colors(grouped, profile)
        self._supplement_photos(grouped, targets, photo_loader, prefetched)
        self._apply_fashion_policy_adjustments(grouped, profile)
        for products in grouped.values():
            products.sort(key=self._sort_key)
        selected = self._select(grouped, targets, limit, profile)
        self.last_search_stats["total_elapsed_seconds"] = round(time.monotonic() - started, 4)
        self.last_search_stats["vision_ranked_products"] = sum(
            bool(product.photo_attributes)
            for products in grouped.values() for product in products
        )
        return selected

    def _drop_wrong_category(self, grouped) -> None:
        """무신사 카테고리가 틀린 상품을 실측표로 걸러낸다. 추가 요청은 없다.

        상품의 부위는 우리가 검색한 카테고리를 그대로 쓴다(`category=category`). 무신사가
        하의를 상의로 올려 두면 하의가 상의 자리에 추천된다. 실측표는 그 옷을 실제로 잰
        표라 부위를 훨씬 잘 말해 준다 — 상의로 올라온 바지도 typeName 이 '바지'다.

        어긋날 때만 뺀다. typeName 이 없는 상품이 4~6% 있고 오버올처럼 양쪽으로 파는 옷도
        있어서, 모르는 경우에는 아무 판단도 하지 않는다.
        """
        stats = {"checked": 0, "dropped": 0}
        for category, products in grouped.items():
            kept = []
            for product in products:
                actual = category_from_type_name(product.measurement_type)
                stats["checked"] += bool(actual)
                if actual and actual != category:
                    stats["dropped"] += 1
                    continue
                kept.append(product)
            grouped[category] = kept
        self.last_search_stats["category_check"] = stats

    def _drop_fully_avoided_colors(self, grouped, profile) -> None:
        """파는 색이 **전부** 회피 색인 상품을 뺀다. 추가 요청은 없다(사이즈표 응답에 같이 온다).

        이 검사만 색 옵션으로 할 수 있다. 어느 색 사진이 카드에 뜨든 그 색은 파는 색 중
        하나이므로, 파는 색이 모두 회피 색이면 사진도 회피 색이다.

        반대로 "파는 색 중에 원하는 색이 있으니 추천"은 하지 않는다. 카드 사진과 가상 피팅은
        대표 사진 한 장으로 돌아가는데, 그 사진이 어느 색인지 모른다 — 첫 컬러칩과 대표 사진
        색이 같은 경우가 345개 중 46%뿐이었다(2026-09-25). 다른 색으로 합성해 주면
        색을 맞춰 추천한 의미가 없다.
        """
        avoided = [value for value in getattr(profile, "avoided_colors", []) or [] if value]
        stats = {"colors_known": 0, "avoided_dropped": 0}
        for category, products in grouped.items():
            kept = []
            for product in products:
                palettes = palettes_for(product.color_options)
                stats["colors_known"] += bool(palettes)
                if palettes and avoided and all(palette in avoided for palette in palettes):
                    stats["avoided_dropped"] += 1
                    continue
                kept.append(product)
            grouped[category] = kept
        self.last_search_stats["color_options"] = stats

    def _photo_shortlist(self, grouped, targets):
        shortlist = []
        for category, products in grouped.items():
            attributes = rule_backed_photo_attributes(targets, category)
            eligible = [
                product for product in products
                if (
                    category == "top"
                    or getattr(self.photo_provider, "validates_named_fit", False) is True
                    or missing_photo_axes(category, product.name, attributes)
                )
            ]
            # Top images also need the design-point check. Bottoms are sampled
            # only when fit/length evidence is missing from their titles. The
            # sample is stratified so the eight slots do not collapse to one
            # popular item type, fit or brand.
            shortlist.extend(self._diverse_photo_sample(eligible, self.photo_candidates))
        return shortlist

    @classmethod
    def _photo_sample_dimensions(cls, product):
        text = cls._normalized(product.name)
        item_type = next((term for term in PHOTO_SAMPLE_ITEM_TERMS.get(product.category, ())
                          if cls._normalized(term) in text), "기타")
        if product.category == "bottom":
            fit = canonical_bottom_fit(product.name, product.photo_attributes) or "핏 미상"
        else:
            fit = next((label for label, terms in PHOTO_SAMPLE_TOP_FIT_TERMS
                        if any(cls._normalized(term) in text for term in terms)), "핏 미상")
        brand = cls._normalized(product.brand) or "브랜드 미상"
        return item_type, fit, brand

    @classmethod
    def _diverse_photo_sample(cls, products, limit):
        remaining = list(enumerate(products))
        selected = []
        seen_types, seen_fits, seen_brands = set(), set(), set()
        while remaining and len(selected) < limit:
            best = max(
                remaining,
                key=lambda item: (
                    4 * (cls._photo_sample_dimensions(item[1])[0] not in seen_types)
                    + 3 * (cls._photo_sample_dimensions(item[1])[1] not in seen_fits)
                    + 1 * (cls._photo_sample_dimensions(item[1])[2] not in seen_brands),
                    -item[0],
                ),
            )
            remaining.remove(best)
            product = best[1]
            selected.append(product)
            item_type, fit, brand = cls._photo_sample_dimensions(product)
            seen_types.add(item_type)
            seen_fits.add(fit)
            seen_brands.add(brand)
        return selected

    def _supplement_photos(self, grouped, targets, loader, prefetched=None):
        self.last_search_stats["photo"] = {
            "candidates": 0, "analyzed_products": 0, "coverage": 0.0,
            "ranking_mode": "title_only", "matched_products": 0, "elapsed_seconds": 0.0,
        }
        if self.photo_provider is None or self.photo_budget <= 0:
            return
        shortlist = self._photo_shortlist(grouped, targets)
        if not shortlist:
            return
        started = time.monotonic()
        try:
            records = self.photo_provider.get_many(shortlist, loader, self.photo_budget, prefetched=prefetched)
        except Exception:
            self.last_search_stats["photo"] = {"candidates": len(shortlist), "failed": True,
                "analyzed_products": 0, "coverage": 0.0, "ranking_mode": "title_only",
                "elapsed_seconds": round(time.monotonic() - started, 4), "matched_products": 0}
            return
        for product in shortlist:
            attributes = rule_backed_photo_attributes(targets, product.category)
            raw_evidence = records.get(product.product_id, {})
            evidence = photo_matches(product.category, product.name, attributes, raw_evidence)
            # An existing brand/title keyword keeps its original score and source.
            evidence = {axis: value for axis, value in evidence.items()
                        if not set(attributes[axis]).intersection(product.matched_keywords)}
            item = {"category": product.category, "goodsName": product.name, "brandName": product.brand}
            before, _ = self._score(item, attributes, 0)
            after, _ = self._score(item, attributes, 0, photo_attributes=evidence)
            product.retrieval_score += after - before
            conflicts = {}
            for axis in photo_validation_axes(
                product.category, product.name, attributes, raw_evidence
            ):
                value = raw_evidence.get(axis, {})
                if axis in evidence or not confident_fit_evidence(value):
                    continue
                # A confident incompatible visual label is negative evidence.
                # It is a ranking penalty, not a public assertion or hard filter.
                conflicts[f"{axis}_conflict"] = dict(value)
                product.retrieval_score -= ATTRIBUTE_WEIGHTS[axis]
            design = raw_evidence.get("design", {})
            product.photo_attributes = {
                **evidence,
                **conflicts,
                **({"design": design} if design.get("source") == "product_photo" else {}),
            }
            if product.photo_attributes:
                product.ranking_evidence_source = "title+photo"
        analyzed = len(records)
        coverage = analyzed / len(shortlist) if shortlist else 0.0
        ranking_mode = (
            "photo_assisted" if coverage >= 0.5
            else "limited_photo_assist" if analyzed
            else "title_only"
        )
        dimensions = {self._photo_sample_dimensions(product) for product in shortlist}
        self.last_search_stats["photo"] = {
            **self.photo_provider.last_stats, "candidates": len(shortlist),
            "analyzed_products": analyzed, "coverage": round(coverage, 4),
            "ranking_mode": ranking_mode, "sample_groups": len(dimensions),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "matched_products": sum(bool(p.photo_attributes) for p in shortlist),
            "rule_matches": sum(any(axis in p.photo_attributes for axis in ("fit", "length")) for p in shortlist),
            "rule_conflicts": sum(any(axis.endswith("_conflict") for axis in p.photo_attributes) for p in shortlist),
        }

    @staticmethod
    def _apply_fashion_policy_adjustments(grouped, profile):
        """Apply small, inspectable trend/design weights before final sorting."""
        for products in grouped.values():
            for product in products:
                adjustments = {}
                trend_value, trend = trend_fit_adjustment(product, profile)
                if trend:
                    adjustments["bottom_fit_trend"] = trend
                tee_value, tee = basic_logo_tee_adjustment(product)
                if tee:
                    adjustments["basic_logo_tee"] = tee
                formal_value, formal = formal_context_adjustment(product, profile)
                if formal:
                    adjustments["formal_context"] = formal
                sporty_value, sporty = sporty_context_adjustment(product, profile)
                if sporty:
                    adjustments["sporty_context"] = sporty
                product.retrieval_score = round(
                    product.retrieval_score + trend_value + tee_value + formal_value + sporty_value, 4
                )
                product.ranking_adjustments = adjustments

    @classmethod
    def _product_matches_material(cls, product: ShoppingProduct, material: str) -> bool:
        normalized = cls._normalized(material)
        if any(cls._normalized(value) == normalized
               for value in (getattr(product, "matched_keywords", None) or [])):
            return True
        text = cls._normalized(f"{product.name} {product.brand}")
        return any(cls._normalized(alias) in text for alias in cls._aliases(material))

    @classmethod
    def _button_shirt(cls, product: ShoppingProduct) -> bool:
        text = cls._normalized(product.name)
        casual_non_button = (
            "티셔츠", "t-shirt", "tshirt", "스웨트셔츠", "sweatshirt",
            "폴로셔츠", "폴로 셔츠", "피케셔츠", "피케 셔츠", "럭비셔츠", "럭비 셔츠",
        )
        if any(cls._normalized(term) in text for term in casual_non_button):
            return False
        return any(cls._normalized(term) in text for term in ("셔츠", "남방", "블라우스", "shirt", "blouse"))

    @staticmethod
    def _casual_shirt_diversity_enabled(profile: UserProfile) -> bool:
        return (
            str(profile.desired_style or "").strip() == "캐주얼"
            and str(profile.purpose or "").strip() not in SHIRT_DIVERSITY_EXEMPT_PURPOSES
            and str(profile.desired_style or "").strip() not in SHIRT_DIVERSITY_EXEMPT_STYLES
            and str(profile.dress_code or "").strip() not in SHIRT_DIVERSITY_EXEMPT_DRESS_CODES
        )

    def _select(self, grouped, targets, limit, profile):
        # 최신 화면 계약: 카테고리마다 최대 limit개. 실측 기반 재정렬도 유지한다.
        selected, seen_ids, brand_counts = [], set(), {}
        requested_materials = {
            category: list(dict.fromkeys(attributes.get("material", [])))
            if category != "shoes" else []
            for category, attributes in targets.targets.items()
        }
        used_materials = {category: set() for category in targets.targets}
        shirt_counts = {category: 0 for category in targets.targets}
        for category in targets.targets:
            for _ in range(limit):
                products = [p for p in grouped.get(category, []) if p.product_id not in seen_ids]
                if not products:
                    break
                # 복수 소재를 선택했다면 각 소재 후보를 최소 한 번 먼저 보여 준다.
                # 요청 소재 후보가 부족한 경우에는 기존 검색 순위로 자연스럽게 fallback한다.
                material_pool = products
                for material in requested_materials[category]:
                    if material in used_materials[category]:
                        continue
                    matching = [
                        product for product in products
                        if self._product_matches_material(product, material)
                    ]
                    if matching:
                        material_pool = matching
                        break
                # 일반 캐주얼의 세 코디가 모두 버튼 셔츠로 수렴하지 않게 한다.
                # 셔츠 한 벌은 허용하되 그 뒤에는 현재 소재 풀, 필요하면 전체
                # 후보에서 티셔츠·니트·맨투맨·재킷 등의 다른 상의가 있으면 우선한다.
                if (
                    category == "top"
                    and self._casual_shirt_diversity_enabled(profile)
                    and shirt_counts[category] >= CASUAL_SHIRT_LIMIT
                ):
                    alternatives = [product for product in material_pool if not self._button_shirt(product)]
                    if not alternatives:
                        alternatives = [product for product in products if not self._button_shirt(product)]
                    if alternatives:
                        material_pool = alternatives
                best = material_pool[0]
                tied = [
                    product for product in material_pool
                    if self._sort_key(product)[0] == self._sort_key(best)[0]
                ]
                best = next((p for p in tied if not p.brand or brand_counts.get(p.brand, 0) < 2), best)
                selected.append(best)
                seen_ids.add(best.product_id)
                shirt_counts[category] += int(category == "top" and self._button_shirt(best))
                brand_counts[best.brand] = brand_counts.get(best.brand, 0) + 1
                used_materials[category].update(
                    material for material in requested_materials[category]
                    if self._product_matches_material(best, material)
                )
        return selected
