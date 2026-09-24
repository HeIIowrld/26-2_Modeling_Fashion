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
from product_measurements import ProductMeasurementClient
from recommendation_keywords import TargetKeywordResult
from schemas import Product, UserProfile
from shopping_http import bounded_results, fetch_json
from size_fit import compare_sizes
from live_product_attributes import missing_photo_axes, photo_matches


API_URL = "https://api.musinsa.com/api2/dp/v2/plp/goods"
CATEGORY_CODES = {"top": "001", "bottom": "003", "shoes": "103"}
CATEGORY_FALLBACK_QUERY = {"top": "베이직 상의", "bottom": "팬츠", "shoes": "신발"}
SORT_CODES = ("POPULAR", "NEW")
PAGE_SIZE = 100
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
    # 무신사가 파는 색 이름 전부(대표 색 하나가 아니다). 사이즈표 응답에서 같이 온다.
    color_options: list[str] = field(default_factory=list)
    color_match: str = ""
    retrieval_score: float = 0.0
    recommendation_reason: str = ""
    recommendation_reason_source: str = "rules"
    size_fit: dict = field(default_factory=dict)
    fit_evidence: list[str] = field(default_factory=list)
    fit_evidence_labels: list[str] = field(default_factory=list)
    reason_rule_ids: list[str] = field(default_factory=list)

    def public_dict(self) -> dict:
        """내부 키워드와 점수는 웹 UI에 보내지 않는다."""
        data = asdict(self)
        data.pop("matched_keywords", None)
        data.pop("retrieval_score", None)
        data.pop("photo_attributes", None)
        data["size_fit"].pop("ranking_bonus", None)
        return data


class MusinsaLiveSearch:
    """무신사 검색을 짧게 캐시하고 실패를 FITTA 분석과 격리한다."""

    def __init__(
        self, timeout: float = 3.0, cache_ttl: float = 300.0, *,
        search_budget: float = 8.0, measurement_budget: float = 5.0,
        measurements: ProductMeasurementClient | None = None,
        candidates_per_category: int = 300, measurement_candidates: int = 8,
        photo_provider=None, photo_candidates: int = 8, photo_budget: float = 0.25,
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

    def _queries(
        self,
        category: str,
        attributes: dict[str, list[str]],
    ) -> list[str]:
        if category == "shoes":
            item_type = self._preferred_term(attributes, "item_type") or "스니커즈"
            color = self._preferred_term(attributes, "color")
            material = self._preferred_term(attributes, "material")
            candidates = list(dict.fromkeys(" ".join(filter(None, terms)) for terms in (
                (color, item_type), (material, item_type), (item_type,),
            )))
            return list(dict.fromkeys(candidates))
        fit = self._preferred_term(attributes, "fit")
        material = self._preferred_term(attributes, "material")
        style = self._preferred_term(attributes, "style")
        color = self._preferred_term(attributes, "color")
        primary = " ".join(value for value in (fit, material) if value)
        noun = CATEGORY_FALLBACK_QUERY[category]
        fits = attributes.get("fit", [])
        alternative_fit = fits[1] if len(fits) > 1 else (self._aliases(fits[0])[-1] if fits else "")
        candidates = [primary, f"{material} {noun}" if material else noun,
                      f"{alternative_fit or fit} {noun}" if fit else "",
                      " ".join(value for value in (color or style, material or noun) if value), noun]
        return list(dict.fromkeys(value.strip() for value in candidates if value.strip()))[:4]

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
                if any(self._normalized(alias) in text for alias in self._aliases(keyword)):
                    score += weight
                    matched.append(keyword)
                    matched_axes.add(attribute)
                    break
        if photo_attributes:
            matches = photo_matches(item.get("category", ""), str(item.get("goodsName") or ""),
                                    attributes, photo_attributes)
            score += sum(ATTRIBUTE_WEIGHTS[axis] for axis in matches if axis not in matched_axes)
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
            # 같은 응답에 색 옵션이 들어 있다. 추가 요청 없이 대표 색 밖의 색을 알게 된다.
            product.color_options = list((record or {}).get("color_options") or [])
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
        self._supplement_colors(grouped, targets, profile)
        selected = self._select(grouped, targets, limit)
        self._supplement_photos(grouped, targets, photo_loader, prefetched)
        self.last_search_stats["total_elapsed_seconds"] = round(time.monotonic() - started, 4)
        self.last_search_stats["photo_protected_positions"] = sum(bool(p.matched_keywords) for p in selected)
        if not any(p.photo_attributes and not p.matched_keywords for products in grouped.values() for p in products):
            return selected
        # Protect the actual pre-photo display positions, including brand diversity
        # and measurement sorting. Photos only compete for title-unmatched slots.
        protected = {p.product_id for p in selected if p.matched_keywords}
        replacements = {category: sorted((p for p in products if not p.matched_keywords), key=self._sort_key)
                        for category, products in grouped.items()}
        used, brand_counts = set(protected), {}
        for product in selected:
            if product.product_id in protected:
                brand_counts[product.brand] = brand_counts.get(product.brand, 0) + 1
        for index, product in enumerate(selected):
            if product.product_id in protected:
                continue
            candidates = [p for p in replacements.get(product.category, []) if p.product_id not in used]
            if not candidates:
                continue
            best = candidates[0]
            tied = [p for p in candidates if self._sort_key(p)[0] == self._sort_key(best)[0]]
            best = next((p for p in tied if not p.brand or brand_counts.get(p.brand, 0) < 2), best)
            selected[index] = best
            used.add(best.product_id)
            brand_counts[best.brand] = brand_counts.get(best.brand, 0) + 1
        self.last_search_stats["total_elapsed_seconds"] = round(time.monotonic() - started, 4)
        self.last_search_stats["photo_protected_positions"] = len(protected)
        return selected

    def _supplement_colors(self, grouped, targets, profile) -> None:
        """상품명에 색이 없어도 무신사가 그 색을 파는지로 판단한다. 추가 요청은 없다.

        상품명 텍스트만 보던 때에는 '그레이'로 저장된 상품이 실제로는 화이트·블랙·베이지로도
        팔리는데 색 조건에서 탈락했고, 싫어하는 색을 빼 달라는 요청도 이름에 색이 없으면
        그냥 통과했다. 대표 사진은 한 색뿐이므로 어떤 색으로 맞췄는지 `color_match` 에 남긴다.
        """
        avoided = [value for value in getattr(profile, "avoided_colors", []) or [] if value]
        stats = {"colors_known": 0, "color_matched": 0, "avoided_dropped": 0}
        for category, products in grouped.items():
            wanted = (targets.targets.get(category) or {}).get("color", [])
            kept = []
            for product in products:
                palettes = palettes_for(product.color_options)
                stats["colors_known"] += bool(palettes)
                # 파는 색이 전부 회피 색일 때만 뺀다. 한 색이라도 남으면 고를 수 있다.
                if palettes and avoided and all(palette in avoided for palette in palettes):
                    stats["avoided_dropped"] += 1
                    continue
                kept.append(product)
                if not wanted or not palettes or set(wanted) & set(product.matched_keywords):
                    continue
                matched = next((keyword for keyword in wanted if keyword in palettes), "")
                if matched:
                    product.retrieval_score += ATTRIBUTE_WEIGHTS["color"]
                    product.color_match = matched
                    stats["color_matched"] += 1
            grouped[category] = kept
        for products in grouped.values():
            products.sort(key=self._sort_key)
        self.last_search_stats["color_options"] = stats

    def _photo_shortlist(self, grouped, targets):
        return [products[index] for index in range(self.photo_candidates)
                for category, products in grouped.items() if index < len(products)
                and missing_photo_axes(category, products[index].name, targets.targets[category])]

    def _supplement_photos(self, grouped, targets, loader, prefetched=None):
        self.last_search_stats["photo"] = {"candidates": 0, "matched_products": 0, "elapsed_seconds": 0.0}
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
                "elapsed_seconds": round(time.monotonic() - started, 4), "matched_products": 0}
            return
        for product in shortlist:
            attributes = targets.targets[product.category]
            evidence = photo_matches(product.category, product.name, attributes, records.get(product.product_id, {}))
            # An existing brand/title keyword keeps its original score and source.
            evidence = {axis: value for axis, value in evidence.items()
                        if not set(attributes[axis]).intersection(product.matched_keywords)}
            item = {"category": product.category, "goodsName": product.name, "brandName": product.brand}
            before, _ = self._score(item, attributes, 0)
            after, _ = self._score(item, attributes, 0, photo_attributes=evidence)
            product.retrieval_score += after - before
            product.photo_attributes = evidence
        self.last_search_stats["photo"] = {
            **self.photo_provider.last_stats, "candidates": len(shortlist),
            "elapsed_seconds": round(time.monotonic() - started, 4),
            "matched_products": sum(bool(p.photo_attributes) for p in shortlist),
        }

    def _select(self, grouped, targets, limit):
        # 최신 화면 계약: 카테고리마다 최대 limit개. 실측 기반 재정렬도 유지한다.
        selected, seen_ids, brand_counts = [], set(), {}
        for category in targets.targets:
            for _ in range(limit):
                products = [p for p in grouped.get(category, []) if p.product_id not in seen_ids]
                if not products:
                    break
                best = products[0]
                tied = [p for p in products if self._sort_key(p)[0] == self._sort_key(best)[0]]
                best = next((p for p in tied if not p.brand or brand_counts.get(p.brand, 0) < 2), best)
                selected.append(best)
                seen_ids.add(best.product_id)
                brand_counts[best.brand] = brand_counts.get(best.brand, 0) + 1
        return selected
