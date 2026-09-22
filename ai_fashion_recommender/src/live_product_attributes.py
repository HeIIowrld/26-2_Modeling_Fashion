"""Conservative product-photo evidence; no model training or title replacement."""
from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait

POLICY_VERSION = "worn-fit-length-v1"
MIN_CONFIDENCE = 0.90
TASKS = {"top": {"fit": "upper_fit", "length": "upper_length"},
         "bottom": {"fit": "lower_fit", "length": "lower_length"}}
# Keep query meanings narrow: wide is not semi-wide; a regular fit is not slim.
QUERY_LABELS = {
    "fit": {"슬림": ("슬림핏",), "슬림핏": ("슬림핏",),
            "레귤러": ("레귤러핏",), "정돈된 핏": ("레귤러핏", "슬림핏"),
            "여유핏": ("여유핏", "오버핏"), "오버핏": ("오버핏",),
            "스트레이트": ("스트레이트핏",), "와이드": ("와이드핏",),
            "테이퍼드": ("테이퍼드핏",), "플레어": ("플레어핏",)},
    "length": {"허리선": ("크롭 기장",), "크롭": ("크롭 기장",),
               "기본 기장": ("기본 기장",), "롱": ("롱 기장", "롱·긴바지 기장"),
               "풀렝스": ("롱·긴바지 기장",), "긴바지": ("롱·긴바지 기장",),
               "쇼츠": ("쇼츠·미니 기장",), "반바지": ("쇼츠·미니 기장",),
               "미니": ("쇼츠·미니 기장",), "무릎": ("무릎 기장",),
               "미디": ("미디·7부 기장",), "7부": ("미디·7부 기장",)},
}
# Check ALL labels on the axis, including a title that contradicts the query.
TITLE_TERMS = {
    "fit": ("슬림", "스키니", "레귤러", "스탠다드", "오버", "루즈", "여유", "와이드",
            "스트레이트", "일자", "테이퍼", "플레어", "부츠컷", "배기", "벌룬",
            "slim", "skinny", "regular", "standard", "oversize", "loose", "wide",
            "straight", "taper", "flare", "bootcut", "relaxed"),
    "length": ("크롭", "숏", "쇼트", "쇼츠", "반바지", "미니", "무릎", "미디",
               "7부", "8부", "9부", "긴바지", "풀렝스", "롱", "기본기장", "기본길이",
               "crop", "short", "mini", "midi", "long", "full length", "ankle"),
}


def title_has_axis(name: str, axis: str) -> bool:
    text = "".join(name.lower().split())
    return any("".join(term.split()) in text for term in TITLE_TERMS.get(axis, ()))


def missing_photo_axes(category: str, name: str, attributes: dict) -> list[str]:
    return [axis for axis in TASKS.get(category, {}) if not title_has_axis(name, axis)
            and any(k in QUERY_LABELS[axis] for k in attributes.get(axis, []))]


def worn_evidence(segmentation, category: str) -> dict:
    """Use only an actual FASHN segmentation, never the benchmark's cut labels."""
    import numpy as np
    mask = np.asarray(segmentation)
    human_fraction = float(np.isin(mask, [1, 2, 12, 13, 14, 16]).mean())
    garment_fraction = float(np.isin(mask, [3, 4] if category == "top" else [4, 5, 6]).mean())
    return {"human_fraction": human_fraction, "garment_fraction": garment_fraction,
            "worn": human_fraction >= 0.01 and garment_fraction >= 0.05}


def accepted_attributes(category: str, context: dict, predictions: dict) -> dict:
    if not context.get("worn"):
        return {}
    accepted = {}
    for axis, task in TASKS.get(category, {}).items():
        prediction = predictions.get(task, {})
        if hasattr(prediction, "to_dict"):
            prediction = prediction.to_dict()
        labels = prediction.get("labels", [])
        confidence = float(prediction.get("confidence", 0))
        if prediction.get("accepted") and len(labels) == 1 and MIN_CONFIDENCE <= confidence <= 1:
            accepted[axis] = {"label": labels[0], "confidence": confidence,
                              "source": "product_photo", "policy": POLICY_VERSION}
    return accepted


def photo_matches(category: str, name: str, attributes: dict, evidence: dict) -> dict:
    matches = {}
    for axis in missing_photo_axes(category, name, attributes):
        value = evidence.get(axis, {})
        if value.get("source") != "product_photo" or value.get("policy") != POLICY_VERSION:
            continue
        if not MIN_CONFIDENCE <= float(value.get("confidence", 0)) <= 1:
            continue
        for keyword in attributes[axis]:
            if value.get("label") in QUERY_LABELS[axis].get(keyword, ()):
                matches[axis] = dict(value, keyword=keyword)
                break
    return matches


class LiveProductAttributes:
    """Reuse loaded models on the caller's inference thread; only downloads run in workers.

    The caller holds the pipeline's model lock. The deadline stops new GPU work;
    an in-flight forward pass finishes before returning (no unsafe background inference).
    Cache lifetime is this predictor instance, so replacing a checkpoint invalidates it.
    """
    def __init__(self, parser, predictor, cache_size: int = 512, cache_ttl: float = 3600):
        self.parser, self.predictor = parser, predictor
        self.cache_size, self.cache_ttl = cache_size, cache_ttl
        self._cache = OrderedDict()
        self._downloads = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fitta-photo")
        self.last_stats = {}

    def close(self):
        self._downloads.shutdown(wait=False, cancel_futures=True)

    @staticmethod
    def _key(product):
        return product.product_id, product.image_url, product.category

    def prefetch(self, products, loader):
        """Overlap bounded image downloads with the existing size-table lookup."""
        if loader is None:
            return {}
        pending = {}
        for product in products:
            key = self._key(product)
            cached = self._cache.get(key)
            if not cached or time.monotonic() - cached[0] >= self.cache_ttl:
                pending[key] = self._downloads.submit(loader, product, timeout=.5)
        return pending

    def predict_image(self, image, category: str) -> dict:
        from schemas import PoseAnalysis
        if category not in TASKS or self.predictor is None or self.parser.backend != "fashn-human-parser":
            return {"context": {}, "predictions": {}, "attributes": {}}
        parsed = self.parser.parse(image, PoseAnalysis(False, 0, "", 0, 0, 0, "", 0))
        context = worn_evidence(parsed["segmentation"], category)
        predictions = (self.predictor.predict(image, tasks=list(TASKS[category].values()))
                       if context["worn"] else {})
        serialized = {key: value.to_dict() for key, value in predictions.items()}
        return {"context": context, "predictions": serialized,
                "attributes": accepted_attributes(category, context, serialized)}

    def get_many(self, products, loader, budget: float, prefetched=None) -> dict:
        started = time.monotonic()
        deadline = started + max(0, budget)
        result, pending = {}, []
        stats = {"candidates": len(products), "cache_hits": 0, "inferred": 0, "failures": 0}
        for product in products:
            key = self._key(product)
            cached = self._cache.get(key)
            if cached and started - cached[0] < self.cache_ttl:
                self._cache.move_to_end(key)
                result[product.product_id] = cached[1]
                stats["cache_hits"] += 1
            else:
                pending.append((product, key))
        if pending and loader and time.monotonic() < deadline:
            # Reserve part of the budget for inference. Slow downloads can finish
            # caching a file, but can never mutate returned products or run models.
            download_budget = min(0.15, max(0, deadline - time.monotonic()) * 0.6)
            prefetched = prefetched or {}
            futures = [prefetched[key] if key in prefetched else
                       self._downloads.submit(loader, product, timeout=download_budget)
                       for product, key in pending]
            done, unfinished = wait(futures, timeout=download_budget)
            for future in unfinished:
                future.cancel()
            paths = []
            for future in futures:
                try:
                    paths.append(future.result() if future in done else None)
                except Exception:
                    paths.append(None)
            for (product, key), path in zip(pending, paths):
                if time.monotonic() >= deadline:
                    break
                if path is None:
                    stats["failures"] += 1
                    continue
                try:
                    evidence = self.predict_image(path, product.category)["attributes"]
                except Exception:
                    stats["failures"] += 1
                    continue
                stats["inferred"] += 1
                self._cache[key] = (time.monotonic(), evidence)
                self._cache.move_to_end(key)
                while len(self._cache) > self.cache_size:
                    self._cache.popitem(last=False)
                result[product.product_id] = evidence
        stats.update(elapsed_seconds=round(time.monotonic() - started, 4),
                     deferred=len(products) - len(result))
        self.last_stats = stats
        return result
