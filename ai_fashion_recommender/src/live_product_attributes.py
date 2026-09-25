"""Conservative product-photo evidence for Fashion-Rule-aware ranking.

The existing trained heads are reused.  Product titles remain useful evidence, but
fit and length rules no longer depend on a keyword being present in the title: a
high-confidence worn-product prediction can take part in candidate ranking.
"""
from __future__ import annotations

import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, wait

POLICY_VERSION = "worn-fit-length-v1"
SHARED_FIT_POLICY_VERSION = "shared-fit-v1"
SUPPORTED_FIT_POLICIES = {POLICY_VERSION, SHARED_FIT_POLICY_VERSION}
DESIGN_POLICY_VERSION = "product-design-v1"
MIN_CONFIDENCE = 0.90
SHARED_FIT_MIN_CONFIDENCE = 0.70
DESIGN_MIN_CONFIDENCE = 0.70
TASKS = {
    "top": {"fit": ("upper_fit",), "length": ("upper_length",)},
    # lower_fit and pant_leg_shape were trained separately.  Requiring their
    # normalized labels to agree when both are confident prevents one head from
    # silently overruling the other.
    "bottom": {
        "fit": ("lower_fit", "pant_leg_shape"),
        "length": ("lower_length",),
    },
}
DESIGN_TASKS = {"top": ("category", "pattern", "detail")}
MODEL_LABELS = {
    "스키니": "슬림핏",
    "스트레이트": "스트레이트핏",
    "테이퍼드": "테이퍼드핏",
    "페그": "테이퍼드핏",
    "부츠컷": "플레어핏",
    "플레어": "플레어핏",
    "와이드": "와이드핏",
    "팔라초": "와이드핏",
}
# Keep query meanings narrow: wide is not semi-wide; a regular fit is not slim.
QUERY_LABELS = {
    "fit": {"슬림": ("슬림핏",), "슬림핏": ("슬림핏",),
            "레귤러": ("레귤러핏",), "정돈된 핏": ("레귤러핏", "슬림핏"),
            "여유핏": ("여유핏", "오버핏"), "오버핏": ("오버핏",),
            "스트레이트": ("스트레이트", "스트레이트핏"),
            "세미와이드": ("세미와이드",),
            "와이드": ("와이드", "와이드핏"),
            "벌룬": ("벌룬", "벌룬핏"),
            "테이퍼드": ("테이퍼드핏",),
            "플레어": ("플레어·부츠컷", "플레어핏"),
            "부츠컷": ("플레어·부츠컷", "플레어핏")},
    "length": {"허리선": ("크롭 기장",), "크롭": ("크롭 기장",),
               "기본 기장": ("기본 기장",), "롱": ("롱 기장", "롱·긴바지 기장"),
               "풀렝스": ("롱·긴바지 기장",), "긴바지": ("롱·긴바지 기장",),
               "쇼츠": ("쇼츠·미니 기장",), "반바지": ("쇼츠·미니 기장",),
               "미니": ("쇼츠·미니 기장",), "무릎": ("무릎 기장",),
               "미디": ("미디·7부 기장",), "7부": ("미디·7부 기장",),
               "앵클": ("크롭·앵클 기장",), "9부": ("크롭·앵클 기장",)},
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


def photo_validation_axes(category: str, name: str, attributes: dict, evidence: dict) -> list[str]:
    """공용 CV 결과는 상품명에 핏이 적혀 있어도 오표기를 재검증한다."""
    missing = set(missing_photo_axes(category, name, attributes))
    return [
        axis for axis in TASKS.get(category, {})
        if (
            axis in missing
            or evidence.get(axis, {}).get("policy") == SHARED_FIT_POLICY_VERSION
        )
        and any(keyword in QUERY_LABELS[axis] for keyword in attributes.get(axis, []))
    ]


def rule_backed_photo_attributes(targets, category: str) -> dict[str, list[str]]:
    """Return only photo-verifiable attributes backed by active Fashion Rules.

    Older callers and evaluation fixtures do not carry provenance.  They retain
    the previous behavior.  Once provenance exists, a visual ranking bonus is
    allowed only for a keyword whose rule is actually active for this request.
    """
    attributes = dict((getattr(targets, "targets", None) or {}).get(category, {}) or {})
    keyword_rules = dict((getattr(targets, "keyword_rules", None) or {}).get(category, {}) or {})
    applied = set(getattr(targets, "applied_rules", None) or [])
    if not keyword_rules and not applied:
        return attributes
    filtered = dict(attributes)
    for axis in TASKS.get(category, {}):
        filtered[axis] = [
            keyword for keyword in attributes.get(axis, [])
            if applied.intersection(keyword_rules.get(keyword, []))
        ]
    return filtered


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
    for axis, tasks in TASKS.get(category, {}).items():
        candidates = []
        for task in tasks:
            prediction = predictions.get(task, {})
            if hasattr(prediction, "to_dict"):
                prediction = prediction.to_dict()
            labels = prediction.get("labels", [])
            confidence = float(prediction.get("confidence", 0))
            if prediction.get("accepted") and len(labels) == 1 and MIN_CONFIDENCE <= confidence <= 1:
                raw_label = labels[0]
                candidates.append((MODEL_LABELS.get(raw_label, raw_label), confidence, task, raw_label))
        # Conflicting confident heads are evidence to abstain, not to pick the
        # slightly larger softmax value.
        if not candidates or len({candidate[0] for candidate in candidates}) != 1:
            continue
        label, confidence, _task, _raw_label = max(candidates, key=lambda item: item[1])
        accepted[axis] = {
            "label": label,
            "confidence": confidence,
            "source": "product_photo",
            "policy": POLICY_VERSION,
        }
    return accepted


def accepted_shared_fit_attributes(category: str, context: dict, predictions: dict) -> dict:
    """공용 predictor의 세부 라벨을 상품 사진 근거로 보존한다."""
    if not context.get("worn"):
        return {}
    from fit_vision_schema import BOTTOM_TO_LEGACY_LENGTH, UPPER_TO_LEGACY_LENGTH

    task_by_axis = {
        "top": {"fit": "upper_fit", "length": "upper_length"},
        "bottom": {"fit": "bottom_silhouette", "length": "bottom_length"},
    }
    length_maps = {
        "top": UPPER_TO_LEGACY_LENGTH,
        "bottom": BOTTOM_TO_LEGACY_LENGTH,
    }
    accepted = {}
    for axis, task in task_by_axis.get(category, {}).items():
        value = predictions.get(task, {})
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        labels = list(value.get("labels") or [])
        confidence = float(value.get("confidence") or 0)
        if not value.get("accepted") or len(labels) != 1 or confidence < SHARED_FIT_MIN_CONFIDENCE:
            continue
        raw_label = labels[0]
        label = length_maps[category].get(raw_label, raw_label) if axis == "length" else raw_label
        accepted[axis] = {
            "label": label,
            "raw_label": raw_label,
            "confidence": confidence,
            "source": "product_photo",
            "policy": SHARED_FIT_POLICY_VERSION,
        }
    return accepted


def confident_fit_evidence(value: dict) -> bool:
    threshold = (
        SHARED_FIT_MIN_CONFIDENCE
        if value.get("policy") == SHARED_FIT_POLICY_VERSION
        else MIN_CONFIDENCE
    )
    return (
        value.get("source") == "product_photo"
        and value.get("policy") in SUPPORTED_FIT_POLICIES
        and threshold <= float(value.get("confidence", 0)) <= 1
    )


def photo_matches(category: str, name: str, attributes: dict, evidence: dict) -> dict:
    matches = {}
    for axis in photo_validation_axes(category, name, attributes, evidence):
        value = evidence.get(axis, {})
        if not confident_fit_evidence(value):
            continue
        for keyword in attributes[axis]:
            if value.get("label") in QUERY_LABELS[axis].get(keyword, ()):
                matches[axis] = dict(value, keyword=keyword)
                break
    return matches


def flat_product_design_metrics(image) -> dict:
    """Estimate the visible design area of a clean product cut-out.

    This is supporting evidence beside the trained pattern/detail heads. Model
    photos with a non-uniform border abstain automatically.
    """
    import numpy as np
    from pathlib import Path
    from PIL import Image

    pil = Image.open(image).convert("RGB") if isinstance(image, (str, Path)) else image.convert("RGB")
    array = np.asarray(pil, dtype=np.float32)
    height, width = array.shape[:2]
    if min(height, width) < 64:
        return {"flat_product": False, "point_area_ratio": 0.0, "color_point": False,
                "wordmark_like": False}
    edge = max(2, min(height, width) // 30)
    border = np.concatenate((
        array[:edge].reshape(-1, 3), array[-edge:].reshape(-1, 3),
        array[:, :edge].reshape(-1, 3), array[:, -edge:].reshape(-1, 3),
    ))
    background = np.median(border, axis=0)
    border_distance = np.linalg.norm(border - background, axis=1)
    if float(np.percentile(border_distance, 90)) > 18:
        return {"flat_product": False, "point_area_ratio": 0.0, "color_point": False,
                "wordmark_like": False}
    foreground = np.linalg.norm(array - background, axis=2) > 30
    fraction = float(foreground.mean())
    if not 0.08 <= fraction <= 0.80:
        return {"flat_product": False, "point_area_ratio": 0.0, "color_point": False,
                "wordmark_like": False}
    ys, xs = np.where(foreground)
    y0, y1, x0, x1 = int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())
    garment_height, garment_width = max(1, y1 - y0 + 1), max(1, x1 - x0 + 1)
    # Centre torso excludes silhouette edges, sleeves and most neck holes.
    iy0 = y0 + int(garment_height * 0.20)
    iy1 = y0 + int(garment_height * 0.78)
    ix0 = x0 + int(garment_width * 0.24)
    ix1 = x0 + int(garment_width * 0.76)
    inner_mask = foreground[iy0:iy1, ix0:ix1]
    inner = array[iy0:iy1, ix0:ix1]
    if inner_mask.sum() < 100:
        return {"flat_product": False, "point_area_ratio": 0.0, "color_point": False,
                "wordmark_like": False}
    base = np.median(inner[inner_mask], axis=0)
    contrast = np.linalg.norm(inner - base, axis=2) > 48
    # A white print can equal the white page background and therefore disappear
    # from ``foreground``. The centre rectangle is inside the torso, so count
    # contrast across that full area while using foreground only to estimate the
    # garment's base colour.
    ratio = float(contrast.mean())
    color_point = float(base.max() - base.min()) >= 35
    wordmark_like = False
    try:
        import cv2
        count, _, stats, centers = cv2.connectedComponentsWithStats(
            contrast.astype("uint8"), 8
        )
        components = []
        minimum_area = max(8, int(contrast.size * 0.00008))
        for index in range(1, count):
            x, y, component_width, component_height, area = stats[index]
            if (
                area >= minimum_area
                and component_height >= contrast.shape[0] * 0.015
                and component_height <= contrast.shape[0] * 0.35
                and component_width <= contrast.shape[1] * 0.50
            ):
                components.append((x, component_width, component_height, centers[index][1]))
        if len(components) >= 3:
            median_height = float(np.median([value[2] for value in components]))
            median_center = float(np.median([value[3] for value in components]))
            aligned = [
                value for value in components
                if 0.45 <= value[2] / max(1, median_height) <= 2.2
                and abs(value[3] - median_center) <= 0.65 * median_height
            ]
            if len(aligned) >= 3:
                coverage = (max(x + width for x, width, _, _ in aligned)
                            - min(x for x, _, _, _ in aligned)) / contrast.shape[1]
                wordmark_like = coverage >= 0.10
    except (ImportError, ValueError, TypeError):
        pass
    return {
        "flat_product": True,
        "point_area_ratio": round(ratio, 4),
        "color_point": bool(color_point),
        "wordmark_like": bool(wordmark_like),
    }


def accepted_design_attributes(category: str, predictions: dict, metrics: dict | None = None) -> dict:
    """Conservatively measure whether a tee has non-logo visual design evidence."""
    if category != "top":
        return {}

    def accepted(task: str) -> tuple[list[str], float]:
        value = predictions.get(task, {})
        if hasattr(value, "to_dict"):
            value = value.to_dict()
        labels = list(value.get("labels") or [])
        confidence = float(value.get("confidence") or 0)
        if not value.get("accepted") or not labels or confidence < DESIGN_MIN_CONFIDENCE:
            return [], confidence
        return labels, confidence

    categories, category_confidence = accepted("category")
    patterns, pattern_confidence = accepted("pattern")
    details, detail_confidence = accepted("detail")
    if not patterns:
        return {}
    metrics = metrics or {"flat_product": False, "point_area_ratio": 0.0,
                          "color_point": False, "wordmark_like": False}
    item_type = categories[0] if len(categories) == 1 else ""
    strong_pattern = any(label != "무지" for label in patterns)
    strong_detail = any(label not in {"디테일 없음", "자수"} for label in details)
    small_point = metrics.get("flat_product") and float(metrics.get("point_area_ratio", 0)) <= 0.035
    plain_basic = not strong_detail and (
        bool(small_point or metrics.get("wordmark_like"))
        if metrics.get("flat_product") else not strong_pattern
    ) and not metrics.get("color_point")
    return {
        "design": {
            "item_type": item_type,
            "patterns": patterns,
            "details": details,
            "plain_basic": bool(plain_basic),
            "flat_product": bool(metrics.get("flat_product")),
            "point_area_ratio": float(metrics.get("point_area_ratio", 0)),
            "color_point": bool(metrics.get("color_point")),
            "wordmark_like": bool(metrics.get("wordmark_like")),
            "confidence": round(max(category_confidence, pattern_confidence, detail_confidence), 4),
            "source": "product_photo",
            "policy": DESIGN_POLICY_VERSION,
        }
    }


class LiveProductAttributes:
    """Reuse loaded models on the caller's inference thread; only downloads run in workers.

    The caller holds the pipeline's model lock. The deadline stops new GPU work;
    an in-flight forward pass finishes before returning (no unsafe background inference).
    Cache lifetime is this predictor instance, so replacing a checkpoint invalidates it.
    """
    def __init__(
        self,
        parser,
        predictor,
        cache_size: int = 512,
        cache_ttl: float = 3600,
        fit_predictor=None,
    ):
        self.parser, self.predictor, self.fit_predictor = parser, predictor, fit_predictor
        self.cache_size, self.cache_ttl = cache_size, cache_ttl
        self._cache = OrderedDict()
        self._downloads = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fitta-photo")
        self.last_stats = {}

    @property
    def validates_named_fit(self) -> bool:
        return self.fit_predictor is not None

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
        if category not in TASKS or (self.predictor is None and self.fit_predictor is None):
            return {"context": {}, "predictions": {}, "attributes": {}}
        context = {"worn": False, "human_fraction": 0.0, "garment_fraction": 0.0}
        garment_mask = None
        if self.parser is not None and self.parser.backend == "fashn-human-parser":
            parsed = self.parser.parse(image, PoseAnalysis(False, 0, "", 0, 0, 0, "", 0))
            context = worn_evidence(parsed["segmentation"], category)
            garment_mask = parsed["upper_mask"] if category == "top" else parsed["lower_mask"]
        tasks = list(DESIGN_TASKS.get(category, ()))
        if context["worn"]:
            tasks.extend(task for axis_tasks in TASKS[category].values() for task in axis_tasks)
        tasks = list(dict.fromkeys(tasks))
        predictions = self.predictor.predict(image, tasks=tasks) if tasks and self.predictor else {}
        serialized = {
            key: value.to_dict() if hasattr(value, "to_dict") else dict(value)
            for key, value in predictions.items()
        }
        attributes = accepted_attributes(category, context, serialized)
        fit_predictions = (
            self.fit_predictor.predict(image, category=category, mask=garment_mask)
            if self.fit_predictor is not None and context["worn"] else {}
        )
        serialized_fit = {
            key: value.to_dict() if hasattr(value, "to_dict") else dict(value)
            for key, value in fit_predictions.items()
        }
        # 새 공용 모델의 세부 핏 결과가 있으면 기존 coarse head보다 우선한다.
        attributes.update(accepted_shared_fit_attributes(category, context, serialized_fit))
        metrics = flat_product_design_metrics(image) if category == "top" else {}
        context["design_metrics"] = metrics
        attributes.update(accepted_design_attributes(category, serialized, metrics))
        return {
            "context": context,
            "predictions": serialized,
            "fit_predictions": serialized_fit,
            "attributes": attributes,
        }

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
