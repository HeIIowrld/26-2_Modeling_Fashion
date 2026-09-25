"""Small, reversible Fashion-Rule ranking adjustments for live products.

This module deliberately keeps temporary taste/trend values separate from the
stable recommendation rules.  A future season can favour straight trousers by
changing only ``BOTTOM_FIT_TREND_WEIGHTS`` rather than rewriting the search
pipeline.
"""
from __future__ import annotations

import re
from typing import Any


TREND_RULE_ID = "R-TREND-01"
DETAIL_RULE_ID = "R-DET-01"
FORMAL_CONTEXT_RULE_ID = "R-CTX-01"
SPORTY_CONTEXT_RULE_ID = "R-CTX-01"

# Temporary retrieval adjustments, not hard filters. A title fit match is worth
# 4 points, so the current policy must also offset the legacy casual default
# (straight/semi-wide) instead of merely adding a tiny wide-fit tie-breaker.
BOTTOM_FIT_TREND_WEIGHTS = {
    "와이드핏": 3.00,
    # Semi-wide receives exactly 70% of the straight-fit deduction.
    "세미와이드": -1.40,
    "스트레이트핏": -2.00,
    "테이퍼드핏": -1.50,
    "슬림핏": -2.00,
    "플레어핏": 0.00,
}
TREND_EXCLUDED_PURPOSES = {"출근", "면접"}
TREND_EXCLUDED_STYLES = {"포멀", "클래식"}
TREND_EXCLUDED_DRESS_CODES = {"포멀", "비즈니스 포멀", "클래식"}

FORMAL_PURPOSES = {"출근", "면접"}
FORMAL_STYLES = {"포멀", "클래식"}
FORMAL_DRESS_CODES = {"비즈니스 캐주얼", "포멀", "비즈니스 포멀", "클래식"}
STRICT_FORMAL_PURPOSES = {"면접"}
STRICT_FORMAL_DRESS_CODES = {"포멀", "비즈니스 포멀"}

# These are context seeds for the existing search slots, not extra searches or
# hard filters.  They prevent a photo-derived material (for example leather)
# from making work/classic and daily/casual collect the same item pool.
FORMAL_CONTEXT_ITEM_TYPES = {
    "top": ("셔츠", "블레이저", "니트"),
    "bottom": ("슬랙스", "치노 팬츠"),
    "shoes": ("더비슈즈", "로퍼"),
}

SPORTY_STYLES = {"스포티"}
SPORTY_PURPOSES = {"운동", "스포츠", "러닝"}
SPORTY_CONTEXT_ITEM_TYPES = {
    "top": ("트랙 재킷", "저지", "아노락"),
    "bottom": ("트랙팬츠", "조거팬츠", "카고팬츠"),
    "shoes": ("러닝화", "스니커즈"),
}

# Brand exemptions remove a category conflict (for example denim) but do not
# by themselves make the whole outfit sporty. Combination-level coverage still
# needs explicit sporty item cues in at least two categories when all three are
# replaced.
SPORTS_BRAND_TERMS = (
    "nike", "나이키", "adidas", "아디다스", "puma", "푸마",
    "new balance", "뉴발란스", "asics", "아식스", "reebok", "리복",
    "under armour", "언더아머", "fila", "휠라", "descente", "데상트",
    "umbro", "엄브로", "mizuno", "미즈노", "salomon", "살로몬",
    "kappa", "카파", "champion", "챔피온", "lululemon", "룰루레몬",
)
SPORTY_POSITIVE_TERMS = {
    "top": {
        "트랙 재킷": 4.0, "트랙자켓": 4.0, "윈드브레이커": 3.5,
        "바람막이": 3.5, "아노락": 3.5, "저지": 3.5, "져지": 3.5,
        "웜업": 3.0, "기능성": 3.0, "러닝": 3.0, "트레이닝": 3.0,
        "풋볼": 2.5, "축구": 2.5, "농구": 2.5, "유니폼": 2.5,
        "스웨트셔츠": 2.0, "맨투맨": 2.0,
    },
    "bottom": {
        "트랙팬츠": 4.0, "트랙 팬츠": 4.0, "조거팬츠": 3.5,
        "조거 팬츠": 3.5, "스웨트팬츠": 3.5, "트레이닝": 3.5,
        "나일론 팬츠": 3.0, "우븐 팬츠": 3.0, "러닝": 3.0,
        "카고팬츠": 2.5, "카고 팬츠": 2.5, "사이드라인": 2.5,
        "사이드 라인": 2.5, "테크": 2.0, "유틸리티": 2.0,
    },
    "shoes": {
        "러닝화": 4.0, "러닝 슈즈": 4.0, "트레이닝화": 3.5,
        "트레이너": 3.0, "스니커즈": 2.0, "운동화": 2.0,
    },
}
SPORTY_NEGATIVE_TERMS = {
    "top": {
        "드레스 셔츠": -4.0, "블레이저": -4.0, "정장 재킷": -4.0,
        "테일러드 재킷": -3.5, "체크 셔츠": -3.0, "체크 오버핏 셔츠": -3.5,
        "셔켓": -2.5, "셔츠": -2.5, "블라우스": -2.5,
    },
    "bottom": {
        "슬랙스": -4.0, "수트 팬츠": -4.0, "정장 바지": -4.0,
        "치노": -2.5, "데님": -2.5, "청바지": -2.5, "진스": -2.5,
    },
    "shoes": {
        "더비": -4.0, "옥스포드": -4.0, "로퍼": -3.5,
        "구두": -3.5, "펌프스": -3.5,
    },
}

# Terms are evaluated only after the normal user-keyword search. They never add
# a query or remove a candidate, which preserves search diversity.
FORMAL_POSITIVE_TERMS = {
    "top": {
        "블레이저": 3.5, "테일러드 재킷": 3.5, "정장 재킷": 3.5,
        "드레스 셔츠": 3.0, "셔츠": 2.5, "블라우스": 2.5,
        "재킷": 2.0, "자켓": 2.0, "코트": 2.0,
        "니트": 1.0, "가디건": 1.0, "폴로": 0.75,
    },
    "bottom": {
        "슬랙스": 3.5, "테일러드 팬츠": 3.5, "수트 팬츠": 3.5,
        "정장 바지": 3.5, "치노": 1.5,
    },
    "shoes": {
        "더비": 3.5, "옥스포드": 3.5, "로퍼": 3.0,
        "구두": 3.0, "펌프스": 3.0, "첼시 부츠": 2.0,
    },
}
FORMAL_CASUAL_TERMS = {
    "top": {
        "후드": -4.5, "맨투맨": -4.0, "스웨트셔츠": -4.0,
        "그래픽 티": -3.0, "프린트 티": -3.0, "반팔 티": -2.5,
        "반팔티": -2.5, "티셔츠": -2.0,
        # A shirt word alone must not make an oversized leather shacket look
        # office/classic. These are soft penalties, so a tailored oversized
        # blazer can still survive when its stronger formal cue wins.
        "셔켓": -3.5, "오버핏": -2.25, "오버사이즈": -2.25,
        "루즈": -1.75, "릴렉스": -1.5, "레더": -1.75, "가죽": -1.75,
        "하프 셔츠": -1.5,
    },
    "bottom": {
        "조거": -4.5, "트레이닝": -4.5, "스웨트팬츠": -4.5,
        "트랙팬츠": -4.0, "카고": -3.5, "반바지": -4.0, "쇼츠": -4.0,
        "디스트로이드": -4.0, "데미지": -4.0, "찢청": -4.0,
        "데님": -2.0, "청바지": -2.0,
    },
    "shoes": {
        "슬리퍼": -5.0, "쪼리": -5.0, "샌들": -4.0,
        "러닝화": -3.0, "운동화": -2.5, "스니커즈": -1.5,
    },
}

# A plain/basic tee receives only a modest deduction.  Visual or textual design
# evidence cancels it, so ringer, colour-block and large graphic tees remain.
BASIC_LOGO_TEE_PENALTY = -1.25
TEE_TERMS = (
    "티셔츠", "반팔티", "반팔 티", "숏슬리브", "t-shirt", "tshirt", "short sleeve tee",
)
DESIGN_POINT_TERMS = (
    "그래픽", "프린트", "아트워크", "레터링", "링거", "배색", "컬러블록", "컬러 블록",
    "스트라이프", "보더", "패턴", "일러스트", "캐릭터", "graphic", "printed", "print",
    "artwork", "ringer", "color block", "colour block", "striped", "illustration", "character",
)
LOGO_ONLY_TERMS = ("로고", "워드마크", "엠블럼", "심볼", "logo", "wordmark", "emblem")


def _normalized(value: Any) -> str:
    return " ".join(str(value or "").lower().split())


def trend_context_enabled(profile: Any) -> bool:
    """Return False for contexts where trend must not override formality."""
    return not (
        _normalized(getattr(profile, "purpose", "")) in TREND_EXCLUDED_PURPOSES
        or _normalized(getattr(profile, "desired_style", "")) in TREND_EXCLUDED_STYLES
        or _normalized(getattr(profile, "dress_code", "")) in TREND_EXCLUDED_DRESS_CODES
    )


def formal_context_enabled(profile: Any) -> bool:
    return (
        _normalized(getattr(profile, "purpose", "")) in FORMAL_PURPOSES
        or _normalized(getattr(profile, "desired_style", "")) in FORMAL_STYLES
        or _normalized(getattr(profile, "dress_code", "")) in FORMAL_DRESS_CODES
    )


def formal_context_item_types(profile: Any, category: str) -> tuple[str, ...]:
    if not formal_context_enabled(profile):
        return ()
    return FORMAL_CONTEXT_ITEM_TYPES.get(category, ())


def sporty_context_enabled(profile: Any) -> bool:
    return (
        _normalized(getattr(profile, "desired_style", "")) in SPORTY_STYLES
        or _normalized(getattr(profile, "purpose", "")) in SPORTY_PURPOSES
    )


def sporty_context_item_types(profile: Any, category: str) -> tuple[str, ...]:
    if not sporty_context_enabled(profile):
        return ()
    return SPORTY_CONTEXT_ITEM_TYPES.get(category, ())


def sports_brand_name(product: Any) -> str:
    brand = _normalized(getattr(product, "brand", ""))
    return next((term for term in SPORTS_BRAND_TERMS if term in brand), "")


def sporty_product_evidence(product: Any) -> dict:
    """Classify explicit sporty cues without treating a brand as a full look."""
    category = str(getattr(product, "category", "") or "")
    name = _normalized(getattr(product, "name", ""))
    brand_term = sports_brand_name(product)
    positives = [
        (term, value) for term, value in SPORTY_POSITIVE_TERMS.get(category, {}).items()
        if term in name
    ]
    negatives = [
        (term, value) for term, value in SPORTY_NEGATIVE_TERMS.get(category, {}).items()
        if term in name
    ]
    # "스웨트셔츠" is a sports cue; do not also count its contained "셔츠"
    # token as a dress-shirt conflict.
    if category == "top" and any(term in name for term in ("스웨트셔츠", "저지 셔츠", "기능성 셔츠")):
        negatives = [(term, value) for term, value in negatives if term != "셔츠"]
    positive = max(positives, key=lambda item: item[1], default=("", 0.0))
    negative = min(negatives, key=lambda item: item[1], default=("", 0.0))
    negative_value = 0.0 if brand_term else float(negative[1])
    if positive[1] >= 2.0:
        tier = "strong"
    elif negative_value < 0:
        tier = "conflict"
    elif brand_term or positive[1] > 0:
        tier = "compatible"
    else:
        tier = "neutral"
    return {
        "tier": tier,
        "positive_term": positive[0],
        "positive": float(positive[1]),
        "negative_term": negative[0],
        "negative": negative_value,
        "sports_brand": brand_term,
        "brand_exempt": bool(brand_term and negative[1] < 0),
    }


def sporty_context_adjustment(product: Any, profile: Any) -> tuple[float, dict]:
    if not sporty_context_enabled(profile):
        return 0.0, {}
    evidence = sporty_product_evidence(product)
    value = round(evidence["positive"] + evidence["negative"], 4)
    if not value and not evidence["brand_exempt"]:
        return 0.0, {}
    return value, {
        "rule_id": SPORTY_CONTEXT_RULE_ID,
        "kind": "sporty_context_guard",
        **evidence,
        "value": value,
    }


def strict_formal_context(profile: Any) -> bool:
    return (
        _normalized(getattr(profile, "purpose", "")) in STRICT_FORMAL_PURPOSES
        or _normalized(getattr(profile, "dress_code", "")) in STRICT_FORMAL_DRESS_CODES
    )


def _photo_fit(photo_attributes: dict | None) -> str:
    photo = photo_attributes or {}
    value = photo.get("fit") or photo.get("fit_conflict") or {}
    if value.get("source") != "product_photo":
        return ""
    return str(value.get("label") or "")


def canonical_bottom_fit(name: str, photo_attributes: dict | None = None) -> str:
    """Resolve the observed fit, preferring high-confidence visual evidence."""
    visual = _photo_fit(photo_attributes)
    if visual in BOTTOM_FIT_TREND_WEIGHTS:
        return visual
    text = re.sub(r"\s+", "", _normalized(name))
    # Order matters: 세미와이드 contains 와이드.
    if any(term in text for term in ("세미와이드", "semiwide", "semi-wide")):
        return "세미와이드"
    if any(term in text for term in ("와이드", "벌룬", "배기", "팔라초", "wide", "balloon", "baggy", "palazzo")):
        return "와이드핏"
    if any(term in text for term in ("스트레이트", "일자", "straight")):
        return "스트레이트핏"
    if any(term in text for term in ("테이퍼", "taper")):
        return "테이퍼드핏"
    if any(term in text for term in ("스키니", "슬림", "skinny", "slim")):
        return "슬림핏"
    if any(term in text for term in ("플레어", "부츠컷", "flare", "bootcut")):
        return "플레어핏"
    return ""


def trend_fit_adjustment(product: Any, profile: Any) -> tuple[float, dict]:
    if getattr(product, "category", "") != "bottom" or not trend_context_enabled(profile):
        return 0.0, {}
    fit = canonical_bottom_fit(
        getattr(product, "name", ""), getattr(product, "photo_attributes", None)
    )
    value = float(BOTTOM_FIT_TREND_WEIGHTS.get(fit, 0.0))
    if not value:
        return 0.0, {}
    return value, {"rule_id": TREND_RULE_ID, "fit": fit, "value": value}


def formal_context_adjustment(product: Any, profile: Any) -> tuple[float, dict]:
    """Rerank clearly formal/casual products without changing search queries."""
    category = str(getattr(product, "category", "") or "")
    if category not in FORMAL_POSITIVE_TERMS or not formal_context_enabled(profile):
        return 0.0, {}
    design = (getattr(product, "photo_attributes", None) or {}).get("design", {})
    text = _normalized(" ".join((
        str(getattr(product, "name", "") or ""),
        str(design.get("item_type") or ""),
    )))
    casual_top = category == "top" and any(
        term in text for term in ("티셔츠", "스웨트셔츠", "맨투맨", "후드")
    )
    positives = [
        value for term, value in FORMAL_POSITIVE_TERMS[category].items()
        if term in text and not (casual_top and term in {"셔츠", "드레스 셔츠"})
    ]
    negatives = [value for term, value in FORMAL_CASUAL_TERMS[category].items() if term in text]
    positive = max(positives, default=0.0)
    negative = min(negatives, default=0.0)
    if strict_formal_context(profile):
        positive *= 1.15
        negative *= 1.25
    value = round(positive + negative, 4)
    if not value:
        return 0.0, {}
    return value, {
        "rule_id": FORMAL_CONTEXT_RULE_ID,
        "kind": "formal_context_guard",
        "strict": strict_formal_context(profile),
        "positive": round(positive, 4),
        "negative": round(negative, 4),
        "value": value,
    }


def title_has_design_point(name: str) -> bool:
    text = _normalized(name)
    return any(term in text for term in DESIGN_POINT_TERMS)


def title_has_logo_only_cue(name: str) -> bool:
    text = _normalized(name)
    return any(term in text for term in LOGO_ONLY_TERMS)


def basic_logo_tee_adjustment(product: Any) -> tuple[float, dict]:
    """Penalise plain basics and logo/wordmark-only tees, regardless of logo size."""
    if getattr(product, "category", "") != "top":
        return 0.0, {}
    name = _normalized(getattr(product, "name", ""))
    design = (getattr(product, "photo_attributes", None) or {}).get("design", {})
    item_type = str(design.get("item_type") or "")
    is_tee = item_type == "티셔츠" or any(term in name for term in TEE_TERMS)
    logo_only = title_has_logo_only_cue(name)
    if not is_tee or (not logo_only and (title_has_design_point(name) or not design.get("plain_basic"))):
        return 0.0, {}
    value = float(BASIC_LOGO_TEE_PENALTY)
    return value, {
        "rule_id": DETAIL_RULE_ID,
        "kind": "plain_or_logo_only_tee",
        "value": value,
    }
