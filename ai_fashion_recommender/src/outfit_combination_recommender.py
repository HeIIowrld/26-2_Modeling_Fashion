"""무신사 후보를 현재 착장과 연결된 세 개의 코디 조합으로 재정렬한다."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from itertools import product as cartesian_product
from typing import Any, Iterable

from recommendation_keywords import TargetKeywordResult
from schemas import OutfitAnalysis, PoseAnalysis, UserProfile


GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
LLM_ENABLED_VALUES = {"1", "true", "yes", "on"}
STYLE_FORMALITY = {"스포티": 1, "스트리트": 1, "캐주얼": 2, "로맨틱": 3, "미니멀": 3, "포멀": 5}
FORBIDDEN_COPY = ("예산", "가격", "할인", "가성비", "비용", "사이즈", "실측")


@dataclass
class CombinationEvidence:
    label: str
    text: str
    rule_ids: tuple[str, ...] = ()


@dataclass
class OutfitCombination:
    combination_id: str
    product_ids: list[str]
    current_items: list[dict[str, str]]
    score: float
    evidence: list[CombinationEvidence] = field(default_factory=list)
    reason: str = ""
    reason_source: str = "rules"

    def public_dict(self) -> dict[str, Any]:
        return {
            "combination_id": self.combination_id,
            "product_ids": list(self.product_ids),
            "current_items": list(self.current_items),
            "reason": self.reason,
            "reason_source": self.reason_source,
            "evidence": [item.text for item in self.evidence],
            "evidence_labels": [item.label for item in self.evidence],
            "rule_ids": list(dict.fromkeys(
                rule_id for item in self.evidence for rule_id in item.rule_ids
            )),
        }


def _usable(value: str) -> bool:
    blocked = ("분석 보류", "분석 불가", "불확실", "해당 없음")
    return bool(value and not any(marker in value for marker in blocked))


def _particle(value: str, with_final: str, without_final: str) -> str:
    """마지막 한글 음절의 받침에 맞는 짧은 조사를 고른다."""
    if not value:
        return without_final
    code = ord(value[-1]) - 0xAC00
    return with_final if 0 <= code <= 11171 and code % 28 else without_final


def _fit_value(value: str, category: str) -> str:
    normalized = value.replace(" ", "").replace("추정", "")
    if not normalized:
        return "분석 보류"
    if category == "top":
        if any(word in normalized for word in ("오버", "루즈", "여유")):
            return "오버핏" if "오버" in normalized else "여유핏"
        if any(word in normalized for word in ("슬림", "세미핏")):
            return "슬림핏"
        return "레귤러핏"
    if any(word in normalized for word in ("와이드", "플레어")):
        return "와이드핏"
    if "테이퍼" in normalized:
        return "테이퍼드핏"
    if "슬림" in normalized:
        return "슬림핏"
    return "스트레이트핏"


def _attribute_value(product: Any, targets: TargetKeywordResult, attribute: str) -> str:
    attributes = targets.targets.get(product.category, {})
    values = list(attributes.get(attribute, []))
    matched = set(getattr(product, "matched_keywords", []) or [])
    return next((value for value in values if value in matched), "")


def _product_garment(product: Any, targets: TargetKeywordResult) -> dict[str, Any]:
    category = product.category
    style = _attribute_value(product, targets, "style")
    fit = _attribute_value(product, targets, "fit")
    return {
        "category": category,
        "color": _attribute_value(product, targets, "color"),
        "style": style,
        "fit": _fit_value(fit, category),
        "length": _attribute_value(product, targets, "length"),
        "pattern": _attribute_value(product, targets, "pattern") or "패턴 불확실",
        "material": _attribute_value(product, targets, "material"),
        "formality": STYLE_FORMALITY.get(style, 3),
    }


def _current_item(category: str, outfit: OutfitAnalysis) -> dict[str, str]:
    summary = outfit.to_summary_dict()
    if category == "top":
        return {
            "category": "top", "label": "현재 상의", "description": summary["상의"],
            "color": outfit.upper_color, "fit": outfit.fit, "material": outfit.material,
        }
    if category == "bottom":
        return {
            "category": "bottom", "label": "현재 하의", "description": summary["하의"],
            "color": outfit.lower_color, "fit": outfit.lower_fit, "material": outfit.lower_material,
        }
    return {
        "category": "shoes", "label": "현재 신발", "description": summary["신발"],
        "color": "", "fit": "", "material": "",
    }


def _shoe_score(product: Any, profile: UserProfile, targets: TargetKeywordResult) -> tuple[float, str]:
    item_type = _attribute_value(product, targets, "item_type")
    if not item_type:
        return 0.60, ""
    preferred = {
        "스트리트": {"스니커즈", "부츠"}, "캐주얼": {"스니커즈", "로퍼"},
        "미니멀": {"로퍼", "더비슈즈", "스니커즈"}, "포멀": {"로퍼", "더비슈즈"},
        "스포티": {"러닝화", "스니커즈"}, "로맨틱": {"메리제인", "로퍼"},
    }.get(profile.desired_style, {"스니커즈"})
    score = 0.92 if item_type in preferred else 0.72
    subject = _particle(item_type, "이", "가")
    return score, f"'{item_type}'{subject} 선택한 {profile.desired_style} 스타일의 코디 흐름을 이어 줍니다."


def _retrieval_rank_scores(groups: dict[str, list[Any]]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for products in groups.values():
        count = max(1, len(products) - 1)
        for index, product in enumerate(products):
            scores[product.product_id] = 1.0 - 0.25 * index / count
    return scores


def _candidate_evidence(
    products: list[Any],
    selected_categories: tuple[str, ...],
    current_items: list[dict[str, str]],
    harmony_reasons: list[str],
    harmony_rules: list[str],
    shoe_reason: str,
    *,
    silhouette_known: bool,
    color_known: bool,
) -> list[CombinationEvidence]:
    evidence: list[CombinationEvidence] = []
    if current_items:
        kept = "·".join(item["label"] for item in current_items)
        object_particle = _particle(kept, "을", "를")
        evidence.append(CombinationEvidence(
            "현재 착장", f"{kept}{object_particle} 유지한 상태에서 교체할 아이템의 조화를 평가했습니다.",
            ("R-CMP-03",),
        ))
    if (
        any(category in selected_categories for category in ("top", "bottom"))
        and harmony_reasons and silhouette_known
    ):
        evidence.append(CombinationEvidence("실루엣", harmony_reasons[0], tuple(harmony_rules[:2])))
        if len(harmony_reasons) >= 3 and color_known:
            evidence.append(CombinationEvidence("색상", harmony_reasons[2], ("R-COL-03",)))
    if shoe_reason and len(evidence) < 3:
        evidence.append(CombinationEvidence("신발", shoe_reason, ("R-CTX-01", "R-ACC-06")))
    if len(evidence) < 3:
        keywords = list(dict.fromkeys(
            keyword for product in products
            for keyword in (getattr(product, "matched_keywords", []) or [])
        ))[:3]
        if keywords:
            evidence.append(CombinationEvidence(
                "검색 조건", f"상품명에서 추천 키워드 '{'·'.join(keywords)}'가 실제로 확인됐습니다.",
            ))
    return evidence[:3]


def _output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    return ""


def _llm_combination_reasons(combinations: list[OutfitCombination]) -> dict[str, str]:
    enabled = os.environ.get("FASHION_LLM_REASONS", "").strip().lower() in LLM_ENABLED_VALUES
    provider = os.environ.get("FASHION_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "openai"
    api_key = os.environ.get("GEMINI_API_KEY" if provider == "gemini" else "OPENAI_API_KEY", "").strip()
    if not enabled or provider not in {"gemini", "openai"} or not api_key or not combinations:
        return {}

    allowed: dict[str, set[str]] = {}
    payload = []
    for combination in combinations:
        facts = []
        ids = set()
        for index, evidence in enumerate(combination.evidence, 1):
            evidence_id = f"{combination.combination_id}-E{index}"
            ids.add(evidence_id)
            facts.append({"evidence_id": evidence_id, "label": evidence.label, "fact": evidence.text,
                          "rule_ids": list(evidence.rule_ids)})
        allowed[combination.combination_id] = ids
        payload.append({"combination_id": combination.combination_id, "allowed_evidence": facts})

    schema = {
        "type": "object", "properties": {"items": {"type": "array", "items": {
            "type": "object", "properties": {
                "combination_id": {"type": "string"}, "summary": {"type": "string"},
                "evidence_ids": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 3},
            }, "required": ["combination_id", "summary", "evidence_ids"], "additionalProperties": False,
        }}}, "required": ["items"], "additionalProperties": False,
    }
    instructions = (
        "당신은 FITTA의 코디 추천 문장 편집자입니다. 각 조합에 이미 확정된 allowed_evidence만 사용해 "
        "현재 유지하는 옷과 교체 상품이 왜 한 코디로 연결되는지 자연스러운 한국어 한 문장으로 쓰세요. "
        "새로운 색상·소재·핏·체형·목적·착용감·사이즈·가격 정보를 만들지 말고, 사용한 evidence_id를 "
        "1~3개 반환하세요. 120자 이내이며 점수와 해시태그는 쓰지 마세요."
    )
    if provider == "gemini":
        body = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps({"combinations": payload}, ensure_ascii=False)}]}],
            "generationConfig": {"responseMimeType": "application/json", "responseJsonSchema": schema},
        }
        request = urllib.request.Request(
            GEMINI_GENERATE_URL.format(model=urllib.parse.quote(os.environ.get("FASHION_LLM_MODEL", "gemini-2.5-flash-lite"), safe="-._")),
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"}, method="POST",
        )
    else:
        body = {
            "model": os.environ.get("FASHION_LLM_MODEL", "gpt-5-mini"), "store": False,
            "instructions": instructions, "input": json.dumps({"combinations": payload}, ensure_ascii=False),
            "text": {"format": {"type": "json_schema", "name": "fitta_outfit_combinations", "strict": True, "schema": schema}},
        }
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL, data=json.dumps(body, ensure_ascii=False).encode(),
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            parsed = json.loads(response.read().decode())
        raw = parsed["candidates"][0]["content"]["parts"][0]["text"] if provider == "gemini" else _output_text(parsed)
        generated = json.loads(raw)
    except (OSError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError, urllib.error.HTTPError):
        return {}

    accepted = {}
    for item in generated.get("items", []):
        combination_id = str(item.get("combination_id") or "")
        summary = str(item.get("summary") or "").strip()[:180]
        evidence_ids = item.get("evidence_ids")
        if (
            combination_id not in allowed or not summary or not isinstance(evidence_ids, list)
            or not 1 <= len(evidence_ids) <= 3 or len(set(evidence_ids)) != len(evidence_ids)
            or any(value not in allowed[combination_id] for value in evidence_ids)
            or any(word in summary for word in FORBIDDEN_COPY)
        ):
            continue
        accepted[combination_id] = summary
    return accepted


def recommend_outfit_combinations(
    products: Iterable[Any],
    profile: UserProfile,
    pose: PoseAnalysis,
    outfit: OutfitAnalysis,
    targets: TargetKeywordResult,
    recommender: Any,
    limit: int = 3,
) -> list[OutfitCombination]:
    """선택 카테고리 상품과 현재 유지할 옷을 합쳐 조화가 높은 코디를 고른다."""
    del pose  # 체형은 상품 검색 키워드에서 이미 반영되며 조합 설명에서 다시 추측하지 않는다.
    selected_categories = tuple(targets.targets)
    grouped = {
        category: [product for product in products if product.category == category]
        for category in selected_categories
    }
    if not selected_categories or any(not grouped[category] for category in selected_categories):
        return []
    rank_scores = _retrieval_rank_scores(grouped)
    current_items = [
        _current_item(category, outfit)
        for category in ("top", "bottom", "shoes")
        if category not in selected_categories and (category != "shoes" or outfit.shoes.get("accepted"))
    ]
    current_by_category = {item["category"]: item for item in current_items}
    candidates: list[OutfitCombination] = []
    for index, selected in enumerate(cartesian_product(*(grouped[category] for category in selected_categories)), 1):
        chosen = list(selected)
        by_category = {product.category: product for product in chosen}
        top = (
            _product_garment(by_category["top"], targets)
            if "top" in by_category else recommender._garment(None, "top", outfit)
        )
        bottom = (
            _product_garment(by_category["bottom"], targets)
            if "bottom" in by_category else recommender._garment(None, "bottom", outfit)
        )
        harmony, _, harmony_reasons, harmony_rules = recommender._outfit_harmony_score(top, bottom, profile)
        shoe_score, shoe_reason = (1.0, "")
        if "shoes" in by_category:
            shoe_score, shoe_reason = _shoe_score(by_category["shoes"], profile, targets)
        retrieval = sum(rank_scores[product.product_id] for product in chosen) / len(chosen)
        overall = 0.50 * retrieval + 0.40 * harmony + 0.10 * shoe_score
        evidence = _candidate_evidence(
            chosen, selected_categories, list(current_by_category.values()),
            harmony_reasons, harmony_rules, shoe_reason,
            silhouette_known=_usable(top["fit"]) and _usable(bottom["fit"]),
            color_known=_usable(top["color"]) and _usable(bottom["color"]),
        )
        candidates.append(OutfitCombination(
            f"OUTFIT-{index}", [product.product_id for product in chosen],
            list(current_by_category.values()), overall, evidence,
        ))

    selected_outfits: list[OutfitCombination] = []
    remaining = list(candidates)
    while remaining and len(selected_outfits) < max(0, limit):
        def diversified_score(candidate: OutfitCombination) -> float:
            overlap = max((
                len(set(candidate.product_ids) & set(chosen.product_ids)) / max(1, len(candidate.product_ids))
                for chosen in selected_outfits
            ), default=0.0)
            return candidate.score - 0.12 * overlap

        best = max(remaining, key=lambda candidate: (diversified_score(candidate), candidate.combination_id))
        remaining.remove(best)
        selected_outfits.append(best)

    for rank, combination in enumerate(selected_outfits, 1):
        combination.combination_id = f"OUTFIT-{rank}"
        combination.reason = " ".join(item.text for item in combination.evidence[:2])
    generated = _llm_combination_reasons(selected_outfits)
    for combination in selected_outfits:
        if combination.combination_id in generated:
            combination.reason = generated[combination.combination_id]
            combination.reason_source = "llm"
    return selected_outfits
