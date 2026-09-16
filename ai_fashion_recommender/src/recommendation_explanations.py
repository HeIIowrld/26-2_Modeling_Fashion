"""웹 결과에 표시할 짧고 개인화된 추천 설명을 만든다.

기본 설명은 Fashion Rule로 만든 검색 키워드와 사용자 조건만 사용해 항상
생성한다. 운영자가 명시적으로 LLM을 켠 경우에는 사진 원본이 아닌 구조화된
분석값만 선택한 LLM API에 보내 자연스러운 한 문장으로 다듬는다.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Iterable

from recommendation_keywords import RecommendationKeywordGenerator, TargetKeywordResult
from schemas import ALL_BODY_SHAPES, BODY_SHAPES, CurrentOutfitEvaluation, OutfitAnalysis, PoseAnalysis, UserProfile


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LLM_ENABLED_VALUES = {"1", "true", "yes", "on"}
CATEGORY_LABELS = {"top": "상의", "bottom": "하의", "shoes": "신발"}
MAX_EVIDENCE = 3
BODY_SHAPE_CONFIDENCE_THRESHOLD = 0.65
SHORT_LEG_RATIO = 0.60
BODY_SHAPE_RULES = {"R-BOD-01": "삼각체형", "R-BOD-06": "삼각체형", "R-BOD-02": "역삼각체형", "R-BOD-03": "사각체형"}
BODY_SHAPE_TEMPLATES = {
    "역삼각체형": {"top": "역삼각체형으로 분석되어 상체가 단순하고 정돈되어 보이는 '{keywords}' 상의를 우선했습니다.", "bottom": "역삼각체형으로 분석되어 상·하체 균형을 위해 하체에 구조감을 더하는 '{keywords}' 실루엣을 우선했습니다."},
    "삼각체형": {"top": "삼각체형으로 분석되어 상체 라인에 구조감을 더해 시선을 위쪽으로 모을 수 있는 '{keywords}' 상의를 우선했습니다.", "bottom": "삼각체형으로 분석되어 하체 볼륨이 과하게 강조되지 않도록 정돈된 '{keywords}' 실루엣을 우선했습니다."},
    "사각체형": {"top": "사각체형으로 분석되어 상·하체 볼륨을 한쪽씩 나눠 줄 '{keywords}' 상의를 우선했습니다.", "bottom": "사각체형으로 분석되어 상의와 볼륨이 겹치지 않도록 정돈된 '{keywords}' 실루엣을 우선했습니다."},
}
MATCHABLE_ATTRIBUTES = ("fit", "length", "waistline", "material", "color", "style", "structure", "silhouette", "function", "purpose")


@dataclass(frozen=True)
class ProductEvidence:
    kind: str
    label: str
    text: str
    rule_ids: tuple[str, ...] = ()
    basis: str = ""


def _shape_is_confident(pose: PoseAnalysis) -> bool:
    return bool(getattr(pose, "valid", False)) and getattr(pose, "body_shape", "") in BODY_SHAPES and float(getattr(pose, "body_shape_confidence", 0.0)) >= BODY_SHAPE_CONFIDENCE_THRESHOLD


def build_product_evidence(product: Any, profile: UserProfile, pose: PoseAnalysis, targets: TargetKeywordResult) -> list[ProductEvidence]:
    category = getattr(product, "category", "")
    attributes = (getattr(targets, "targets", None) or {}).get(category, {}) or {}
    keyword_rules = (getattr(targets, "keyword_rules", None) or {}).get(category, {}) or {}
    applied = set(getattr(targets, "applied_rules", None) or [])
    sources = getattr(targets, "sources", None) or {}
    matched = list(dict.fromkeys(getattr(product, "matched_keywords", None) or []))

    def attribute_of(keyword: str) -> str:
        return next((name for name in MATCHABLE_ATTRIBUTES if keyword in attributes.get(name, [])), "")

    def rules_of(keyword: str) -> list[str]:
        return [rule_id for rule_id in keyword_rules.get(keyword, []) if rule_id in applied]

    def quoted(values: list[str]) -> str:
        return "·".join(values)

    evidence: list[ProductEvidence] = []
    cited: set[str] = set()
    shape_keywords = [keyword for keyword in matched if any(rule in BODY_SHAPE_RULES for rule in rules_of(keyword))]
    if shape_keywords:
        rule_ids = tuple(dict.fromkeys(rule for keyword in shape_keywords for rule in rules_of(keyword) if rule in BODY_SHAPE_RULES))
        shapes = {BODY_SHAPE_RULES[rule] for rule in rule_ids}
        if _shape_is_confident(pose) and shapes == {pose.body_shape}:
            text = BODY_SHAPE_TEMPLATES[pose.body_shape].get(category, "{shape}으로 분석되어 균형을 고려한 '{keywords}' 실루엣을 우선했습니다.").format(shape=pose.body_shape, keywords=quoted(shape_keywords))
            basis = "체형"
        else:
            text = f"사진에서 추정된 상·하체 비율을 참고해 균형감 있는 '{quoted(shape_keywords)}' 실루엣을 우선했습니다."
            basis = "체형 비율"
        evidence.append(ProductEvidence("body_shape", "체형", text, rule_ids, basis))
        cited.update(shape_keywords)

    proportion = [keyword for keyword in matched if keyword not in cited and "R-BOD-05" in rules_of(keyword)]
    if proportion:
        goal = profile.silhouette_goal if RecommendationKeywordGenerator._provided(profile, "silhouette_goal") else ""
        if goal or (getattr(pose, "valid", False) and 0 < pose.leg_ratio < SHORT_LEG_RATIO):
            basis = f"‘{goal}’ 목표" if goal else "다리 비율"
            text = f"{basis}에 맞춰 허리선과 세로선이 길게 이어지는 '{quoted(proportion)}' 디자인을 우선했습니다."
            evidence.append(ProductEvidence("proportion", "비율", text, ("R-BOD-05",), basis))
            cited.update(proportion)

    style_keywords = [keyword for keyword in matched if keyword not in cited and attribute_of(keyword) == "style"]
    if style_keywords:
        evidence.append(ProductEvidence("style", "스타일", f"선택한 {style_keywords[0]} 스타일과 잘 맞는 상품입니다.", tuple(rules_of(style_keywords[0]))))
        cited.update(style_keywords)
    purpose_keywords = [keyword for keyword in matched if keyword not in cited and attribute_of(keyword) == "purpose" and "R-CTX-01" in rules_of(keyword)]
    if purpose_keywords:
        evidence.append(ProductEvidence("purpose", "목적", f"선택한 {purpose_keywords[0]} 목적에 맞는 상품입니다.", ("R-CTX-01",)))
        cited.update(purpose_keywords)
    details = [(keyword, attribute_of(keyword)) for keyword in matched if keyword not in cited and attribute_of(keyword) in {"material", "color", "function"}]
    if details:
        labels = {"material": "소재", "color": "색상", "function": "기능"}
        names = "·".join(keyword if attr == "function" else f"{keyword} {labels[attr]}" for keyword, attr in details)
        evidence.append(ProductEvidence("detail", "·".join(dict.fromkeys(labels[attr] for _, attr in details)), f"선택한 {names} 조건에 맞는 상품입니다.", tuple(dict.fromkeys(rule for keyword, _ in details for rule in rules_of(keyword)))))
    return evidence[:MAX_EVIDENCE]
MATRIX_LABELS = {
    "body_fit": "체형 적합도",
    "situation_fit": "상황 적합도",
    "style_fit": "스타일 적합도",
}


def build_outfit_summary_points(
    outfit: OutfitAnalysis,
    evaluation: CurrentOutfitEvaluation,
) -> list[str]:
    """현재 상·하의 특징과 2×3 평가를 정확히 세 문장으로 요약한다."""
    descriptions = outfit.to_summary_dict()
    matrix = evaluation.diagnostic_matrix

    def item_line(category: str, label: str) -> str:
        scores = matrix.get(category, {})
        score_copy = " · ".join(
            f"{MATRIX_LABELS[key]} {float(scores.get(key, 0)):.0f}점"
            for key in ("body_fit", "situation_fit", "style_fit")
        )
        return f"{label}: {descriptions.get(label, '분석 보류')} — {score_copy}."

    harmony = float(evaluation.harmony_score)
    return [
        item_line("top", "상의"),
        item_line("bottom", "하의"),
        f"상·하의 조화는 {harmony:.0f}점이며, {evaluation.verdict}.",
    ]


def _fallback_reason(product: Any, profile: UserProfile, pose: PoseAnalysis, evidence: Iterable[ProductEvidence] = (), targets: TargetKeywordResult | None = None) -> str:
    evidence = list(evidence)
    if evidence:
        bases = [item.basis for item in evidence if item.basis]
        attributes = (getattr(targets, "targets", None) or {}).get(product.category, {}) if targets else {}
        style = next(iter(attributes.get("style", [])), "")
        subjects = bases[:1] + ([f"{style} 스타일"] if style else [])
        if subjects:
            category = CATEGORY_LABELS.get(product.category, "상품")
            return f"{'과 '.join(subjects)} 조건에 잘 맞는 {category}라 추천했어요."
    keywords = list(getattr(product, "search_keywords", []) or [])[:3]
    keyword_copy = "·".join(keywords) or CATEGORY_LABELS.get(product.category, "상품")
    contexts = [value for value in (profile.purpose, profile.desired_style) if value and value != "자동"]
    context_copy = "·".join(dict.fromkeys(contexts)) or "원하시는 분위기"
    category = CATEGORY_LABELS.get(product.category, "아이템")
    category_with_particle = f"{category}이라" if category == "신발" else f"{category}라"
    return f"{context_copy}에 잘 어울리는 {keyword_copy} 포인트의 {category_with_particle} 추천해요."


def _output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    return ""


def _llm_reasons(
    products: list[Any],
    profile: UserProfile,
    pose: PoseAnalysis,
    targets: TargetKeywordResult,
    fallbacks: dict[str, str],
) -> dict[str, str]:
    enabled = os.environ.get("FASHION_LLM_REASONS", "").strip().lower() in LLM_ENABLED_VALUES
    provider = os.environ.get("FASHION_LLM_PROVIDER", "").strip().lower()
    if not provider:
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "openai"
    api_key = (
        os.environ.get("GEMINI_API_KEY", "").strip()
        if provider == "gemini"
        else os.environ.get("OPENAI_API_KEY", "").strip()
    )
    if not enabled or provider not in {"gemini", "openai"} or not api_key or not products:
        return {}

    product_context = [
        {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "search_keywords": list(product.search_keywords)[:3],
        }
        for product in products
    ]
    input_payload = {
        "user_context": {
            "gender": profile.gender,
            "purpose": profile.purpose,
            "desired_style": profile.desired_style,
            "silhouette_goal": profile.silhouette_goal,
            "season": profile.season,
            "dress_code": profile.dress_code,
            "activity_level": profile.activity_level,
            "preferred_colors": profile.preferred_colors,
            "personal_tone": profile.personal_tone,
            "preferred_materials": profile.preferred_materials,
        },
        "body_analysis": {
            "body_shape": pose.body_shape,
            "body_shape_confidence": pose.body_shape_confidence,
            "leg_ratio": pose.leg_ratio,
        },
        "fashion_rule_search_targets": targets.targets,
        "products": product_context,
    }
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "reason": {"type": "string"},
                    },
                    "required": ["product_id", "reason"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    instructions = (
        "당신은 고객에게 직접 옷을 추천하는 친절하고 감각적인 옷가게 점원입니다. "
        "고객에게 말하듯 각 상품의 추천 이유를 자연스러운 한국어 한 문장으로 작성하세요. "
        "사용자의 목적과 원하는 스타일, Fashion Rule 검색 조건, 상품의 실제 search_keywords 중 "
        "도움이 되는 내용을 골라 상품마다 조금씩 다르게 설명하세요. 정보를 기계적으로 나열하거나 "
        "고정된 문장 틀을 반복하지 마세요. 제공된 정보에 없는 소재·핏·기능·효과를 지어내면 안 됩니다. "
        "예산, 가격, 할인, 가성비, 비용 등 금액과 관련된 이야기는 절대 하지 마세요. "
        "신발은 체형이나 다리 길이를 추천 근거로 사용하지 마세요. 해시태그와 점수도 쓰지 마세요. "
        "각 문장은 120자 이내로 작성하세요."
    )
    if provider == "gemini":
        model = os.environ.get("FASHION_LLM_MODEL", "gemini-2.5-flash-lite")
        request_body = {
            "systemInstruction": {"parts": [{"text": instructions}]},
            "contents": [{
                "role": "user",
                "parts": [{"text": json.dumps(input_payload, ensure_ascii=False)}],
            }],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
            },
        }
        request = urllib.request.Request(
            GEMINI_GENERATE_URL.format(model=urllib.parse.quote(model, safe="-._")),
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
            method="POST",
        )
    else:
        request_body = {
            "model": os.environ.get("FASHION_LLM_MODEL", "gpt-5-mini"),
            "store": False,
            "instructions": instructions,
            "input": json.dumps(input_payload, ensure_ascii=False),
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "fitta_product_reasons",
                    "strict": True,
                    "schema": schema,
                }
            },
        }
        request = urllib.request.Request(
            OPENAI_RESPONSES_URL,
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
    try:
        with urllib.request.urlopen(request, timeout=10.0) as response:
            parsed = json.loads(response.read().decode("utf-8"))
        if provider == "gemini":
            text = parsed["candidates"][0]["content"]["parts"][0]["text"]
        else:
            text = _output_text(parsed)
        generated = json.loads(text)
    except (OSError, TimeoutError, ValueError, KeyError, IndexError, json.JSONDecodeError, urllib.error.HTTPError):
        return {}

    products_by_id = {product.product_id: product for product in products}
    accepted: dict[str, str] = {}
    for item in generated.get("items", []):
        product_id = str(item.get("product_id"))
        reason = str(item.get("reason", "")).strip()[:180]
        product = products_by_id.get(product_id)
        if not product or not reason:
            continue
        forbidden = ("예산", "가격", "할인", "가성비", "비용", "만원", "원대")
        # 말투와 구성은 LLM에 맡기고, 사용자가 금지한 금액 관련 표현만 확실히 제외한다.
        if any(word in reason for word in forbidden):
            continue
        accepted[product_id] = reason
    return accepted


def add_product_recommendation_reasons(
    products: Iterable[Any],
    profile: UserProfile,
    pose: PoseAnalysis,
    targets: TargetKeywordResult,
) -> None:
    """검색 상품 객체에 항상 표시 가능한 추천 이유와 생성 출처를 붙인다."""
    product_list = list(products)
    evidence = {
        product.product_id: build_product_evidence(product, profile, pose, targets)
        for product in product_list
    }
    fallbacks = {
        product.product_id: _fallback_reason(product, profile, pose, evidence[product.product_id], targets)
        for product in product_list
    }
    generated = _llm_reasons(product_list, profile, pose, targets, fallbacks)
    for product in product_list:
        product.recommendation_reason = generated.get(product.product_id, fallbacks[product.product_id])
        product.recommendation_reason_source = "llm" if product.product_id in generated else "rules"
        items = evidence[product.product_id]
        product.fit_evidence = [item.text for item in items]
        product.fit_evidence_labels = [item.label for item in items]
        product.reason_rule_ids = list(dict.fromkeys(rule for item in items for rule in item.rule_ids))
