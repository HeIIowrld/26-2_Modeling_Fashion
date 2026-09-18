"""웹 결과에 표시할 짧고 개인화된 추천 설명을 만든다.

서버가 실제 상품명과 매칭된 키워드, 적용 규칙, 분석 신뢰도를 먼저 검증한다.
LLM은 이 검증을 통과한 근거만 자연스러운 한 문장으로 다듬으며 새로운 추천
근거를 판단하거나 추가하지 않는다.
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
from schemas import ALL_BODY_SHAPES, CurrentOutfitEvaluation, OutfitAnalysis, PoseAnalysis, UserProfile


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LLM_ENABLED_VALUES = {"1", "true", "yes", "on"}
MAX_EVIDENCE = 3
BODY_SHAPE_CONFIDENCE_THRESHOLD = 0.65
PROPORTION_CONFIDENCE_THRESHOLD = 0.65
SHORT_LEG_RATIO = 0.60
BODY_SHAPE_RULES = {
    "R-BOD-01": "삼각체형",
    "R-BOD-06": "삼각체형",
    "R-BOD-02": "역삼각체형",
    "R-BOD-03": {"사각체형", "모래시계체형"},
    "R-BOD-07": "둥근체형",
    "R-BOD-08": "마름모꼴체형",
}
BODY_SHAPE_TEMPLATES = {
    "역삼각체형": {"top": "역삼각체형으로 분석되어 상체가 단순하고 정돈되어 보이는 '{keywords}' 상의를 우선했습니다.", "bottom": "역삼각체형으로 분석되어 상·하체 균형을 위해 하체에 구조감을 더하는 '{keywords}' 실루엣을 우선했습니다."},
    "삼각체형": {"top": "삼각체형으로 분석되어 상체 라인에 구조감을 더해 시선을 위쪽으로 모을 수 있는 '{keywords}' 상의를 우선했습니다.", "bottom": "삼각체형으로 분석되어 하체 볼륨이 과하게 강조되지 않도록 정돈된 '{keywords}' 실루엣을 우선했습니다."},
    "사각체형": {"top": "사각체형으로 분석되어 상·하체 볼륨을 한쪽씩 나눠 줄 '{keywords}' 상의를 우선했습니다.", "bottom": "사각체형으로 분석되어 상의와 볼륨이 겹치지 않도록 정돈된 '{keywords}' 실루엣을 우선했습니다."},
    "모래시계체형": {"top": "모래시계체형으로 분석되어 허리 기준점을 만들고 상·하체 볼륨을 나눠 줄 '{keywords}' 상의를 우선했습니다.", "bottom": "모래시계체형으로 분석되어 상·하체 볼륨을 한쪽씩 나눠 줄 '{keywords}' 실루엣을 우선했습니다."},
    "둥근체형": {"top": "둥근체형으로 분석되어 상체에 세로 방향과 정돈된 여유를 더할 '{keywords}' 상의를 우선했습니다.", "bottom": "둥근체형으로 분석되어 하체를 정돈할 '{keywords}' 실루엣을 우선했습니다."},
    "마름모꼴체형": {"top": "마름모꼴체형으로 분석되어 어깨와 넥라인에 구조를 더할 '{keywords}' 상의를 우선했습니다.", "bottom": "마름모꼴체형으로 분석되어 상·하체 균형을 정돈할 '{keywords}' 실루엣을 우선했습니다."},
}
MATCHABLE_ATTRIBUTES = (
    "item_type", "fit", "length", "waistline", "material", "color",
    "style", "structure", "silhouette", "function",
)
BODY_LANGUAGE = ("체형", "다리", "상체", "하체", "신체 비율", "허리선")
FORBIDDEN_REASON_LANGUAGE = (
    "예산", "가격", "할인", "가성비", "비용", "만원", "원대",
    "사이즈", "실측", "정사이즈",
)
UNSUPPORTED_PURPOSE_LANGUAGE = ("데일리", "데이트", "출근", "면접", "하객", "여행")


@dataclass(frozen=True)
class ProductEvidence:
    kind: str
    label: str
    text: str
    rule_ids: tuple[str, ...] = ()
    basis: str = ""
    keywords: tuple[str, ...] = ()
    source: str = ""


def _shape_is_confident(pose: PoseAnalysis) -> bool:
    return bool(getattr(pose, "valid", False)) and getattr(pose, "body_shape", "") in ALL_BODY_SHAPES and float(getattr(pose, "body_shape_confidence", 0.0)) >= BODY_SHAPE_CONFIDENCE_THRESHOLD


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

    def source_of(attribute: str) -> str:
        return str(sources.get(f"{category}.{attribute}") or sources.get(attribute) or "")

    evidence: list[ProductEvidence] = []
    cited: set[str] = set()
    shape_keywords = [
        keyword for keyword in matched
        if any(rule in BODY_SHAPE_RULES for rule in rules_of(keyword))
    ]
    if category != "shoes" and shape_keywords and _shape_is_confident(pose):
        rule_ids = tuple(dict.fromkeys(rule for keyword in shape_keywords for rule in rules_of(keyword) if rule in BODY_SHAPE_RULES))
        shapes = set()
        for rule_id in rule_ids:
            mapped_shapes = BODY_SHAPE_RULES[rule_id]
            shapes.update(mapped_shapes if isinstance(mapped_shapes, set) else {mapped_shapes})
        if pose.body_shape in shapes:
            text = BODY_SHAPE_TEMPLATES[pose.body_shape].get(category, "{shape}으로 분석되어 균형을 고려한 '{keywords}' 실루엣을 우선했습니다.").format(shape=pose.body_shape, keywords=quoted(shape_keywords))
            evidence.append(ProductEvidence(
                "body_shape", "체형", text, rule_ids, "체형",
                tuple(shape_keywords), "photo_analysis",
            ))
            cited.update(shape_keywords)

    proportion = [keyword for keyword in matched if keyword not in cited and "R-BOD-05" in rules_of(keyword)]
    if category != "shoes" and proportion:
        goal = profile.silhouette_goal if RecommendationKeywordGenerator._provided(profile, "silhouette_goal") else ""
        reliable_ratio = (
            getattr(pose, "valid", False)
            and float(getattr(pose, "full_body_score", 0.0)) >= PROPORTION_CONFIDENCE_THRESHOLD
            and 0 < pose.leg_ratio < SHORT_LEG_RATIO
        )
        if goal or reliable_ratio:
            basis = f"‘{goal}’ 목표" if goal else "다리 비율"
            text = f"{basis}에 맞춰 허리선과 세로선이 길게 이어지는 '{quoted(proportion)}' 디자인을 우선했습니다."
            evidence.append(ProductEvidence(
                "proportion", "비율", text, ("R-BOD-05",), basis,
                tuple(proportion), "user_input" if goal else "photo_analysis",
            ))
            cited.update(proportion)

    fit_attributes = {"fit", "length", "waistline", "structure", "silhouette"}
    fit_keywords = [
        keyword for keyword in matched
        if keyword not in cited and attribute_of(keyword) in fit_attributes and rules_of(keyword)
    ]
    if category != "shoes" and fit_keywords:
        fit_rules = tuple(dict.fromkeys(rule for keyword in fit_keywords for rule in rules_of(keyword)))
        fit_source = source_of(attribute_of(fit_keywords[0]))
        text = f"추천 규칙에서 도출된 '{quoted(fit_keywords)}' 핏이 상품명과 일치합니다."
        evidence.append(ProductEvidence(
            "fit", "핏", text, fit_rules, "핏 규칙", tuple(fit_keywords), fit_source,
        ))
        cited.update(fit_keywords)

    item_keywords = [
        keyword for keyword in matched
        if keyword not in cited and attribute_of(keyword) == "item_type" and rules_of(keyword)
    ]
    if item_keywords:
        item_rules = tuple(dict.fromkeys(rule for keyword in item_keywords for rule in rules_of(keyword)))
        evidence.append(ProductEvidence(
            "item_type", "종류",
            f"추천 조건에서 도출된 '{quoted(item_keywords)}' 종류가 상품명과 일치합니다.",
            item_rules, "상품 종류", tuple(item_keywords), source_of("item_type"),
        ))
        cited.update(item_keywords)

    style_keywords = [keyword for keyword in matched if keyword not in cited and attribute_of(keyword) == "style"]
    if style_keywords:
        style_source = source_of("style")
        prefix = "선택한" if style_source == "user_input" else "현재 착장에서 확인된"
        evidence.append(ProductEvidence(
            "style", "스타일",
            f"{prefix} '{style_keywords[0]}' 스타일 키워드가 상품명과 일치합니다.",
            tuple(rules_of(style_keywords[0])), "스타일", tuple(style_keywords), style_source,
        ))
        cited.update(style_keywords)
    details = [(keyword, attribute_of(keyword)) for keyword in matched if keyword not in cited and attribute_of(keyword) in {"material", "color", "function"}]
    if details:
        labels = {"material": "소재", "color": "색상", "function": "기능"}
        names = "·".join(keyword if attr == "function" else f"{keyword} {labels[attr]}" for keyword, attr in details)
        detail_sources = [source_of(attribute) for _, attribute in details]
        prefix = "현재 착장에서 확인된" if detail_sources and all(source == "photo_fallback" for source in detail_sources) else "선택한"
        evidence.append(ProductEvidence(
            "detail", "·".join(dict.fromkeys(labels[attr] for _, attr in details)),
            f"{prefix} {names} 조건이 상품명과 일치합니다.",
            tuple(dict.fromkeys(rule for keyword, _ in details for rule in rules_of(keyword))),
            "상품 속성", tuple(keyword for keyword, _ in details),
            ",".join(dict.fromkeys(detail_sources)),
        ))
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


def _fallback_reason(evidence: Iterable[ProductEvidence] = ()) -> str:
    """검증된 첫 근거만 사용한다. 근거가 없으면 설명을 지어내지 않는다."""
    evidence = list(evidence)
    return evidence[0].text if evidence else ""


def _output_text(response: dict[str, Any]) -> str:
    for item in response.get("output", []):
        for content in item.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                return str(content["text"])
    return ""


def _llm_reasons(
    products: list[Any],
    evidence_by_product: dict[str, list[ProductEvidence]],
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

    allowed_by_product: dict[str, dict[str, ProductEvidence]] = {}
    product_context = []
    for product in products:
        items = evidence_by_product.get(product.product_id, [])[:MAX_EVIDENCE]
        if not items:
            continue
        allowed: dict[str, ProductEvidence] = {}
        payload_items = []
        for index, evidence in enumerate(items, 1):
            evidence_id = f"{product.product_id}-E{index}"
            allowed[evidence_id] = evidence
            payload_items.append({
                "evidence_id": evidence_id,
                "label": evidence.label,
                "fact": evidence.text,
                "matched_keywords": list(evidence.keywords),
                "rule_ids": list(evidence.rule_ids),
                "source": evidence.source,
            })
        allowed_by_product[product.product_id] = allowed
        product_context.append({
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "allowed_evidence": payload_items,
        })
    if not product_context:
        return {}
    input_payload = {"products": product_context}
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "product_id": {"type": "string"},
                        "summary": {"type": "string"},
                        "evidence_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "maxItems": MAX_EVIDENCE,
                        },
                    },
                    "required": ["product_id", "summary", "evidence_ids"],
                    "additionalProperties": False,
                },
            }
        },
        "required": ["items"],
        "additionalProperties": False,
    }
    instructions = (
        "당신은 FITTA가 이미 검증한 상품 추천 근거를 자연스러운 한국어로 표현하는 편집자입니다. "
        "추천 여부와 근거는 서버가 결정했으므로 새로운 근거를 판단하거나 추가하지 마세요. "
        "각 상품의 allowed_evidence에 있는 사실과 matched_keywords만 사용하고, 사용한 근거의 "
        "evidence_id를 evidence_ids에 1~3개 반환하세요. 상품명에 매칭되지 않은 속성, 사용자 목적, "
        "예산·가격·할인·가성비, 실제 신체 치수, 상품 실측, 사이즈, 확인되지 않은 착용감·소재·기능·효과를 "
        "언급하지 마세요. 체형 근거가 제공되지 않았다면 체형이나 신체 특징을 말하지 말고, 신발에는 체형·다리 "
        "보정 표현을 쓰지 마세요. 입력 근거가 하나면 하나만 설명하고, 근거가 없는 상품은 출력하지 마세요. "
        "matched_keywords의 표현을 최소 하나 포함해 친절한 옷가게 점원의 한 문장으로 작성하되 해시태그와 점수는 "
        "쓰지 말고 120자 이내로 작성하세요."
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
        reason = str(item.get("summary", "")).strip()[:180]
        evidence_ids = item.get("evidence_ids", [])
        product = products_by_id.get(product_id)
        allowed = allowed_by_product.get(product_id, {})
        if (
            not product or not reason or not isinstance(evidence_ids, list)
            or not 1 <= len(evidence_ids) <= MAX_EVIDENCE
            or len(set(evidence_ids)) != len(evidence_ids)
            or any(not isinstance(evidence_id, str) or evidence_id not in allowed for evidence_id in evidence_ids)
        ):
            continue
        selected = [allowed[evidence_id] for evidence_id in evidence_ids]
        if any(word in reason for word in FORBIDDEN_REASON_LANGUAGE + UNSUPPORTED_PURPOSE_LANGUAGE):
            continue
        has_body_evidence = any(evidence.kind in {"body_shape", "proportion"} for evidence in selected)
        if (product.category == "shoes" or not has_body_evidence) and any(word in reason for word in BODY_LANGUAGE):
            continue
        normalized_reason = "".join(reason.lower().split())
        matched_keywords = list(dict.fromkeys(
            keyword for evidence in selected for keyword in evidence.keywords if keyword
        ))
        if not matched_keywords or not any("".join(keyword.lower().split()) in normalized_reason for keyword in matched_keywords):
            continue
        accepted[product_id] = reason
    return accepted


def add_product_recommendation_reasons(
    products: Iterable[Any],
    profile: UserProfile,
    pose: PoseAnalysis,
    targets: TargetKeywordResult,
    *,
    use_llm: bool = True,
) -> None:
    """검색 상품 객체에 항상 표시 가능한 추천 이유와 생성 출처를 붙인다."""
    product_list = list(products)
    evidence = {
        product.product_id: build_product_evidence(product, profile, pose, targets)
        for product in product_list
    }
    fallbacks = {
        product.product_id: _fallback_reason(evidence[product.product_id])
        for product in product_list
    }
    generated = _llm_reasons(product_list, evidence) if use_llm else {}
    for product in product_list:
        product.recommendation_reason = generated.get(product.product_id, fallbacks[product.product_id])
        product.recommendation_reason_source = "llm" if product.product_id in generated else "rules"
        items = evidence[product.product_id]
        product.fit_evidence = [item.text for item in items]
        product.fit_evidence_labels = [item.label for item in items]
        product.reason_rule_ids = list(dict.fromkeys(rule for item in items for rule in item.rule_ids))
