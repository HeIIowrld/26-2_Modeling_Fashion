"""웹 결과에 표시할 짧고 개인화된 추천 설명을 만든다.

기본 설명은 Fashion Rule로 만든 검색 키워드와 사용자 조건만 사용해 항상
생성한다. 운영자가 명시적으로 LLM을 켠 경우에는 사진 원본이 아닌 구조화된
분석값만 OpenAI Responses API에 보내 자연스러운 한 문장으로 다듬는다.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Iterable

from recommendation_keywords import TargetKeywordResult
from schemas import CurrentOutfitEvaluation, GOAL_NONE, OutfitAnalysis, PoseAnalysis, UserProfile


OPENAI_RESPONSES_URL = "https://api.openai.com/v1/responses"
GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
LLM_ENABLED_VALUES = {"1", "true", "yes", "on"}
CATEGORY_LABELS = {"top": "상의", "bottom": "하의"}
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


def _fallback_reason(product: Any, profile: UserProfile, pose: PoseAnalysis) -> str:
    keywords = list(getattr(product, "search_keywords", []) or [])[:3]
    keyword_copy = "·".join(keywords) or CATEGORY_LABELS.get(product.category, "상품")
    contexts = [value for value in (profile.purpose, profile.desired_style) if value and value != "자동"]
    context_copy = "·".join(dict.fromkeys(contexts)) or "입력한 코디"

    if profile.silhouette_goal and profile.silhouette_goal != GOAL_NONE:
        body_copy = f"‘{profile.silhouette_goal}’ 실루엣 목표"
    elif pose.body_shape_confidence >= 0.65 and "불" not in pose.body_shape:
        body_copy = f"분석된 {pose.body_shape} 실루엣"
    else:
        body_copy = "사진에서 확인한 현재 실루엣"

    budget_copy = "예산 범위에도 들어오는 " if profile.min_budget is not None or profile.max_budget is not None else ""
    category = CATEGORY_LABELS.get(product.category, "아이템")
    return (
        f"{context_copy} 조건의 {keyword_copy} 기준과 {body_copy}까지 함께 고려했고, "
        f"{budget_copy}{category}라 추천했어요."
    )


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
            "price": product.price,
            "search_keywords": list(product.search_keywords),
            "fact_based_draft": fallbacks[product.product_id],
        }
        for product in products
    ]
    input_payload = {
        "user_filters": {
            "purpose": profile.purpose,
            "desired_style": profile.desired_style,
            "silhouette_goal": profile.silhouette_goal,
            "season": profile.season,
            "activity_level": profile.activity_level,
            "preferred_colors": profile.preferred_colors,
            "preferred_materials": profile.preferred_materials,
            "budget": [profile.min_budget, profile.max_budget or profile.budget],
        },
        "body_analysis": {
            "shape": pose.body_shape,
            "confidence": pose.body_shape_confidence,
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
        "당신은 한국 패션 추천 서비스 FITTA의 카피라이터입니다. 제공된 사실만 사용해 "
        "각 상품의 추천 이유를 친근한 한국어 한 문장(최대 100자)으로 쓰세요. "
        "사용자 필터, 체형 또는 실루엣 목표, 상품 키워드 중 최소 두 가지를 연결하고 "
        "측정되지 않은 효과나 상품 속성은 만들지 마세요. 해시태그와 점수는 쓰지 마세요."
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

    known_ids = set(fallbacks)
    return {
        str(item.get("product_id")): str(item.get("reason", "")).strip()[:180]
        for item in generated.get("items", [])
        if str(item.get("product_id")) in known_ids and str(item.get("reason", "")).strip()
    }


def add_product_recommendation_reasons(
    products: Iterable[Any],
    profile: UserProfile,
    pose: PoseAnalysis,
    targets: TargetKeywordResult,
) -> None:
    """검색 상품 객체에 항상 표시 가능한 추천 이유와 생성 출처를 붙인다."""
    product_list = list(products)
    fallbacks = {
        product.product_id: _fallback_reason(product, profile, pose)
        for product in product_list
    }
    generated = _llm_reasons(product_list, profile, pose, targets, fallbacks)
    for product in product_list:
        product.recommendation_reason = generated.get(product.product_id, fallbacks[product.product_id])
        product.recommendation_reason_source = "llm" if product.product_id in generated else "rules"
