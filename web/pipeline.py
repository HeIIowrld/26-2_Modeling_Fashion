"""main.ipynb의 실행 순서를 웹 요청에서 재사용할 수 있게 감싼 래퍼."""

from __future__ import annotations

import sys
import threading
import inspect
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parents[1] / "ai_fashion_recommender"
# 런타임 모듈은 src/에 모여 있고 평면 임포트를 유지한다(커밋 014b384).
for _path in (PROJECT_DIR, PROJECT_DIR / "src"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from clothing_parser import ClothingParser
from config import (
    DATA_DIR,
    ENABLE_VTON,
    FASHION_ATTRIBUTE_HEADS_PATH,
    OUTPUT_DIR,
    resolve_catalog,
)
from fashion_model import FashionClassifier
from fashion_prompts import STYLE_PROMPTS
from feedback_store import FeedbackStore
from outfit_analyzer import COLOR_PALETTE, OutfitAnalyzer, _dominant_palette
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import CHANGE_SCOPE_MAP, PURPOSE_STYLES, RecommendationEngine
from musinsa_live_search import MusinsaLiveSearch
from body_shape import classify
from schemas import GOAL_NONE, SILHOUETTE_GOAL_CHOICES, UserProfile, WardrobeItem
from virtual_tryon import TryOnNotReady, VirtualTryOnAdapter

RULES_PATH = PROJECT_DIR / "FASHION_RULES_MASTER.md"
ATTRIBUTE_HEADS_PATH = FASHION_ATTRIBUTE_HEADS_PATH

PURPOSES = list(PURPOSE_STYLES)
STYLES = [
    "캐주얼",
    "미니멀",
    {"value": "스트리트", "label": "스트릿"},
    {"value": "포멀", "label": "클래식"},
    "스포티",
    "기타",
]
CHANGE_SCOPES = [scope for scope in CHANGE_SCOPE_MAP if scope != "현재 유지"]
SEASONS = [
    {"value": "자동", "label": "자동 (현재 계절)"},
    "봄",
    "여름",
    "가을",
    "겨울",
]
SILHOUETTE_GOALS = [{"value": value, "label": label} for value, label in SILHOUETTE_GOAL_CHOICES]
DRESS_CODES = ["자동", "캐주얼", "스마트 캐주얼", "비즈니스 캐주얼", "포멀"]
ACTIVITY_LEVELS = [
    {"value": "낮음", "label": "적음"},
    "보통",
    {"value": "높음", "label": "많음"},
]
MATERIALS = ["면·일상 소재", "데님", "니트", "얇은 소재", "가죽"]
MATERIAL_PREFERENCE_MAP = {
    "면·일상 소재": ["코튼", "폴리에스터"],
    "데님": ["데님"],
    "니트": ["니트"],
    "얇은 소재": ["린넨", "쉬폰"],
    "가죽": ["가죽"],
}

STAGES = [
    ("prepare", "GPU 모델·상품 데이터 준비"),
    ("wardrobe", "보유 옷 사진 확인"),
    ("pose", "전신 관절·자세 찾기"),
    ("quality", "해상도·선명도 검사"),
    ("body", "체형·실루엣 비율 계산"),
    ("segment", "상의·하의 영역 분리"),
    ("attributes", "색상·핏·소재 인식"),
    ("candidates", "추천 키워드 생성"),
    ("scoring", "무신사 실시간 상품 검색"),
    ("preview", "검색 결과 카드 준비"),
    ("finalize", "추천 결과 정리"),
]


__all__ = [
    "STAGES",
    "PipelineError",
    "PipelineResult",
    "TryOnNotReady",
    "analyze_wardrobe_items",
    "build_profile",
    "form_options",
    "generate_tryon",
    "get_engine",
    "run_pipeline",
    "rule_titles",
    "save_feedback",
    "tryon_status",
]


class PipelineError(RuntimeError):
    """사용자에게 그대로 보여줄 수 있는 실패 사유."""


@dataclass
class PipelineResult:
    """화면에 보낼 JSON과, 이후 예상 착장샷 생성에 필요한 원본 객체를 함께 담는다."""

    payload: dict
    recommendations: list
    person_image: Path
    tryon_context: dict = field(default_factory=dict)


@dataclass
class Engine:
    """무거운 체크포인트를 프로세스당 한 번만 올려 요청 간에 재사용한다."""

    pose_analyzer: PoseAnalyzer
    quality_checker: QualityChecker
    outfit_analyzer: OutfitAnalyzer
    recommender: RecommendationEngine
    product_search: MusinsaLiveSearch
    tryon: VirtualTryOnAdapter
    device: str
    trained_heads: bool
    parser_backend: str


_engine: Engine | None = None
_engine_lock = threading.Lock()
# MediaPipe·SegFormer·CatVTON 세션은 동시 호출에 안전하지 않다. 분석 도중의
# 자동 미리보기와 결과 화면의 추가 합성이 겹치지 않도록 하나의 재진입 잠금을 쓴다.
_analysis_lock = threading.RLock()


def get_engine() -> Engine:
    global _engine
    with _engine_lock:
        if _engine is None:
            _engine = _build_engine()
        return _engine


def _build_tryon() -> VirtualTryOnAdapter:
    """생성 모델을 쓸 수 있으면 CatVTON을, 아니면 비활성 어댑터를 준다.

    config.ENABLE_VTON 이 꺼져 있으면 아예 시도하지 않는다. 켜져 있어도 CatVTON
    저장소나 GPU가 없는 환경이 있으므로, 실패하면 조용히 비활성으로 떨어지고
    이유를 남긴다. 여기서 예외를 올리면 웹 서버 자체가 안 뜬다.
    """
    if not ENABLE_VTON:
        return VirtualTryOnAdapter(enabled=False)
    try:
        from catvton_tryon import CatVTONTryOn

        return CatVTONTryOn()
    except Exception as error:  # 저장소 없음·의존성 없음·GPU 없음 모두 여기로 온다
        print(f"[VTON] 생성 모델을 켜지 못해 비활성으로 실행합니다: {type(error).__name__}: {error}")
        return VirtualTryOnAdapter(enabled=False)


def _build_engine() -> Engine:
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    clothing_parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(
        enabled=True,
        attribute_checkpoint=ATTRIBUTE_HEADS_PATH if ATTRIBUTE_HEADS_PATH.is_file() else None,
    )
    # 어떤 CSV를 쓸지는 config.resolve_catalog 한 곳에서 정한다.
    # 상품 사진이 있는 크롤링 카탈로그(products_musinsa_enriched.csv)가 있으면
    # 그쪽을 쓰고, 없으면 손으로 만든 products.csv 로 떨어진다.
    catalog = ProductCatalog(resolve_catalog(DATA_DIR))
    return Engine(
        pose_analyzer=pose_analyzer,
        quality_checker=QualityChecker(pose_analyzer),
        outfit_analyzer=OutfitAnalyzer(clothing_parser, classifier),
        recommender=RecommendationEngine(RULES_PATH, catalog),
        product_search=MusinsaLiveSearch(),
        tryon=_build_tryon(),
        device=classifier.device,
        trained_heads=classifier.trained_attributes_enabled,
        parser_backend=clothing_parser.backend,
    )


def analyze_wardrobe_items(profile: UserProfile, image_paths: list[Path]) -> None:
    """보유 옷 사진에서 사용자가 고르지 않은 색상과 스타일을 채운다.

    상품 단독 사진은 전신 포즈가 없으므로 의류 파서를 쓰지 않는다. 가장자리의
    배경색과 다른 영역을 옷으로 보고 대표색을 구하고, 스타일은 웹 분석과 같은
    FashionSigLIP 분류기를 재사용한다.
    """
    if not profile.owned_items or not image_paths:
        return
    classifier = get_engine().outfit_analyzer.classifier
    for item, image_path in zip(profile.owned_items, image_paths):
        with Image.open(image_path) as opened:
            rgb = np.asarray(opened.convert("RGB"))
        height, width = rgb.shape[:2]
        edge = max(1, min(height, width) // 20)
        corners = np.concatenate(
            (
                rgb[:edge, :edge].reshape(-1, 3),
                rgb[:edge, -edge:].reshape(-1, 3),
                rgb[-edge:, :edge].reshape(-1, 3),
                rgb[-edge:, -edge:].reshape(-1, 3),
            ),
            axis=0,
        )
        background = np.median(corners, axis=0)
        distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
        garment_mask = distance >= 24
        if float(garment_mask.mean()) < 0.04:
            garment_mask = np.ones((height, width), dtype=bool)
        item.color = _dominant_palette(rgb, garment_mask, max_colors=1)[0]["name"]
        item.style = classifier.best_mapped_label(image_path, STYLE_PROMPTS)[0]


def build_profile(payload: dict) -> UserProfile:
    def number(key: str) -> float | None:
        value = payload.get(key)
        if value in (None, ""):
            return None
        return float(value)

    def string_list(key: str) -> list[str]:
        value = payload.get(key) or []
        return [str(item).strip() for item in value if str(item).strip()]

    def current_season() -> str:
        month = datetime.now().month
        if month in (3, 4, 5):
            return "봄"
        if month in (6, 7, 8):
            return "여름"
        if month in (9, 10, 11):
            return "가을"
        return "겨울"

    selected_materials = string_list("preferred_materials")
    preferred_materials = list(dict.fromkeys(
        model_label
        for selected in selected_materials
        for model_label in MATERIAL_PREFERENCE_MAP.get(selected, [selected])
    ))

    owned_items = [
        WardrobeItem(
            item_id=str(item.get("item_id") or f"OWN-{index:02d}"),
            category=str(item.get("category") or "top"),
            color=str(item.get("color") or ""),
            style=str(item.get("style") or ""),
            season=str(item.get("season") or "사계절"),
        )
        for index, item in enumerate(payload.get("owned_items") or [], start=1)
        if str(item.get("color") or "").strip() or item.get("image_index") is not None
    ]

    # budget: support min_budget/max_budget from the UI while keeping backward compatibility
    def _to_int_or_none(val):
        try:
            if val is None or str(val) == "":
                return None
            return int(float(val))
        except Exception:
            return None

    min_b = _to_int_or_none(payload.get("min_budget"))
    max_b = _to_int_or_none(payload.get("max_budget"))
    if min_b is not None and max_b is not None:
        budget_val = int((min_b + max_b) / 2)
    else:
        budget_val = _to_int_or_none(payload.get("budget")) or 150_000

    return UserProfile(
        purpose=payload.get("purpose") or "데일리",
        desired_style=payload.get("desired_style") or "캐주얼",
        budget=budget_val,
        min_budget=min_b,
        max_budget=max_b,
        change_scope=payload.get("change_scope") or "전체 변경",
        height_cm=number("height_cm"),
        weight_kg=number("weight_kg"),
        chest_cm=number("chest_cm"),
        waist_cm=number("waist_cm"),
        hip_cm=number("hip_cm"),
        usual_top_size=payload.get("usual_top_size"),
        usual_bottom_size=payload.get("usual_bottom_size"),
        season=(
            current_season()
            if payload.get("season") in (None, "", "자동")
            else payload.get("season")
        ),
        silhouette_goal=payload.get("silhouette_goal") or GOAL_NONE,
        dress_code=payload.get("dress_code") or "자동",
        activity_level=payload.get("activity_level") or "보통",
        preferred_colors=string_list("preferred_colors"),
        avoided_colors=string_list("avoided_colors"),
        preferred_materials=preferred_materials,
        avoided_materials=string_list("avoided_materials"),
        excluded_item_types=string_list("excluded_item_types"),
        temperature_c=number("temperature_c"),
        feels_like_c=number("feels_like_c"),
        humidity=number("humidity"),
        precipitation_probability=number("precipitation_probability"),
        wind_mps=number("wind_mps"),
        uv_index=number("uv_index"),
        owned_items=owned_items,
        provided_fields=[
            key for key, value in payload.items()
            if value not in (None, "", [], {})
        ],
    )


def run_pipeline(
    image_path: Path,
    profile: UserProfile,
    output_dir: Path,
    on_stage: Callable[[str], None],
    body_image_path: Path | None = None,
) -> PipelineResult:
    """main.ipynb의 2~7번 셀과 같은 순서로 실행하고 JSON 직렬화 가능한 결과를 만든다."""
    engine = get_engine()
    output_dir.mkdir(parents=True, exist_ok=True)

    with _analysis_lock:
        on_stage("pose")
        pose_result = engine.pose_analyzer.analyze(image_path)
        on_stage("quality")
        input_quality = engine.quality_checker.check_input(image_path, pose=pose_result)
        if not input_quality["passed"]:
            raise PipelineError(
                "전신사진 품질 기준을 통과하지 못했습니다: "
                + " / ".join(input_quality["issues"])
            )

        if not pose_result.valid:
            raise PipelineError("유효한 전신 포즈를 찾지 못했습니다. 정면 전신사진을 사용하세요.")
        landmark_path = output_dir / "pose_landmarks.jpg"
        engine.pose_analyzer.draw_landmarks(image_path, analysis=pose_result).save(
            landmark_path, quality=92
        )
        # 둘레를 입력했으면 사진 추정보다 정확하므로 그 값으로 덮어쓴다.
        # 여기서 확정해야 추천 엔진과 화면이 같은 체형을 본다.
        # 체형 파악용 사진이 있으면 그쪽을 쓴다. 몸이 드러날수록 실루엣 폭이 정확하다.
        on_stage("body")
        pose_result.body_shape, body_shape_basis = classify(
            profile, pose_result, person_image=body_image_path or image_path
        )

        analyze_outfit = engine.outfit_analyzer.analyze
        if "on_stage" in inspect.signature(analyze_outfit).parameters:
            outfit_result, parsed = analyze_outfit(image_path, pose_result, on_stage=on_stage)
        else:  # 간단한 테스트 대역·구버전 어댑터 호환
            on_stage("segment")
            outfit_result, parsed = analyze_outfit(image_path, pose_result)
            on_stage("attributes")
        segmentation_path = output_dir / "segmentation.jpg"
        engine.outfit_analyzer.parser.colorize(parsed["segmentation"]).save(
            segmentation_path, quality=92
        )

        # CSV 카탈로그 추천은 더 이상 실행하지 않는다. 사진·사용자 조건에서 만든
        # 키워드를 곧바로 무신사 실시간 검색에 전달한다.
        on_stage("candidates")
        target_keywords = engine.recommender.generate_target_keywords(profile, pose_result, outfit_result)
        on_stage("scoring")
        product_search = getattr(engine, "product_search", None)
        shopping_results = []
        if product_search is not None:
            try:
                shopping_results = product_search.search(
                    target_keywords,
                    profile,
                    limit=3,
                )
            except Exception as exc:  # 외부 검색 장애가 본 분석까지 실패시키지 않게 격리한다.
                print(f"[MUSINSA] live search unavailable: {exc}")

        on_stage("preview")

    on_stage("finalize")
    pose_dict = pose_result.to_dict()
    pose_dict.pop("landmarks", None)
    pose_dict["body_shape_basis"] = body_shape_basis
    payload = {
        "input_quality": input_quality,
        "pose": pose_dict,
        "outfit": outfit_result.to_dict(),
        "outfit_summary": outfit_result.to_summary_dict(),
        "shopping_results": [item.public_dict() for item in shopping_results],
        "rules": {
            "implemented": len(engine.recommender.active_rule_ids),
            "documented": len(engine.recommender.documented_rule_ids),
            "scoring": len(engine.recommender.scoring_rule_ids),
            "unsupported": [
                {"id": rule_id, "reason": engine.recommender.UNSUPPORTED_RULE_REASONS[rule_id]}
                for rule_id in engine.recommender.unsupported_rule_ids
            ],
        },
        "engine": {
            "device": engine.device,
            "trained_heads": engine.trained_heads,
            "parser_backend": engine.parser_backend,
            "vton_enabled": engine.tryon.enabled,
            "product_color_audits": len(
                getattr(getattr(engine.recommender, "catalog", None), "color_audits", {})
            ),
            "product_color_overrides": getattr(
                getattr(engine.recommender, "catalog", None), "color_override_count", 0
            ),
        },
        "request": _request_summary(profile),
        "images": {
            "original": "original.jpg",
            "landmarks": landmark_path.name,
            "segmentation": segmentation_path.name,
        },
    }
    return PipelineResult(
        payload=payload,
        recommendations=[],
        person_image=image_path,
    )


def tryon_status() -> dict:
    """예상 착장샷 생성이 가능한지와, 불가능하면 그 사유를 알려준다."""
    adapter = get_engine().tryon
    return {
        "available": adapter.available,
        "reason": "" if adapter.available else adapter.NOT_READY_REASON,
        "warnings": list(getattr(adapter, "last_warnings", [])),
    }


def generate_tryon(
    person_image: Path,
    recommendation,
    output_path: Path,
    context: dict | None = None,
) -> Path:
    """추천 코디 하나에 대한 예상 착장샷을 만든다.

    생성 모델이 없으면 추천 보드로 몰래 대체하지 않고 TryOnNotReady를 올린다.
    """
    if not recommendation.products:
        raise TryOnNotReady("현재 코디를 유지하는 조건이라 새로 생성할 착장샷이 없습니다.")
    with _analysis_lock:
        return get_engine().tryon.synthesize(
            person_image=person_image,
            recommendation=recommendation,
            output_path=output_path,
            context=context,
        )


def _recommendation_dict(recommendation) -> dict:
    data = recommendation.to_dict()
    data["products"] = [
        {
            "product_id": product.product_id,
            "name": product.name,
            "category": product.category,
            "color": product.color,
            "color_rgb": list(COLOR_PALETTE.get(product.color, (160, 160, 160))),
            "style": product.style,
            "price": product.price,
            "season": product.season,
            "url": product.url,
            "item_type": product.item_type,
            "fit": product.fit,
            "length": product.length,
            "pattern": product.pattern,
            "material": product.material,
            "neckline": product.neckline,
            "formality": product.formality,
            "catalog_color": product.catalog_color or product.color,
            "image_color": product.image_color,
            "image_color_confidence": product.image_color_confidence,
            "color_source": product.color_source,
        }
        for product in recommendation.products
    ]
    return data


def _request_summary(profile: UserProfile) -> dict:
    """사용자가 고른 조건이 결과에 어떻게 넘어갔는지 화면·QA에서 확인한다."""
    return {
        "purpose": profile.purpose,
        "desired_style": profile.desired_style,
        "change_scope": profile.change_scope,
        "min_budget": profile.min_budget,
        "max_budget": profile.max_budget,
        "season": profile.season,
        "activity_level": profile.activity_level,
        "preferred_colors": list(profile.preferred_colors),
        "avoided_colors": list(profile.avoided_colors),
        "preferred_materials": list(profile.preferred_materials),
    }


def rule_titles() -> dict[str, str]:
    rules = get_engine().recommender.rule_book.rules
    return {
        rule_id: rule.title
        for rule_id, rule in rules.items()
    }


def save_feedback(rank: int, action: str, note: str = "") -> dict:
    return FeedbackStore(OUTPUT_DIR / "feedback.jsonl").append(rank, action, note)


def form_options() -> dict:
    return {
        "purposes": PURPOSES,
        "styles": STYLES,
        "change_scopes": CHANGE_SCOPES,
        "seasons": SEASONS,
        "silhouette_goals": SILHOUETTE_GOALS,
        "dress_codes": DRESS_CODES,
        "activity_levels": ACTIVITY_LEVELS,
        "colors": [
            {"name": name, "rgb": list(rgb)} for name, rgb in COLOR_PALETTE.items()
        ],
        "materials": MATERIALS,
        "stages": [{"key": key, "label": label} for key, label in STAGES],
    }
