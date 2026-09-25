"""main.ipynb의 실행 순서를 웹 요청에서 재사용할 수 있게 감싼 래퍼."""

from __future__ import annotations

import hashlib
import io
import os
import sys
import threading
import inspect
import urllib.parse
import urllib.request
import uuid
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, UnidentifiedImageError

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
    FASHION_FIT_VISION_HEADS_PATH,
    OUTPUT_DIR,
    garment_image_path,
    resolve_catalog,
)
from fashion_model import FashionClassifier
from fashion_prompts import STYLE_PROMPTS
from recommendation_explanations import (
    add_product_recommendation_reasons,
    build_outfit_summary_points,
)
from outfit_combination_recommender import recommend_outfit_combinations
from feedback_store import FeedbackStore
from outfit_analyzer import COLOR_PALETTE, OutfitAnalyzer, _dominant_palette
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import CHANGE_SCOPE_MAP, PURPOSE_STYLES, RecommendationEngine
from musinsa_live_search import MusinsaLiveSearch
from live_product_attributes import LiveProductAttributes
from product_measurements import ProductMeasurementClient
from size_fit import validate_references
from body_shape import classify
from body_visibility import body_shape_analysis_allowed, with_body_visibility
from schemas import (
    BASIS_PHOTO,
    GOAL_NONE,
    SHAPE_UNCERTAIN,
    SILHOUETTE_GOAL_CHOICES,
    Product,
    UserProfile,
    WardrobeItem,
)
from virtual_tryon import TryOnNotReady, VirtualTryOnAdapter
from catvton_tryon import (
    USER_BOTTOM_LENGTH_OPTIONS,
    bottom_length_warnings,
    current_bottom_length,
    is_bottom_length_warning,
)

RULES_PATH = PROJECT_DIR / "FASHION_RULES_MASTER.md"
ATTRIBUTE_HEADS_PATH = FASHION_ATTRIBUTE_HEADS_PATH
FIT_VISION_HEADS_PATH = FASHION_FIT_VISION_HEADS_PATH

PURPOSES = list(PURPOSE_STYLES)
GENDERS = ["남성", "여성"]
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
TRYON_PRODUCT_CATEGORIES = {"top", "bottom"}
MAX_SHOPPING_IMAGE_BYTES = 12 * 1024 * 1024
MAX_SHOPPING_IMAGE_PIXELS = 30_000_000
SHOPPING_IMAGE_HOST_SUFFIX = ".msscdn.net"

STAGES = [
    ("prepare", "GPU 모델·상품 데이터 준비"),
    ("wardrobe", "보유 옷 사진 확인"),
    ("pose", "전신 관절·자세 찾기"),
    ("quality", "해상도·선명도 검사"),
    ("segment", "상의·하의 영역 분리"),
    ("attributes", "색상·핏·소재 인식"),
    ("body", "체형·실루엣 비율 계산"),
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
    "generate_tryon_with_warnings",
    "get_engine",
    "length_check_status",
    "reference_bottom_lengths",
    "refresh_bottom_length_warnings",
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
    shopping_tryon_products: dict[str, Product] = field(default_factory=dict)


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

        preset = os.environ.get("FASHION_VTON_PRESET", "standard").strip().lower()
        if preset == "fast":
            adapter = CatVTONTryOn.fast()
        elif preset in {"high", "high_detail"}:
            adapter = CatVTONTryOn.high_detail()
        else:
            if preset not in {"", "standard"}:
                print(f"[VTON] 알 수 없는 프리셋 {preset!r}; standard를 사용합니다.")
            adapter = CatVTONTryOn()
        shoe_model = os.environ.get("FASHION_SHOE_MODEL_PATH", "").strip()
        if shoe_model:
            from shoe_tryon import OutfitTryOn, ShoeTryOn
            editor = ShoeTryOn(shoe_model)
            adapter.transition_editor = editor
            return OutfitTryOn(adapter, editor)
        return adapter
    except Exception as error:  # 저장소 없음·의존성 없음·GPU 없음 모두 여기로 온다
        print(f"[VTON] 생성 모델을 켜지 못해 비활성으로 실행합니다: {type(error).__name__}: {error}")
        return VirtualTryOnAdapter(enabled=False)


def _build_engine() -> Engine:
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    clothing_parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(
        enabled=True,
        attribute_checkpoint=ATTRIBUTE_HEADS_PATH if ATTRIBUTE_HEADS_PATH.is_file() else None,
        fit_checkpoint=FIT_VISION_HEADS_PATH if FIT_VISION_HEADS_PATH.is_file() else None,
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
        product_search=MusinsaLiveSearch(
            measurements=ProductMeasurementClient(DATA_DIR / "cache" / "product_measurements"),
            photo_provider=LiveProductAttributes(
                clothing_parser,
                classifier.attribute_predictor,
                fit_predictor=classifier.fit_predictor,
            )
            if (classifier.trained_attributes_enabled or classifier.trained_fit_enabled) else None),
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
        change_categories=payload.get("change_categories"),
        gender=str(payload.get("gender") or ""),
        height_cm=number("height_cm"),
        weight_kg=number("weight_kg"),
        chest_cm=number("chest_cm"),
        waist_cm=number("waist_cm"),
        hip_cm=number("hip_cm"),
        usual_top_size=payload.get("usual_top_size"),
        usual_bottom_size=payload.get("usual_bottom_size"),
        reference_measurements=validate_references(payload.get("reference_measurements")),
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


def _shopping_image_host_allowed(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    hostname = (parsed.hostname or "").lower()
    return parsed.scheme == "https" and (
        hostname == SHOPPING_IMAGE_HOST_SUFFIX.removeprefix(".")
        or hostname.endswith(SHOPPING_IMAGE_HOST_SUFFIX)
    )


def _cache_live_shopping_image(product, output_dir: Path, *, timeout: float = 6.0) -> Path | None:
    """무신사 API가 돌려준 공개 상품 이미지를 현재 세션에만 안전하게 저장한다."""
    if not _shopping_image_host_allowed(product.image_url):
        return None
    suffix = hashlib.sha256(f"{product.product_id}:{product.image_url}".encode("utf-8")).hexdigest()[:16]
    target = output_dir / f"shopping_{suffix}.png"
    if target.is_file():
        return target
    request = urllib.request.Request(
        product.image_url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; FITTA/1.0)"},
    )
    try:
        with urllib.request.urlopen(request, timeout=max(0.01, timeout)) as response:
            if not _shopping_image_host_allowed(response.geturl()):
                return None
            declared = int(response.headers.get("Content-Length") or 0)
            if declared > MAX_SHOPPING_IMAGE_BYTES:
                return None
            raw = response.read(MAX_SHOPPING_IMAGE_BYTES + 1)
        if not raw or len(raw) > MAX_SHOPPING_IMAGE_BYTES:
            return None
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.width * opened.height > MAX_SHOPPING_IMAGE_PIXELS:
                return None
            # Downloads can overlap; publish only a complete decoded image.
            temporary = target.with_name(f"{target.stem}.{uuid.uuid4().hex}.tmp")
            try:
                # Preserve decoded pixels: benchmark and live inference see the
                # same image, without another lossy JPEG compression pass.
                opened.convert("RGB").save(temporary, "PNG")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    except (OSError, ValueError, UnidentifiedImageError):
        return None
    return target


def _shopping_tryon_payloads(
    shopping_results: list,
    catalog_products: list[Product],
    output_dir: Path,
    *,
    adapter_available: bool,
    supported_categories: set[str] | None = None,
    shoe_unavailable_reason: str = "",
) -> tuple[list[dict], dict[str, Product]]:
    """검색 상품을 공개 응답과 실제 VTON에 쓸 수 있는 Product 객체로 나눈다."""
    catalog_by_id = {product.product_id: product for product in catalog_products}
    payloads: list[dict] = []
    tryon_products: dict[str, Product] = {}
    for item in shopping_results:
        payload = item.public_dict()
        resolved = catalog_by_id.get(item.product_id)
        reason = ""
        if item.category not in (supported_categories if supported_categories is not None else TRYON_PRODUCT_CATEGORIES):
            reason = "현재 실제 합성은 상의와 하의만 지원합니다."
            resolved = None
        elif item.category == "shoes" and shoe_unavailable_reason:
            reason = shoe_unavailable_reason
            resolved = None
        elif not adapter_available:
            reason = "현재 합성 GPU를 사용할 수 없습니다."
            resolved = None
        elif resolved is not None:
            path = garment_image_path(resolved.image_path) if resolved.image_path else None
            if path is None or not path.is_file():
                reason = "GPU 서버에 이 상품 이미지가 준비되지 않았습니다."
                resolved = None
        else:
            cached = _cache_live_shopping_image(item, output_dir)
            if cached is None:
                reason = "실시간 상품 이미지를 GPU 세션에 준비하지 못했습니다."
            else:
                resolved = Product(
                    product_id=item.product_id,
                    name=item.name,
                    category=item.category,
                    color="",
                    style="",
                    purposes=[],
                    body_shapes=[],
                    price=item.price,
                    season="사계절",
                    stock=True,
                    url=item.url,
                    brand=item.brand,
                    gender=item.gender,
                    image_url=item.image_url,
                    image_path=str(cached),
                )
        if resolved is not None:
            tryon_products[item.product_id] = resolved
        payload["tryon_available"] = resolved is not None
        payload["tryon_reason"] = reason
        payloads.append(payload)
    return payloads, tryon_products


def validate_input_photo(image_path: Path, engine=None) -> dict:
    """Share the input policy with preflight; serialize model access with analysis."""
    engine = engine or get_engine()
    with _analysis_lock:
        pose = engine.pose_analyzer.analyze(image_path)
        quality = engine.quality_checker.check_input(image_path, pose=pose)
        if not quality["passed"]:
            return quality
        outfit, parsed = engine.outfit_analyzer.analyze(image_path, pose)
        return with_body_visibility(quality, outfit, parsed)


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
        analyze_outfit = engine.outfit_analyzer.analyze
        if "on_stage" in inspect.signature(analyze_outfit).parameters:
            outfit_result, parsed = analyze_outfit(image_path, pose_result, on_stage=on_stage)
        else:  # 간단한 테스트 대역·구버전 어댑터 호환
            on_stage("segment")
            outfit_result, parsed = analyze_outfit(image_path, pose_result)
            on_stage("attributes")

        # Check before silhouette measurement, recommendations or VTON. Calling
        # /api/analyze directly must not bypass the upload preflight policy.
        input_quality = with_body_visibility(input_quality, outfit_result, parsed)
        if not input_quality["passed"]:
            raise PipelineError(" / ".join(input_quality["issues"]))
        if body_image_path is not None:
            body_pose = engine.pose_analyzer.analyze(body_image_path)
            body_quality = engine.quality_checker.check_input(body_image_path, pose=body_pose)
            if body_quality["passed"]:
                body_outfit, body_parsed = analyze_outfit(body_image_path, body_pose)
                body_quality = with_body_visibility(body_quality, body_outfit, body_parsed)
            if not body_quality["passed"]:
                raise PipelineError("체형 파악용 사진: " + " / ".join(body_quality["issues"]))
        on_stage("body")
        shape_pose = body_pose if body_image_path is not None else pose_result
        shape_quality = body_quality if body_image_path is not None else input_quality
        body_shape_reliable = body_shape_analysis_allowed(
            shape_quality,
            has_circumferences=bool(getattr(profile, "has_circumferences", False)),
        )
        if body_shape_reliable:
            pose_result.body_shape, body_shape_basis = classify(
                profile,
                shape_pose,
                person_image=body_image_path or image_path,
            )
        else:
            # 몸선을 가리는 옷은 입력을 막지 않되 사진 폭을 실제 체형으로 사용하지 않는다.
            pose_result.body_shape = SHAPE_UNCERTAIN
            pose_result.body_shape_confidence = 0.0
            body_shape_basis = BASIS_PHOTO

        # Fashion Rules의 기존 2×3 진단기를 현재 착장 결과에 다시 연결한다.
        current_outfit_evaluation = engine.recommender.evaluate_current_outfit(
            profile,
            pose_result,
            outfit_result,
        )
        # 구형 CSV 추천에는 쓰지 않지만, 사용자가 고른 무신사 상품을 VTON으로
        # 합성할 때 현재 착장의 마스크·분석 결과가 필요하다.
        tryon_context = {
            key: parsed.get(key)
            for key in (
                "upper_mask",
                "lower_mask",
                "upper_style_mask",
                "lower_style_mask",
                "segmentation",
            )
        }
        tryon_context.update(
            {
                "outfit": outfit_result,
                "classifier": getattr(engine.outfit_analyzer, "classifier", None),
                "pose": pose_result,
            }
        )
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
                    limit=6,
                    photo_loader=lambda product, timeout: _cache_live_shopping_image(
                        product, output_dir, timeout=timeout),
                )
            except Exception as exc:  # 외부 검색 장애가 본 분석까지 실패시키지 않게 격리한다.
                print(f"[MUSINSA] live search unavailable: {exc}")
        shopping_outfits = recommend_outfit_combinations(
            shopping_results,
            profile,
            pose_result,
            outfit_result,
            target_keywords,
            engine.recommender,
            limit=3,
        )
        if shopping_outfits:
            selected_ids = list(dict.fromkeys(
                product_id
                for combination in shopping_outfits
                for product_id in combination.product_ids
            ))
            by_id = {product.product_id: product for product in shopping_results}
            shopping_results = [by_id[product_id] for product_id in selected_ids if product_id in by_id]
        add_product_recommendation_reasons(
            shopping_results,
            profile,
            pose_result,
            target_keywords,
            # The LLM may only rephrase the verified rule/keyword evidence;
            # it cannot add purpose, budget, sizing or unsupported claims.
            use_llm=True,
        )
        supported_categories = set(getattr(engine.tryon, "supported_categories", TRYON_PRODUCT_CATEGORIES))
        shoe_reason = ""
        if "shoes" in supported_categories:
            from shoe_tryon import foot_edit_mask
            try:
                foot_edit_mask(parsed["segmentation"], pose_result)
            except TryOnNotReady as exc:
                shoe_reason = str(exc)
        shopping_payloads, shopping_tryon_products = _shopping_tryon_payloads(
            shopping_results,
            engine.recommender.catalog.products,
            output_dir,
            adapter_available=engine.tryon.available,
            supported_categories=supported_categories,
            shoe_unavailable_reason=shoe_reason,
        )
        shopping_payload_by_id = {
            item["product_id"]: item for item in shopping_payloads
        }
        shopping_outfit_payloads = []
        for combination in shopping_outfits:
            combination_payload = combination.public_dict()
            combination_payload["products"] = [
                shopping_payload_by_id[product_id]
                for product_id in combination.product_ids
                if product_id in shopping_payload_by_id
            ]
            shopping_outfit_payloads.append(combination_payload)

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
        "current_outfit_evaluation": {
            **current_outfit_evaluation.to_dict(),
            "summary_points": build_outfit_summary_points(
                outfit_result,
                current_outfit_evaluation,
            ),
        },
        "shopping_results": shopping_payloads,
        "shopping_outfits": shopping_outfit_payloads,
        "rules": {
            "implemented": len(engine.recommender.active_rule_ids),
            "documented": len(engine.recommender.documented_rule_ids),
            "scoring": len(engine.recommender.scoring_rule_ids),
            "unsupported": [
                {
                    "id": rule_id,
                    "reason": engine.recommender.UNSUPPORTED_RULE_REASONS.get(
                        rule_id,
                        "문서에는 정의되어 있지만 실행 코드가 아직 연결되지 않았습니다.",
                    ),
                }
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
            "product_search": dict(
                getattr(product_search, "last_search_stats", {}) or {}
            ),
        },
        "tryon": _adapter_tryon_status(engine.tryon),
        "length_check": length_check_status(
            outfit_result,
            has_bottom=parsed.get("lower_mask") is not None and bool(np.any(parsed.get("lower_mask"))),
        ),
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
        tryon_context=tryon_context,
        shopping_tryon_products=shopping_tryon_products,
    )


def length_check_status(outfit, has_bottom: bool, context: dict | None = None) -> dict:
    """현재 하의 기장을 사진에서 쟀는지, 사용자 입력이 필요한지를 화면에 알린다.

    사진 밖으로 밑단이 잘리면 기장 차이 경고를 낼 근거가 없다. 추측해서 통과시키지 않고
    밑단이 보이는 사진이나 사용자 입력을 요청한다.
    """
    current, source = current_bottom_length(outfit, context)
    if not has_bottom:
        status, message = "no_bottom", ""
    elif source == "photo":
        status, message = "measured", ""
    elif source == "user":
        status, message = "user_input", "입력한 현재 하의 기장으로 기장 차이를 확인해요."
    else:
        status = "needs_input"
        message = (
            "사진에서 하의 밑단이 잘려 지금 입은 옷의 기장을 확인하지 못했어요. "
            "기장을 알려주시면 기장 차이로 합성이 어색해질 조합을 미리 표시해요. "
            "밑단이 보이게 다시 찍으면 자동으로 판정해요."
        )
    return {
        "status": status,
        "value": current,
        "options": list(USER_BOTTOM_LENGTH_OPTIONS),
        "message": message,
    }


def reference_bottom_lengths() -> dict[str, str]:
    """합성 어댑터가 캐시한 상품 기장. 엔진이 아직 없으면 적재하지 않고 빈 값을 준다."""
    engine = _engine
    if engine is None:
        return {}
    return dict(getattr(engine.tryon, "reference_bottom_lengths", {}) or {})


def refresh_bottom_length_warnings(
    warnings: list[str],
    products: list[Product],
    outfit,
    context: dict | None,
    reference_lengths: dict[str, str],
) -> list[str] | None:
    """이미 만든 합성 결과의 기장 경고만 새 현재 기장 기준으로 다시 계산한다.

    reference_lengths는 합성할 때 어댑터가 캐시한 상품 기장이라 모델 추론이나 재합성이
    없다. 캐시가 없는 결과는 None을 돌려 기존 경고를 그대로 두게 한다.
    """
    bottoms = [product for product in products if product.category == "bottom"]
    if not bottoms:
        return list(warnings)
    cache = reference_lengths
    if any(product.product_id not in cache for product in bottoms):
        return None
    current, source = current_bottom_length(outfit, context)
    kept = [message for message in warnings if not is_bottom_length_warning(message)]
    for product in bottoms:
        kept.extend(bottom_length_warnings(current, source, cache[product.product_id]))
    return kept


def _adapter_tryon_status(
    adapter: VirtualTryOnAdapter,
    *,
    preview_kind: str | None = None,
) -> dict:
    raw_warnings = getattr(adapter, "last_warnings", [])
    warnings = list(raw_warnings) if isinstance(raw_warnings, (list, tuple)) else []
    status = {
        "available": adapter.available,
        "reason": "" if adapter.available else adapter.NOT_READY_REASON,
        "warnings": warnings,
    }
    if preview_kind is not None:
        status["preview_kind"] = preview_kind
    return status


def tryon_status() -> dict:
    """예상 착장샷 생성이 가능한지와, 불가능하면 그 사유를 알려준다."""
    return _adapter_tryon_status(get_engine().tryon)


def generate_tryon(
    person_image: Path,
    recommendation,
    output_path: Path,
    context: dict | None = None,
) -> Path:
    """추천 코디 하나에 대한 예상 착장샷을 만든다.

    생성 모델이 없으면 추천 보드로 몰래 대체하지 않고 TryOnNotReady를 올린다.
    """
    output, _warnings = generate_tryon_with_warnings(
        person_image,
        recommendation,
        output_path,
        context=context,
    )
    return output


def generate_tryon_with_warnings(
    person_image: Path,
    recommendation,
    output_path: Path,
    context: dict | None = None,
) -> tuple[Path, list[str]]:
    """합성 결과와 그 호출에서 나온 품질 경고를 원자적으로 돌려준다."""
    if not recommendation.products:
        raise TryOnNotReady("현재 코디를 유지하는 조건이라 새로 생성할 착장샷이 없습니다.")
    strict_context = dict(context or {})
    strict_context["strict_vton"] = True
    with _analysis_lock:
        adapter = get_engine().tryon
        output = adapter.synthesize(
            person_image=person_image,
            recommendation=recommendation,
            output_path=output_path,
            context=strict_context,
        )
        raw_warnings = getattr(adapter, "last_warnings", [])
        warnings = list(raw_warnings) if isinstance(raw_warnings, (list, tuple)) else []
        return output, warnings


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
        "gender": profile.gender,
        "change_scope": profile.change_scope,
        "change_categories": profile.change_categories,
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
        "genders": GENDERS,
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
