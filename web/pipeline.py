"""main.ipynb의 실행 순서를 웹 요청에서 재사용할 수 있게 감싼 래퍼."""

from __future__ import annotations

import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

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
from feedback_store import FeedbackStore
from outfit_analyzer import COLOR_PALETTE, OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import CHANGE_SCOPE_MAP, PURPOSE_STYLES, RecommendationEngine
from body_shape import classify
from schemas import GOAL_NONE, SILHOUETTE_GOAL_CHOICES, UserProfile, WardrobeItem
from virtual_tryon import TryOnNotReady, VirtualTryOnAdapter

RULES_PATH = PROJECT_DIR / "FASHION_RULES_MASTER.md"
ATTRIBUTE_HEADS_PATH = FASHION_ATTRIBUTE_HEADS_PATH

PURPOSES = list(PURPOSE_STYLES)
STYLES = ["캐주얼", "미니멀", "포멀", "스포티", "스트리트", "로맨틱"]
CHANGE_SCOPES = list(CHANGE_SCOPE_MAP)
SEASONS = ["사계절", "봄", "여름", "가을", "겨울"]
SILHOUETTE_GOALS = [{"value": value, "label": label} for value, label in SILHOUETTE_GOAL_CHOICES]
DRESS_CODES = ["자동", "캐주얼", "스마트 캐주얼", "비즈니스 캐주얼", "포멀"]
ACTIVITY_LEVELS = ["낮음", "보통", "높음"]
MATERIALS = ["코튼", "린넨", "데님", "니트", "울", "가죽", "폴리에스터", "쉬폰"]

STAGES = [
    ("quality", "사진 품질 검사"),
    ("pose", "체형·자세 분석"),
    ("outfit", "현재 착장 분석"),
    ("recommend", "코디 후보 순위 결정"),
    ("preview", "결과 이미지 생성"),
]


__all__ = [
    "STAGES",
    "PipelineError",
    "PipelineResult",
    "TryOnNotReady",
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


@dataclass
class Engine:
    """무거운 체크포인트를 프로세스당 한 번만 올려 요청 간에 재사용한다."""

    pose_analyzer: PoseAnalyzer
    quality_checker: QualityChecker
    outfit_analyzer: OutfitAnalyzer
    recommender: RecommendationEngine
    tryon: VirtualTryOnAdapter
    device: str
    trained_heads: bool
    parser_backend: str


_engine: Engine | None = None
_engine_lock = threading.Lock()
# MediaPipe와 SegFormer 세션은 동시 호출에 안전하지 않아 분석 자체를 직렬화한다.
_analysis_lock = threading.Lock()


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
        tryon=_build_tryon(),
        device=classifier.device,
        trained_heads=classifier.trained_attributes_enabled,
        parser_backend=clothing_parser.backend,
    )


def build_profile(payload: dict) -> UserProfile:
    def number(key: str) -> float | None:
        value = payload.get(key)
        if value in (None, ""):
            return None
        return float(value)

    def string_list(key: str) -> list[str]:
        value = payload.get(key) or []
        return [str(item).strip() for item in value if str(item).strip()]

    owned_items = [
        WardrobeItem(
            item_id=str(item.get("item_id") or f"OWN-{index:02d}"),
            category=str(item.get("category") or "top"),
            color=str(item.get("color") or ""),
            style=str(item.get("style") or ""),
            season=str(item.get("season") or "사계절"),
        )
        for index, item in enumerate(payload.get("owned_items") or [], start=1)
        if str(item.get("color") or "").strip()
    ]

    return UserProfile(
        purpose=payload.get("purpose") or "데일리",
        desired_style=payload.get("desired_style") or "캐주얼",
        budget=int(payload.get("budget") or 150_000),
        change_scope=payload.get("change_scope") or "전체 변경",
        height_cm=number("height_cm"),
        weight_kg=number("weight_kg"),
        chest_cm=number("chest_cm"),
        waist_cm=number("waist_cm"),
        hip_cm=number("hip_cm"),
        season=payload.get("season") or "사계절",
        silhouette_goal=payload.get("silhouette_goal") or GOAL_NONE,
        dress_code=payload.get("dress_code") or "자동",
        activity_level=payload.get("activity_level") or "보통",
        preferred_colors=string_list("preferred_colors"),
        avoided_colors=string_list("avoided_colors"),
        avoided_materials=string_list("avoided_materials"),
        excluded_item_types=string_list("excluded_item_types"),
        temperature_c=number("temperature_c"),
        feels_like_c=number("feels_like_c"),
        humidity=number("humidity"),
        precipitation_probability=number("precipitation_probability"),
        wind_mps=number("wind_mps"),
        uv_index=number("uv_index"),
        owned_items=owned_items,
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
        on_stage("quality")
        pose_result = engine.pose_analyzer.analyze(image_path)
        input_quality = engine.quality_checker.check_input(image_path, pose=pose_result)
        if not input_quality["passed"]:
            raise PipelineError(
                "전신사진 품질 기준을 통과하지 못했습니다: "
                + " / ".join(input_quality["issues"])
            )

        on_stage("pose")
        if not pose_result.valid:
            raise PipelineError("유효한 전신 포즈를 찾지 못했습니다. 정면 전신사진을 사용하세요.")
        landmark_path = output_dir / "pose_landmarks.jpg"
        engine.pose_analyzer.draw_landmarks(image_path, analysis=pose_result).save(
            landmark_path, quality=92
        )
        # 둘레를 입력했으면 사진 추정보다 정확하므로 그 값으로 덮어쓴다.
        # 여기서 확정해야 추천 엔진과 화면이 같은 체형을 본다.
        # 체형 파악용 사진이 있으면 그쪽을 쓴다. 몸이 드러날수록 실루엣 폭이 정확하다.
        pose_result.body_shape, body_shape_basis = classify(
            profile, pose_result, person_image=body_image_path or image_path
        )

        on_stage("outfit")
        outfit_result, parsed = engine.outfit_analyzer.analyze(image_path, pose_result)
        segmentation_path = output_dir / "segmentation.jpg"
        engine.outfit_analyzer.parser.colorize(parsed["segmentation"]).save(
            segmentation_path, quality=92
        )

        on_stage("recommend")
        try:
            recommendations = engine.recommender.recommend(profile, pose_result, outfit_result, top_k=3)
        except ValueError as exc:
            raise PipelineError(str(exc)) from exc

        on_stage("preview")
        preview_path = output_dir / "preview.jpg"
        if recommendations[0].products:
            engine.tryon.generate(
                person_image=image_path,
                recommendation=recommendations[0],
                output_path=preview_path,
            )
        else:
            Image.open(image_path).convert("RGB").save(preview_path, quality=92)

    pose_dict = pose_result.to_dict()
    pose_dict.pop("landmarks", None)
    pose_dict["body_shape_basis"] = body_shape_basis
    payload = {
        "input_quality": input_quality,
        "pose": pose_dict,
        "outfit": outfit_result.to_dict(),
        "outfit_summary": outfit_result.to_summary_dict(),
        "recommendations": [_recommendation_dict(item) for item in recommendations],
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
        },
        "tryon": tryon_status(),
        "images": {
            "original": "original.jpg",
            "landmarks": landmark_path.name,
            "segmentation": segmentation_path.name,
            "preview": preview_path.name,
        },
    }
    return PipelineResult(payload=payload, recommendations=recommendations, person_image=image_path)


def tryon_status() -> dict:
    """예상 착장샷 생성이 가능한지와, 불가능하면 그 사유를 알려준다."""
    adapter = get_engine().tryon
    return {
        "available": adapter.available,
        "reason": "" if adapter.available else adapter.NOT_READY_REASON,
    }


def generate_tryon(person_image: Path, recommendation, output_path: Path) -> Path:
    """추천 코디 하나에 대한 예상 착장샷을 만든다.

    생성 모델이 없으면 추천 보드로 몰래 대체하지 않고 TryOnNotReady를 올린다.
    """
    if not recommendation.products:
        raise TryOnNotReady("현재 코디를 유지하는 조건이라 새로 생성할 착장샷이 없습니다.")
    return get_engine().tryon.synthesize(
        person_image=person_image,
        recommendation=recommendation,
        output_path=output_path,
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
        }
        for product in recommendation.products
    ]
    return data


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
