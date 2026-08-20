from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from clothing_parser import ClothingParser
from config import ATTRIBUTE_CONFIDENCE_THRESHOLDS
from fashion_attribute_model import AttributePrediction, fuse_measured_and_learned, geometry_vector
from fashion_attribute_schema import LOWER_CATEGORIES, UPPER_CATEGORIES
from fashion_prompts import (
    LOWER_TYPE_PROMPTS,
    MATERIAL_PROMPTS,
    NECKLINE_PROMPTS,
    PATTERN_PROMPTS,
    STYLE_PROMPTS,
    UPPER_TYPE_PROMPTS,
)
from fashion_model import FashionClassifier
from garment_attribute_analyzer import GarmentAttributeAnalyzer
from pose_analyzer import _to_rgb_array
from schemas import OutfitAnalysis, PoseAnalysis


COLOR_PALETTE = {
    "블랙": (25, 25, 25),
    "화이트": (235, 235, 235),
    "그레이": (130, 130, 130),
    "네이비": (35, 50, 90),
    "블루": (55, 110, 190),
    "브라운": (115, 75, 45),
    "베이지": (205, 185, 145),
    "레드": (185, 45, 45),
    "핑크": (220, 125, 155),
    "그린": (60, 130, 75),
    "옐로": (220, 185, 50),
    "퍼플": (115, 70, 145),
    "오렌지": (220, 115, 45),
    "카키": (105, 105, 55),
    "버건디": (115, 35, 55),
}
NEUTRALS = {"블랙", "화이트", "그레이", "네이비", "브라운", "베이지"}


def _dominant_palette(rgb: np.ndarray, mask: np.ndarray, max_colors: int = 3) -> list[dict]:
    pixels = rgb[mask]
    if len(pixels) == 0:
        return [{"name": "그레이", "rgb": [128, 128, 128], "proportion": 1.0}]
    # 지나치게 밝거나 어두운 그림자/하이라이트를 줄인 뒤 대표 색 군집을 찾는다.
    brightness = pixels.mean(axis=1)
    filtered = pixels[(brightness > 20) & (brightness < 245)]
    if len(filtered) < 20:
        filtered = pixels
    if len(filtered) > 5000:
        indices = np.linspace(0, len(filtered) - 1, 5000, dtype=int)
        filtered = filtered[indices]
    # CIELAB에서 군집화해 RGB 거리보다 사람의 색 차이 인식에 가깝게 만든다.
    samples_rgb = filtered.astype(np.uint8)
    samples = cv2.cvtColor(samples_rgb[np.newaxis, :, :], cv2.COLOR_RGB2LAB)[0].astype(np.float32)
    cluster_count = min(max_colors, len(samples))
    _, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    center_lab = np.clip(centers, 0, 255).astype(np.uint8)
    center_rgb = cv2.cvtColor(center_lab[np.newaxis, :, :], cv2.COLOR_LAB2RGB)[0]
    order = np.argsort(counts)[::-1]
    total = max(int(counts.sum()), 1)
    return [
        {
            "name": _nearest_color(tuple(int(value) for value in center_rgb[index])),
            "rgb": [int(value) for value in center_rgb[index]],
            "proportion": round(float(counts[index] / total), 3),
        }
        for index in order
    ]


def _dominant_rgb(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    dominant = _dominant_palette(rgb, mask, max_colors=3)[0]["rgb"]
    return tuple(dominant)


def _nearest_color(rgb: tuple[int, int, int]) -> str:
    sample = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
    return min(
        COLOR_PALETTE,
        key=lambda name: np.linalg.norm(
            sample - cv2.cvtColor(np.uint8([[COLOR_PALETTE[name]]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float32)
        ),
    )


def color_harmony(first: str, second: str) -> str:
    if first == second:
        return "톤온톤"
    if first in NEUTRALS or second in NEUTRALS:
        return "안정적인 무채색 조합"
    rgb1, rgb2 = COLOR_PALETTE[first], COLOR_PALETTE[second]
    hue1 = colorsys.rgb_to_hsv(*(value / 255 for value in rgb1))[0]
    hue2 = colorsys.rgb_to_hsv(*(value / 255 for value in rgb2))[0]
    distance = min(abs(hue1 - hue2), 1 - abs(hue1 - hue2))
    if distance < 0.10:
        return "유사색 조합"
    if distance > 0.38:
        return "대비색 조합"
    return "보통 조합"


def _garment_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """의류 영역을 학습 데이터와 같은 방식(배경을 남긴 bbox crop)으로 자른다.

    학습 표본은 Fashionpedia bbox crop이라 배경이 남아 있다. 추론에서만 배경을
    흰색으로 지우면 같은 옷의 임베딩이 크게 달라져(코사인 0.81) 신뢰도가 무너진다.
    """
    if not mask.any():
        return Image.fromarray(rgb)
    y1, y2, x1, x2 = _mask_bounds(mask)
    return Image.fromarray(rgb[y1:y2, x1:x2])


def _mask_bounds(mask: np.ndarray) -> tuple[int, int, int, int]:
    """마스크를 감싸는 최소 사각형. 학습의 bbox와 같은 의미가 되도록 여백을 두지 않는다."""
    ys, xs = np.where(mask)
    return int(ys.min()), int(ys.max()) + 1, int(xs.min()), int(xs.max()) + 1


def _crop_geometry(mask: np.ndarray) -> list[float]:
    """학습과 같은 정의로 crop의 기하 특징을 만든다."""
    if not mask.any():
        return geometry_vector(1, 1, tight_crop=False)
    y1, y2, x1, x2 = _mask_bounds(mask)
    return geometry_vector(x2 - x1, y2 - y1, tight_crop=True)


def _masked_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """배경을 흰색으로 지운 crop. 색 분석 등 배경을 빼야 하는 곳에서만 쓴다."""
    if not mask.any():
        return Image.fromarray(rgb)
    y1, y2, x1, x2 = _mask_bounds(mask)
    padding = 12
    x1, x2 = max(0, x1 - padding), min(rgb.shape[1], x2 + padding)
    y1, y2 = max(0, y1 - padding), min(rgb.shape[0], y2 + padding)
    crop = rgb[y1:y2, x1:x2].copy()
    crop[~mask[y1:y2, x1:x2]] = 255
    return Image.fromarray(crop)


class OutfitAnalyzer:
    def __init__(self, parser: ClothingParser, classifier: FashionClassifier) -> None:
        self.parser = parser
        self.classifier = classifier
        self.attribute_analyzer = GarmentAttributeAnalyzer()

    def analyze(self, image: str | Path | Image.Image, pose: PoseAnalysis) -> tuple[OutfitAnalysis, dict]:
        rgb = _to_rgb_array(image)
        parsed = self.parser.parse(image, pose)
        attributes = self.attribute_analyzer.analyze(parsed["segmentation"], pose)
        parsed["attributes"] = attributes
        upper_palette = _dominant_palette(rgb, parsed["upper_mask"])
        lower_palette = _dominant_palette(rgb, parsed["lower_mask"])
        upper_color = upper_palette[0]["name"]
        lower_color = lower_palette[0]["name"]
        style, style_confidence = self.classifier.best_mapped_label(image, STYLE_PROMPTS)
        if style_confidence < ATTRIBUTE_CONFIDENCE_THRESHOLDS["style"]:
            style = "스타일 불확실"
        attribute_sources: dict[str, str] = {}
        sleeve_shape = collar = silhouette = "분석 보류"
        details: list[str] = []
        lower_details: list[str] = []
        lower_subtype = "분석 보류"
        pant_leg_shape = "분석 보류"
        pant_length = "분석 보류"
        learned_upper = {}
        learned_lower = {}
        if self.classifier.enabled:
            upper_crop = _garment_crop(rgb, parsed["upper_mask"])
            upper_geometry = _crop_geometry(parsed["upper_mask"])
            learned_upper, upper_results = self.classifier.analyze_crop(
                upper_crop,
                tasks=[
                    "category", "sleeve_length", "sleeve_shape", "upper_length", "neckline",
                    "collar", "upper_fit", "silhouette", "pattern", "material", "detail",
                ],
                prompt_groups={
                    "category": UPPER_TYPE_PROMPTS,
                    "pattern": PATTERN_PROMPTS,
                    "material": MATERIAL_PROMPTS,
                    "neckline": NECKLINE_PROMPTS,
                },
                geometry=upper_geometry,
            )
            upper_type, upper_type_confidence = upper_results["category"]
            pattern, pattern_confidence = upper_results["pattern"]
            material, material_confidence = upper_results["material"]
            neckline, neckline_confidence = upper_results["neckline"]
            learned_category = learned_upper.get("category")
            if (
                learned_category
                and learned_category.accepted
                and learned_category.labels[0] in UPPER_CATEGORIES
            ):
                attributes["upper_type"] = learned_category.labels[0]
                attribute_sources["upper_type"] = "trained_head"
            elif upper_type_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["category"]:
                attributes["upper_type"] = upper_type
                attribute_sources["upper_type"] = "zero_shot"

            learned_pattern = learned_upper.get("pattern")
            if learned_pattern and learned_pattern.accepted:
                attributes["pattern"] = "|".join(learned_pattern.labels)
                attribute_sources["pattern"] = "trained_head"
            else:
                attributes["pattern"] = pattern if pattern_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["pattern"] else "패턴 불확실"
                attribute_sources["pattern"] = "zero_shot"

            learned_material = learned_upper.get("material")
            if learned_material and learned_material.accepted:
                attributes["material"] = "|".join(learned_material.labels)
                attribute_sources["material"] = "trained_head"
            else:
                attributes["material"] = material if material_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["material"] else "소재 불확실"
                attribute_sources["material"] = "zero_shot"

            upper_area_ratio = float(parsed["upper_mask"].mean())
            shoulder_visibility = min(
                pose.landmarks.get("left_shoulder", (0, 0, 0))[2],
                pose.landmarks.get("right_shoulder", (0, 0, 0))[2],
            )
            neckline_visible = upper_area_ratio >= 0.02 and shoulder_visibility >= 0.55
            learned_neckline = learned_upper.get("neckline") if neckline_visible else None
            if learned_neckline and learned_neckline.accepted:
                attributes["neckline"] = learned_neckline.labels[0]
                attribute_sources["neckline"] = "trained_head"
            elif neckline_visible:
                attributes["neckline"] = neckline if neckline_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["neckline"] else "네크라인 불확실"
                attribute_sources["neckline"] = "zero_shot"
            else:
                attributes["neckline"] = "네크라인 분석 보류"
                attribute_sources["neckline"] = "not_visible"

            for task_name, attribute_name in (
                ("sleeve_length", "sleeve_length"),
                ("upper_length", "upper_length"),
                ("upper_fit", "fit"),
            ):
                fused, _, source = fuse_measured_and_learned(
                    attributes[attribute_name],
                    attributes["attribute_confidence"],
                    learned_upper.get(task_name),
                )
                attributes[attribute_name] = fused
                attribute_sources[attribute_name] = source
            if learned_upper.get("sleeve_shape") and learned_upper["sleeve_shape"].accepted:
                sleeve_shape = learned_upper["sleeve_shape"].labels[0]
                attribute_sources["sleeve_shape"] = "trained_head"
            if learned_upper.get("collar") and learned_upper["collar"].accepted and neckline_visible:
                collar = learned_upper["collar"].labels[0]
                attribute_sources["collar"] = "trained_head"
            refined_upper_type = _refine_upper_type(attributes["upper_type"], collar)
            if refined_upper_type != attributes["upper_type"]:
                attributes["upper_type"] = refined_upper_type
                attribute_sources["upper_type"] = "derived_category_collar"
            if learned_upper.get("silhouette") and learned_upper["silhouette"].accepted:
                silhouette = learned_upper["silhouette"].labels[0]
                attribute_sources["silhouette"] = "trained_head"
            if learned_upper.get("detail") and learned_upper["detail"].accepted:
                details = learned_upper["detail"].labels
                attribute_sources["details"] = "trained_head"

            if attributes["lower_type"] == "원피스":
                lower_type_confidence = lower_pattern_confidence = lower_material_confidence = 0.0
                lower_pattern = attributes["pattern"]
                lower_material = attributes["material"]
                lower_subtype = "해당 없음"
                pant_leg_shape = "해당 없음"
                pant_length = "해당 없음"
            else:
                lower_crop = _garment_crop(rgb, parsed["lower_mask"])
                lower_geometry = _crop_geometry(parsed["lower_mask"])
                learned_lower, lower_results = self.classifier.analyze_crop(
                    lower_crop,
                    tasks=[
                        "category", "lower_subtype", "pant_leg_shape", "pant_length", "lower_detail",
                        "lower_length", "lower_fit", "silhouette", "pattern", "material", "detail",
                    ],
                    prompt_groups={
                        "category": LOWER_TYPE_PROMPTS,
                        "pattern": PATTERN_PROMPTS,
                        "material": MATERIAL_PROMPTS,
                    },
                    geometry=lower_geometry,
                )
                lower_type, lower_type_confidence = lower_results["category"]
                lower_pattern_value, lower_pattern_confidence = lower_results["pattern"]
                lower_material_value, lower_material_confidence = lower_results["material"]
                learned_lower_category = learned_lower.get("category")
                if (
                    learned_lower_category
                    and learned_lower_category.accepted
                    and learned_lower_category.labels[0] in LOWER_CATEGORIES
                ):
                    attributes["lower_type"] = learned_lower_category.labels[0]
                    attribute_sources["lower_type"] = "trained_head"
                elif lower_type_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["category"]:
                    attributes["lower_type"] = lower_type
                    attribute_sources["lower_type"] = "zero_shot"
                learned_lower_subtype = learned_lower.get("lower_subtype")
                if learned_lower_subtype and learned_lower_subtype.accepted:
                    lower_subtype = learned_lower_subtype.labels[0]
                    attribute_sources["lower_subtype"] = "trained_head"
                elif attributes["lower_type"] == "청바지":
                    lower_subtype = "청바지"
                    attribute_sources["lower_subtype"] = "derived_category"
                learned_pant_leg_shape = learned_lower.get("pant_leg_shape")
                if learned_pant_leg_shape and learned_pant_leg_shape.accepted:
                    pant_leg_shape = learned_pant_leg_shape.labels[0]
                    attribute_sources["pant_leg_shape"] = "trained_head"
                learned_pant_length = learned_lower.get("pant_length")
                if learned_pant_length and learned_pant_length.accepted:
                    pant_length = learned_pant_length.labels[0]
                    attribute_sources["pant_length"] = "trained_head"
                learned_lower_pattern = learned_lower.get("pattern")
                if learned_lower_pattern and learned_lower_pattern.accepted:
                    lower_pattern = "|".join(learned_lower_pattern.labels)
                    attribute_sources["lower_pattern"] = "trained_head"
                else:
                    lower_pattern = (
                        lower_pattern_value
                        if lower_pattern_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["pattern"]
                        else "패턴 불확실"
                    )
                    attribute_sources["lower_pattern"] = "zero_shot"
                learned_lower_material = learned_lower.get("material")
                if learned_lower_material and learned_lower_material.accepted:
                    lower_material = "|".join(learned_lower_material.labels)
                    attribute_sources["lower_material"] = "trained_head"
                else:
                    lower_material = (
                        lower_material_value
                        if lower_material_confidence >= ATTRIBUTE_CONFIDENCE_THRESHOLDS["material"]
                        else "소재 불확실"
                    )
                    attribute_sources["lower_material"] = "zero_shot"
                lower_length_prediction = _adapt_pant_length_prediction(learned_pant_length)
                if lower_length_prediction is None or not lower_length_prediction.accepted:
                    lower_length_prediction = _adapt_lower_length_prediction(
                        learned_lower.get("lower_length"), attributes["lower_type"]
                    )
                for prediction, attribute_name in (
                    (lower_length_prediction, "bottom_length"),
                    (learned_lower.get("lower_fit"), "lower_fit"),
                ):
                    fused, _, source = fuse_measured_and_learned(
                        attributes[attribute_name], attributes["attribute_confidence"], prediction
                    )
                    attributes[attribute_name] = fused
                    attribute_sources[attribute_name] = source
                if learned_lower.get("lower_detail") and learned_lower["lower_detail"].accepted:
                    lower_details = learned_lower["lower_detail"].labels
                    attribute_sources["lower_details"] = "trained_lower_detail_head"
                elif learned_lower.get("detail") and learned_lower["detail"].accepted:
                    lower_details = learned_lower["detail"].labels
                    attribute_sources["lower_details"] = "trained_head"
        else:
            style_confidence = pattern_confidence = material_confidence = neckline_confidence = 0.0
            upper_type_confidence = lower_type_confidence = lower_pattern_confidence = lower_material_confidence = 0.0
            lower_pattern = lower_material = "분석 보류"
            attributes["neckline"] = "분석 보류"
        notes = []
        if self.parser.backend == "pose-guided-fallback":
            notes.append("포즈 기반 임시 마스크입니다. 의류 길이와 종류를 정식 분석에 사용하지 마세요.")
        if not self.classifier.enabled:
            notes.append("FashionSigLIP이 꺼져 있어 스타일·패턴·소재·네크라인은 분석하지 않았습니다.")
        else:
            notes.append(
                "FashionSigLIP 후보 간 상대점수: "
                f"스타일 {style_confidence:.2f}, 상의 종류 {upper_type_confidence:.2f}, "
                f"상의 패턴 {pattern_confidence:.2f}, 상의 소재 {material_confidence:.2f}, "
                f"하의 대분류 {lower_type_confidence:.2f}, 하의 패턴 {lower_pattern_confidence:.2f}, "
                f"하의 소재 {lower_material_confidence:.2f}, 네크라인 {neckline_confidence:.2f}. "
                "이 값은 보정된 정답 확률이 아닙니다."
            )
            if self.classifier.trained_attributes_enabled:
                accepted = [
                    f"{name}={','.join(prediction.labels)}({prediction.confidence:.2f})"
                    for results in (learned_upper, learned_lower)
                    for name, prediction in results.items()
                    if prediction.accepted
                ]
                notes.append(
                    "학습 속성 헤드 결과: " + (", ".join(accepted) if accepted else "임계값을 넘은 속성 없음")
                )
        return (
            OutfitAnalysis(
                parser_backend=self.parser.backend,
                upper_color=upper_color,
                lower_color=lower_color,
                color_harmony=color_harmony(upper_color, lower_color),
                detected_items=list(parsed["present_labels"]),
                style=style,
                upper_type=attributes["upper_type"],
                lower_type=attributes["lower_type"],
                lower_subtype=lower_subtype,
                pant_leg_shape=pant_leg_shape,
                pant_length=pant_length,
                sleeve_length=attributes["sleeve_length"],
                upper_length=attributes["upper_length"],
                bottom_length=attributes["bottom_length"],
                fit=attributes["fit"],
                lower_fit=attributes["lower_fit"],
                neckline=attributes["neckline"],
                pattern=attributes["pattern"],
                material=attributes["material"],
                lower_pattern=lower_pattern,
                lower_material=lower_material,
                sleeve_shape=sleeve_shape,
                collar=collar,
                silhouette=silhouette,
                details=details,
                lower_details=lower_details,
                attribute_sources=attribute_sources,
                upper_palette=upper_palette,
                lower_palette=lower_palette,
                attribute_confidence=attributes["attribute_confidence"],
                notes=notes,
            ),
            parsed,
        )


def _adapt_lower_length_prediction(
    prediction: AttributePrediction | None,
    lower_type: str,
) -> AttributePrediction | None:
    if prediction is None or not prediction.accepted:
        return prediction
    is_pants = any(word in lower_type for word in ("바지", "팬츠", "청바지", "쇼츠"))
    mappings = {
        "쇼츠·미니 기장": "반바지" if is_pants else "미니 기장",
        "무릎 기장": "무릎 기장 바지" if is_pants else "무릎 기장",
        "미디·7부 기장": "크롭·7부 바지" if is_pants else "미디 기장",
        "롱·긴바지 기장": "긴바지" if is_pants else "롱·맥시 기장",
    }
    mapped = mappings.get(prediction.labels[0], prediction.labels[0])
    return AttributePrediction([mapped], prediction.scores, prediction.confidence, prediction.accepted)


def _adapt_pant_length_prediction(
    prediction: AttributePrediction | None,
) -> AttributePrediction | None:
    if prediction is None or not prediction.accepted:
        return prediction
    mappings = {
        "카프리·7부": "크롭·7부 바지",
        "크롭·앵클": "크롭·7부 바지",
        "풀렝스": "긴바지",
    }
    mapped = mappings.get(prediction.labels[0], prediction.labels[0])
    return AttributePrediction([mapped], prediction.scores, prediction.confidence, prediction.accepted)


def _refine_upper_type(category: str, collar: str) -> str:
    """카테고리와 독립 칼라 헤드가 합의할 때 한국 쇼핑 분류명으로 구체화한다."""
    if category == "티셔츠" and collar == "폴로 칼라":
        return "폴로 셔츠"
    return category
