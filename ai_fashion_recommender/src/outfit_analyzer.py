from __future__ import annotations

import colorsys
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from clothing_parser import ClothingParser
from config import ATTRIBUTE_CONFIDENCE_THRESHOLDS
from fashion_attribute_model import AttributePrediction, fuse_measured_and_learned, geometry_vector
from fashion_attribute_schema import LOWER_CATEGORIES, UPPER_CATEGORIES
from fashion_prompts import (
    LAYERING_PROMPTS,
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
from wear_state_analyzer import infer_layering_state, infer_sleeve_state


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

# 색 이름마다 하나의 RGB만 두면 같은 카키도 조명과 명도에 따라 브라운·베이지로
# 이동한다. 실제 의류에서 자주 보이는 밝기 범위를 여러 기준점으로 표현한다.
COLOR_REFERENCE_RGB = {
    "블랙": ((10, 10, 10), (28, 28, 28), (48, 48, 48)),
    "화이트": ((245, 245, 245), (225, 225, 225), (242, 238, 228)),
    "그레이": ((85, 85, 85), (130, 130, 130), (180, 180, 180)),
    "네이비": ((23, 23, 43), (35, 50, 90), (52, 62, 92)),
    "블루": ((45, 90, 155), (55, 110, 190), (100, 155, 215)),
    "브라운": ((70, 45, 30), (115, 75, 45), (145, 95, 60), (105, 85, 65)),
    "베이지": ((170, 150, 115), (205, 185, 145), (225, 207, 179)),
    "레드": ((145, 30, 30), (185, 45, 45), (220, 65, 55)),
    "핑크": ((190, 90, 125), (220, 125, 155), (240, 175, 190)),
    "그린": ((35, 95, 50), (60, 130, 75), (95, 155, 100)),
    "옐로": ((185, 145, 25), (220, 185, 50), (240, 210, 90)),
    "퍼플": ((80, 45, 110), (115, 70, 145), (155, 105, 180)),
    "오렌지": ((180, 80, 30), (220, 115, 45), (235, 145, 70)),
    "카키": ((79, 75, 54), (105, 105, 55), (125, 119, 91), (135, 130, 85)),
    "버건디": ((75, 25, 40), (115, 35, 55), (145, 50, 70)),
}


def _interior_mask(mask: np.ndarray) -> np.ndarray:
    """분할 경계의 배경·피부 픽셀이 대표색에 섞이지 않도록 내부만 남긴다."""
    binary = np.asarray(mask, dtype=np.uint8)
    if int(binary.sum()) < 400:
        return binary.astype(bool)
    eroded = cv2.erode(binary, np.ones((3, 3), dtype=np.uint8), iterations=1)
    if int(eroded.sum()) >= int(binary.sum() * 0.65):
        return eroded.astype(bool)
    return binary.astype(bool)


def _white_balance_rgb(rgb: np.ndarray) -> np.ndarray:
    """사진 속 밝은 무채색 영역으로 약한 화이트밸런스 보정을 수행한다.

    확실한 흰색·밝은 회색 후보가 충분하지 않으면 원본을 그대로 사용하고,
    과도한 보정을 막기 위해 채널 이득을 10% 이내로 제한한다.
    """
    image = np.asarray(rgb, dtype=np.uint8)
    flat = image.reshape(-1, 3).astype(np.float32)
    maximum = flat.max(axis=1)
    minimum = flat.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    neutral = flat[(maximum >= 165) & (maximum <= 250) & (saturation <= 0.10)]
    minimum_count = max(100, int(len(flat) * 0.01))
    if len(neutral) < minimum_count:
        return image
    reference = np.percentile(neutral, 75, axis=0)
    target = float(reference.mean())
    gains = np.clip(target / np.maximum(reference, 1.0), 0.90, 1.10)
    return np.clip(image.astype(np.float32) * gains, 0, 255).astype(np.uint8)


def _rgb_to_cielab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    encoded = cv2.cvtColor(np.uint8([[rgb]]), cv2.COLOR_RGB2LAB)[0, 0].astype(np.float64)
    return float(encoded[0] * 100 / 255), float(encoded[1] - 128), float(encoded[2] - 128)


def _ciede2000(first: tuple[float, float, float], second: tuple[float, float, float]) -> float:
    """두 CIELAB 색의 CIEDE2000 지각 색차를 계산한다."""
    l1, a1, b1 = first
    l2, a2, b2 = second
    c1, c2 = math.hypot(a1, b1), math.hypot(a2, b2)
    c_bar = (c1 + c2) / 2
    g = 0.5 * (1 - math.sqrt(c_bar**7 / (c_bar**7 + 25**7)))
    a1_prime, a2_prime = (1 + g) * a1, (1 + g) * a2
    c1_prime, c2_prime = math.hypot(a1_prime, b1), math.hypot(a2_prime, b2)
    h1_prime = math.degrees(math.atan2(b1, a1_prime)) % 360
    h2_prime = math.degrees(math.atan2(b2, a2_prime)) % 360

    delta_l = l2 - l1
    delta_c = c2_prime - c1_prime
    if c1_prime * c2_prime == 0:
        delta_h_angle = 0.0
    elif abs(h2_prime - h1_prime) <= 180:
        delta_h_angle = h2_prime - h1_prime
    elif h2_prime <= h1_prime:
        delta_h_angle = h2_prime - h1_prime + 360
    else:
        delta_h_angle = h2_prime - h1_prime - 360
    delta_h = 2 * math.sqrt(c1_prime * c2_prime) * math.sin(math.radians(delta_h_angle / 2))

    l_bar = (l1 + l2) / 2
    c_prime_bar = (c1_prime + c2_prime) / 2
    if c1_prime * c2_prime == 0:
        h_bar = h1_prime + h2_prime
    elif abs(h1_prime - h2_prime) <= 180:
        h_bar = (h1_prime + h2_prime) / 2
    elif h1_prime + h2_prime < 360:
        h_bar = (h1_prime + h2_prime + 360) / 2
    else:
        h_bar = (h1_prime + h2_prime - 360) / 2
    t = (
        1
        - 0.17 * math.cos(math.radians(h_bar - 30))
        + 0.24 * math.cos(math.radians(2 * h_bar))
        + 0.32 * math.cos(math.radians(3 * h_bar + 6))
        - 0.20 * math.cos(math.radians(4 * h_bar - 63))
    )
    s_l = 1 + 0.015 * (l_bar - 50) ** 2 / math.sqrt(20 + (l_bar - 50) ** 2)
    s_c = 1 + 0.045 * c_prime_bar
    s_h = 1 + 0.015 * c_prime_bar * t
    delta_theta = 30 * math.exp(-((h_bar - 275) / 25) ** 2)
    r_c = 2 * math.sqrt(c_prime_bar**7 / (c_prime_bar**7 + 25**7))
    r_t = -r_c * math.sin(math.radians(2 * delta_theta))
    return math.sqrt(
        (delta_l / s_l) ** 2
        + (delta_c / s_c) ** 2
        + (delta_h / s_h) ** 2
        + r_t * (delta_c / s_c) * (delta_h / s_h)
    )


def _semantic_color(rgb: tuple[int, int, int]) -> str:
    """다중 기준색 가운데 지각 색차가 가장 작은 색 이름을 반환한다."""
    sample = _rgb_to_cielab(tuple(int(value) for value in rgb))
    return min(
        COLOR_REFERENCE_RGB,
        key=lambda name: min(
            _ciede2000(sample, _rgb_to_cielab(reference))
            for reference in COLOR_REFERENCE_RGB[name]
        ),
    )


def _dominant_palette(rgb: np.ndarray, mask: np.ndarray, max_colors: int = 3) -> list[dict]:
    pixels = _white_balance_rgb(rgb)[_interior_mask(mask)]
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
    # 명도만 다른 군집이 같은 실제 색으로 판정되면 하나로 합친다. 예를 들어
    # 네이비의 밝은 주름과 어두운 주름이 별도 군집이어도 대표색은 네이비다.
    merged: dict[str, dict[str, np.ndarray | int]] = {}
    for index, count in enumerate(counts):
        if count <= 0:
            continue
        center = center_rgb[index].astype(np.float64)
        name = _semantic_color(tuple(int(value) for value in center))
        entry = merged.setdefault(name, {"count": 0, "rgb_sum": np.zeros(3, dtype=np.float64)})
        entry["count"] = int(entry["count"]) + int(count)
        entry["rgb_sum"] = np.asarray(entry["rgb_sum"]) + center * int(count)

    total = max(sum(int(entry["count"]) for entry in merged.values()), 1)
    palette = []
    for name, entry in merged.items():
        count = int(entry["count"])
        representative = np.rint(np.asarray(entry["rgb_sum"]) / max(count, 1)).astype(int)
        palette.append(
            {
                "name": name,
                "rgb": representative.tolist(),
                "proportion": round(float(count / total), 3),
            }
        )
    return sorted(palette, key=lambda item: item["proportion"], reverse=True)


def _dominant_rgb(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    dominant = _dominant_palette(rgb, mask, max_colors=3)[0]["rgb"]
    return tuple(dominant)


def _nearest_color(rgb: tuple[int, int, int]) -> str:
    return _semantic_color(rgb)


def _nearest_color_lab(rgb: tuple[int, int, int]) -> str:
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


def _neck_roi_crop(rgb: np.ndarray, pose: PoseAnalysis) -> Image.Image:
    """셔츠 칼라와 바깥 V넥 경계가 함께 보이는 목·윗가슴 ROI."""
    if not pose.landmarks or not all(
        name in pose.landmarks for name in ("left_shoulder", "right_shoulder")
    ):
        return Image.fromarray(rgb)
    height, width = rgb.shape[:2]
    left = pose.landmarks["left_shoulder"]
    right = pose.landmarks["right_shoulder"]
    shoulder_width = max(abs(right[0] - left[0]) * width, width * 0.12)
    center_x = (left[0] + right[0]) * width / 2
    shoulder_y = (left[1] + right[1]) * height / 2
    x1 = max(0, int(center_x - shoulder_width * 0.72))
    x2 = min(width, int(center_x + shoulder_width * 0.72))
    y1 = max(0, int(shoulder_y - shoulder_width * 0.55))
    y2 = min(height, int(shoulder_y + shoulder_width * 0.85))
    if x2 <= x1 or y2 <= y1:
        return Image.fromarray(rgb)
    return Image.fromarray(rgb[y1:y2, x1:x2])


def _layering_zero_shot(
    classifier: FashionClassifier,
    upper_crop: Image.Image,
    neck_crop: Image.Image,
) -> tuple[str, float]:
    """상의 전체와 목 ROI의 레이어드 상대점수를 결합한다."""
    prompts = list(LAYERING_PROMPTS.values())
    upper = classifier.classify(upper_crop, prompts)
    neck = classifier.classify(neck_crop, prompts)
    combined = {
        label: 0.65 * upper[prompt] + 0.35 * neck[prompt]
        for label, prompt in LAYERING_PROMPTS.items()
    }
    label = max(combined, key=combined.get)
    return label, float(combined[label])


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
        layering_state = "판단 보류"
        upper_items: list[str] = []
        inner_category = outer_category = "해당 없음"
        layering_confidence = 0.0
        layering_reason = "FashionSigLIP 비활성화"
        layering_prediction = None
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

            layering_label, layering_score = _layering_zero_shot(
                self.classifier, upper_crop, _neck_roi_crop(rgb, pose)
            )
            layering_prediction = self.classifier.predict_layering(upper_crop)
            layering = infer_layering_state(
                attributes["upper_type"],
                collar,
                attributes["neckline"],
                attributes["material"],
                trained_prediction=layering_prediction,
                zero_shot_label=layering_label,
                zero_shot_confidence=layering_score,
            )
            layering_state = layering.state
            upper_items = layering.upper_items
            inner_category = layering.inner_category
            outer_category = layering.outer_category
            layering_confidence = layering.confidence
            layering_reason = layering.reason
            attribute_sources["layering_state"] = (
                "trained_multi_roi_head"
                if layering_prediction is not None and layering_prediction.accepted
                else (
                    "derived_attribute_conflict"
                    if layering.state == "레이어드"
                    else "zero_shot_roi"
                )
            )

            refined_upper_type = _refine_upper_type(
                attributes["upper_type"],
                collar,
                layering_state=layering_state,
                material=attributes["material"],
                neckline=attributes["neckline"],
            )
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

        sleeve = infer_sleeve_state(
            attributes.get("measurements", {}),
            learned_upper.get("sleeve_length"),
            attributes["sleeve_length"],
            pose_landmarks=pose.landmarks,
        )
        attributes["sleeve_length"] = sleeve.designed_length
        attributes["visible_sleeve_length"] = sleeve.visible_length
        attributes["sleeve_state"] = sleeve.state
        attributes["layering_state"] = layering_state
        attributes["upper_items"] = upper_items or [attributes["upper_type"]]
        attribute_sources["sleeve_state"] = "visible_mask_and_trained_head"
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
        notes.append(
            f"소매 착용 상태: {sleeve.state} ({sleeve.reason}, 신뢰도 {sleeve.confidence:.2f})."
        )
        notes.append(
            f"레이어드 상태: {layering_state} ({layering_reason}, 신뢰도 {layering_confidence:.2f})."
        )
        if layering_prediction is not None:
            notes.append(
                "멀티 ROI 레이어드 헤드: "
                f"단일 {layering_prediction.scores['단일 옷']:.2f}, "
                f"레이어드 {layering_prediction.scores['겹쳐입음']:.2f}; "
                f"안옷 {layering_prediction.inner_category}, 겉옷 {layering_prediction.outer_category}."
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
                visible_sleeve_length=sleeve.visible_length,
                sleeve_state=sleeve.state,
                input_valid=not sleeve.requires_retake,
                input_error_code=sleeve.error_code,
                input_error_message=sleeve.error_message,
                layering_state=layering_state,
                upper_items=upper_items or [attributes["upper_type"]],
                inner_category=inner_category,
                outer_category=outer_category,
                wear_state_confidence={
                    "sleeve_state": round(sleeve.confidence, 3),
                    "layering_state": round(layering_confidence, 3),
                },
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


def _refine_upper_type(
    category: str,
    collar: str,
    *,
    layering_state: str = "단일 상의",
    material: str = "",
    neckline: str = "",
) -> str:
    """카테고리와 독립 칼라 헤드가 합의할 때 한국 쇼핑 분류명으로 구체화한다."""
    if layering_state != "단일 상의":
        return category
    if "니트" in material or neckline == "V넥":
        return category
    if category == "티셔츠" and collar == "폴로 칼라":
        return "폴로 셔츠"
    return category
