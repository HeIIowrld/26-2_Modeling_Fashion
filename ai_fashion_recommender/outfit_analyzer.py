from __future__ import annotations

import colorsys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from clothing_parser import ClothingParser
from config import ATTRIBUTE_CONFIDENCE_THRESHOLD
from fashion_prompts import MATERIAL_PROMPTS, NECKLINE_PROMPTS, PATTERN_PROMPTS, STYLE_PROMPTS
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
}
NEUTRALS = {"블랙", "화이트", "그레이", "네이비", "브라운", "베이지"}


def _dominant_rgb(rgb: np.ndarray, mask: np.ndarray) -> tuple[int, int, int]:
    pixels = rgb[mask]
    if len(pixels) == 0:
        return 128, 128, 128
    # 지나치게 밝거나 어두운 그림자/하이라이트를 줄인 뒤 대표 색 군집을 찾는다.
    brightness = pixels.mean(axis=1)
    filtered = pixels[(brightness > 20) & (brightness < 245)]
    if len(filtered) < 20:
        filtered = pixels
    if len(filtered) > 5000:
        indices = np.linspace(0, len(filtered) - 1, 5000, dtype=int)
        filtered = filtered[indices]
    samples = filtered.astype(np.float32)
    cluster_count = min(3, len(samples))
    _, labels, centers = cv2.kmeans(
        samples,
        cluster_count,
        None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2),
        5,
        cv2.KMEANS_PP_CENTERS,
    )
    counts = np.bincount(labels.ravel(), minlength=cluster_count)
    hsv = cv2.cvtColor(np.uint8(centers[np.newaxis, :, :]), cv2.COLOR_RGB2HSV)[0]
    saturation = hsv[:, 1] / 255.0
    # 유색 옷은 무채색 하이라이트가 조금 더 많아도 실제 색 군집을 선택한다.
    scores = counts * (1.0 + 0.35 * saturation)
    return tuple(np.clip(centers[int(np.argmax(scores))], 0, 255).astype(int))


def _nearest_color(rgb: tuple[int, int, int]) -> str:
    return min(COLOR_PALETTE, key=lambda name: np.linalg.norm(np.array(rgb) - np.array(COLOR_PALETTE[name])))


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


def _masked_crop(rgb: np.ndarray, mask: np.ndarray) -> Image.Image:
    """배경 영향을 줄이기 위해 선택한 의류만 흰 배경 위에 남긴다."""
    if not mask.any():
        return Image.fromarray(rgb)
    ys, xs = np.where(mask)
    padding = 12
    x1, x2 = max(0, xs.min() - padding), min(rgb.shape[1], xs.max() + padding + 1)
    y1, y2 = max(0, ys.min() - padding), min(rgb.shape[0], ys.max() + padding + 1)
    crop = rgb[y1:y2, x1:x2].copy()
    crop_mask = mask[y1:y2, x1:x2]
    crop[~crop_mask] = 255
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
        upper_color = _nearest_color(_dominant_rgb(rgb, parsed["upper_mask"]))
        lower_color = _nearest_color(_dominant_rgb(rgb, parsed["lower_mask"]))
        style, style_confidence = self.classifier.best_mapped_label(image, STYLE_PROMPTS)
        if self.classifier.enabled:
            garment_crop = _masked_crop(rgb, parsed["upper_mask"])
            garment_results = self.classifier.best_mapped_labels(
                garment_crop,
                {"pattern": PATTERN_PROMPTS, "material": MATERIAL_PROMPTS, "neckline": NECKLINE_PROMPTS},
            )
            pattern, pattern_confidence = garment_results["pattern"]
            material, material_confidence = garment_results["material"]
            neckline, neckline_confidence = garment_results["neckline"]
            threshold = ATTRIBUTE_CONFIDENCE_THRESHOLD
            attributes["pattern"] = pattern if pattern_confidence >= threshold else "패턴 불확실"
            attributes["material"] = material if material_confidence >= threshold else "소재 불확실"
            attributes["neckline"] = neckline if neckline_confidence >= threshold else "네크라인 불확실"
        else:
            style_confidence = pattern_confidence = material_confidence = neckline_confidence = 0.0
            attributes["neckline"] = "분석 보류"
        notes = []
        if self.parser.backend == "pose-guided-fallback":
            notes.append("포즈 기반 임시 마스크입니다. 의류 길이와 종류를 정식 분석에 사용하지 마세요.")
        if not self.classifier.enabled:
            notes.append("FashionSigLIP이 꺼져 있어 스타일·패턴·소재·네크라인은 분석하지 않았습니다.")
        else:
            notes.append(
                f"FashionSigLIP 신뢰도: 스타일 {style_confidence:.2f}, 패턴 {pattern_confidence:.2f}, "
                f"소재 {material_confidence:.2f}, 네크라인 {neckline_confidence:.2f}."
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
                sleeve_length=attributes["sleeve_length"],
                upper_length=attributes["upper_length"],
                bottom_length=attributes["bottom_length"],
                fit=attributes["fit"],
                neckline=attributes["neckline"],
                pattern=attributes["pattern"],
                material=attributes["material"],
                attribute_confidence=attributes["attribute_confidence"],
                notes=notes,
            ),
            parsed,
        )
