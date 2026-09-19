"""무신사 상품 이미지에서 의류 색상의 웜/쿨 성향을 추정한다.

상품 상세 색상 데이터가 없는 검색 결과용 휴리스틱이다. 일반 의류는 지각 색상
팔레트와 비교하고, 데님은 워싱·스티치·금속 디테일 규칙을 별도로 반영한다.
"""

from __future__ import annotations

import colorsys
import io
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
from PIL import Image, UnidentifiedImageError


WARM_TONE = "웜톤"
COOL_TONE = "쿨톤"
VALID_PERSONAL_TONES = {WARM_TONE, COOL_TONE}

# 이름이 아니라 실제 RGB 기준색과 비교한다. 각 계열 안에서도 명도·채도가 다른
# 기준색을 여러 개 둬서 베이지/카키/코랄과 네이비/블루/라벤더를 함께 다룬다.
WARM_PALETTE = np.array([
    (244, 226, 190), (207, 180, 132), (180, 126, 72), (116, 76, 47),
    (128, 126, 67), (102, 112, 57), (214, 161, 47), (211, 105, 79),
    (190, 73, 49), (205, 112, 48), (63, 122, 120),
], dtype=np.float64)
COOL_PALETTE = np.array([
    (233, 239, 245), (146, 154, 167), (55, 62, 75), (37, 55, 91),
    (48, 88, 174), (84, 127, 188), (168, 202, 226), (137, 119, 181),
    (184, 82, 135), (151, 45, 78), (53, 119, 151),
], dtype=np.float64)

WARM_WORDS = ("웜", "크림", "베이지", "카멜", "브라운", "카키", "올리브", "머스타드", "코랄", "테라코타", "오렌지", "골드", "브론즈", "구리")
COOL_WORDS = ("쿨", "차콜", "라벤더", "라일락", "코발트", "실버", "그레이시", "아이스 블루")
DENIM_WORDS = ("데님", "denim", "청바지", "진팬츠", "jeans", " jean")
WARM_DENIM_WORDS = ("빈티지", "티스테인", "옐로 워싱", "브라운 워싱", "오렌지 스티치", "브라운 스티치", "골드 버튼", "브론즈", "구리")
COOL_DENIM_WORDS = ("아이스 블루", "딥 인디고", "그레이시 블루", "그레이 워싱", "실버 버튼", "실버 리벳", "화이트 스티치")
WARM_DENIM_PALETTE = np.array([(101, 119, 121), (115, 126, 121), (91, 105, 105), (126, 114, 92)], dtype=np.float64)
COOL_DENIM_PALETTE = np.array([(161, 197, 223), (86, 126, 178), (43, 64, 104), (112, 128, 151)], dtype=np.float64)


@dataclass(frozen=True)
class ProductToneResult:
    tone: str = "판단 불가"
    confidence: float = 0.0
    is_denim: bool = False
    evidence: list[str] = field(default_factory=list)

    def matches(self, personal_tone: str, threshold: float = 0.18) -> bool:
        return personal_tone in VALID_PERSONAL_TONES and self.tone == personal_tone and self.confidence >= threshold


def _foreground_pixels(image: Image.Image, lower_focus: bool = False) -> np.ndarray:
    image = image.convert("RGB")
    image.thumbnail((240, 240), Image.Resampling.LANCZOS)
    rgb = np.asarray(image, dtype=np.uint8)
    if lower_focus and rgb.shape[0] >= 20:
        # 착용 사진의 얼굴·머리카락이 데님 워싱 색으로 집계되지 않게 하의 영역에 집중한다.
        rgb = rgb[int(rgb.shape[0] * 0.22):]
    h, w = rgb.shape[:2]
    if h < 2 or w < 2:
        return rgb.reshape(-1, 3)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1])).astype(np.float64)
    background = np.median(border, axis=0)
    flat = rgb.reshape(-1, 3).astype(np.float64)
    distance = np.linalg.norm(flat - background, axis=1)
    maximum, minimum = flat.max(axis=1), flat.min(axis=1)
    saturation = (maximum - minimum) / np.maximum(maximum, 1.0)
    keep = (distance > 24) & ~((maximum > 242) & (saturation < 0.08))
    pixels = flat[keep]
    if len(pixels) < max(80, len(flat) // 20):
        pixels = flat
    if len(pixels) > 6000:
        pixels = pixels[np.linspace(0, len(pixels) - 1, 6000, dtype=int)]
    return pixels


def _palette_vote(pixels: np.ndarray, warm: np.ndarray, cool: np.ndarray) -> tuple[str, float]:
    # 밝기 차이의 영향을 줄이고 색 방향을 더 보도록 RGB를 정규화한다.
    scale = np.maximum(pixels.mean(axis=1, keepdims=True), 24.0)
    normalized = pixels / scale
    warm_n = warm / np.maximum(warm.mean(axis=1, keepdims=True), 24.0)
    cool_n = cool / np.maximum(cool.mean(axis=1, keepdims=True), 24.0)
    warm_d = np.sqrt(((normalized[:, None, :] - warm_n[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    cool_d = np.sqrt(((normalized[:, None, :] - cool_n[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    margin = float(np.median(cool_d - warm_d))
    confidence = min(0.92, abs(margin) / 0.34)
    return (WARM_TONE if margin > 0 else COOL_TONE), confidence


def classify_product_tone(
    image: Image.Image,
    product_name: str = "",
    search_keywords: Iterable[str] = (),
) -> ProductToneResult:
    text = " ".join((product_name, *[str(value) for value in search_keywords])).lower()
    is_denim = any(word in text for word in DENIM_WORDS)
    pixels = _foreground_pixels(image, lower_focus=is_denim)
    if not len(pixels):
        return ProductToneResult()
    warm_words = WARM_DENIM_WORDS if is_denim else WARM_WORDS
    cool_words = COOL_DENIM_WORDS if is_denim else COOL_WORDS
    warm_hits = [word for word in warm_words if word in text]
    cool_hits = [word for word in cool_words if word in text]
    chroma = (pixels.max(axis=1) - pixels.min(axis=1)) / np.maximum(pixels.max(axis=1), 1.0)
    useful = pixels[chroma >= 0.10]
    if len(useful) < 50:
        text_score = 0.34 * (len(warm_hits) - len(cool_hits))
        evidence = ["데님 전용 워싱·디테일 규칙" if is_denim else "상품명 색상 단서"]
        evidence.extend(f"상품명 단서: {word}" for word in (warm_hits + cool_hits)[:2])
        if not text_score:
            return ProductToneResult(is_denim=is_denim, evidence=["무채색 또는 색상 정보 부족"])
        return ProductToneResult(
            WARM_TONE if text_score > 0 else COOL_TONE,
            round(min(0.90, abs(text_score)), 3),
            is_denim,
            evidence,
        )

    warm_palette = WARM_DENIM_PALETTE if is_denim else WARM_PALETTE
    cool_palette = COOL_DENIM_PALETTE if is_denim else COOL_PALETTE
    tone, image_confidence = _palette_vote(useful, warm_palette, cool_palette)
    score = image_confidence if tone == WARM_TONE else -image_confidence
    score += 0.32 * len(warm_hits) - 0.32 * len(cool_hits)

    evidence = ["데님 전용 워싱·디테일 규칙" if is_denim else "상품 이미지 대표색"]
    evidence.extend(f"상품명 단서: {word}" for word in (warm_hits + cool_hits)[:2])
    if is_denim:
        # 전체는 파란색이어도 소량의 오렌지/브라운 스티치·틴팅이 있으면 웜 근거다.
        warm_accents = 0
        for red, green, blue in useful.astype(np.uint8):
            hue, sat, value = colorsys.rgb_to_hsv(red / 255, green / 255, blue / 255)
            degrees = hue * 360
            if 8 <= degrees <= 55 and sat >= 0.30 and 0.20 <= value <= 0.92:
                warm_accents += 1
        accent_ratio = warm_accents / max(len(useful), 1)
        if 0.004 <= accent_ratio <= 0.20:
            score += min(0.24, accent_ratio * 2.0)
            evidence.append("오렌지·브라운 계열 워싱/스티치 포인트")
        gray_detail = pixels[(chroma < 0.08) & (pixels.mean(axis=1) > 105) & (pixels.mean(axis=1) < 238)]
        gray_ratio = len(gray_detail) / max(len(pixels), 1)
        if 0.004 <= gray_ratio <= 0.18:
            score -= min(0.12, gray_ratio)
            evidence.append("화이트·그레이 계열 워싱/스티치 포인트")

    confidence = min(0.98, abs(score))
    if confidence < 0.12:
        return ProductToneResult(is_denim=is_denim, evidence=evidence)
    return ProductToneResult(WARM_TONE if score > 0 else COOL_TONE, round(confidence, 3), is_denim, evidence)


def classify_product_tone_url(
    image_url: str,
    product_name: str = "",
    search_keywords: Iterable[str] = (),
    timeout: float = 3.5,
) -> ProductToneResult:
    parsed = urllib.parse.urlparse(image_url)
    host = (parsed.hostname or "").lower()
    if parsed.scheme != "https" or not (host == "msscdn.net" or host.endswith(".msscdn.net")):
        return ProductToneResult(evidence=["허용되지 않은 이미지 주소"])
    request = urllib.request.Request(image_url, headers={"User-Agent": "Mozilla/5.0 (compatible; FITTA/1.0)"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = response.read(6 * 1024 * 1024 + 1)
        if len(payload) > 6 * 1024 * 1024:
            return ProductToneResult(evidence=["이미지 용량 초과"])
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 30_000_000:
                return ProductToneResult(evidence=["이미지 해상도 초과"])
            return classify_product_tone(image, product_name, search_keywords)
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError):
        return ProductToneResult(evidence=["상품 이미지 분석 실패"])
