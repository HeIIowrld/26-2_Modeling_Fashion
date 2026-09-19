from __future__ import annotations

"""가상 피팅 합성 후 품질 검사.

합성 결과를 FASHN 파서로 다시 분할해 사람 눈 대신 수치로 점검한다. README 7단계의
"얼굴·체형 유지 → 옷 반영 확인 → 기준 미달 시에만 1장 추가 생성" 규칙의 검사 부분이다.

검사는 두 종류로 나눈다.
- 재생성 대상(retryable): 같은 마스크에서 시드만 바꿔도 달라지는 생성 실패.
  예) 마스크 안인데 배를 맨살로 그림, 상품과 다른 옷을 그림.
- 구조적 한계: 마스크·입력 모양에서 오는 실패라 다시 뽑아도 거의 같다.
  예) 원래 크롭탑 밑단에서 마스크가 끝나 긴 후드도 짧게 잘림. 경고만 남긴다.

기준값은 data/tryon_quality_thresholds.json에 있고 모두 잠정값이다. 근거와 검증 범위는
reports/vton_quality/ 의 해당 날짜 리포트를 따른다.
"""

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from config import DATA_DIR

THRESHOLDS_PATH = DATA_DIR / "tryon_quality_thresholds.json"
DEFAULT_THRESHOLDS = {
    "outside_psnr_min": 30.0,
    "face_psnr_min": 30.0,
    "coverage_min": 0.5,
    "torso_skin_inside_max": 0.08,
    "torso_skin_outside_max": 0.05,
    "garment_leak_max": 0.05,
    "fidelity_delta_min": -0.03,
    "color_de_max": 25.0,
    "length_gap_max": 1.0,
    "sharpness_min": 25.0,
}

FACE_LABELS = (1, 2)
TORSO_SKIN_LABEL = 16
# 배를 드러내는 게 상품의 의도인 경우. 이름은 판매자가 붙인 사실상의 정답이다.
EXPOSED_MIDRIFF_NAME = re.compile(
    r"크롭|crop|브라|bra\b|bralette|뷔스티에|bustier|컷아웃|cut-?out|튜브|tube", re.IGNORECASE
)


@dataclass
class QualityCheck:
    name: str
    value: float | None
    threshold: float
    passed: bool
    retryable: bool
    message: str = ""


@dataclass
class TryOnQualityReport:
    category: str
    checks: list[QualityCheck] = field(default_factory=list)
    skipped: str = ""

    @property
    def failed(self) -> list[QualityCheck]:
        return [check for check in self.checks if not check.passed]

    @property
    def retry_recommended(self) -> bool:
        return any(check.retryable for check in self.failed)

    def penalty(self) -> float:
        """시도끼리 비교할 점수. 재생성으로 달라지는 실패에 더 큰 무게를 둔다."""
        return sum(1.0 if check.retryable else 0.25 for check in self.failed)

    def warnings(self) -> list[str]:
        return [check.message for check in self.failed if check.message]

    def to_dict(self) -> dict:
        return {"category": self.category, "skipped": self.skipped,
                "checks": [asdict(check) for check in self.checks]}


def load_thresholds(path: Path = THRESHOLDS_PATH) -> dict[str, float]:
    thresholds = dict(DEFAULT_THRESHOLDS)
    try:
        stored = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return thresholds
    thresholds.update({key: float(value) for key, value in stored.get("thresholds", {}).items()})
    return thresholds


def region_psnr(first: np.ndarray, second: np.ndarray, region: np.ndarray) -> float | None:
    if region.sum() < 100:
        return None
    diff = first[region].astype(np.float64) - second[region].astype(np.float64)
    mse = float((diff ** 2).mean())
    return 99.0 if mse < 1e-10 else float(10 * np.log10(255.0 ** 2 / mse))


def _dilate(mask: np.ndarray, ratio: float) -> np.ndarray:
    import cv2

    size = max(3, int(min(mask.shape[:2]) * ratio) | 1)
    return cv2.dilate(mask.astype(np.uint8), np.ones((size, size), np.uint8)) > 0


def body_band(
    landmarks_px: dict[str, tuple[float, float]] | None,
    shape: tuple[int, int],
    category: str,
) -> np.ndarray | None:
    """어떤 상품이든 옷이 덮어야 하는 몸통 핵심 영역.

    상의: 가슴 아래~골반 관절 위(어깨→골반 45~90%), 좌우는 두 골반 관절 사이.
    골반 관절은 몸 안쪽에 있어 옆구리·팔 사이 배경이 섞이지 않는다. 깊은 V넥·원숄더도
    이 높이까지는 내려오지 않는다.
    하의: 골반 관절~무릎 35% 높이. 미니 기장 쇼츠·치마도 이 띠는 덮는다.
    """
    if not landmarks_px:
        return None
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if category == "bottom":
        names = ("left_hip", "right_hip", "left_knee", "right_knee")
    if any(name not in landmarks_px for name in names):
        return None
    height, width = shape
    first_y = (landmarks_px[names[0]][1] + landmarks_px[names[1]][1]) / 2
    second_y = (landmarks_px[names[2]][1] + landmarks_px[names[3]][1]) / 2
    if second_y <= first_y:
        return None
    if category == "top":
        top, bottom = first_y + 0.45 * (second_y - first_y), first_y + 0.90 * (second_y - first_y)
        xs = (landmarks_px["left_hip"][0], landmarks_px["right_hip"][0])
    else:
        top, bottom = first_y, first_y + 0.35 * (second_y - first_y)
        xs = (landmarks_px["left_hip"][0], landmarks_px["right_hip"][0])
    band = np.zeros((height, width), dtype=bool)
    y1, y2 = int(max(0, round(top))), int(min(height, round(bottom)))
    x1, x2 = int(max(0, round(min(xs)))), int(min(width, round(max(xs))))
    if y2 - y1 < 3 or x2 - x1 < 3:
        return None
    band[y1:y2, x1:x2] = True
    return band


def dominant_color_distance(
    reference_rgb: np.ndarray, reference_mask: np.ndarray, rgb: np.ndarray, mask: np.ndarray
) -> float | None:
    from outfit_analyzer import _ciede2000, _dominant_rgb, _rgb_to_cielab

    if reference_mask.sum() < 400 or mask.sum() < 400:
        return None
    first = _rgb_to_cielab(_dominant_rgb(reference_rgb, reference_mask))
    second = _rgb_to_cielab(_dominant_rgb(rgb, mask))
    return float(_ciede2000(first, second))


def rendered_bottom_length(
    segmentation: np.ndarray, landmarks_px: dict[str, tuple[float, float]]
) -> str:
    """합성 결과의 하의 기장을 현재 착장 분석과 같은 측정기로 잰다."""
    from types import SimpleNamespace

    from garment_attribute_analyzer import GarmentAttributeAnalyzer

    height, width = segmentation.shape
    # landmarks_px는 이미 보이는 관절만 남긴 값이다. 측정기의 가시도 검사를 통과하도록 1.0을 준다.
    pose = SimpleNamespace(landmarks={
        name: (x / width, y / height, 1.0) for name, (x, y) in landmarks_px.items()
    })
    areas = {label: int((segmentation == label).sum()) for label in (4, 5, 6)}
    if max(areas.values()) == 0:
        return "분석 불가"
    label = max(areas, key=areas.get)
    garment_type = {4: "원피스", 5: "치마", 6: "바지"}[label]
    return GarmentAttributeAnalyzer()._bottom_length(
        (segmentation == label).astype(np.uint8), pose, garment_type
    )


def _bottom_order_gap(reference: str, rendered: str) -> int | None:
    from catvton_tryon import BOTTOM_LENGTH_ORDER

    first = BOTTOM_LENGTH_ORDER.get((reference or "").replace(" 추정", ""))
    second = BOTTOM_LENGTH_ORDER.get((rendered or "").replace(" 추정", ""))
    if first is None or second is None:
        return None
    return abs(first - second)


def assess_tryon(
    *,
    category: str,
    before: np.ndarray,
    after: np.ndarray,
    edit_mask: np.ndarray,
    after_seg: np.ndarray,
    target_labels: tuple[int, ...],
    before_seg: np.ndarray | None = None,
    landmarks_px: dict[str, tuple[float, float]] | None = None,
    product_name: str = "",
    reference_rgb: np.ndarray | None = None,
    reference_mask: np.ndarray | None = None,
    embed: Callable[[Image.Image], np.ndarray] | None = None,
    sharpness: float | None = None,
    reference_length: str = "",
    thresholds: dict[str, float] | None = None,
) -> TryOnQualityReport:
    """한 카테고리 합성 단계의 결과를 점검한다. 모든 배열은 같은 해상도여야 한다."""
    from outfit_analyzer import _masked_crop

    limits = dict(DEFAULT_THRESHOLDS if thresholds is None else thresholds)
    report = TryOnQualityReport(category=category)
    edit = edit_mask.astype(bool)
    if before.shape != after.shape or edit.shape != after.shape[:2] or after_seg.shape != edit.shape:
        report.skipped = "해상도 불일치"
        return report
    add = report.checks.append
    label = "상의" if category == "top" else "하의"

    # 1) 보존: repaint가 마스크 밖을 원본으로 되돌리므로 여기서 떨어지면 좌표·크기 버그다.
    outside = ~_dilate(edit, 0.015)
    value = region_psnr(before, after, outside)
    if value is not None:
        add(QualityCheck("outside_preservation", round(value, 2), limits["outside_psnr_min"],
                         value >= limits["outside_psnr_min"], False,
                         "합성 영역 밖의 원본 사진이 달라졌습니다. 결과 이미지 정렬을 확인해야 합니다."))
    if before_seg is not None:
        face = np.isin(before_seg, FACE_LABELS) & ~edit
        value = region_psnr(before, after, face)
        if value is not None:
            add(QualityCheck("face_preservation", round(value, 2), limits["face_psnr_min"],
                             value >= limits["face_psnr_min"], False,
                             "얼굴·머리 영역이 원본과 달라졌습니다."))

    target_after = np.isin(after_seg, target_labels)
    band = body_band(landmarks_px, edit.shape, category)

    # 2) 옷이 몸통 핵심 영역을 덮었는가
    if band is not None and (band & edit).sum() >= 200:
        core = band & edit
        value = float(target_after[core].mean())
        add(QualityCheck("garment_coverage", round(value, 4), limits["coverage_min"],
                         value >= limits["coverage_min"], True,
                         f"{label} 상품이 몸통을 충분히 덮지 못했습니다(덮인 비율 {value:.0%})."))

    # 3) 상의 크롭화: 의도하지 않은 배 노출. 마스크 안(생성 실패)과 밖(마스크 구조)을 나눈다.
    if category == "top" and band is not None and band.sum() >= 200 \
            and not EXPOSED_MIDRIFF_NAME.search(product_name or ""):
        skin = after_seg == TORSO_SKIN_LABEL
        inside = float((skin & band & edit).sum()) / float(band.sum())
        outside_skin = float((skin & band & ~edit).sum()) / float(band.sum())
        add(QualityCheck("torso_skin_inside_mask", round(inside, 4), limits["torso_skin_inside_max"],
                         inside <= limits["torso_skin_inside_max"], True,
                         f"상의가 상품보다 짧거나 트이게 그려져 배·가슴 피부가 드러났습니다({inside:.0%})."))
        add(QualityCheck("torso_skin_outside_mask", round(outside_skin, 4), limits["torso_skin_outside_max"],
                         outside_skin <= limits["torso_skin_outside_max"], False,
                         "원래 옷이 짧거나 앞이 트여 있어 새 상의가 그 부분을 덮지 못했습니다"
                         f"({outside_skin:.0%}). 몸통이 가려진 사진에서 더 정확합니다."))

    # 4) 원래 옷 잔류: repaint 뒤 마스크 밖에 남은 대상 옷은 원래 옷이다.
    if target_after.sum() >= 400:
        value = float((target_after & ~_dilate(edit, 0.02)).sum()) / float(target_after.sum())
        add(QualityCheck("garment_leak", round(value, 4), limits["garment_leak_max"],
                         value <= limits["garment_leak_max"], False,
                         f"합성 영역 밖에 원래 {label}가 일부 남았습니다({value:.0%})."))

    # 5) 상품 충실도: 결과 옷이 원래 옷보다 상품 레퍼런스에 더 가까워야 한다.
    if embed is not None and reference_rgb is not None and target_after.sum() >= 400:
        before_garment = (np.isin(before_seg, target_labels) if before_seg is not None else edit) & edit
        if before_garment.sum() >= 400:
            ref_mask = reference_mask if reference_mask is not None else np.ones(reference_rgb.shape[:2], bool)
            ref_vec = embed(_masked_crop(reference_rgb, ref_mask))
            sim_after = float(embed(_masked_crop(after, target_after)) @ ref_vec)
            sim_before = float(embed(_masked_crop(before, before_garment)) @ ref_vec)
            value = sim_after - sim_before
            # 원래 옷과 똑같이 남은 결과(차이 0)도 실패여야 하므로 초과 비교다.
            add(QualityCheck("reference_fidelity", round(value, 4), limits["fidelity_delta_min"],
                             value > limits["fidelity_delta_min"], True,
                             f"합성된 {label}가 추천 상품보다 원래 옷에 더 가깝습니다."))
    if reference_rgb is not None and reference_mask is not None:
        value = dominant_color_distance(reference_rgb, reference_mask, after, target_after)
        if value is not None:
            add(QualityCheck("color_fidelity", round(value, 2), limits["color_de_max"],
                             value <= limits["color_de_max"], True,
                             f"합성된 {label} 색이 상품 사진과 다릅니다(색차 {value:.0f})."))

    # 6) 하의 기장·종류 충실도: 상품은 긴바지인데 원래 치마·반바지 마스크 모양대로 짧게
    #    그려지거나, 반바지인데 원래 긴바지 자리까지 레깅스처럼 채우는 실패(2026-09-15 확인).
    if category == "bottom" and reference_length and landmarks_px:
        rendered = rendered_bottom_length(after_seg, landmarks_px)
        gap = _bottom_order_gap(reference_length, rendered)
        if gap is not None:
            add(QualityCheck("bottom_length_fidelity", float(gap), limits["length_gap_max"],
                             gap <= limits["length_gap_max"], False,
                             f"상품은 {reference_length.replace(' 추정', '')}인데 합성 결과는 {rendered}로 "
                             "그려졌습니다. 원래 하의의 모양을 따라가 기장이 달라졌습니다."))

    if sharpness is not None:
        add(QualityCheck("sharpness", round(sharpness, 1), limits["sharpness_min"],
                         sharpness >= limits["sharpness_min"], True,
                         f"합성된 {label}가 흐릿합니다."))
    return report
