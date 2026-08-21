from __future__ import annotations

"""CatVTON 기반 실제 가상 피팅 어댑터.

CatVTON(비상업 CC BY-NC-SA 4.0)의 디퓨전 파이프라인으로 추천 상품 이미지를
사람 사진에 합성한다. CatVTON 원본의 AutoMasker(detectron2 필요) 대신
프로젝트의 FASHN Human Parser 마스크를 사용하므로 Windows에서도 동작한다.

요구사항:
- third_party/CatVTON 저장소 클론 (모델 코드)
- GPU torch 환경 (CPU도 동작하지만 장당 수 분 이상 소요)
- 첫 실행 시 HuggingFace에서 체크포인트 자동 다운로드 (약 5GB)

디테일 보존 처리:
- 무신사 대표 이미지에는 다른 컬러웨이·색상 스와치·모델이 입은 다른 옷이 함께
  나오는 경우가 많다. FASHN 파서로 대상 카테고리(상/하의)의 옷 영역만 남기고
  나머지는 흰 배경으로 지운 정제 이미지를 만들어 CatVTON에 넣는다.
- CatVTON 공식 inference.py/app.py와 동일하게 마스크를 블러 처리해 파이프라인에
  넣고, 생성 결과는 블러 마스크로 원본과 다시 합성(repaint)한다. VAE 왕복으로
  얼굴·배경 디테일이 흐려지는 것을 막고 경계 이음매를 부드럽게 만든다.
- 마스크 영역의 선명도(라플라시안 분산)가 기준에 못 미치면 시드를 바꿔 한 번 더
  생성하고 더 선명한 쪽을 선택한다(README 7단계의 "기준 미달 시에만 재생성" 규칙).
"""

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from config import GARMENT_CLEAN_DIR, PROJECT_DIR, garment_image_path
from schemas import Recommendation
from virtual_tryon import VirtualTryOnAdapter

CATVTON_REPO = PROJECT_DIR.parent / "third_party" / "CatVTON"
BASE_CKPT = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT = "zhengchong/CatVTON"
GARMENT_CACHE_DIR = GARMENT_CLEAN_DIR

# clothing_parser.LABELS 기준: 3=top, 4=dress, 5=skirt, 6=pants, 7=belt, 10=scarf
GARMENT_TARGET_LABELS = {
    "top": (3, 4, 10),
    "bottom": (4, 5, 6, 7),
}

# 인페인팅 마스크에서 항상 제외해 원본을 보존하는 라벨:
# 1=face, 2=hair, 8=bag, 9=hat, 11=glasses, 13=hands, 15=feet
PROTECT_LABELS = (1, 2, 8, 9, 11, 13, 15)

# 하의 기장 순서. VTON은 마스크 모양대로 옷을 그리는 경향이 강해서, 원래 옷보다
# 짧은 하의를 입히면 마스크 하단(원래 바지가 있던 종아리)을 맨다리가 아니라
# 옷 비슷한 것으로 채운다(퍼지 레그워머·니삭스 환각).
BOTTOM_LENGTH_ORDER = {
    "쇼츠·미니 기장": 0,
    "미니 기장": 0,
    "반바지": 0,
    "무릎 기장": 1,
    "무릎 기장 바지": 1,
    "미디·7부 기장": 2,
    "미디 기장": 2,
    "크롭·7부 바지": 2,
    "롱·긴바지 기장": 3,
    "롱·맥시 기장": 3,
    "긴바지": 3,
}
# 학습된 category 헤드는 상품 레퍼런스에서 신뢰도가 높다(쇼츠 0.92 / 팬츠 0.88).
# 반면 pant_length 헤드는 상품 crop에서 거의 항상 보류라 기장 판정에 쓰지 않는다.
SHORT_BOTTOM_CATEGORIES = {"쇼츠"}

SLEEVE_LENGTH_ORDER = {"민소매": 0, "반팔": 1, "7부 소매": 2, "긴팔": 3}

# 실측 기준(female_012 통제 실험): gap=1은 정상 합성, gap=3은 마스크 전체가
# 옷 텍스처로 채워지는 실패. gap=2는 미검증이라 보수적으로 경고에 포함한다.
UNRELIABLE_LENGTH_GAP = 2


def _length_gap(order: dict[str, int], current: str, target: str) -> int | None:
    first = order.get((current or "").replace(" 추정", ""))
    second = order.get((target or "").replace(" 추정", ""))
    if first is None or second is None:
        return None
    return first - second


def bottom_length_gap(current_length: str, target_length: str) -> int | None:
    """원래 하의 대비 새 하의가 몇 단계 짧은지 센다. 판정 불가면 None.

    양수면 새 옷이 더 짧아 맨다리를 새로 그려야 하는 어려운 합성이다.
    """
    return _length_gap(BOTTOM_LENGTH_ORDER, current_length, target_length)


def sleeve_length_gap(current_length: str, target_length: str) -> int | None:
    """원래 상의 대비 새 상의 소매가 몇 단계 짧은지 센다.

    하의와 같은 원리다. 마스크가 긴팔 모양이면 반팔 상품을 넣어도 모델이
    소매를 채워 긴팔로 그린다(female_012 A안에서 확인).
    """
    return _length_gap(SLEEVE_LENGTH_ORDER, current_length, target_length)


def classify_reference_bottom_length(classifier, image) -> str:
    """상품 레퍼런스 이미지의 하의 기장을 학습된 속성 헤드로 추정한다.

    상품 crop에서는 category 헤드가 가장 신뢰도가 높고(쇼츠 0.92), lower_length는
    보조로만 쓴다. 판정하지 못하면 빈 문자열을 돌려 게이트를 건너뛴다.
    """
    if classifier is None or not getattr(classifier, "trained_attributes_enabled", False):
        return ""
    prediction = classifier.predict_trained_attributes(
        image, tasks=["category", "lower_length"]
    )
    category = prediction.get("category")
    if category and category.accepted and category.labels[0] in SHORT_BOTTOM_CATEGORIES:
        return "쇼츠·미니 기장"
    length = prediction.get("lower_length")
    if length and length.accepted:
        return length.labels[0]
    return ""


def classify_reference_sleeve_length(classifier, image) -> str:
    """상품 레퍼런스의 소매 길이를 추정한다. 판정 불가면 빈 문자열."""
    if classifier is None or not getattr(classifier, "trained_attributes_enabled", False):
        return ""
    prediction = classifier.predict_trained_attributes(image, tasks=["sleeve_length"])
    sleeve = prediction.get("sleeve_length")
    return sleeve.labels[0] if sleeve and sleeve.accepted else ""


def _solidify_mask(mask: np.ndarray) -> np.ndarray:
    """마스크의 오목한 홈(라펠·자락·주름)을 메워 뭉툭하게 만든다.

    인페인팅 마스크가 원래 옷의 윤곽을 그대로 드러내면 모델이 그 모양의 옷을 다시
    그리는 경향이 있어 학습 데이터의 agnostic 마스크처럼 다듬는다. 볼록 껍질은 팔과
    몸통 사이 공간까지 메워 망토 모양 아티팩트를 만들었으므로(20장 배치에서 확인),
    국소 오목부만 메우고 팔-몸통 간격은 남기는 모폴로지 닫힘을 쓴다.
    """
    import cv2

    binary = (mask > 0).astype(np.uint8)
    size = max(3, int(min(mask.shape[:2]) * 0.05) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)


def _dilate_mask(mask: np.ndarray, ratio: float = 0.03) -> np.ndarray:
    """마스크 경계에 여유를 준다. 인셋 repaint가 경계 블렌딩 밴드를 마스크 안쪽에
    만들기 때문에, 원래 옷의 윤곽이 밴드보다 깊이 덮이도록 약간의 여유가 필요하다."""
    import cv2

    binary = (mask > 0).astype(np.uint8) * 255
    kernel_size = max(3, int(min(mask.shape[:2]) * ratio) | 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    return cv2.dilate(binary, kernel, iterations=1)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    """나란히 놓인 다른 컬러웨이나 색상 스와치 점을 걸러내고 가장 큰 옷 덩어리만 남긴다."""
    import cv2

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if num_labels <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = 1 + int(np.argmax(areas))
    return labels == keep


def _refine_garment_mask(mask: np.ndarray) -> np.ndarray:
    """세그멘테이션 경계를 다듬는다: 닫힘 연산으로 파인 홈을 메우고, 침식으로
    경계에 혼입된 배경 픽셀(어두운 테두리)을 깎아낸다."""
    import cv2

    binary = mask.astype(np.uint8)
    size = min(mask.shape[:2])
    close_size = max(3, int(size * 0.01) | 1)
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    smoothed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    eroded = cv2.erode(smoothed, erode_kernel, iterations=max(1, round(size * 0.004)))
    return (eroded if eroded.any() else smoothed).astype(bool)


def _paste_on_white(rgb: np.ndarray, mask: np.ndarray, padding: int = 16) -> Image.Image:
    """옷 픽셀만 남기고 흰 배경과 부드럽게 섞는다. 하드 컷은 계단 현상과 어두운
    테두리를 남겨 합성 결과에 그대로 배어나온다."""
    import cv2

    ys, xs = np.where(mask)
    x1, x2 = max(0, xs.min() - padding), min(rgb.shape[1], xs.max() + padding + 1)
    y1, y2 = max(0, ys.min() - padding), min(rgb.shape[0], ys.max() + padding + 1)
    crop = rgb[y1:y2, x1:x2].astype(np.float32)
    alpha = mask[y1:y2, x1:x2].astype(np.float32)
    # 시그마가 크면 반투명 테두리가 생겨 모델이 시스루 소재로 오해할 수 있다.
    sigma = max(1.0, min(rgb.shape[:2]) * 0.001)
    alpha = cv2.GaussianBlur(alpha, (0, 0), sigma)[..., None]
    blended = crop * alpha + 255.0 * (1.0 - alpha)
    return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))


def evaluate_garment_reference(
    rgb: np.ndarray, mask: np.ndarray, *, target_pixels: int
) -> dict[str, float]:
    """레퍼런스 의류 조각이 합성에 쓸 만한지 수치화한다.

    CatVTON은 이 조각을 조건 입력 해상도로 리사이즈하므로, 원본에 실제로 존재하는
    의류 픽셀 수가 조건 해상도에 크게 못 미치면 업스케일된 흐린 텍스처가 들어가고
    결과가 뭉개지거나 시스루로 렌더링된다. 전신 착용컷에서 잘라낸 작은 상의 조각이
    대표적이다(MS6797005: coverage 0.15 → 여성 10장 전원 시스루).

    - coverage: 의류 픽셀 수 / 조건 입력 픽셀 수. 1.0 이상이면 축소, 낮을수록 확대.
    - fill: 바운딩 박스 대비 의류 비율. 낮으면 구겨지거나 팔이 벌어진 착용 자세다.
    - contrast: 흰 배경과의 밝기 차. 낮으면 흰 옷이 배경에 묻혀 실루엣이 흐려진다.
    """
    pixels = int(mask.sum())
    if pixels == 0:
        return {"garment_pixels": 0.0, "coverage": 0.0, "fill": 0.0, "contrast": 0.0}
    ys, xs = np.where(mask)
    bbox = float((ys.max() - ys.min() + 1) * (xs.max() - xs.min() + 1))
    return {
        "garment_pixels": float(pixels),
        "coverage": pixels / float(target_pixels),
        "fill": pixels / bbox,
        "contrast": float(255.0 - rgb[mask].astype(np.float32).mean()),
    }


def _border_color(image: Image.Image) -> tuple[int, int, int]:
    """가장자리 픽셀의 중앙값. 레터박스 여백을 배경과 비슷하게 채운다."""
    rgb = np.asarray(image.convert("RGB"))
    edges = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]])
    return tuple(int(value) for value in np.median(edges, axis=0))


def pad_to_aspect(
    image: Image.Image, size: tuple[int, int], fill
) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """대상 종횡비에 맞게 여백을 덧대고, 원본이 놓인 영역을 함께 돌려준다.

    CatVTON의 `resize_and_crop`은 대상 비율에 맞춰 **가운데를 잘라낸다**. 인스타
    수집본처럼 세로로 긴 사진(중앙값 약 1:2)을 그대로 넣으면 머리와 발이 잘리고
    허리 근처만 남는다. 먼저 비율을 맞춰 두면 잘림 없이 전신이 보존된다.
    """
    width, height = image.size
    target_ratio = size[0] / size[1]
    if width / height < target_ratio:
        new_width, new_height = max(width, round(height * target_ratio)), height
    else:
        new_width, new_height = width, max(height, round(width / target_ratio))
    if (new_width, new_height) == (width, height):
        return image, (0, 0, width, height)
    offset_x, offset_y = (new_width - width) // 2, (new_height - height) // 2
    canvas = Image.new(image.mode, (new_width, new_height), fill)
    canvas.paste(image, (offset_x, offset_y))
    return canvas, (offset_x, offset_y, offset_x + width, offset_y + height)


def unpad_result(
    result: Image.Image,
    content_box: tuple[int, int, int, int],
    padded_size: tuple[int, int],
    original_size: tuple[int, int],
) -> Image.Image:
    """레터박스로 덧댄 여백을 걷어내고 원본 크기로 되돌린다."""
    if content_box[:2] == (0, 0) and content_box[2:] == padded_size:
        return result
    scale_x = result.width / padded_size[0]
    scale_y = result.height / padded_size[1]
    box = (
        round(content_box[0] * scale_x), round(content_box[1] * scale_y),
        round(content_box[2] * scale_x), round(content_box[3] * scale_y),
    )
    return result.crop(box).resize(original_size, Image.LANCZOS)


def _odd_blur_radius(height: int, divisor: int = 50) -> int:
    """CatVTON inference.py의 repaint()와 동일한 규칙: 세로 길이에 비례한 홀수 반경."""
    radius = max(3, height // divisor)
    return radius if radius % 2 == 1 else radius + 1


class CatVTONTryOn(VirtualTryOnAdapter):
    """FASHN 마스크 + CatVTON으로 실제 착장 합성을 수행한다."""

    def __init__(
        self,
        device: str = "auto",
        num_inference_steps: int = 50,  # CatVTON inference.py 기본값
        guidance_scale: float = 2.5,
        width: int = 768,
        height: int = 1024,
        seed: int = 42,
        pipeline_mask_blur: int = 9,
        clean_garment_refs: bool = True,
        garment_cache_dir: str | Path = GARMENT_CACHE_DIR,
        max_retries: int = 1,
        min_sharpness: float = 25.0,
        repaint_inset: bool = True,
        repaint_blur_divisor: int = 300,
        min_reference_coverage: float = 0.25,
    ) -> None:
        super().__init__(enabled=True)
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.width = width
        self.height = height
        self.seed = seed
        self.pipeline_mask_blur = pipeline_mask_blur
        self.clean_garment_refs = clean_garment_refs
        self.garment_cache_dir = Path(garment_cache_dir)
        self.max_retries = max_retries
        self.min_sharpness = min_sharpness
        # repaint_inset: 경계 블렌딩 밴드를 마스크 안쪽으로 침식시켜 마스크 밖으로
        # 새 옷이 반투명하게 번지는 halo를 없앤다. False면 CatVTON 공식 방식
        # (마스크 경계 중심의 대칭 블러, height//50 반경)을 그대로 쓴다.
        self.repaint_inset = repaint_inset
        # 블렌딩 반경은 height // divisor다. divisor가 클수록 밴드가 좁고 원래 옷 색이
        # 덜 번진다. 2026-08-21 A/B에서 원래 옷 색 잔류가 75→21.1% / 150→10.7% /
        # 300→9.5%였고 다른 파라미터는 전부 10.6% 부근이었다. halo는 이 값에만 반응한다.
        # 300에서 시간 비용은 없다. reports/vton_quality/param_tuning_2026-08-21.md 참고.
        self.repaint_blur_divisor = repaint_blur_divisor
        self.min_reference_coverage = min_reference_coverage
        self.reference_reports: dict[str, dict[str, float]] = {}
        self.last_warnings: list[str] = []
        self._quality_cache_data: dict[str, dict[str, float]] | None = None
        self._device_request = device
        self._pipeline = None
        self._garment_parser = None  # False면 사용 불가로 확정, None이면 미확인

    @classmethod
    def high_detail(cls, **overrides) -> "CatVTONTryOn":
        """텍스처·패턴이 더 잘 보이도록 해상도와 스텝을 올린 프리셋. GPU 메모리를 더 쓴다."""
        params = dict(width=832, height=1152, num_inference_steps=50, guidance_scale=2.5)
        params.update(overrides)
        return cls(**params)

    @classmethod
    def fast(cls, **overrides) -> "CatVTONTryOn":
        """스텝을 줄여 반복 평가용으로 빠르게 돌리는 프리셋.

        2026-08-21 A/B에서 25스텝은 장당 24.5초로 기본값 50스텝(44.0초)의 절반이고
        육안 차이는 청바지 질감이 약간 평평해지는 정도였다. 75스텝(65.4초)은 50과
        구분되지 않아 값을 못 한다. 전후 비교나 배치 평가처럼 여러 번 돌리는
        작업에 쓰고, 데모용 최종본은 기본값(50)으로 다시 뽑는다.
        """
        params = dict(num_inference_steps=25)
        params.update(overrides)
        return cls(**params)

    def _load_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        if not CATVTON_REPO.exists():
            raise FileNotFoundError(
                f"CatVTON 저장소가 없습니다: {CATVTON_REPO}\n"
                "git clone https://github.com/Zheng-Chong/CatVTON.git third_party/CatVTON"
            )
        import torch

        if str(CATVTON_REPO) not in sys.path:
            sys.path.append(str(CATVTON_REPO))
        from model.pipeline import CatVTONPipeline

        device = self._device_request
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        self._pipeline = CatVTONPipeline(
            base_ckpt=BASE_CKPT,
            attn_ckpt=ATTN_CKPT,
            attn_ckpt_version="mix",
            weight_dtype=dtype,
            device=device,
            skip_safety_check=True,
        )
        self.device = device
        return self._pipeline

    def _get_garment_parser(self):
        if self._garment_parser is False:
            return None
        if self._garment_parser is None:
            from clothing_parser import ClothingParser

            try:
                self._garment_parser = ClothingParser(use_fashn=True)
            except RuntimeError as exc:
                print(f"상품 이미지 정제용 FASHN 파서를 불러오지 못해 원본 이미지를 그대로 사용합니다: {exc}")
                self._garment_parser = False
                return None
        return self._garment_parser

    def _prepare_garment_reference(self, garment_path: Path, category: str) -> Image.Image:
        """상품 대표 이미지에서 대상 카테고리의 옷만 남기고 나머지(다른 컬러웨이, 색상
        스와치, 모델이 입은 다른 옷)는 흰 배경으로 지운 뒤 캐시해 재사용한다."""
        original = Image.open(garment_path).convert("RGB")
        if not self.clean_garment_refs:
            return original
        target_labels = GARMENT_TARGET_LABELS.get(category)
        if not target_labels:
            return original
        cache_path = self.garment_cache_dir / garment_path.name
        if cache_path.exists():
            # 정제 결과를 재사용할 때도 품질 지표는 사이드카에서 복원해 게이트가 동작하게 한다.
            cached_report = self._quality_cache().get(garment_path.name)
            if cached_report:
                self.reference_reports[garment_path.name] = cached_report
                self._warn_low_coverage(garment_path.name, cached_report)
            return Image.open(cache_path).convert("RGB")
        parser = self._get_garment_parser()
        if parser is None or parser.backend != "fashn-human-parser":
            return original
        try:
            parsed = parser.parse(original, pose=None)
            segmentation = parsed["segmentation"]
            mask = np.isin(segmentation, target_labels)
            if mask.sum() < 0.02 * mask.size:
                return original
            mask = _largest_component(mask)
            mask = _refine_garment_mask(mask)
            # 닫힘 연산이 옷 위를 가로지르는 가방끈·머리카락·팔 픽셀을 다시 포함시켜
            # 검은 줄무늬 아티팩트로 남는 것을 막는다 (착용컷 레퍼런스에서 흔함).
            occluders = np.isin(segmentation, (1, 2, 8, 9, 11, 12, 13, 17))
            mask = np.logical_and(mask, ~occluders)
            if mask.sum() < 0.02 * mask.size:
                return original
            rgb = np.array(original)
            report = evaluate_garment_reference(
                rgb, mask, target_pixels=self.width * self.height
            )
            self.reference_reports[garment_path.name] = report
            self._warn_low_coverage(garment_path.name, report)
            cleaned = _paste_on_white(rgb, mask)
        except Exception as exc:  # 정제 실패 시 원본으로 안전하게 대체한다.
            print(f"상품 이미지 정제 실패({garment_path.name}), 원본을 사용합니다: {exc}")
            return original
        self.garment_cache_dir.mkdir(parents=True, exist_ok=True)
        cleaned.save(cache_path, quality=95)
        self._store_quality(garment_path.name, report)
        return cleaned

    def _quality_cache(self) -> dict[str, dict[str, float]]:
        if self._quality_cache_data is None:
            path = self.garment_cache_dir / "_reference_quality.json"
            try:
                self._quality_cache_data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                self._quality_cache_data = {}
        return self._quality_cache_data

    def _store_quality(self, name: str, report: dict[str, float]) -> None:
        cache = self._quality_cache()
        cache[name] = report
        path = self.garment_cache_dir / "_reference_quality.json"
        try:
            path.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
        except OSError as exc:  # 캐시 저장 실패가 합성을 막아서는 안 된다.
            print(f"레퍼런스 품질 캐시 저장 실패: {exc}")

    def _warn_low_coverage(self, name: str, report: dict[str, float]) -> None:
        if report.get("coverage", 1.0) < self.min_reference_coverage:
            self._add_warning(
                f"레퍼런스 해상도 낮음({name}): coverage={report['coverage']:.2f} < "
                f"{self.min_reference_coverage}. 조건 입력으로 확대되므로 텍스처가 뭉개지고 "
                "레퍼런스와 다른 옷이 그려질 수 있습니다(참고 지표. 시스루 예측기로는 "
                "검증되지 않았습니다)."
            )

    def _add_warning(self, message: str) -> None:
        self.last_warnings.append(message)
        print(message)

    def _check_length_gap(self, garment: Image.Image, category: str, context: dict) -> None:
        """새 옷이 원래 옷보다 짧으면 마스크 모양 프라이어로 합성이 깨진다는 것을 알린다.

        마스크는 원래 옷 기준으로 만들어지므로, 더 짧은 옷을 넣으면 모델이 남는
        마스크 영역을 맨살이 아니라 옷 텍스처로 채운다(긴바지→쇼츠에서 다리 전체가
        니트로 덮이는 실패를 통제 실험으로 확인).
        """
        classifier = context.get("classifier")
        outfit = context.get("outfit")
        if classifier is None or outfit is None:
            return
        if category == "bottom":
            target = classify_reference_bottom_length(classifier, garment)
            gap = bottom_length_gap(getattr(outfit, "bottom_length", ""), target)
            label, unit = "하의 기장", "다리"
        else:
            target = classify_reference_sleeve_length(classifier, garment)
            gap = sleeve_length_gap(getattr(outfit, "sleeve_length", ""), target)
            label, unit = "소매 길이", "팔"
        if gap is not None and gap >= UNRELIABLE_LENGTH_GAP:
            self._add_warning(
                f"{label} 차이가 큽니다(현재보다 {gap}단계 짧음: → {target}). "
                f"마스크가 원래 옷 모양이라 {unit} 영역이 옷 텍스처로 채워질 수 있습니다."
            )

    def _blur_mask(self, mask: Image.Image, radius: int) -> Image.Image:
        return mask.filter(ImageFilter.GaussianBlur(radius)) if radius > 0 else mask

    def _repaint(self, person: Image.Image, mask: Image.Image, result: Image.Image, radius: int) -> Image.Image:
        """마스크 밖 픽셀은 원본을 유지해 VAE 왕복으로 인한 얼굴·배경 디테일 손실을 막는다.

        repaint_inset=True면 마스크를 반경만큼 침식한 뒤 작은 블러를 적용해 블렌딩
        밴드를 마스크 '안쪽'에 만든다. 공식 방식(경계 중심 대칭 블러)은 밴드가 마스크
        밖 배경까지 걸쳐 새 옷의 반투명 잔상(halo·베일)을 남기는데, 밴드를 안쪽으로
        옮기면 마스크 밖은 100% 원본이 유지된다. 마스크는 원래 옷보다 dilate 여유가
        있어 침식해도 원래 옷 윤곽이 다시 드러나지 않는다."""
        person_np = np.asarray(person, dtype=np.float32)
        result_np = np.asarray(result, dtype=np.float32)
        if self.repaint_inset and radius > 0:
            import cv2

            binary = (np.asarray(mask) > 127).astype(np.uint8)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
            eroded = cv2.erode(binary, kernel)
            if eroded.any():
                sigma = max(radius * 0.6, 1.0)
                alpha = cv2.GaussianBlur(eroded.astype(np.float32), (0, 0), sigma)[..., None]
                alpha = np.clip(alpha, 0.0, 1.0)
                blended = person_np * (1 - alpha) + result_np * alpha
                return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))
            # 침식으로 마스크가 사라질 만큼 얇으면 공식 방식으로 대체한다.
        blurred = self._blur_mask(mask, radius)
        alpha = np.asarray(blurred, dtype=np.float32)[..., None] / 255.0
        blended = person_np * (1 - alpha) + result_np * alpha
        return Image.fromarray(np.clip(blended, 0, 255).astype(np.uint8))

    def _garment_sharpness(self, image: Image.Image, mask: Image.Image) -> float:
        import cv2

        mask_np = np.asarray(mask) > 127
        if mask_np.sum() < 100:
            return 0.0
        gray = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F)[mask_np].var())

    def _tryon_once(self, person: Image.Image, garment: Image.Image, mask: Image.Image) -> Image.Image:
        import torch

        pipeline = self._load_pipeline()
        pipeline_mask = self._blur_mask(mask, self.pipeline_mask_blur)
        repaint_radius = _odd_blur_radius(
            person.size[1], self.repaint_blur_divisor if self.repaint_inset else 50
        )

        best_result, best_score = None, -1.0
        for attempt in range(self.max_retries + 1):
            generator = torch.Generator(device=self.device).manual_seed(self.seed + attempt)
            raw = pipeline(
                image=person,
                condition_image=garment,
                mask=pipeline_mask,
                num_inference_steps=self.num_inference_steps,
                guidance_scale=self.guidance_scale,
                width=self.width,
                height=self.height,
                generator=generator,
            )[0]
            repainted = self._repaint(person, mask, raw, repaint_radius)
            score = self._garment_sharpness(repainted, mask)
            if score > best_score:
                best_result, best_score = repainted, score
            if best_score >= self.min_sharpness:
                break
        return best_result

    def generate(
        self,
        person_image: str | Path,
        recommendation: Recommendation,
        output_path: str | Path,
        context: dict | None = None,
    ) -> Path:
        """추천 상품(상의/하의)을 순서대로 합성한다.

        context에는 outfit_analyzer가 만든 FASHN 마스크가 필요하다:
        {"upper_mask": ..., "lower_mask": ...} — 팔/다리를 포함한
        upper/lower_style_mask가 함께 오면 그것을 우선 사용한다(원래 옷 실루엣
        잔류와 소매·기장 변경 문제를 막는다). "segmentation"이 함께 오면
        얼굴·헤어·손·발·가방 영역을 마스크에서 제외해 원본을 보존한다.

        "outfit"(OutfitAnalysis)과 "classifier"(FashionClassifier)를 함께 넘기면
        합성 신뢰도를 점검해 `last_warnings`에 사유를 남긴다. 결과 이미지 옆에
        이 경고를 함께 보여주면 사용자가 깨진 합성을 사실로 오해하지 않는다.
        """
        context = context or {}
        self.last_warnings = []
        person = Image.open(person_image).convert("RGB")

        jobs = []  # (garment 이미지 경로, 원본 크기 마스크, 카테고리)
        for product in recommendation.products:
            garment_path = garment_image_path(product.image_path) if product.image_path else None
            if garment_path is None or not garment_path.exists():
                continue
            mask_keys = (
                ("upper_style_mask", "upper_mask")
                if product.category == "top"
                else ("lower_style_mask", "lower_mask")
            )
            mask = next((context[key] for key in mask_keys if context.get(key) is not None), None)
            if mask is None or not np.any(mask):
                continue
            jobs.append((garment_path, mask, product.category))

        if not jobs:
            # 합성 재료가 없으면 기존 추천 보드로 대체한다.
            return self._make_preview(person_image, recommendation, output_path)

        self._load_pipeline()
        # 얼굴·헤어·손·발·가방 등은 어떤 마스크에서도 제외해 원본을 보존한다.
        segmentation = context.get("segmentation")
        protect = (
            np.isin(segmentation, PROTECT_LABELS) if segmentation is not None else None
        )
        # CatVTON 입력 규격에 맞춰 사람/마스크를 같은 방식으로 맞춘다. 세로로 긴 사진은
        # 먼저 여백을 덧대 비율을 맞춰야 전신이 잘리지 않는다(인스타 수집본은 약 1:2).
        from utils import resize_and_crop  # CatVTON 저장소의 유틸 (sys.path는 _load_pipeline에서 등록)
        target = (self.width, self.height)
        person_padded, content_box = pad_to_aspect(person, target, _border_color(person))
        padded_size = person_padded.size
        result = resize_and_crop(person_padded, target)
        for garment_path, raw_mask, category in jobs:
            garment = self._prepare_garment_reference(garment_path, category)
            self._check_length_gap(garment, category, context)
            mask_np = _dilate_mask(_solidify_mask(raw_mask))
            if protect is not None:
                mask_np[protect] = 0
            # 여백은 합성 대상이 아니므로 마스크는 0으로 채운다.
            mask_padded, _ = pad_to_aspect(Image.fromarray(mask_np).convert("L"), target, 0)
            mask = resize_and_crop(mask_padded, target)
            result = self._tryon_once(result, garment, mask)

        result = unpad_result(result, content_box, padded_size, person.size)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, quality=95)
        return output
