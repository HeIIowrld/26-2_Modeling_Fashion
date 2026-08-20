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

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from config import PROJECT_DIR
from schemas import Recommendation
from virtual_tryon import VirtualTryOnAdapter

CATVTON_REPO = PROJECT_DIR.parent / "third_party" / "CatVTON"
BASE_CKPT = "booksforcharlie/stable-diffusion-inpainting"
ATTN_CKPT = "zhengchong/CatVTON"
GARMENT_CACHE_DIR = PROJECT_DIR / "data" / "musinsa_images_clean"

# clothing_parser.LABELS 기준: 3=top, 4=dress, 5=skirt, 6=pants, 7=belt, 10=scarf
GARMENT_TARGET_LABELS = {
    "top": (3, 4, 10),
    "bottom": (4, 5, 6, 7),
}

# 인페인팅 마스크에서 항상 제외해 원본을 보존하는 라벨:
# 1=face, 2=hair, 8=bag, 9=hat, 11=glasses, 13=hands, 15=feet
PROTECT_LABELS = (1, 2, 8, 9, 11, 13, 15)


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
        repaint_blur_divisor: int = 150,
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
        self.repaint_blur_divisor = repaint_blur_divisor
        self._device_request = device
        self._pipeline = None
        self._garment_parser = None  # False면 사용 불가로 확정, None이면 미확인

    @classmethod
    def high_detail(cls, **overrides) -> "CatVTONTryOn":
        """텍스처·패턴이 더 잘 보이도록 해상도와 스텝을 올린 프리셋. GPU 메모리를 더 쓴다."""
        params = dict(width=832, height=1152, num_inference_steps=50, guidance_scale=2.5)
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
            cleaned = _paste_on_white(np.array(original), mask)
        except Exception as exc:  # 정제 실패 시 원본으로 안전하게 대체한다.
            print(f"상품 이미지 정제 실패({garment_path.name}), 원본을 사용합니다: {exc}")
            return original
        self.garment_cache_dir.mkdir(parents=True, exist_ok=True)
        cleaned.save(cache_path, quality=95)
        return cleaned

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
        """
        context = context or {}
        person = Image.open(person_image).convert("RGB")

        jobs = []  # (garment 이미지 경로, 원본 크기 마스크, 카테고리)
        for product in recommendation.products:
            garment_path = PROJECT_DIR / product.image_path if product.image_path else None
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
        # CatVTON 입력 규격에 맞춰 사람/마스크를 같은 방식으로 잘라 크기를 통일한다.
        from utils import resize_and_crop  # CatVTON 저장소의 유틸 (sys.path는 _load_pipeline에서 등록)
        result = resize_and_crop(person, (self.width, self.height))
        for garment_path, raw_mask, category in jobs:
            garment = self._prepare_garment_reference(garment_path, category)
            mask_np = _dilate_mask(_solidify_mask(raw_mask))
            if protect is not None:
                mask_np[protect] = 0
            mask = Image.fromarray(mask_np).convert("L")
            mask = resize_and_crop(mask, (self.width, self.height))
            result = self._tryon_once(result, garment, mask)

        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        result.save(output, quality=95)
        return output
