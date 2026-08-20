import os
from pathlib import Path


# 모듈은 src/ 아래에 있고 data·models·outputs는 그 상위(프로젝트 폴더)에 있다.
PROJECT_DIR = Path(__file__).resolve().parent.parent


def resolve_path(value: str | Path | None, default: str | Path, base_dir: str | Path = PROJECT_DIR) -> Path:
    """상대경로는 프로젝트 폴더 기준, 절대경로는 그대로 해석한다."""
    selected = Path(value).expanduser() if value else Path(default).expanduser()
    if not selected.is_absolute():
        selected = Path(base_dir).expanduser() / selected
    return selected.resolve()


# 다른 PC에서는 환경변수만 바꿔도 데이터와 출력 위치를 변경할 수 있다.
# 설정하지 않으면 프로젝트 내부 data/, outputs/를 사용한다.
DATA_DIR = resolve_path(os.environ.get("FASHION_DATA_DIR"), "data")
OUTPUT_DIR = resolve_path(os.environ.get("FASHION_OUTPUT_DIR"), "outputs")
FONT_PATH = os.environ.get("FASHION_FONT_PATH", "").strip()

# 이미지 자산은 코드 밖 datasets/에 역할별로 나눠 둔다.
#   people   : 옷을 갈아입힐 사람 사진 (합성 대상)
#   garments : 입힐 상품 이미지 (합성 타겟). clean은 파서로 옷만 남긴 정제본이다.
REPO_DIR = PROJECT_DIR.parent
DATASETS_DIR = resolve_path(os.environ.get("FASHION_DATASETS_DIR"), "datasets", REPO_DIR)
PEOPLE_DIR = DATASETS_DIR / "people"
GARMENT_RAW_DIR = DATASETS_DIR / "garments" / "raw"
GARMENT_CLEAN_DIR = DATASETS_DIR / "garments" / "clean"


def garment_image_path(image_name: str) -> Path:
    """카탈로그 CSV의 image_path(파일명)를 실제 상품 이미지 경로로 바꾼다."""
    return GARMENT_RAW_DIR / Path(image_name).name

# 정식 분석 경로는 두 모델을 모두 사용한다. 메모리·네트워크 점검 때만 False로 바꾼다.
ENABLE_FASHN_PARSER = True
ENABLE_FASHION_SIGLIP = True
ENABLE_VTON = False

FASHN_PARSER_MODEL_ID = "fashn-ai/fashn-human-parser"
FASHION_SIGLIP_MODEL_ID = "Marqo/marqo-fashionSigLIP"
FASHION_ATTRIBUTE_HEADS_PATH = resolve_path(
    os.environ.get("FASHION_ATTRIBUTE_HEADS_PATH"),
    "models/fashion_attribute_heads.pt",
)
# FashionSigLIP 점수는 후보 프롬프트 사이의 상대 점수다. 실제 확률로 해석하지 않는다.
ATTRIBUTE_CONFIDENCE_THRESHOLDS = {
    "style": 0.40,
    "category": 0.40,
    "pattern": 0.45,
    "material": 0.50,
    "neckline": 0.45,
}

# 2D 사진에서 계산한 비율은 실제 신체 치수가 아니라 추천용 참고값이다.
MIN_LANDMARK_VISIBILITY = 0.40
MIN_FULL_BODY_SCORE = 0.55
MIN_BODY_SHAPE_CONFIDENCE = 0.65

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
