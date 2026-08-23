from __future__ import annotations

"""인스타 수집 전신사진으로 추천~합성 파이프라인을 검증한다.

개인 전신사진 없이도 `../datasets/people/{men,women}`의 정면 전신 사진을 테스트
인물로 바로 써볼 수 있다. 이 이미지들은 "1·2단계 개발과 검증" 용도이므로,
카탈로그를 새로 크롤링하거나 CatVTON 합성 디테일 로직(마스크 블러+repaint,
상품 이미지 정제, 선명도 재생성)을 바꿀 때마다 여기서 빠르게 전후를 비교한다.

주의: 인스타 스크린샷에는 오른쪽 UI(좋아요·댓글)와 옷 위에 겹친 텍스트
오버레이가 있다. 오버레이가 의류 마스크에 섞이면 합성 결과에 글자가 남을 수
있으므로, 결과가 이상하면 원본에 텍스트가 겹쳤는지 먼저 확인한다.

사용법:
    python test_tryon.py                  # 경량 모드: 추천 보드만 생성(무거운 모델 불필요)
    python test_tryon.py --vton           # 실제 CatVTON 합성(GPU 권장, 첫 실행 시 체크포인트 다운로드)
    python test_tryon.py --vton --high-detail   # CatVTONTryOn.high_detail() 프리셋 사용
    python test_tryon.py --count 5 --gender female
"""

import argparse
from pathlib import Path

import sys

# 런타임 모듈은 src/에 있다. 임포트 전에 경로를 등록한다.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (
    DATA_DIR,
    FASHION_ATTRIBUTE_HEADS_PATH,
    OUTPUT_DIR,
    PEOPLE_DIR,
    PROJECT_DIR,
    resolve_catalog,
)
from clothing_parser import ClothingParser
from fashion_model import FashionClassifier
from outfit_analyzer import OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import RecommendationEngine
from schemas import UserProfile

SAMPLE_ROOT = PEOPLE_DIR
# 수집본은 확장자가 섞여 있다(jpeg/png).
GENDER_FOLDERS = {"male": "men", "female": "women"}
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")


def _sample_images(gender: str, count: int) -> list[Path]:
    """성별 폴더에서 파일명 순으로 앞에서부터 count장씩 고른다.

    인스타 수집본은 게시물별로 IMG 번호가 이어져 있어, 같은 게시물의 여러 장이
    연달아 뽑히지 않도록 폴더 전체에 고르게 퍼뜨려 고른다.
    """
    genders = ["male", "female"] if gender == "all" else [gender]
    images: list[Path] = []
    for name in genders:
        folder = SAMPLE_ROOT / GENDER_FOLDERS[name]
        if not folder.exists():
            continue
        found = sorted(
            path for path in folder.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES
        )
        if not found:
            continue
        if count >= len(found):
            images.extend(found)
            continue
        stride = len(found) / count
        images.extend(found[int(index * stride)] for index in range(count))
    return images


def catalog_path() -> Path:
    # 고르는 규칙은 config.resolve_catalog 한 곳에만 둔다(웹·Notebook과 동일).
    return resolve_catalog(DATA_DIR)


def main() -> None:
    parser = argparse.ArgumentParser(description="유튜브 캡처 프레임으로 추천~합성 파이프라인 테스트")
    parser.add_argument("--gender", choices=["male", "female", "all"], default="all")
    parser.add_argument("--count", type=int, default=2, help="성별별 테스트할 이미지 수")
    parser.add_argument("--light", action="store_true", help="FASHN/SigLIP 없이 포즈 기반 근사 마스크만 사용")
    parser.add_argument("--vton", action="store_true", help="추천 보드 대신 실제 CatVTON 합성을 실행")
    parser.add_argument("--high-detail", action="store_true", help="--vton과 함께 CatVTONTryOn.high_detail() 프리셋 사용")
    parser.add_argument("--skip-existing", action="store_true", help="결과 파일이 이미 있는 이미지는 건너뛴다(배치 확장 시 중복 합성 방지)")
    args = parser.parse_args()

    if args.vton and args.light:
        parser.error("--vton은 FASHN 마스크가 필요해 --light와 함께 쓸 수 없습니다.")

    images = _sample_images(args.gender, args.count)
    if not images:
        raise SystemExit(
            f"{SAMPLE_ROOT}에서 테스트 이미지를 찾지 못했습니다. "
            "먼저 collect/ 파이프라인으로 프레임을 모으세요 (README '데이터 수집' 참고)."
        )

    pose_analyzer = PoseAnalyzer()
    quality = QualityChecker(pose_analyzer)
    clothing_parser = ClothingParser(use_fashn=not args.light)
    # 학습된 의류 속성 헤드가 있으면 옷 종류·속성 인식에 사용한다.
    heads = FASHION_ATTRIBUTE_HEADS_PATH if FASHION_ATTRIBUTE_HEADS_PATH.is_file() else None
    classifier = FashionClassifier(
        enabled=not args.light,
        attribute_checkpoint=None if args.light else heads,
    )
    outfit_analyzer = OutfitAnalyzer(clothing_parser, classifier)
    engine = RecommendationEngine(PROJECT_DIR / "FASHION_RULES_MASTER.md", ProductCatalog(catalog_path()))

    if args.vton:
        from catvton_tryon import CatVTONTryOn

        vton = CatVTONTryOn.high_detail() if args.high_detail else CatVTONTryOn()
    else:
        from virtual_tryon import VirtualTryOnAdapter

        vton = VirtualTryOnAdapter(enabled=False)

    out_dir = OUTPUT_DIR / "tryon_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    gender_labels = {"men": "남성", "women": "여성"}

    for image_path in images:
        if args.skip_existing and (out_dir / f"{image_path.stem}_result.jpg").exists():
            continue
        # 폴더 이름(men/women)으로 성별을 정해 반대 성별 상품이 추천되지 않게 한다.
        profile = UserProfile(gender=gender_labels.get(image_path.parent.name, ""))
        print(f"\n=== {image_path.name} ({profile.gender or '성별 무관'}) ===")
        pose = pose_analyzer.analyze(image_path)
        report = quality.check_input(image_path, pose)
        if not pose.valid:
            print(f"  건너뜀: 전신이 충분히 보이지 않습니다. {report['issues']}")
            continue

        outfit, parsed = outfit_analyzer.analyze(image_path, pose)
        recommendations = engine.recommend(profile, pose, outfit, top_k=1)
        if not recommendations:
            print("  건너뜀: 조건에 맞는 추천 상품이 없습니다.")
            continue

        output_path = out_dir / f"{image_path.stem}_result.jpg"
        result_path = vton.generate(
            image_path,
            recommendations[0],
            output_path,
            context={
                "upper_mask": parsed.get("upper_mask"),
                "lower_mask": parsed.get("lower_mask"),
                "upper_style_mask": parsed.get("upper_style_mask"),
                "lower_style_mask": parsed.get("lower_style_mask"),
                "segmentation": parsed.get("segmentation"),
                # 합성 신뢰도 점검(레퍼런스 해상도·기장 차이)에 필요한 재료
                "outfit": outfit,
                "classifier": classifier,
            },
        )
        print(f"  체형 {pose.body_shape} / 선명도 {report['sharpness']:.1f}")
        for product in recommendations[0].products:
            print(f"  추천: {product.name} ({product.color}, {product.price:,}원)")
        for warning in getattr(vton, "last_warnings", []):
            print(f"  ⚠️ {warning}")
        print(f"  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
