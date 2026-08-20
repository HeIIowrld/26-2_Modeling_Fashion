from __future__ import annotations

"""collect/ 파이프라인으로 모은 유튜브 룩북 프레임으로 추천~합성 파이프라인을 검증한다.

개인 전신사진 없이도 `../data/images/{male,female}`의 정면 전신 프레임을 테스트
인물로 바로 써볼 수 있다. README가 밝히듯 이 이미지들은 "1·2단계 개발과 검증"
용도이므로, 카탈로그를 새로 크롤링하거나 CatVTON 합성 디테일 로직(마스크
블러+repaint, 상품 이미지 정제, 선명도 재생성)을 바꿀 때마다 여기서 빠르게
전후를 비교한다.

사용법:
    python test_tryon.py                  # 경량 모드: 추천 보드만 생성(무거운 모델 불필요)
    python test_tryon.py --vton           # 실제 CatVTON 합성(GPU 권장, 첫 실행 시 체크포인트 다운로드)
    python test_tryon.py --vton --high-detail   # CatVTONTryOn.high_detail() 프리셋 사용
    python test_tryon.py --count 5 --gender female
"""

import argparse
import re
from pathlib import Path

from config import DATA_DIR, OUTPUT_DIR, PROJECT_DIR
from clothing_parser import ClothingParser
from fashion_model import FashionClassifier
from outfit_analyzer import OutfitAnalyzer
from pose_analyzer import PoseAnalyzer
from product_catalog import ProductCatalog
from quality_checker import QualityChecker
from recommendation_engine import RecommendationEngine
from schemas import UserProfile

SAMPLE_ROOT = PROJECT_DIR.parent / "data" / "images"


def _spread_by_video(files: list[Path], count: int) -> list[Path]:
    """같은 영상의 프레임만 몰아 뽑지 않도록 영상별로 돌아가며 하나씩 고른다.

    파일명 형식: {gender}_{번호}_{유튜브영상ID}_{프레임}.jpg — 영상 ID에는 '-'와
    '_'가 들어갈 수 있어 앞뒤 고정 패턴으로 가운데를 통째로 잡는다.
    """
    groups: dict[str, list[Path]] = {}
    for file in files:
        match = re.match(r"^(?:male|female)_\d+_(.+)_\d+$", file.stem)
        groups.setdefault(match.group(1) if match else file.stem, []).append(file)
    queues = [groups[key] for key in sorted(groups)]
    picked: list[Path] = []
    while len(picked) < count and any(queues):
        for queue in queues:
            if queue and len(picked) < count:
                picked.append(queue.pop(0))
    return picked


def _sample_images(gender: str, count: int) -> list[Path]:
    genders = ["male", "female"] if gender == "all" else [gender]
    images: list[Path] = []
    for name in genders:
        folder = SAMPLE_ROOT / name
        if not folder.exists():
            continue
        images.extend(_spread_by_video(sorted(folder.glob("*.jpg")), count))
    return images


def catalog_path() -> Path:
    musinsa = DATA_DIR / "products_musinsa.csv"
    return musinsa if musinsa.exists() else DATA_DIR / "products.csv"


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
    classifier = FashionClassifier(enabled=not args.light)
    outfit_analyzer = OutfitAnalyzer(clothing_parser, classifier)
    engine = RecommendationEngine(DATA_DIR / "fashion_rules.json", ProductCatalog(catalog_path()))

    if args.vton:
        from catvton_tryon import CatVTONTryOn

        vton = CatVTONTryOn.high_detail() if args.high_detail else CatVTONTryOn()
    else:
        from virtual_tryon import VirtualTryOnAdapter

        vton = VirtualTryOnAdapter(enabled=False)

    out_dir = OUTPUT_DIR / "tryon_tests"
    out_dir.mkdir(parents=True, exist_ok=True)
    gender_labels = {"male": "남성", "female": "여성"}

    for image_path in images:
        if args.skip_existing and (out_dir / f"{image_path.stem}_result.jpg").exists():
            continue
        # 폴더 이름(male/female)으로 성별을 정해 반대 성별 상품이 추천되지 않게 한다.
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
            },
        )
        print(f"  체형 {pose.body_shape} / 선명도 {report['sharpness']:.1f}")
        for product in recommendations[0].products:
            print(f"  추천: {product.name} ({product.color}, {product.price:,}원)")
        print(f"  결과 저장: {result_path}")


if __name__ == "__main__":
    main()
