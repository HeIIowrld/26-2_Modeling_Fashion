"""사진 한 장을 학습된 속성 헤드에 넣어본다.

사용법
    python predict_image.py <이미지경로>
    python predict_image.py <이미지경로> --checkpoint models/fashion_attribute_heads.pt   # 기존 모델과 비교
    python predict_image.py <이미지경로> --upper 0.18,0.58 --lower 0.46,0.97              # crop 비율 조정
    python predict_image.py <이미지경로> --bbox 120,300,260,420                            # 직접 bbox 지정

⚠️ 중요 — 이 스크립트의 한계
학습 데이터는 **의류 하나만 잘라낸 crop**(Fashionpedia bbox / 쇼핑몰 상품컷)입니다.
전신 사진을 그대로 넣으면 학습 분포와 달라 성능이 떨어집니다.
정식 경로는 `clothing_parser.py`(FASHN Human Parser)로 의류 마스크를 만들어 crop하는 것인데
이 환경에는 FASHN 파서가 설치돼 있지 않습니다.
그래서 여기서는 **세로 비율 기반 근사 crop**을 씁니다. 참고용 결과로만 보세요.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

from config import FASHION_SIGLIP_MODEL_ID  # noqa: E402
from fashion_attribute_model import FashionAttributePredictor  # noqa: E402
from fashion_attribute_schema import (  # noqa: E402
    ATTRIBUTE_TASKS,
    LOWER_CATEGORIES,
    LOWER_ONLY_TASKS,
    UPPER_CATEGORIES,
    UPPER_ONLY_TASKS,
)

SHARED_TASKS = set(ATTRIBUTE_TASKS) - UPPER_ONLY_TASKS - LOWER_ONLY_TASKS


def crop_ratio(image: Image.Image, top: float, bottom: float) -> Image.Image:
    height = image.height
    return image.crop((0, int(height * top), image.width, int(height * bottom)))


def show(title: str, predictions: dict, tasks: list[str]) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'task':<16}{'예측':<22}{'confidence':>12}{'출력여부':>10}")
    print("-" * 72)
    for task_name in tasks:
        prediction = predictions.get(task_name)
        if prediction is None:
            continue
        labels = ", ".join(prediction.labels) if prediction.labels else "— (분석 보류)"
        state = "출력" if prediction.accepted else "보류"
        print(f"{task_name:<16}{labels:<22}{prediction.confidence:>12.3f}{state:>10}")


def top_scores(predictions: dict, task_name: str, limit: int = 4) -> str:
    prediction = predictions.get(task_name)
    if prediction is None:
        return ""
    ordered = sorted(prediction.scores.items(), key=lambda kv: -kv[1])[:limit]
    return "  ".join(f"{label} {score:.2f}" for label, score in ordered if score > 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument(
        "--checkpoint",
        default=str(PROJECT_DIR / "models" / "fashion_attribute_heads_augmented.pt"),
    )
    parser.add_argument("--upper", default="0.18,0.58", help="상의 crop 세로 비율 top,bottom")
    parser.add_argument("--lower", default="0.46,0.97", help="하의 crop 세로 비율 top,bottom")
    parser.add_argument("--bbox", default="", help="x,y,w,h 직접 지정 (지정 시 이 영역만 분석)")
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--save-crops", action="store_true", help="사용한 crop을 outputs/에 저장")
    args = parser.parse_args()

    torch.set_num_threads(args.threads)
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise SystemExit(f"이미지가 없습니다: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"입력: {image_path.name}  {image.width}x{image.height}")
    print(f"체크포인트: {Path(args.checkpoint).name}")

    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(
        f"hf-hub:{FASHION_SIGLIP_MODEL_ID}", device="cpu"
    )
    model.eval()
    predictor = FashionAttributePredictor(
        args.checkpoint,
        image_encoder=model,
        preprocess=preprocess,
        model_id=FASHION_SIGLIP_MODEL_ID,
        device="cpu",
    )

    regions: list[tuple[str, Image.Image]] = []
    if args.bbox:
        x, y, width, height = (int(v) for v in args.bbox.split(","))
        regions.append(("지정 영역", image.crop((x, y, x + width, y + height))))
    else:
        upper_top, upper_bottom = (float(v) for v in args.upper.split(","))
        lower_top, lower_bottom = (float(v) for v in args.lower.split(","))
        regions.append((f"상의 추정 crop ({args.upper})", crop_ratio(image, upper_top, upper_bottom)))
        regions.append((f"하의 추정 crop ({args.lower})", crop_ratio(image, lower_top, lower_bottom)))
        regions.append(("전체 이미지", image))

    for name, region in regions:
        predictions = predictor.predict(region)
        category = predictions["category"].labels[0] if predictions["category"].labels else ""
        if category in UPPER_CATEGORIES:
            tasks = ["category"] + sorted(UPPER_ONLY_TASKS) + sorted(SHARED_TASKS - {"category"})
        elif category in LOWER_CATEGORIES:
            tasks = ["category"] + sorted(LOWER_ONLY_TASKS) + sorted(SHARED_TASKS - {"category"})
        else:
            tasks = list(ATTRIBUTE_TASKS)
        show(f"{name}  →  주 카테고리: {category or '판단 보류'}", predictions, tasks)
        print()
        print("  주요 후보 점수")
        for task_name in ("category", "material", "pattern", "detail"):
            scores = top_scores(predictions, task_name)
            if scores:
                print(f"    {task_name:<10} {scores}")
        if args.save_crops:
            output = PROJECT_DIR / "outputs" / f"predict_{name.split()[0]}.jpg"
            output.parent.mkdir(parents=True, exist_ok=True)
            region.save(output)
            print(f"    crop 저장: {output}")


if __name__ == "__main__":
    main()
