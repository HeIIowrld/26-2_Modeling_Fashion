from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fit_vision_training import FitVisionTrainingConfig, prepare_fit_caches, train_fit_vision_heads
from fit_vision_model import FIT_FEATURE_MODES


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="사용자/상품 사진 공용 상·하의 핏 모델을 준비합니다.")
    commands = root.add_subparsers(dest="command", required=True)
    cache = commands.add_parser("cache")
    cache.add_argument("--annotations-csv", required=True)
    cache.add_argument("--image-root", required=True)
    cache.add_argument("--train-cache", required=True)
    cache.add_argument("--val-cache", required=True)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--batch-size", type=int, default=32)
    train = commands.add_parser("train")
    train.add_argument("--train-cache", required=True)
    train.add_argument("--val-cache", required=True)
    train.add_argument("--output-checkpoint", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=40)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--feature-mode", choices=FIT_FEATURE_MODES,
                       default="rgb_mask_geometry")
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "cache":
        outputs = prepare_fit_caches(
            args.annotations_csv, args.image_root, args.train_cache, args.val_cache,
            device=args.device, batch_size=args.batch_size,
        )
        print("train cache:", outputs[0])
        print("val cache:", outputs[1])
        return
    summary = train_fit_vision_heads(
        args.train_cache, args.val_cache, args.output_checkpoint,
        config=FitVisionTrainingConfig(
            epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
            feature_mode=args.feature_mode,
        ), device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
