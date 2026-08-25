from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from layering_training import LayeringTrainingConfig, prepare_layering_caches, train_layering_heads


def main() -> None:
    parser = argparse.ArgumentParser(description="멀티 ROI 레이어드·안옷·겉옷 헤드 학습")
    parser.add_argument("--csv", required=True, help="레이어드 라벨 CSV")
    parser.add_argument("--image-root", required=True, help="CSV 이미지 경로의 기준 폴더")
    parser.add_argument("--output", default="models/layering_heads.pt")
    parser.add_argument("--cache-dir", default="data/layering_cache")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args()

    cache_dir = (ROOT / args.cache_dir).resolve()
    train_cache, val_cache = cache_dir / "train.pt", cache_dir / "val.pt"
    if args.rebuild_cache or not train_cache.is_file() or not val_cache.is_file():
        prepare_layering_caches(
            args.csv,
            args.image_root,
            train_cache,
            val_cache,
            device=args.device,
            batch_size=args.embedding_batch_size,
        )
    summary = train_layering_heads(
        train_cache,
        val_cache,
        ROOT / args.output,
        config=LayeringTrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        ),
        device=args.device,
    )
    print("레이어드 학습 완료:", ROOT / args.output)
    print("검증 지표:", summary["metrics"])


if __name__ == "__main__":
    main()
