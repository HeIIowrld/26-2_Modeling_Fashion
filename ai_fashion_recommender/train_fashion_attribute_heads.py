from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import FASHION_SIGLIP_MODEL_ID
from fashion_attribute_dataset import convert_fashionpedia_instances
from fashion_attribute_training import (
    TrainingConfig,
    filter_embedding_cache,
    merge_embedding_caches,
    prepare_embedding_caches,
    train_attribute_heads,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="고정 FashionSigLIP 임베딩 위에 다중 의류 속성 분류 헤드를 학습합니다."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    convert = subparsers.add_parser("convert-fashionpedia", help="Fashionpedia JSON을 학습 CSV로 변환")
    convert.add_argument("--annotations", required=True)
    convert.add_argument("--output-csv", required=True)
    convert.add_argument("--split", required=True, choices=("train", "val", "test"))
    convert.add_argument(
        "--image-prefix",
        default="",
        help="image-root 아래의 split 폴더명. 예: train 또는 val",
    )

    cache = subparsers.add_parser("cache", help="고정 백본 임베딩을 train/val 캐시로 저장")
    cache.add_argument("--annotations-csv", required=True)
    cache.add_argument("--image-root", required=True)
    cache.add_argument("--train-cache", required=True)
    cache.add_argument("--val-cache", required=True)
    cache.add_argument("--model-id", default=FASHION_SIGLIP_MODEL_ID)
    cache.add_argument("--device", default="auto")
    cache.add_argument("--batch-size", type=int, default=64)
    cache.add_argument(
        "--reuse-caches",
        nargs="*",
        default=[],
        help="같은 이미지의 기존 FashionSigLIP 특징을 재사용할 캐시",
    )

    merge = subparsers.add_parser("merge-cache", help="같은 백본으로 만든 임베딩 캐시를 합침")
    merge.add_argument("--inputs", nargs="+", required=True)
    merge.add_argument("--output", required=True)

    filter_cache = subparsers.add_parser("filter-cache", help="기존 캐시에서 특정 데이터 경로를 제외")
    filter_cache.add_argument("--input", required=True)
    filter_cache.add_argument("--output", required=True)
    filter_cache.add_argument("--exclude-path-fragments", nargs="+", required=True)

    train = subparsers.add_parser("train", help="저장된 임베딩으로 작은 속성 헤드만 학습")
    train.add_argument("--train-cache", required=True)
    train.add_argument("--val-cache", required=True)
    train.add_argument("--output-checkpoint", required=True)
    train.add_argument("--device", default="auto")
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--learning-rate", type=float, default=1e-3)
    train.add_argument("--hidden-dim", type=int, default=256)
    train.add_argument("--dropout", type=float, default=0.15)
    train.add_argument("--patience", type=int, default=5)
    train.add_argument("--minimum-label-examples", type=int, default=5)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "convert-fashionpedia":
        result = convert_fashionpedia_instances(
            args.annotations,
            args.output_csv,
            split=args.split,
            image_prefix=args.image_prefix,
        )
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    if args.command == "cache":
        outputs = prepare_embedding_caches(
            args.annotations_csv,
            args.image_root,
            args.train_cache,
            args.val_cache,
            model_id=args.model_id,
            device=args.device,
            batch_size=args.batch_size,
            reuse_cache_paths=args.reuse_caches,
        )
        print("train cache:", outputs[0])
        print("val cache:", outputs[1])
        return
    if args.command == "merge-cache":
        output = merge_embedding_caches(args.inputs, args.output)
        print("merged cache:", output)
        return
    if args.command == "filter-cache":
        output = filter_embedding_cache(
            args.input,
            args.output,
            exclude_path_fragments=args.exclude_path_fragments,
        )
        print("filtered cache:", output)
        return
    config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        patience=args.patience,
        minimum_label_examples=args.minimum_label_examples,
    )
    summary = train_attribute_heads(
        args.train_cache,
        args.val_cache,
        args.output_checkpoint,
        config=config,
        device=args.device,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
