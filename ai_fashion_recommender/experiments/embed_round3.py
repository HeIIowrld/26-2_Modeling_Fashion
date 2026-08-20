"""3차 보강 crop 임베딩.

기존과 동일한 조건을 유지한다.
  - Marqo/marqo-fashionSigLIP, backbone freeze
  - encode_image(normalize=True) → 768차원 L2 정규화
  - 기존 preprocess 그대로
  - 20배치마다 진행 상황 저장 → 중단 후 재개 가능
완료 후 crop 수와 embedding 수 일치, NaN/Inf 여부를 확인한다.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parent.parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR / "src"))

from fashion_attribute_dataset import encode_record_targets, load_attribute_csv  # noqa: E402
from fashion_attribute_schema import ATTRIBUTE_TASKS  # noqa: E402
from fashion_attribute_training import FrozenFashionSigLIPEncoder  # noqa: E402

from embed_train_subset import ImageCropper  # noqa: E402

BATCH = 16
SAVE_EVERY = 20
DATASET_DIR = PROJECT_DIR / "data" / "fashionpedia_train_r3"
OUTPUT = PROJECT_DIR / "data" / "cache" / "fashion_attribute_embeddings_r3.pt"
PROGRESS = PROJECT_DIR / "data" / "cache" / "fashion_attribute_embeddings_r3.progress.pt"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    records = load_attribute_csv(DATASET_DIR / "fashion_attribute_annotations.csv",
                                 DATASET_DIR, split="train")
    total = len(records)
    print(f"crop {total:,}  threads={args.threads}", flush=True)

    encoder = FrozenFashionSigLIPEncoder(device="cpu")
    frozen = all(not p.requires_grad for p in encoder.model.parameters())
    print(f"encoder {encoder.model_id}  backbone freeze={frozen}", flush=True)

    done_features: list[torch.Tensor] = []
    start = 0
    if PROGRESS.is_file():
        payload = torch.load(PROGRESS, map_location="cpu", weights_only=False)
        if payload.get("total") == total:
            done_features = [payload["features"]]
            start = int(payload["count"])
            print(f"  중간 결과 {start:,}건에서 재개", flush=True)

    cropper = ImageCropper()
    began = time.time()
    since_save = 0
    from concurrent.futures import ThreadPoolExecutor

    for offset in range(start, total, BATCH):
        indices = range(offset, min(offset + BATCH, total))
        crops = [cropper.crop(records[i]) for i in indices]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tensors = list(pool.map(encoder.preprocess, crops))
        batch = torch.stack(tensors).to(encoder.device)
        with torch.inference_mode():
            done_features.append(
                encoder.model.encode_image(batch, normalize=True).float().cpu())
        since_save += 1
        completed = min(offset + BATCH, total)
        if since_save >= SAVE_EVERY or completed == total:
            torch.save({"features": torch.cat(done_features, dim=0),
                        "count": completed, "total": total}, PROGRESS)
            since_save = 0
            rate = (completed - start) / max(time.time() - began, 1e-6)
            print(f"  {completed:,}/{total:,}  {rate:.2f} crop/s  "
                  f"남은 {(total - completed) / max(rate, 1e-6) / 60:.0f}분", flush=True)

    features = torch.cat(done_features, dim=0)

    print("\n검증")
    ok_count = len(features) == total
    print(f"  [{'PASS' if ok_count else 'FAIL'}] crop 수 == embedding 수: {len(features):,} / {total:,}")
    finite = bool(torch.isfinite(features).all())
    print(f"  [{'PASS' if finite else 'FAIL'}] NaN/Inf 없음: "
          f"NaN {int(torch.isnan(features).sum())}, Inf {int(torch.isinf(features).sum())}")
    norms = features.norm(dim=1)
    normalized = bool(((norms - 1.0).abs() < 1e-3).all())
    print(f"  [{'PASS' if normalized else 'FAIL'}] L2 정규화: "
          f"norm 최소 {float(norms.min()):.6f} 최대 {float(norms.max()):.6f}")
    print(f"  차원: {tuple(features.shape)}")
    if not (ok_count and finite and normalized):
        raise SystemExit("임베딩 검증 실패")

    rows = [encode_record_targets(record) for record in records]
    targets, valid = {}, {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        targets[task_name] = torch.tensor(
            [row[0][task_name] for row in rows],
            dtype=torch.float32 if task.multi_label else torch.long)
        valid[task_name] = torch.tensor([row[1][task_name] for row in rows], dtype=torch.bool)

    torch.save({
        "version": 1,
        "backbone_model_id": encoder.model_id,
        "features": features,
        "targets": targets,
        "valid": valid,
        "image_paths": [str(record.image_path) for record in records],
    }, OUTPUT)
    if PROGRESS.is_file():
        PROGRESS.unlink()
    print(f"\n저장: {OUTPUT.relative_to(PROJECT_DIR).as_posix()}  "
          f"({OUTPUT.stat().st_size / 1e6:.1f} MB, {(time.time() - began) / 60:.1f}분)")


if __name__ == "__main__":
    main()
