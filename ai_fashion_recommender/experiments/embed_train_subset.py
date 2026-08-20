"""7단계 D — 새 crop을 FashionSigLIP으로 임베딩한다. 기존 캐시 포맷과 100% 동일하게 저장.

기존 build_embedding_cache 대비 두 가지만 바꿨다 (결과는 동일).
1. 같은 원본 이미지를 연속으로 crop하는 경우가 많다(crop 8,513 / 이미지 4,500).
   방금 디코드한 이미지를 재사용해 PIL 디코드를 절반 가까이 줄인다.
2. 이미지 로드·crop·preprocess를 스레드 풀에서 미리 준비해 모델 forward와 겹친다.
또한 부분 저장을 매 배치가 아니라 20배치마다 해서 O(n^2) 쓰기를 줄인다.
"""
from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from PIL import Image

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))
from fashion_attribute_dataset import load_attribute_csv, encode_record_targets  # noqa: E402
from fashion_attribute_schema import ATTRIBUTE_TASKS  # noqa: E402
from fashion_attribute_training import FrozenFashionSigLIPEncoder  # noqa: E402

DATASET_DIR = Path(sys.argv[1])
OUT_DIR = Path(sys.argv[2])
THREADS = int(sys.argv[3]) if len(sys.argv) > 3 else 8
BATCH = 16
SAVE_EVERY = 20


class ImageCropper:
    """직전 이미지 하나만 캐시한다. CSV가 이미지별로 묶여 있어 적중률이 높다."""

    def __init__(self) -> None:
        self._path = None
        self._image = None
        self.hits = 0
        self.misses = 0

    def crop(self, record):
        path = str(record.image_path)
        if path != self._path:
            self._image = Image.open(path).convert("RGB")
            self._path = path
            self.misses += 1
        else:
            self.hits += 1
        image = self._image
        if record.bbox is None:
            return image.copy()
        x, y, width, height = record.bbox
        left, top = max(0, math.floor(x)), max(0, math.floor(y))
        right = min(image.width, math.ceil(x + width))
        bottom = min(image.height, math.ceil(y + height))
        if right <= left or bottom <= top:
            raise ValueError(f"이미지 범위를 벗어난 bbox: {path} {record.bbox}")
        return image.crop((left, top, right, bottom))


def build(records, encoder, output: Path):
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_path = output.with_suffix(output.suffix + ".partial.pt")
    features_done: list[torch.Tensor] = []
    start_index = 0
    if partial_path.is_file():
        payload = torch.load(partial_path, map_location="cpu", weights_only=False)
        if payload.get("count", 0) <= len(records):
            features_done = [payload["features"]]
            start_index = int(payload["count"])
            print(f"  중간 결과 {start_index}건 재개", flush=True)

    cropper = ImageCropper()
    started = time.time()

    def prepare(indices):
        # crop 자체는 순서 의존(직전 이미지 캐시)이라 메인 스레드에서, preprocess만 병렬로
        crops = [cropper.crop(records[i]) for i in indices]
        with ThreadPoolExecutor(max_workers=4) as pool:
            tensors = list(pool.map(lambda im: encoder.preprocess(im), crops))
        return torch.stack(tensors)

    total = len(records)
    batches_since_save = 0
    for start in range(start_index, total, BATCH):
        indices = list(range(start, min(start + BATCH, total)))
        batch = prepare(indices).to(encoder.device)
        with torch.inference_mode():
            features_done.append(encoder.model.encode_image(batch, normalize=True).float().cpu())
        batches_since_save += 1
        done = min(start + BATCH, total)
        if batches_since_save >= SAVE_EVERY or done == total:
            torch.save(
                {"features": torch.cat(features_done, dim=0), "count": done}, partial_path
            )
            batches_since_save = 0
            rate = (done - start_index) / max(time.time() - started, 1e-6)
            remaining = (total - done) / max(rate, 1e-6)
            print(
                f"  {done:,}/{total:,}  {rate:.2f} crop/s  남은 {remaining/60:.0f}분 "
                f"(이미지 재사용 {cropper.hits}/{cropper.hits + cropper.misses})",
                flush=True,
            )

    features = torch.cat(features_done, dim=0)
    if len(features) != total:
        raise SystemExit(f"임베딩 {len(features)}개 != 레코드 {total}개")

    rows = [encode_record_targets(record) for record in records]
    targets, valid = {}, {}
    for task_name, task in ATTRIBUTE_TASKS.items():
        values = [row[0][task_name] for row in rows]
        targets[task_name] = torch.tensor(
            values, dtype=torch.float32 if task.multi_label else torch.long
        )
        valid[task_name] = torch.tensor([row[1][task_name] for row in rows], dtype=torch.bool)
    torch.save(
        {
            "version": 1,
            "backbone_model_id": encoder.model_id,
            "features": features,
            "targets": targets,
            "valid": valid,
            "image_paths": [str(record.image_path) for record in records],
        },
        output,
    )
    if partial_path.is_file():
        partial_path.unlink()
    return output


def main() -> None:
    torch.set_num_threads(THREADS)
    csv_path = DATASET_DIR / "fashion_attribute_annotations.csv"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    encoder = FrozenFashionSigLIPEncoder(device="cpu")
    print(f"encoder ready ({encoder.model_id}), threads={THREADS}", flush=True)
    for split in ("train", "val_train_split"):
        records = load_attribute_csv(csv_path, DATASET_DIR, split=split)
        print(f"\n[{split}] records {len(records):,}", flush=True)
        started = time.time()
        out = build(records, encoder, OUT_DIR / f"fp_train_{split}.pt")
        print(f"[{split}] 완료 {(time.time()-started)/60:.1f}분 -> {out}", flush=True)


if __name__ == "__main__":
    main()
