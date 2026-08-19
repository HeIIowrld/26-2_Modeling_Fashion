"""7단계 B — 선택한 image_id의 이미지만 parquet shard에서 뽑아 저장한다.

메모리가 1GB 남짓이라 shard 전체(480MB, row_group 1개)를 한 번에 읽지 않고
iter_batches 로 흘려 읽으면서 필요한 행만 디스크에 쓴다.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

SELECTION = Path(sys.argv[1])
ANNOTATION = Path(sys.argv[2])
OUTPUT_DIR = Path(sys.argv[3])
SHARDS = [int(v) for v in sys.argv[4].split(",")]

BASE = "datasets/detection-datasets/fashionpedia@refs%2Fconvert%2Fparquet/default/train"


def main() -> None:
    wanted = set(json.loads(SELECTION.read_text())["image_ids"])
    print(f"target images: {len(wanted):,}", flush=True)

    import ijson
    file_names: dict[int, str] = {}
    with ANNOTATION.open("rb") as handle:
        for item in ijson.items(handle, "images.item", use_float=True):
            image_id = int(item["id"])
            if image_id in wanted:
                file_names[image_id] = item["file_name"]
    print(f"file names resolved: {len(file_names):,}", flush=True)

    image_dir = OUTPUT_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    fs = HfFileSystem()
    written = skipped = 0
    started = time.time()
    for shard in SHARDS:
        path = f"{BASE}/{shard:04d}.parquet"
        print(f"\n[shard {shard:04d}] 시작", flush=True)
        with fs.open(path, "rb") as handle:
            parquet_file = pq.ParquetFile(handle)
            for batch in parquet_file.iter_batches(batch_size=32, columns=["image_id", "image"]):
                ids = batch["image_id"].to_pylist()
                if not any(i in wanted for i in ids):
                    continue
                images = batch["image"].to_pylist()
                for image_id, image in zip(ids, images):
                    if image_id not in wanted:
                        continue
                    name = file_names.get(int(image_id))
                    if not name:
                        skipped += 1
                        continue
                    target = image_dir / name
                    payload = image["bytes"] if isinstance(image, dict) else image
                    if target.is_file() and target.stat().st_size == len(payload):
                        written += 1
                        continue
                    target.write_bytes(payload)
                    written += 1
                    if written % 250 == 0:
                        rate = written / max(time.time() - started, 1e-6)
                        print(f"  {written:,}/{len(wanted):,}  {rate:.0f} img/s", flush=True)
    print(f"\n완료: {written:,}장 저장, {skipped}장 건너뜀, {time.time() - started:.0f}s")
    print(f"저장 위치: {image_dir}")


if __name__ == "__main__":
    main()
