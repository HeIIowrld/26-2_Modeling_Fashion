"""DeepFashion-MultiModal에서 평가용 전신 사진 표본만 받아온다.

전체 이미지 압축본(6.8GB)을 받지 않고 HTTP Range 요청으로 zip 목차를 읽어 필요한
파일만 꺼낸다. 사람 라벨(shape annotations)은 공식 배포본의 labels.zip(575KB)을 쓴다.

라이선스: DeepFashion-MultiModal은 비상업 연구 목적 전용이다. 받은 사진·라벨은
저장소에 커밋하지 않는다.

    python fetch_deepfashion_mm_sample.py --out /data1/dsl01/eval/deepfashion_mm --per-stratum 12

표본은 전신(`_full`) 사진 중 하의 기장 라벨(짧음·중간·7부·긴·NA)×성별로 층화 추출한다.
NA는 '하의가 사진에 보이지 않음'이라는 정답이라 기장 보류 판정 검증에 필요하다.
"""
from __future__ import annotations

import argparse
import io
import json
import random
import urllib.request
import zipfile
from collections import defaultdict
from pathlib import Path

IMAGES_ZIP = ("https://huggingface.co/datasets/huanngzh/DeepFashion-MultiModal-Parts2Whole"
              "/resolve/main/images.zip")
SEGM_ZIP = ("https://huggingface.co/datasets/huanngzh/DeepFashion-MultiModal-Parts2Whole"
            "/resolve/main/segm.zip")
LABELS_ZIP = ("https://drive.usercontent.google.com/download?"
              "id=11WoM5ZFwWpVjrIvZajW0g8EmQCNKMAWH&export=download&confirm=t")
SEED = 20260915

# 공식 README의 shape annotation 정의
SHAPE_FIELDS = (
    "sleeve_length", "lower_length", "socks", "hat", "glasses", "neckwear",
    "wrist", "ring", "waist_accessory", "neckline", "outer_cardigan", "covers_navel",
)
LOWER_LENGTH = {0: "three-point", 1: "medium short", 2: "three-quarter", 3: "long", 4: "NA"}


class HttpRangeFile(io.RawIOBase):
    """zipfile이 필요한 부분만 Range 요청으로 읽는 읽기 전용 파일 객체."""

    def __init__(self, url: str) -> None:
        self.source_url = url
        self._resolve()
        self.position = 0

    def _resolve(self) -> None:
        # HF resolve 주소는 매번 CDN으로 리다이렉트된다. 서명된 최종 주소를 한 번만 받아 쓴다.
        request = urllib.request.Request(self.source_url, method="HEAD")
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.url
            self.size = int(response.headers["Content-Length"])

    def seekable(self) -> bool:
        return True

    def readable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def read(self, size: int = -1) -> bytes:
        if self.position >= self.size:
            return b""
        end = self.size - 1 if size is None or size < 0 else min(self.size, self.position + size) - 1
        for attempt in range(5):
            request = urllib.request.Request(self.url, headers={"Range": f"bytes={self.position}-{end}"})
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 4:
                    raise
                self._resolve()  # 서명 주소 만료 대비
        self.position += len(data)
        return data

    def readinto(self, buffer) -> int:
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def read_shape_labels(labels_zip: Path) -> dict[str, dict[str, int]]:
    with zipfile.ZipFile(labels_zip) as archive:
        name = next(n for n in archive.namelist() if n.endswith("shape_anno_all.txt"))
        text = archive.read(name).decode("utf-8")
    labels = {}
    for line in text.splitlines():
        parts = line.split()
        if len(parts) == len(SHAPE_FIELDS) + 1:
            labels[parts[0]] = dict(zip(SHAPE_FIELDS, map(int, parts[1:])))
    return labels


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-stratum", type=int, default=12)
    opts = ap.parse_args()
    out = opts.out
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "segm").mkdir(parents=True, exist_ok=True)

    labels_zip = out / "labels.zip"
    if not labels_zip.is_file():
        urllib.request.urlretrieve(LABELS_ZIP, labels_zip)
    labels = read_shape_labels(labels_zip)
    print(f"[라벨] {len(labels)}장", flush=True)

    # 헤더의 작은 읽기들이 파일 본문과 한 번의 요청으로 묶이도록 버퍼를 둔다.
    images = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(IMAGES_ZIP), buffer_size=1 << 20))
    segm = zipfile.ZipFile(io.BufferedReader(HttpRangeFile(SEGM_ZIP), buffer_size=1 << 18))
    image_names = {Path(n).name: n for n in images.namelist() if n.endswith(".jpg")}
    segm_names = {Path(n).name: n for n in segm.namelist() if n.endswith(".png")}
    print(f"[목차] 이미지 {len(image_names)} · 파싱 {len(segm_names)}", flush=True)

    # 사람(id)마다 한 장만 뽑아 같은 인물이 표본을 채우지 않게 한다.
    strata: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for name, shape in labels.items():
        if "_full" not in name or name not in image_names:
            continue
        person = name.split("-id_")[1].split("-")[0] if "-id_" in name else name
        gender = name.split("-")[0]
        strata[(gender, shape["lower_length"])].setdefault(person, name)
    rng = random.Random(SEED)
    chosen = []
    for key in sorted(strata):
        names = sorted(strata[key].values())
        rng.shuffle(names)
        chosen.extend(names[:opts.per_stratum])
        print(f"[층] {key[0]}/{LOWER_LENGTH[key[1]]}: 후보 {len(names)} → {min(len(names), opts.per_stratum)}",
              flush=True)

    manifest = []
    for index, name in enumerate(chosen, 1):
        target = out / "images" / name
        if not target.is_file():
            target.write_bytes(images.read(image_names[name]))
        segm_name = name.replace(".jpg", "_segm.png")
        if segm_name in segm_names and not (out / "segm" / segm_name).is_file():
            (out / "segm" / segm_name).write_bytes(segm.read(segm_names[segm_name]))
        manifest.append({"image": name, "segm": segm_name if segm_name in segm_names else None,
                         **labels[name]})
        if index % 20 == 0:
            print(f"[받기] {index}/{len(chosen)}", flush=True)
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[완료] {len(manifest)}장 → {out}", flush=True)


if __name__ == "__main__":
    main()
