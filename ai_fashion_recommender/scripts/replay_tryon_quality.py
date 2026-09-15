"""기존 합성 결과에 합성 후 품질 검사(tryon_quality)를 재생한다. 새 합성은 없다.

eval_vton_fit.py가 남긴 폴더 구조를 그대로 읽는다.
  <root>/people/*.png|jpg
  <root>/out/images/<pair>.jpg, <pair>_seg.npz (orig·res 분할, edit 마스크)
  <root>/out/refs/<product_id>.jpg (정제 레퍼런스)
  <root>/out/<results>.jsonl

    python replay_tryon_quality.py <root> [--results results.jsonl] [--prefix ""] [--out quality_replay.jsonl]
        [--people-dir <사람 폴더>] [--refs-root <refs.json이 있는 out 폴더>]

운영 경로와 달리 원본 해상도에서 측정한다(결과 이미지가 원본 크기로 저장돼 있다).
검사 정의는 운영과 같은 tryon_quality.assess_tryon을 호출한다.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

import cv2  # noqa: E402

from catvton_tryon import GARMENT_TARGET_LABELS, landmarks_to_pixels  # noqa: E402
from clothing_parser import ClothingParser  # noqa: E402
from fashion_attribute_model import PREPROCESS_SQUASH  # noqa: E402
from fashion_model import FashionClassifier  # noqa: E402
from pose_analyzer import PoseAnalyzer  # noqa: E402
from tryon_quality import assess_tryon, load_thresholds  # noqa: E402

def reference_length(ref: dict) -> str:
    """운영의 classify_reference_bottom_length를 refs.json에 저장된 헤드 확률로 근사한다.

    명시적 반바지 이름 → category 쇼츠 → lower_length(0.5 이상) 순서. 운영 기준과 달리
    헤드별 수락 임계값 대신 0.5를 쓰므로 재생 전용 근사다.
    """
    from types import SimpleNamespace

    from catvton_tryon import classify_reference_bottom_length

    short = classify_reference_bottom_length(None, None, product=SimpleNamespace(name=ref.get("name", "")))
    if short:
        return short
    judge = ref.get("judge") or {}
    category = judge.get("category") or {}
    if category and max(category, key=category.get) == "쇼츠" and max(category.values()) >= 0.5:
        return "쇼츠·미니 기장"
    lengths = judge.get("lower_length") or {}
    if not lengths:
        return ""
    label = max(lengths, key=lengths.get)
    return label if lengths[label] >= 0.5 else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--results", default="results.jsonl")
    ap.add_argument("--prefix", default="", help="마스크 변종 이미지 접두어(예: wide__)")
    ap.add_argument("--out", default="quality_replay.jsonl")
    ap.add_argument("--no-embed", action="store_true")
    ap.add_argument("--people-dir", type=Path, help="사람 사진 폴더(기본 <root>/people)")
    ap.add_argument("--refs-root", type=Path, help="refs.json·refs/가 있는 out 폴더(기본 <root>/out)")
    opts = ap.parse_args()
    root = opts.root
    out_dir = root / "out"
    refs_dir = opts.refs_root or out_dir
    people_dir = opts.people_dir or root / "people"
    records = [json.loads(line) for line in (out_dir / opts.results).read_text(encoding="utf-8").splitlines()]
    latest = {}
    for record in records:
        if "error" not in record:
            latest[record["pair"]] = record
    ref_meta = {ref["product_id"]: ref for ref in json.loads((refs_dir / "refs.json").read_text(encoding="utf-8"))}

    parser = ClothingParser(use_fashn=True)
    if parser.backend != "fashn-human-parser":
        raise RuntimeError("FASHN 파서가 필요합니다.")
    classifier = None if opts.no_embed else FashionClassifier(enabled=True)
    embed = None if classifier is None else (
        lambda image: classifier._encode_image(image, PREPROCESS_SQUASH).cpu().numpy()[0])
    thresholds = load_thresholds()
    pose_analyzer = PoseAnalyzer()
    poses, refs = {}, {}

    sink_path = out_dir / opts.out
    with sink_path.open("w", encoding="utf-8") as sink:
        for index, (pair, record) in enumerate(sorted(latest.items()), 1):
            person = people_dir / record["person"]
            category, pid = record["category"], record["product_id"]
            image_path = out_dir / "images" / f"{opts.prefix}{pair}.jpg"
            seg_path = image_path.with_name(image_path.stem + "_seg.npz")
            if not image_path.is_file() or not seg_path.is_file():
                continue
            before = np.asarray(Image.open(person).convert("RGB"))
            after = np.asarray(Image.open(image_path).convert("RGB"))
            segs = np.load(seg_path)
            if person.name not in poses:
                pose = pose_analyzer.analyze(person)
                poses[person.name] = landmarks_to_pixels(pose.landmarks, before.shape[1], before.shape[0])
            if pid not in refs:
                ref_rgb = np.asarray(Image.open(refs_dir / "refs" / f"{pid}.jpg").convert("RGB"))
                ref_seg = parser.parse(ref_rgb, pose=None)["segmentation"]
                refs[pid] = (ref_rgb, np.isin(ref_seg, GARMENT_TARGET_LABELS[category]))
            ref_rgb, ref_mask = refs[pid]
            edit = segs["edit"].astype(bool)
            gray = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
            sharp = float(cv2.Laplacian(gray, cv2.CV_64F)[edit].var()) if edit.sum() >= 100 else None
            report = assess_tryon(
                category=category, before=before, after=after, edit_mask=edit,
                after_seg=segs["res"], before_seg=segs["orig"],
                target_labels=GARMENT_TARGET_LABELS[category], landmarks_px=poses[person.name],
                product_name=ref_meta.get(pid, {}).get("name", ""),
                reference_rgb=ref_rgb, reference_mask=ref_mask if ref_mask.sum() >= 400 else None,
                embed=embed, sharpness=sharp, thresholds=thresholds,
                reference_length=reference_length(ref_meta.get(pid, {})) if category == "bottom" else "",
            )
            row = {"pair": pair, "person": record["person"], "product_id": pid, "category": category,
                   "prefix": opts.prefix, **report.to_dict(),
                   "retry_recommended": report.retry_recommended, "penalty": report.penalty()}
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if index % 40 == 0:
                print(f"[{index}/{len(latest)}]", flush=True)
    print(f"[완료] {sink_path}")


if __name__ == "__main__":
    main()
