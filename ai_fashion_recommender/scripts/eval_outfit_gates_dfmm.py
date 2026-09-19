"""DeepFashion-MultiModal 사람 라벨로 현재 착장 판정(기장·배꼽·카디건)의 일반화를 잰다.

fetch_deepfashion_mm_sample.py가 받은 표본을 쓴다. 합성은 하지 않는다.
사람이 직접 단 shape 라벨이 정답이다. 특히 하의 기장 NA(사진에 안 보임)는
'판정 보류가 맞는 사진'의 정답이라 보류 판정의 특이도를 잴 수 있다.

    python eval_outfit_gates_dfmm.py --root /data1/dsl01/eval/deepfashion_mm --out out/dfmm_outfit.jsonl
    python eval_outfit_gates_dfmm.py --summary out/dfmm_outfit.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from collections import Counter, defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

# 공식 정의: 0 three-point, 1 medium short, 2 three-quarter, 3 long, 4 NA
GT_LOWER_ORDER = {0: 0, 1: 1, 2: 2, 3: 3, 4: None}
GT_LOWER_NAME = {0: "짧음(3부)", 1: "무릎 부근(5부)", 2: "7부", 3: "긴 기장", 4: "보이지 않음"}


def run(root: Path, out: Path) -> None:
    from catvton_tryon import BOTTOM_LENGTH_ORDER, outerwear_level
    from clothing_parser import ClothingParser
    from fashion_model import FashionClassifier
    from outfit_analyzer import OutfitAnalyzer
    from pose_analyzer import PoseAnalyzer

    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(enabled=True, attribute_checkpoint=os.environ.get("FASHION_ATTRIBUTE_HEADS_PATH"))
    if parser.backend != "fashn-human-parser" or not classifier.trained_attributes_enabled:
        raise RuntimeError("정식 FASHN 파서와 속성 헤드가 필요합니다.")
    analyzer = OutfitAnalyzer(parser, classifier)
    pose_analyzer = PoseAnalyzer()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as sink:
        for index, item in enumerate(manifest, 1):
            path = root / "images" / item["image"]
            row = {"image": item["image"], "gt": item}
            try:
                pose = pose_analyzer.analyze(path)
                row["pose_valid"] = pose.valid
                outfit, _ = analyzer.analyze(path, pose)
                length = (outfit.bottom_length or "").replace(" 추정", "")
                row.update({
                    "bottom_length": outfit.bottom_length,
                    "bottom_order": BOTTOM_LENGTH_ORDER.get(length),
                    "bottom_source": outfit.attribute_sources.get("bottom_length"),
                    "lower_type": outfit.lower_type,
                    "upper_type": outfit.upper_type,
                    "upper_length": outfit.upper_length,
                    "layering_state": outfit.layering_state,
                    "visible_sleeve_length": outfit.visible_sleeve_length,
                    "outerwear_level": outerwear_level(outfit.upper_type),
                })
            except Exception as error:  # 한 장의 실패로 표본 전체를 멈추지 않는다.
                row["error"] = f"{type(error).__name__}: {error}"
                row["traceback"] = traceback.format_exc()[-800:]
            sink.write(json.dumps(row, ensure_ascii=False) + "\n")
            sink.flush()
            if index % 25 == 0:
                print(f"[{index}/{len(manifest)}]", flush=True)


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    ok = [r for r in rows if "error" not in r]
    result: dict = {"images": len(rows), "errors": len(rows) - len(ok)}

    by_gt = defaultdict(list)
    for row in ok:
        by_gt[row["gt"]["lower_length"]].append(row)
    table = {}
    for gt, items in sorted(by_gt.items()):
        expected = GT_LOWER_ORDER[gt]
        judged = [r for r in items if r["bottom_order"] is not None]
        entry = {"n": len(items), "judged": len(judged),
                 "predictions": dict(Counter(r["bottom_length"] for r in items).most_common())}
        if expected is None:
            entry["abstained"] = len(items) - len(judged)
        elif judged:
            entry["exact"] = sum(r["bottom_order"] == expected for r in judged)
            entry["within_one"] = sum(abs(r["bottom_order"] - expected) <= 1 for r in judged)
            # 기장 경고(gap≥2)를 잘못 켜거나 끄게 만드는 큰 오판
            entry["off_by_two_or_more"] = sum(abs(r["bottom_order"] - expected) >= 2 for r in judged)
        table[GT_LOWER_NAME[gt]] = entry
    result["bottom_length"] = table
    visible = [r for r in ok if GT_LOWER_ORDER[r["gt"]["lower_length"]] is not None]
    judged = [r for r in visible if r["bottom_order"] is not None]
    result["bottom_length_visible_total"] = {
        "n": len(visible), "judged": len(judged),
        "exact": sum(r["bottom_order"] == GT_LOWER_ORDER[r["gt"]["lower_length"]] for r in judged),
        "off_by_two_or_more": sum(abs(r["bottom_order"] - GT_LOWER_ORDER[r["gt"]["lower_length"]]) >= 2
                                  for r in judged),
    }
    na = [r for r in ok if r["gt"]["lower_length"] == 4]
    result["bottom_length_na"] = {"n": len(na), "abstained": sum(r["bottom_order"] is None for r in na)}

    # 카디건(0=예) 정답과 아우터 soft/hard 판정
    cardigan = [r for r in ok if r["gt"]["outer_cardigan"] in (0, 1)]
    result["cardigan"] = {
        "gt_yes": sum(r["gt"]["outer_cardigan"] == 0 for r in cardigan),
        "gt_yes_flagged": sum(r["gt"]["outer_cardigan"] == 0 and r["outerwear_level"] is not None for r in cardigan),
        "gt_no": sum(r["gt"]["outer_cardigan"] == 1 for r in cardigan),
        "gt_no_flagged": sum(r["gt"]["outer_cardigan"] == 1 and r["outerwear_level"] is not None for r in cardigan),
    }
    navel = [r for r in ok if r["gt"]["covers_navel"] in (0, 1)]
    result["covers_navel"] = {
        "gt_exposed": sum(r["gt"]["covers_navel"] == 0 for r in navel),
        "gt_exposed_judged_crop": sum(r["gt"]["covers_navel"] == 0 and "크롭" in (r["upper_length"] or "") for r in navel),
        "gt_covered": sum(r["gt"]["covers_navel"] == 1 for r in navel),
        "gt_covered_judged_crop": sum(r["gt"]["covers_navel"] == 1 and "크롭" in (r["upper_length"] or "") for r in navel),
    }
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--summary", type=Path)
    opts = ap.parse_args()
    if opts.summary:
        print(json.dumps(summarize(opts.summary), ensure_ascii=False, indent=2))
    else:
        run(opts.root, opts.out)
        print(json.dumps(summarize(opts.out), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
