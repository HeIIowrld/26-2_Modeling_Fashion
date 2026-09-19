"""기존 합성 결과를 재사용해 품질 게이트와 선택적 마스크 후보를 재평가한다.

eval_vton_fit.py와 같은 환경변수, --baseline-root <기존 평가 폴더> 필요.
새 합성 실험은 아니며, 기존 native/wide 결과를 조합한 정책의 사후 분석이다.
"""
import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np
from PIL import Image

import eval_vton_fit as ev
from catvton_tryon import classify_reference_bottom_length
from garment_attribute_analyzer import GarmentAttributeAnalyzer
from vton_eval_utils import is_short_bottom, is_wide_bottom, load_dedup


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-root", type=Path, required=True)
    args = ap.parse_args()
    base = args.baseline_root
    old_src = base / "repo/ai_fashion_recommender/src"
    old_measure = load_module("old_measure", old_src / "garment_attribute_analyzer.py")
    old_outfit = load_module("old_outfit", old_src / "outfit_analyzer.py")
    old_outfit.GarmentAttributeAnalyzer = old_measure.GarmentAttributeAnalyzer
    old_vton = load_module("old_vton", old_src / "catvton_tryon.py")
    parser = ev.ClothingParser(use_fashn=True)
    classifier = ev.FashionClassifier(enabled=True, attribute_checkpoint=ev.os.environ["FASHION_ATTRIBUTE_HEADS_PATH"])
    pose_analyzer = ev.PoseAnalyzer()
    native, _ = load_dedup(base / "out/results.jsonl")
    wide, _ = load_dedup(base / "out/results_wide.jsonl")
    wide = {r["pair"]: r for r in wide}
    refs = json.loads((base / "out/refs.json").read_text(encoding="utf-8"))
    ref_by_id = {r["product_id"]: r for r in refs}
    catalog = {r["product_id"]: r for r in ev.csv.DictReader(open(ev.CATALOG, encoding="utf-8-sig"))}
    report = {"people": [], "references": [], "paired": []}
    for ref in refs:
        if ref["category"] != "bottom":
            continue
        raw = Image.open(ev.GARMENT_RAW / Path(catalog[ref["product_id"]]["image_path"]).name).convert("RGB")
        cache = base / "out/ref_cache" / Path(catalog[ref["product_id"]]["image_path"]).name
        clean = Image.open(cache if cache.is_file() else base / "out/refs" / f"{ref['product_id']}.jpg").convert("RGB")
        report["references"].append({"product_id": ref["product_id"], "name": ref["name"],
            "old": old_vton.classify_reference_bottom_length(classifier, clean),
            "new": classify_reference_bottom_length(classifier, clean, source_image=raw,
                                                       product=SimpleNamespace(name=ref["name"]))})
    for path in sorted((base / "people").iterdir()):
        if path.suffix.lower() not in {".png", ".jpg"}:
            continue
        pose = pose_analyzer.analyze(path)
        old, parsed = old_outfit.OutfitAnalyzer(parser, classifier).analyze(path, pose)
        with patch.object(parser, "parse", return_value=parsed):
            new, _ = ev.OutfitAnalyzer(parser, classifier).analyze(path, pose)
        old_gate, new_gate = old_vton.CatVTONTryOn(), ev.CatVTONTryOn()
        jobs = [(None, None, SimpleNamespace(category="top"))]
        old_gate._apply_outerwear_policy(jobs, {"outfit": old})
        new_gate._apply_outerwear_policy(jobs, {"outfit": new})
        item = {"person": path.name, "landmarks": pose.landmarks,
                "old_length": old.bottom_length, "new_length": new.bottom_length,
                "new_pant_length": new.pant_length,
                "old_measured": old_measure.GarmentAttributeAnalyzer().analyze(parsed["segmentation"], pose)["bottom_length"],
                "new_measured": GarmentAttributeAnalyzer().analyze(parsed["segmentation"], pose)["bottom_length"],
                "upper_type": new.upper_type, "layering": new.layering_state,
                "visible_sleeve": new.visible_sleeve_length,
                "old_notes": old.notes, "new_notes": new.notes,
                "old_sources": old.attribute_sources, "new_sources": new.attribute_sources,
                "old_outerwear": old_gate.last_warnings, "new_outerwear": new_gate.last_warnings}
        report["people"].append(item)
        print(json.dumps(item, ensure_ascii=False), flush=True)
        orig = np.asarray(Image.open(path).convert("RGB"))
        if path.stem == "Simon_1":
            ev.OUT.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(ev.OUT / "Simon_1_masks.npz",
                                segmentation=parsed["segmentation"], lower=parsed["lower_mask"])
        # 두 결과를 동일한 native 마스크 밖에서 측정한다. 의도한 확장도 차이에 포함된다.
        for r in [r for r in native if r["person"] == path.name and r["category"] == "bottom"]:
            w = wide.get(r["pair"])
            ref = ref_by_id[r["product_id"]]
            if w is None or is_short_bottom(ref):
                continue
            band = "shin" if r["width_orig"].get("shin") is not None else "thigh"
            if r["width_res"].get(band) is None or w["width_res"].get(band) is None:
                continue
            use_wide = is_wide_bottom(ref)
            selected = w if use_wide else r
            edit = ev.edit_mask(parsed, "bottom")
            outside = ~(cv2.dilate(edit.astype(np.uint8), np.ones((15, 15), np.uint8)) > 0)
            image_native = np.asarray(Image.open(base / "out/images" / f"{r['pair']}.jpg").convert("RGB"))
            image_selected = np.asarray(Image.open(base / "out/images" / f"{'wide__' if use_wide else ''}{r['pair']}.jpg").convert("RGB"))
            report["paired"].append({"person": r["person"], "pair": r["pair"], "wide": use_wide,
                "seller_fit": r["seller_fit"], "native": r["width_res"][band], "selected": selected["width_res"][band],
                "rank_native": r.get("rank_res"), "rank_selected": selected.get("rank_res"),
                "psnr_native_fixed": ev.psnr(orig, image_native, outside),
                "psnr_selected_fixed": ev.psnr(orig, image_selected, outside)})
        ev.OUT.mkdir(parents=True, exist_ok=True)
        (ev.OUT / "gate_audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
