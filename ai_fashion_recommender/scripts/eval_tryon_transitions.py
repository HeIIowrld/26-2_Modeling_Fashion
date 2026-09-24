"""Paired transition smoke test; keeps production releases and images untouched.

python eval_tryon_transitions.py MANIFEST.json OUTPUT --runtime-repo RELEASE
Manifest: {"people": [absolute paths], "products": [{"id", "name", "category", "path"}]}.
Images stay outside version control. Metrics complement, not replace, visual review.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
from PIL import Image, ImageDraw, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("manifest", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--runtime-repo", type=Path, required=True)
    ap.add_argument("--transition-model", type=Path)
    ap.add_argument("--max-retries", type=int, choices=(0, 1), default=0)
    args = ap.parse_args()
    import catvton_tryon
    from clothing_parser import ClothingParser
    from fashion_model import FashionClassifier
    from outfit_analyzer import OutfitAnalyzer
    from pose_analyzer import PoseAnalyzer
    from schemas import Product, Recommendation

    catvton_tryon.CATVTON_REPO = args.runtime_repo / "third_party" / "CatVTON"
    manifest = json.loads(args.manifest.read_text(encoding="utf-8-sig"))
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "results.jsonl").exists():
        raise FileExistsError("Use a fresh output directory to avoid mixing evaluation runs.")
    sources = [Path(p) for p in manifest["people"]] + [Path(p["path"]) for p in manifest["products"]]
    code = Path(__file__).resolve().parents[1] / "src"
    (args.output / "inputs.json").write_text(json.dumps({
        "manifest": manifest, "sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "seed": 42, "catvton_steps": 25, "reference_editor_steps": 4, "max_retries": args.max_retries,
        "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                        (code / name for name in ("catvton_tryon.py", "tryon_transition.py", "shoe_tryon.py", "tryon_quality.py"))},
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    parser = ClothingParser(use_fashn=True)
    classifier = FashionClassifier(enabled=True, attribute_checkpoint=(args.runtime_repo /
                                     "ai_fashion_recommender/models/fashion_attribute_heads_augmented.pt"))
    analyzer = OutfitAnalyzer(parser, classifier)
    pose_analyzer = PoseAnalyzer(model_complexity=1)
    adapter = catvton_tryon.CatVTONTryOn.fast(garment_cache_dir=args.output / "cache", max_retries=args.max_retries)
    adapter._garment_parser = parser
    if args.transition_model:
        from shoe_tryon import ShoeTryOn
        adapter.transition_editor = ShoeTryOn(args.transition_model)
    for path in manifest["people"]:
        person = Path(path)
        pose = pose_analyzer.analyze(person)
        outfit, parsed = analyzer.analyze(person, pose)
        context = {k: parsed.get(k) for k in ("upper_mask", "lower_mask", "upper_style_mask",
                                              "lower_style_mask", "segmentation")}
        context.update(outfit=outfit, classifier=classifier, pose=pose, parser=parser)
        for item in manifest["products"]:
            product = Product(item["id"], item["name"], item["category"], "", "", [], [], 0, "", True,
                              image_path=item["path"])
            rec = Recommendation(rank=1, products=[product], total_score=0, score_breakdown={}, reasons=[])
            images = [("source", Image.open(person).convert("RGB")),
                      ("product", Image.open(item["path"]).convert("RGB"))]
            for enabled in (False, True):
                adapter.transition_correction = enabled
                variant = "corrected" if enabled else "baseline"
                target = args.output / f"{person.stem}__{product.product_id}__{variant}.png"
                record = {"person": person.name, "product": item, "variant": variant}
                start = time.perf_counter()
                try:
                    adapter.generate(person, rec, target, context=context)
                    image = Image.open(target).convert("RGB")
                    images.append((variant, image))
                    seg = parser.parse(image, pose=None)["segmentation"]
                    record.update(notes=adapter.last_mask_notes, warnings=adapter.last_warnings,
                                  quality=adapter.last_quality_reports,
                                  original_lengths={"bottom": outfit.bottom_length, "sleeve": outfit.sleeve_length},
                                  target_pixels=int(np.isin(seg, (3,) if product.category == "top" else (5, 6)).sum()),
                                  new_skin_pixels=int((np.isin(seg, (12, 14)) & np.isin(parsed["segmentation"], (3, 6))).sum()))
                except Exception:
                    record["error"] = traceback.format_exc()
                    images.append((variant + " FAILED", Image.new("RGB", (320, 470), "white")))
                record["seconds"] = round(time.perf_counter() - start, 2)
                with (args.output / "results.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(record, ensure_ascii=False) + "\n")
                print(json.dumps(record, ensure_ascii=False), flush=True)
            sheet = Image.new("RGB", (320 * len(images), 500), "white")
            draw = ImageDraw.Draw(sheet)
            for index, (label, image) in enumerate(images):
                draw.text((index * 320 + 10, 5), label, fill="black")
                sheet.paste(ImageOps.contain(image, (320, 470)), (index * 320, 25))
            sheet.save(args.output / f"{person.stem}__{product.product_id}__sheet.jpg")
    pose_analyzer.close()


if __name__ == "__main__":
    main()
