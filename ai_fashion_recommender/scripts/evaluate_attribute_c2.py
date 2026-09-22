"""Replay the C-2 runtime policy on the frozen C-1 products (AI references)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import benchmark_product_attributes as benchmark
from live_product_attributes import LiveProductAttributes, POLICY_VERSION
from live_product_attributes import accepted_attributes, photo_matches
from evaluate_attribute_c1_ai_review import pair_predictions
from musinsa_live_search import MusinsaLiveSearch


def make_provider(args):
    import open_clip
    import torch
    from clothing_parser import ClothingParser
    from fashion_attribute_model import FashionAttributePredictor
    if benchmark.digest(args.checkpoint) != args.expected_sha256:
        raise ValueError("Checkpoint differs from the deployed checkpoint")
    torch.set_num_threads(4)
    model, _, preprocess = open_clip.create_model_and_transforms(
        f"local-dir:{args.backbone_dir}", device=args.device)
    model.eval()
    predictor = FashionAttributePredictor(args.checkpoint, image_encoder=model, preprocess=preprocess,
        model_id="Marqo/marqo-fashionSigLIP", device=args.device)
    return LiveProductAttributes(ClothingParser(use_fashn=True), predictor)


def predict(args):
    provider = make_provider(args)
    results = []
    started = time.monotonic()
    try:
        for row in benchmark.read_csv(args.labels):
            path = args.images / f"{row['product_id']}.jpg"
            if benchmark.digest(path) != row["image_sha256"]:
                raise ValueError("Frozen image changed")
            tick = time.monotonic()
            evidence = provider.predict_image(path, row["category"])
            results.append({"product_id": row["product_id"], "category": row["category"],
                            "image_sha256": row["image_sha256"], **evidence,
                            "elapsed_seconds": time.monotonic() - tick})
            print(row["product_id"], evidence["context"], evidence["attributes"], flush=True)
    finally:
        provider.close()
    benchmark.write_json(args.output, {"policy": POLICY_VERSION,
        "checkpoint_sha256": benchmark.digest(args.checkpoint), "device": args.device,
        "labels_sha256": benchmark.digest(args.labels),
        "policy_sha256": benchmark.digest(benchmark.ROOT / "src/live_product_attributes.py"),
        "created_at": benchmark.now(), "elapsed_seconds": time.monotonic() - started,
        "results": results})


def report(args):
    rows = benchmark.read_csv(args.labels)
    original = json.loads(args.baseline.read_text())
    records, pairs = pair_predictions(rows, original)
    payload = json.loads(args.predictions.read_text())
    if payload["checkpoint_sha256"] != original["checkpoint_sha256"]:
        raise ValueError("Checkpoints differ")
    if payload["labels_sha256"] != benchmark.digest(args.labels):
        raise ValueError("References changed")
    photos = {p["product_id"]: p for p in payload["results"]}
    if len(photos) != 100 or len(payload["results"]) != 100:
        raise ValueError("Require 100 distinct photo results")
    search = MusinsaLiveSearch()
    output = []
    try:
        for row in rows:
            photo = photos[row["product_id"]]
            if photo["image_sha256"] != row["image_sha256"] or photo["category"] != row["category"]:
                raise ValueError("Photo input changed")
            if photo["attributes"] != accepted_attributes(row["category"], photo["context"], photo["predictions"]):
                raise ValueError("Stored decision differs from runtime policy")
            for axis in benchmark.AXES:
                vocabulary = benchmark.vocabulary(row["category"], axis)
                attributes = {axis: list(dict.fromkeys(k for _, k in vocabulary))}
                item = {"goodsName": row["name"], "brandName": row["brand_name"],
                        "brand": row["brand"], "category": row["category"]}
                before, matched = search._score(item, attributes, 0)
                after, preserved = search._score(item, attributes, 0, photo_attributes=photo["attributes"])
                if matched != preserved or after < before:
                    raise ValueError("Existing name match changed")
                labels = records[row["product_id"]]["title"][axis]["labels"]
                addition = photo_matches(row["category"], row["name"], attributes, photo["attributes"])
                supplemented = not labels and after > before and axis in addition
                after_labels = [addition[axis]["label"]] if supplemented else labels
                gold = row[f"gold_{axis}"]
                evaluable = gold not in benchmark.SKIP_LABELS
                output.append({"product_id": row["product_id"], "category": row["category"],
                    "cut_type": row["cut_type"], "axis": axis, "reference": gold, "evaluable": evaluable,
                    "no_title_keyword": records[row["product_id"]]["title"][axis]["no_title_keyword"],
                    "before": labels, "after": after_labels, "supplemented": supplemented,
                    "before_correct": evaluable and set(labels) == set(gold.split("|")),
                    "after_correct": evaluable and set(after_labels) == set(gold.split("|")),
                    "before_score": before, "after_score": after})
    finally:
        search.close()
    lines = ["# C-2 같은 100개 전후 비교", "",
        "AI 참조 라벨과의 일치율이며 사람 정답 정확도가 아니다. 같은 표본을 보고 설계한 탐색적 평가다.",
        "상위 후보로 선택되었을 때의 속성 보충을 실제 `_score()`로 재생한다. 한 검색에서 100개를 모두 추론하지 않는다.", ""]
    summary = {}
    for missing, heading in ((False, "전체"), (True, "상품명 키워드 없음")):
        lines += [f"## {heading}", "", "| 축 | 상품명 | C-2 | 새 일치 / 새 판정 | 새 불일치 |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        values = {}
        for axis in benchmark.AXES:
            chosen = [p for p in output if p["axis"] == axis and p["evaluable"]
                      and (not missing or p["no_title_keyword"])]
            added = [p for p in chosen if p["supplemented"]]
            values[axis] = {"n": len(chosen), "before": sum(p["before_correct"] for p in chosen),
                "after": sum(p["after_correct"] for p in chosen), "added": len(added),
                "added_correct": sum(p["after_correct"] for p in added)}
            v = values[axis]
            lines.append(f"| {axis} | {v['before']}/{v['n']} | {v['after']}/{v['n']} | "
                         f"{v['added_correct']}/{v['added']} | {v['added'] - v['added_correct']} |")
        summary[heading] = values
        lines.append("")
    lines += ["## 컷별", "", "| 컷 | 축 | 상품명 | C-2 |", "| --- | --- | ---: | ---: |"]
    for cut in (*benchmark.CUTS, "other"):
        for axis in benchmark.AXES:
            chosen = [p for p in output if p["cut_type"] == cut and p["axis"] == axis and p["evaluable"]]
            lines.append(f"| {cut} | {axis} | {sum(p['before_correct'] for p in chosen)}/{len(chosen)} | "
                         f"{sum(p['after_correct'] for p in chosen)}/{len(chosen)} |")
    args.output.write_text("\n".join(lines) + "\n")
    benchmark.write_json(args.output.with_suffix(".json"), {"policy": POLICY_VERSION,
        "inputs": {name: benchmark.digest(getattr(args, name)) for name in ("labels", "baseline", "predictions")},
        "summary": summary, "pairs": output})
    print(json.dumps(summary, ensure_ascii=False))


def runtime(args):
    """Freeze candidate input, run real GPU inference and optionally real CDN downloads."""
    import tempfile
    from functools import partial
    from recommendation_keywords import TargetKeywordResult
    from schemas import UserProfile
    from unittest.mock import patch
    sys.path.insert(0, str(benchmark.ROOT.parent))
    from web.pipeline import _cache_live_shopping_image
    provider = make_provider(args)
    search = MusinsaLiveSearch(photo_provider=provider)
    rows = benchmark.read_csv(args.labels)
    products = {category: [{"goodsNo": r["product_id"].removeprefix("MS"), "goodsName": r["name"],
        "brandName": r["brand_name"], "brand": r["brand"], "thumbnail": r["image_url"], "finalPrice": 50000}
        for r in rows if r["category"] == category] for category in benchmark.CATEGORIES}
    cases = {
        "all_axes": {c: {a: list(dict.fromkeys(k for _, k in benchmark.vocabulary(c, a))) for a in benchmark.AXES}
                     for c in benchmark.CATEGORIES},
        "fit": {"top": {"fit": ["여유핏", "레귤러"]}, "bottom": {"fit": ["와이드", "스트레이트"]}},
        "length": {"top": {"length": ["허리선", "기본 기장", "롱"]},
                   "bottom": {"length": ["풀렝스", "쇼츠"]}},
    }
    measurements = []
    live_measurements = []
    try:
        for case, attributes in cases.items():
            targets = TargetKeywordResult("mixed", attributes)
            with patch.object(search, "_fetch", side_effect=lambda c, *a, **kw: products[c]):
                for transport in ("local_images", "musinsa_cdn"):
                    with tempfile.TemporaryDirectory() as directory:
                        loader = (lambda product, timeout: args.images / f"{product.product_id}.jpg") if transport == "local_images" else partial(
                            _cache_live_shopping_image, output_dir=Path(directory))
                        provider._cache.clear()
                        for run in ("baseline", "cold", "repeat1", "repeat2", "repeat3", "repeat4", "repeat5", "repeat6"):
                            search.photo_budget = 0 if run == "baseline" else 0.25
                            start = time.monotonic()
                            selected = search.search(targets, UserProfile(), limit=6, photo_loader=loader)
                            elapsed = time.monotonic() - start
                            if run == "baseline":
                                protected = {i: p.product_id for i, p in enumerate(selected) if p.matched_keywords}
                            regressions = [pid for i, pid in protected.items() if selected[i].product_id != pid]
                            record = {"case": case, "transport": transport, "run": run,
                                "elapsed_seconds": elapsed, "stats": search.last_search_stats,
                                "selected": [p.product_id for p in selected], "protected_positions": protected,
                                "rank_regressions": regressions,
                                "photo_evidence": {p.product_id: p.photo_attributes for p in selected if p.photo_attributes}}
                            measurements.append(record)
                            print(case, transport, run, round(elapsed, 4), search.last_search_stats.get("photo"), flush=True)
                            if regressions:
                                raise ValueError("Title-matched result lost its position")
        if args.live_api:
            from product_measurements import ProductMeasurementClient
            targets = TargetKeywordResult("mixed", {
                "top": {"fit": ["여유핏", "레귤러"], "length": ["기본 기장"], "material": ["코튼"]},
                "bottom": {"fit": ["와이드", "스트레이트"], "length": ["풀렝스"], "material": ["데님"]},
                "shoes": {"item_type": ["스니커즈"]}})
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                search.measurements = ProductMeasurementClient(root / "sizes")
                loader = partial(_cache_live_shopping_image, output_dir=root)
                for run in ("warmup", "baseline1", "photo1", "baseline2", "photo2", "baseline3", "photo3",
                            "baseline4", "photo4", "baseline5", "photo5", "baseline6", "photo6"):
                    search.photo_budget = .25 if run.startswith("photo") else 0
                    start = time.monotonic()
                    selected = search.search(targets, UserProfile(), limit=6, photo_loader=loader)
                    record = {"run": run, "elapsed_seconds": time.monotonic() - start,
                        "stats": search.last_search_stats, "selected": [p.product_id for p in selected],
                        "photo_evidence": {p.product_id: p.photo_attributes for p in selected if p.photo_attributes}}
                    live_measurements.append(record)
                    print("live_api", run, round(record["elapsed_seconds"], 4), search.last_search_stats, flush=True)
                for enabled in (False, True):
                    search._cache.clear()
                    provider._cache.clear()
                    cold_root = root / ("cold_photo" if enabled else "cold_baseline")
                    cold_root.mkdir()
                    search.measurements = ProductMeasurementClient(cold_root / "sizes")
                    search.photo_budget = .25 if enabled else 0
                    start = time.monotonic()
                    selected = search.search(targets, UserProfile(), limit=6,
                        photo_loader=partial(_cache_live_shopping_image, output_dir=cold_root))
                    record = {"run": cold_root.name, "elapsed_seconds": time.monotonic() - start,
                        "stats": search.last_search_stats, "selected": [p.product_id for p in selected],
                        "photo_evidence": {p.product_id: p.photo_attributes for p in selected if p.photo_attributes}}
                    live_measurements.append(record)
                    print("live_api", cold_root.name, round(record["elapsed_seconds"], 4), search.last_search_stats, flush=True)
    finally:
        search.close()
    benchmark.write_json(args.output, {"created_at": benchmark.now(), "policy": POLICY_VERSION,
        "device": args.device, "checkpoint_sha256": benchmark.digest(args.checkpoint),
        "source_sha256": {name: benchmark.digest(benchmark.ROOT / "src" / name)
                          for name in ("live_product_attributes.py", "musinsa_live_search.py")},
        "labels_sha256": benchmark.digest(args.labels),
        "scope": "Fixed C-1 100 candidate rows; real model inference and CDN; search API and measurements excluded",
        "results": measurements, "live_api_results": live_measurements})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for command, function in (("predict", predict), ("runtime", runtime)):
        p = sub.add_parser(command)
        for name in ("labels", "images", "checkpoint", "backbone-dir", "output"):
            p.add_argument(f"--{name}", type=Path, required=True)
        p.add_argument("--expected-sha256", required=True)
        p.add_argument("--device", default="cuda")
        if command == "runtime":
            p.add_argument("--live-api", action="store_true", help="Also measure real search and size APIs")
        p.set_defaults(run=function)
    p = sub.add_parser("report")
    for name in ("labels", "baseline", "predictions", "output"):
        p.add_argument(f"--{name}", type=Path, required=True)
    p.set_defaults(run=report)
    args = parser.parse_args()
    args.run(args)


if __name__ == "__main__":
    main()
