"""Build and inspect a resumable outfit library without serving user uploads."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import asdict, fields, replace
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "src"))

import numpy as np

from offline_outfit_library import (CATEGORIES, OutfitLibrary, atomic_json, build_library,
                                   default_scenarios, digest, shoe_features)
from schemas import Product, UserProfile, PoseAnalysis
from offline_render_quality import (assess_preservation, basic_quality_status, pants_geometry,
                                    quality_fingerprint, reference_rank, render_is_eligible)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def render_quality_status(reports, input_valid):
    return basic_quality_status(reports, input_valid)


def status(root):
    metadata = read(root / "index.json")
    planned = read(root / "render_plan.json")["items"] if (root / "render_plan.json").exists() else []
    records = [read(p) for p in (root / "renders").rglob("*.json")]
    records = [r for r in records if r.get("library_identity") == metadata["identity"]]
    finished = {r.get("plan_key") or (r.get("combination_id", "") + "-" + digest(r["reference_condition"])[:8])
                for r in records if r.get("plan_key") or r.get("reference_condition")}
    return {"library_identity": metadata["identity"], "combinations": metadata["combination_count"],
            "scores": metadata["score_count"], "scenarios": len(metadata["scenarios"]),
            "render_plan": len(planned), "render_status": dict(Counter(r["status"] for r in records)),
            "verified": sum(render_is_eligible(r) for r in records),
            "pending": len({p["render_key"] for p in planned} - finished),
            "render_seconds": round(sum(r.get("seconds", 0) for r in records), 2)}


def fetch(root):
    from musinsa_crawler import API_URL, HEADERS
    root.mkdir(parents=True, exist_ok=True)
    for category, code in (("top", "001"), ("bottom", "003"), ("shoes", "103"), ("anchors", "001001")):
        path = root / f"popular_{category}.json"
        if path.exists():
            continue  # A run is an immutable snapshot; use a new directory to refresh.
        url = API_URL + "?" + urllib.parse.urlencode(dict(gf="A", category=code, sortCode="POPULAR",
                                                         page=1, size=100, caller="CATEGORY"))
        with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=20) as response:
            payload = json.load(response)
        items = payload.get("data", {}).get("list", [])
        if not items:
            raise ValueError(f"Empty {category} snapshot")
        atomic_json(path, {"url": url, "fetched_at": time.time(), "items": items})
        print(f"snapshot {category}: {len(items)}", flush=True)
        time.sleep(1)


def _shoe_type(name):
    groups = (("메리제인", ("메리제인", "mary jane")), ("로퍼", ("로퍼", "loafer")),
              ("더비슈즈", ("더비", "옥스포드", "derby", "oxford")),
              ("부츠", ("부츠", "boots")), ("러닝화", ("러닝", "running")),
              ("스니커즈", ("스니커", "sneaker", "운동화", "올드스쿨", "old skool", "척테일러")),
              ("샌들", ("샌들", "sandal")), ("슬리퍼", ("슬리퍼", "slide")))
    return next((label for label, words in groups if any(w in name.lower() for w in words)), "")


def prepare(root, device):
    from PIL import Image
    from config import FASHION_ATTRIBUTE_HEADS_PATH, FASHION_SIGLIP_MODEL_ID
    from fashion_model import FashionClassifier
    from musinsa_crawler import parse_item, COLOR_KEYWORDS, _match_keyword, HEADERS
    from enrich_catalog import (load_derivation, predict_for, derive, normalize, derive_style,
                                derive_purposes, derive_body_shapes)

    classifier = FashionClassifier(enabled=True, device=device, attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH)
    table = load_derivation()
    checkpoint_hash = hashlib.sha256(FASHION_ATTRIBUTE_HEADS_PATH.read_bytes()).hexdigest()
    records, failed = [], []
    items = [(c, rank, item, False) for c in CATEGORIES
             for rank, item in enumerate(read(root / f"popular_{c}.json")["items"], 1)]
    if (root / "popular_anchors.json").is_file():
        anchors = [item for item in read(root / "popular_anchors.json")["items"]
                   if re.search(r"화이트|white|백색", item.get("goodsName", ""), re.I)]
        # Real references, explicitly outside the overall popularity top 100.
        for gender in ("공용", "남성", "여성"):
            anchor = next((p for p in anchors if p.get("displayGenderText") == gender), None)
            if anchor:
                items.append(("top", 0, anchor, True))
    seen = set()
    for category, rank, item, anchor in items:
        product = parse_item(item, category)
        if product is None or product.product_id in seen:
            continue
        seen.add(product.product_id)
        cache_key = digest({"item": item, "category": category, "model": FASHION_SIGLIP_MODEL_ID,
                            "checkpoint": checkpoint_hash,
                            "code": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
        artifact = root / "products" / f"{product.product_id}-{cache_key[:12]}.json"
        embedding = artifact.with_suffix(".npy")
        image_path = root / "images" / f"{product.product_id}.jpg"
        try:
            if artifact.exists() and embedding.exists() and image_path.exists():
                record = read(artifact)
            else:
                image_path.parent.mkdir(parents=True, exist_ok=True)
                cached_image_valid = False
                if image_path.exists():
                    try:
                        with Image.open(image_path) as opened:
                            opened.load()
                        cached_image_valid = True
                    except (OSError, ValueError):
                        pass
                if not cached_image_valid:
                    if not product.image_url.startswith("https://image.msscdn.net/"):
                        raise ValueError("Unexpected or missing product image URL")
                    url = re.sub(r"_\d+(?=\.[a-z]+(?:\?|$))", "_big", product.image_url)
                    request = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(request, timeout=30) as response:
                        content = response.read()
                    temporary_image = image_path.with_suffix(".download")
                    temporary_image.write_bytes(content)
                    with Image.open(temporary_image) as opened:
                        opened.load()
                    temporary_image.replace(image_path)
                    time.sleep(.1)
                with Image.open(image_path) as opened:
                    rgb = opened.convert("RGB")
                color = _match_keyword(product.name, COLOR_KEYWORDS, "")
                row = asdict(product)
                row["color"] = color
                attributes, sources = {}, {"color": "title" if color else "unknown"}
                if category == "shoes":
                    kind = _shoe_type(product.name)
                    row.update(item_type=kind, style="", fit="", length="", material="", pattern="",
                               formality={"로퍼": 3, "더비슈즈": 5,
                                          "메리제인": 3, "부츠": 3}.get(kind, 1 if kind else 3))
                    sources["item_type"] = "title" if kind else "unknown"
                else:
                    predicted = predict_for(classifier, image_path, category == "top")
                    attributes["sleeve_length"] = predicted.get("_sleeve", "")
                    derived = derive(row, predicted, table)
                    predicted = normalize(predicted, table)
                    row.update({k: v for k, v in predicted.items() if not k.startswith("_")})
                    row.update(derived)
                    row["style"] = derive_style(row.get("item_type", ""), row["formality"], "", table)
                    row["purposes"] = derive_purposes(row.get("item_type", ""), row["formality"], table)
                    row["body_shapes"] = derive_body_shapes(category == "top", row, table)
                    sources.update(attributes="trained_heads", formality="catalog_derivation")
                    if anchor:
                        # Source category 001001 supplies short sleeve evidence.
                        row["item_type"] = "티셔츠"
                        attributes["sleeve_length"] = "반팔"
                        sources["item_type"] = sources["sleeve_length"] = "musinsa_category_001001"
                for key in ("purposes", "body_shapes", "activity_tags"):
                    value = row.get(key, [])
                    row[key] = value.split("|") if isinstance(value, str) and value else (value or [])
                row["image_path"] = str(image_path.resolve())
                row["catalog_color"] = color
                row["color_source"] = sources["color"]
                valid = {f.name for f in fields(Product)}
                normalized = Product(**{k: v for k, v in row.items() if k in valid})
                vector = classifier._encode_image(rgb).detach().cpu().numpy()[0]
                artifact.parent.mkdir(parents=True, exist_ok=True)
                np.save(embedding, vector.astype(np.float32), allow_pickle=False)
                record = {"product": asdict(normalized), "attributes": attributes, "sources": sources,
                          "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
                          "embedding": str(embedding.relative_to(root)), "cache_key": cache_key}
                atomic_json(artifact, record)
            record.update(popularity_rank=rank, reference_anchor=anchor)
            records.append(record)
            print(f"prepared {len(records)}/{len(items)} {product.product_id}", flush=True)
        except Exception as exc:
            failed.append({"id": product.product_id, "error": repr(exc)})
            print(f"product failed {product.product_id}: {exc}", flush=True)
    catalog = {"products": records, "failed": failed, "embedding_model": FASHION_SIGLIP_MODEL_ID,
               "checkpoint_sha256": checkpoint_hash,
               "snapshots": {p.name: digest(read(p)) for p in root.glob("popular_*.json")}}
    atomic_json(root / "catalog.json", catalog)
    print(f"catalog: {len(records)} prepared, {len(failed)} failures", flush=True)
    if not all(any(r["product"]["category"] == c for r in records) for c in CATEGORIES):
        raise ValueError("A category has no usable products")


def make_engine():
    from product_catalog import ProductCatalog
    from recommendation_engine import RecommendationEngine
    return RecommendationEngine(PROJECT / "FASHION_RULES_MASTER.md", ProductCatalog(PROJECT / "data/products.csv"))


def build(root, all_contexts):
    from recommendation_engine import PURPOSE_STYLES, PURPOSE_FORMALITY
    styles = sorted({s for values in PURPOSE_STYLES.values() for s in values})
    contexts = [(p, s) for p in PURPOSE_FORMALITY for s in styles] if all_contexts else None
    build_library(read(root / "catalog.json"), default_scenarios(contexts), root, make_engine())


def plan(root, per_scenario):
    library = OutfitLibrary(root)
    # Deduplicate combinations across contexts, retaining all requested contexts.
    planned = {}
    for scenario in library.metadata["scenarios"]:
        for gender in ("남성", "여성"):
            results = library.query(scenario["id"], {"top": {"color": "화이트", "item_type": "티셔츠",
                                                               "sleeve_length": "반팔"}},
                                    limit=per_scenario, gender=gender)
            for result in results:
                reference_condition = {"body_shape": scenario["pose"]["body_shape"],
                                       "proportion": scenario["id"].rsplit("/", 1)[1], "gender": gender}
                render_key = result["combination_id"] + "-" + digest(reference_condition)[:8]
                entry = planned.setdefault(render_key, {**result, "render_key": render_key,
                                           "reference_condition": reference_condition, "contexts": []})
                entry["contexts"].append({"scenario": scenario["id"], "gender": gender})
    entries = list(planned.values())
    atomic_json(root / "render_plan.json", {"library_identity": library.metadata["identity"],
                "anchor": "white_short_sleeved_tshirt", "per_scenario": per_scenario, "items": entries})
    print(f"render plan: {len(entries)} outfit/reference combinations across {len(library.metadata['scenarios'])} contexts", flush=True)
    if not entries:
        raise ValueError("No white short-sleeved anchor matches; inspect catalog metadata")


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def render_manifest(shoe_model, lower_backend):
    from config import FASHION_ATTRIBUTE_HEADS_PATH
    from shoe_tryon import MODEL_ID, MODEL_REVISION
    import os
    # Local weight file identity plus pinned upstream revision. Replacing a
    # checkpoint changes this identity, even when its configured path is reused.
    weights = {str(p.relative_to(shoe_model)): [p.stat().st_size, p.stat().st_mtime_ns]
               for p in shoe_model.rglob("*") if p.is_file()}
    hf_refs = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    revisions = {str(p.relative_to(hf_refs)): p.read_text().strip()
                 for model in ("models--zhengchong--CatVTON", "models--booksforcharlie--stable-diffusion-inpainting")
                 for p in (hf_refs / model / "refs").glob("*") if p.is_file()}
    return {"code": {name: file_sha(PROJECT / "src" / name) for name in
            ("catvton_tryon.py", "shoe_tryon.py", "offline_lower_tryon.py", "offline_render_quality.py",
             "clothing_parser.py", "outfit_analyzer.py", "pose_analyzer.py", "tryon_quality.py")},
            "script": file_sha(__file__), "attribute_checkpoint": file_sha(FASHION_ATTRIBUTE_HEADS_PATH),
            "shoe_model": MODEL_ID, "shoe_revision": MODEL_REVISION, "local_weights": weights,
            "catvton_cached_revisions": revisions, "lower_backend": lower_backend,
            "seed": 42, "clothing_preset": "fast", "max_retries": 1}


def audit_evidence(root, evidence):
    catalog = read(root / "catalog.json")
    products = {p["product"]["product_id"]: p["product"] for p in catalog["products"]}
    rows = []
    for path in sorted((evidence / "renders").glob("*.json")):
        cached = read(path)
        original = cached["original"]
        selected = [products[pid] for pid in original["product_ids"]]
        bottom_id = original["product_ids"][1]
        with np.load(evidence / "products" / f"{bottom_id}.npz", allow_pickle=False) as data:
            target = pants_geometry(data["segmentation"])
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as data:
            report = assess_preservation(selected, cached["outfit"], data["segmentation"], target,
                                         original.get("quality_reports", []))
        row = {"name": path.stem, "previous_status": original["status"], "preservation": report,
               "reference": original["reference"], "product_ids": original["product_ids"]}
        rows.append(row)
    result = {"quality_fingerprint": quality_fingerprint(), "count": len(rows),
              "previous_status": dict(Counter(r["previous_status"] for r in rows)),
              "new_status": dict(Counter(r["preservation"]["status"] for r in rows)),
              "failures": dict(Counter(f for r in rows for f in r["preservation"]["failed"])),
              "items": rows}
    atomic_json(evidence.parent / "audit.json", result)
    print(json.dumps({k: v for k, v in result.items() if k != "items"}, ensure_ascii=False, indent=2))


def record_visual_review(root, record_path, decision, reviewer, notes):
    record_path = record_path.resolve()
    if not record_path.is_relative_to((root / "renders").resolve()):
        raise ValueError("Review record must belong to this library's renders directory")
    record = read(record_path)
    actual_sha = file_sha(record_path.with_suffix(".png"))
    if actual_sha != record.get("output_sha256"):
        raise ValueError("Image changed since automatic assessment; render/reassess again")
    quality = record.get("preservation", {})
    if decision == "accepted" and (quality.get("status") != "auto_passed"
                                     or quality.get("fingerprint") != quality_fingerprint()):
        raise ValueError("An accepted review requires a current automatic preservation pass")
    record["visual_review"] = {"decision": decision, "reviewer": reviewer, "notes": notes,
                               "output_sha256": actual_sha, "reviewed_at": time.time()}
    record["status"] = "complete" if decision == "accepted" else "visual_rejected"
    atomic_json(record_path, record)


def render(root, people, shoe_model, max_renders, *, evidence=None, lower_backend="catvton",
           reference_name=None, combination_ids=None):
    from PIL import Image
    from config import FASHION_ATTRIBUTE_HEADS_PATH
    from fashion_model import FashionClassifier
    from clothing_parser import ClothingParser
    from pose_analyzer import PoseAnalyzer
    from outfit_analyzer import OutfitAnalyzer
    from catvton_tryon import CatVTONTryOn
    from shoe_tryon import OutfitTryOn, ShoeTryOn, foot_edit_mask
    from schemas import Recommendation

    library, engine = OutfitLibrary(root), make_engine()
    render_plan = read(root / "render_plan.json")
    if render_plan["library_identity"] != library.metadata["identity"]:
        raise ValueError("Render plan does not match library")
    parser, poses = ClothingParser(use_fashn=True), PoseAnalyzer()
    classifier = FashionClassifier(enabled=True, attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH)
    analyzer = OutfitAnalyzer(parser, classifier)
    manifest = render_manifest(shoe_model, lower_backend)
    renderer_id = digest(manifest)
    atomic_json(root / "render_manifests" / f"{renderer_id}.json", manifest)
    clothing = CatVTONTryOn.fast(garment_cache_dir=root / "clean" / renderer_id[:16], max_retries=1)
    clothing._garment_parser = parser
    footwear_editor = ShoeTryOn(shoe_model)
    if lower_backend == "flux":
        from offline_lower_tryon import OfflineLowerTryOn
        clothing = OfflineLowerTryOn(clothing, footwear_editor, parser)
    adapter = OutfitTryOn(clothing, footwear_editor, parser)
    if not adapter.available or "shoes" not in adapter.supported_categories:
        raise ValueError("All three try-on categories are required")
    if evidence is None:
        raise ValueError("Run inspect_offline_outfits.py first and pass --evidence")
    references = []
    for cached_path in sorted((evidence / "people").glob("*.json")):
        cached = read(cached_path)
        path = people / cached["name"]
        if path.is_file() and file_sha(path) == cached["image_sha256"]:
            with np.load(cached_path.with_suffix(".npz"), allow_pickle=False) as data:
                references.append((cached, path, data["segmentation"]))
    if not references:
        raise ValueError("No current image evidence; run the inspection stage first")
    analyzed = {}
    selections = []
    completed = 0
    for entry in render_plan["items"]:
        if combination_ids and entry["combination_id"] not in combination_ids:
            continue
        if max_renders and completed >= max_renders:
            break
        condition = entry["reference_condition"]
        products = [Product(**p) for p in entry["products"]]
        bottom_path = evidence / "products" / f"{products[1].product_id}.npz"
        target_geometry = {"measurable": False, "reason": "no_product_evidence"}
        if bottom_path.is_file():
            product_evidence = read(bottom_path.with_suffix(".json"))
            if product_evidence["image_sha256"] != file_sha(products[1].image_path):
                raise ValueError("Product image changed; rebuild the image evidence before rendering")
            with np.load(bottom_path, allow_pickle=False) as data:
                target_geometry = pants_geometry(data["segmentation"])
        candidates = []
        reasons = Counter()
        for cached, path, segmentation in references:
            p = cached["pose"]
            if (cached["gender"] != condition["gender"] or p["body_shape"] != condition["body_shape"]
                    or ("short" if .40 <= p["leg_ratio"] < .60 else "normal") != condition["proportion"]):
                continue
            rank = reference_rank(cached, entry["products"], target_geometry, segmentation)
            reasons.update(rank["reasons"])
            if reference_name:
                if path.name == reference_name:
                    candidates.append((rank["penalty"], path.name, cached, path, rank))
            elif rank["eligible"]:
                candidates.append((rank["penalty"], path.name, cached, path, rank))
        missing_path = root / "renders" / entry["combination_id"] / f"{entry['render_key']}-missing.json"
        if not candidates:
            if reference_name:
                continue  # Explicit A/B subset, not a claim of missing references elsewhere.
            record_path = missing_path
            atomic_json(record_path, {"status": "missing_reference", "combination_id": entry["combination_id"],
                        "plan_key": entry["render_key"], "library_identity": library.metadata["identity"],
                        "reference_condition": condition, "rejection_reasons": dict(reasons)})
            continue
        _, _, cached, person, rank = min(candidates, key=lambda r: r[:2])
        gender = cached["gender"]
        inputs = {"renderer": renderer_id, "library": library.metadata["identity"],
                  "reference": cached["image_sha256"], "products": [file_sha(p.image_path) for p in products]}
        render_id = digest(inputs)
        record_path = root / "renders" / entry["combination_id"] / f"{entry['render_key']}-{render_id[:12]}.json"
        if record_path.exists():
            previous = read(record_path)
            if (previous.get("status") in {"complete", "auto_passed", "quality_failed", "unassessed", "visual_rejected"}
                    and previous.get("render_fingerprint") == render_id
                    and record_path.with_suffix(".png").exists()
                    and previous.get("output_sha256") == file_sha(record_path.with_suffix(".png"))):
                continue
        missing_path.unlink(missing_ok=True)
        if person not in analyzed:
            pose = poses.analyze(person)
            outfit, parsed = analyzer.analyze(person, pose)
            foot_edit_mask(parsed["segmentation"], pose)
            analyzed[person] = pose, outfit, parsed
        pose, outfit, parsed = analyzed[person]
        if (pose.body_shape != condition["body_shape"]
                or ("short" if .40 <= pose.leg_ratio < .60 else "normal") != condition["proportion"]):
            raise ValueError("Reference analysis changed; rebuild the evidence cache")
        current_rank = reference_rank({**cached, "pose": asdict(pose), "outfit": asdict(outfit)},
                                      entry["products"], target_geometry, parsed["segmentation"])
        if not reference_name and not current_rank["eligible"]:
            raise ValueError("Reference eligibility changed; rebuild the evidence cache")
        selections.append({"plan_key": entry["render_key"], "reference": person.name, "rank": current_rank})
        record = {"combination_id": entry["combination_id"], "product_ids": entry["product_ids"],
                  "library_identity": library.metadata["identity"], "reference": person.name,
                  "plan_key": entry["render_key"], "render_fingerprint": render_id, "render_inputs": inputs,
                  "renderer_manifest": renderer_id, "lower_backend": lower_backend,
                  "reference_selection": current_rank, "explicit_evaluation_reference": bool(reference_name),
                  "reference_pose": asdict(pose), "reference_gender": gender,
                  "reference_condition": condition,
                  "visual_review": "pending", "shoe_score_source": "product_metadata_not_visual_recognition"}
        start = time.monotonic()
        output = record_path.with_suffix(".png")
        try:
            adapter.synthesize(person, Recommendation(1, products, entry["score"], {}, []), output,
                               {**parsed, "pose": pose, "outfit": outfit, "classifier": classifier, "strict_vton": True})
            rendered, rendered_parsed = analyzer.analyze(output, pose)
            scores = {}
            for scenario in library.metadata["scenarios"]:
                if (scenario["pose"]["body_shape"] != condition["body_shape"]
                        or scenario["id"].rsplit("/", 1)[1] != condition["proportion"]):
                    continue
                profile, scenario_pose = UserProfile(**scenario["profile"]), PoseAnalysis(**scenario["pose"])
                top = engine._garment(None, "top", rendered)
                bottom = engine._garment(None, "bottom", rendered)
                diagnostic = engine._diagnose_pair(top, bottom, profile, scenario_pose)
                observed = [replace(products[i], color=g["color"], fit=g["fit"], length=g["length"],
                                    item_type=g["item_type"], formality=g["formality"])
                            for i, g in enumerate((top, bottom))]
                footwear = shoe_features(*observed, products[2], profile, engine)
                scores[scenario["id"]] = round(.85 * diagnostic["overall_score"] + .15 * float(footwear.mean()), 3)
            vector = classifier._encode_image(output).detach().cpu().numpy()[0]
            np.save(output.with_suffix(".npy"), vector.astype(np.float32), allow_pickle=False)
            preservation = assess_preservation(entry["products"], asdict(rendered),
                rendered_parsed["segmentation"], target_geometry, adapter.last_quality_reports)
            np.savez_compressed(output.with_suffix(".npz"), segmentation=rendered_parsed["segmentation"])
            record.update(status=preservation["status"], preservation=preservation, output_sha256=file_sha(output),
                          output=output.name, embedding=output.with_suffix(".npy").name,
                          observed_outfit=asdict(rendered), scores=scores,
                          warnings=adapter.last_warnings, quality_reports=adapter.last_quality_reports,
                          score_transfer="nearest_body_shape_and_leg_ratio_reference_not_personal_fit")
        except Exception as exc:
            import traceback
            traceback.print_exc()
            record.update(status="failed", error=repr(exc))
        record["seconds"] = round(time.monotonic() - start, 2)
        atomic_json(record_path, record)
        completed += 1
        print(f"render {completed}: {entry['combination_id']} {record['status']} {record['seconds']}s", flush=True)
    atomic_json(root / "references.json", {"candidate_count": len(references), "selected": selections})
    poses.close()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("stage", choices=("fetch", "prepare", "index", "plan", "render", "query", "status", "audit", "review"))
    cli.add_argument("--root", type=Path, required=True)
    cli.add_argument("--device", default="auto")
    cli.add_argument("--all-contexts", action="store_true")
    cli.add_argument("--per-scenario", type=int, default=3)
    cli.add_argument("--people", type=Path)
    cli.add_argument("--shoe-model", type=Path)
    cli.add_argument("--max-renders", type=int, default=0, help="0 means resume all planned renders")
    cli.add_argument("--evidence", type=Path)
    cli.add_argument("--lower-backend", choices=("catvton", "flux"), default="catvton")
    cli.add_argument("--reference-name", help="Explicit reference for controlled evaluation only")
    cli.add_argument("--combination-ids", nargs="+")
    cli.add_argument("--record", type=Path)
    cli.add_argument("--decision", choices=("accepted", "rejected"))
    cli.add_argument("--reviewer")
    cli.add_argument("--notes")
    cli.add_argument("--scenario")
    cli.add_argument("--kept", default='{"top":{"color":"화이트","item_type":"티셔츠","sleeve_length":"반팔"}}')
    cli.add_argument("--gender", default="")
    cli.add_argument("--max-budget", type=int)
    cli.add_argument("--verified-only", action="store_true")
    args = cli.parse_args()
    root = args.root.resolve()
    if args.stage == "fetch": fetch(root)
    elif args.stage == "prepare": prepare(root, args.device)
    elif args.stage == "index": build(root, args.all_contexts)
    elif args.stage == "plan": plan(root, args.per_scenario)
    elif args.stage == "status": print(json.dumps(status(root), ensure_ascii=False, indent=2))
    elif args.stage == "audit":
        if not args.evidence: cli.error("audit requires --evidence")
        audit_evidence(root, args.evidence)
    elif args.stage == "review":
        if not all((args.record, args.decision, args.reviewer, args.notes)):
            cli.error("review requires --record, --decision, --reviewer and --notes")
        record_visual_review(root, args.record, args.decision, args.reviewer, args.notes)
    elif args.stage == "render":
        if not args.people or not args.shoe_model:
            cli.error("render requires --people and --shoe-model")
        render(root, args.people, args.shoe_model, args.max_renders, evidence=args.evidence,
               lower_backend=args.lower_backend, reference_name=args.reference_name,
               combination_ids=args.combination_ids)
    else:
        library = OutfitLibrary(root)
        scenario = args.scenario or library.metadata["scenarios"][0]["id"]
        print(json.dumps(library.query(scenario, json.loads(args.kept), limit=args.per_scenario,
                         gender=args.gender, max_budget=args.max_budget, verified_only=args.verified_only),
                         ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
