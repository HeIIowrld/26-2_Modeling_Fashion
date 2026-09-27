"""Persistent top × bottom × shoes scores; lookup never runs a renderer.

The shoe contribution is an explicit, uncalibrated metadata heuristic. Image
embeddings are retrieval features, not a learned compatibility score.
"""
from __future__ import annotations

import hashlib
import ast
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np

from schemas import BODY_SHAPES, OutfitAnalysis, PoseAnalysis, Product, UserProfile
from offline_render_quality import render_is_eligible

CATEGORIES = ("top", "bottom", "shoes")
VERSION = 1
FEATURE_NAMES = ("top.body_fit", "top.situation_fit", "top.style_fit",
                 "bottom.body_fit", "bottom.situation_fit", "bottom.style_fit",
                 "outfit.harmony", "shoe.style", "shoe.formality", "shoe.color")
SHOE_STYLES = {
    "캐주얼": {"스니커즈", "로퍼"}, "미니멀": {"스니커즈", "로퍼", "더비슈즈"},
    "포멀": {"로퍼", "더비슈즈"}, "스포티": {"스니커즈", "러닝화"},
    "스트리트": {"스니커즈", "부츠"}, "로맨틱": {"메리제인", "로퍼"},
}


def digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode()).hexdigest()


def scoring_fingerprint() -> str:
    root = Path(__file__).resolve().parent
    paths = [root / name for name in ("recommendation_engine.py",
             "outfit_analyzer.py", "schemas.py", "config.py", "fashion_rules.py")]
    paths.append(root.parent / "FASHION_RULES_MASTER.md")
    # Query/review changes must not invalidate 216 million unchanged scores.
    source = Path(__file__).read_text()
    scoring_nodes = [node for node in ast.parse(source).body
                     if isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef)
                     and node.name in {"default_scenarios", "empty_outfit", "shoe_features", "score_triple", "build_library"}]
    return digest({**{p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                   "offline_scoring": [ast.get_source_segment(source, n) for n in scoring_nodes]})


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def default_scenarios(styles=None):
    scenarios = []
    for purpose, style in styles or [("데일리", "캐주얼")]:
        for shape in BODY_SHAPES:
            for proportion, leg_ratio in (("normal", .65), ("short", .52)):
                scenarios.append({
                    "id": f"{purpose}/{style}/{shape}/{proportion}",
                    "profile": asdict(UserProfile(purpose=purpose, desired_style=style)),
                    "pose": asdict(PoseAnalysis(True, 1.0, shape, 0, 0, leg_ratio,
                                                "rule_template", 1.0)),
                    "source": "rule_template_not_measured_person",
                })
    return scenarios


def empty_outfit():
    return OutfitAnalysis("offline", "", "", "", [], "")


def shoe_features(top, bottom, shoe, profile, engine):
    """0..100 metadata scores; unknown attributes are neutral, never invented."""
    known = bool(shoe.item_type)
    style = (92 if shoe.item_type in SHOE_STYLES.get(profile.desired_style, set()) else 72) if known else 50
    formality = ({0: 100, 1: 90, 2: 72}.get(abs(bottom.formality - shoe.formality), 40)
                 if known and bottom.item_type else 50)
    colors = []
    for garment in (top, bottom):
        if not garment.color or not shoe.color:
            colors.append(50)
        else:
            colors.append({"안정적인 무채색 조합": 92, "톤온톤": 90,
                           "유사색 조합": 88, "대비색 조합": 76,
                           "보통 조합": 68}[engine._safe_harmony(garment.color, shoe.color)])
    return np.asarray([style, formality, sum(colors) / 2], dtype=np.float32)


def score_triple(top, bottom, shoe, profile, pose, engine):
    diagnostic = engine._diagnose_pair(engine._garment(top, "top", empty_outfit()),
                                      engine._garment(bottom, "bottom", empty_outfit()), profile, pose)
    footwear = shoe_features(top, bottom, shoe, profile, engine)
    score = .85 * diagnostic["overall_score"] + .15 * float(footwear.mean())
    features = [diagnostic["matrix"][category][key] for category in ("top", "bottom")
                for key in ("body_fit", "situation_fit", "style_fit")]
    return score, np.asarray(features + [diagnostic["harmony_score"], *footwear], dtype=np.float32)


def build_library(catalog: dict, scenarios: list[dict], output: Path, engine) -> dict:
    records = catalog["products"]
    ids = [r["product"]["product_id"] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate product IDs")
    groups = {c: [r for r in records if r["product"]["category"] == c] for c in CATEGORIES}
    if not scenarios or len({s["id"] for s in scenarios}) != len(scenarios):
        raise ValueError("Scenario IDs must be nonempty and unique")
    if any(not groups[c] for c in CATEGORIES):
        raise ValueError("All three categories require products")
    products = {c: [Product(**r["product"]) for r in groups[c]] for c in CATEGORIES}
    nt, nb, ns = (len(products[c]) for c in CATEGORIES)
    scores = np.empty((len(scenarios), nt, nb, ns), dtype=np.float32)
    # Store clothing and shoe factors once; a triple's vector is their concatenation.
    clothing = np.empty((len(scenarios), nt, nb, 7), dtype=np.float32)
    shoe_style = np.empty((len(scenarios), ns), dtype=np.float32)
    shoe_formality = np.empty((nb, ns), dtype=np.float32)
    top_shoe_color = np.empty((nt, ns), dtype=np.float32)
    bottom_shoe_color = np.empty((nb, ns), dtype=np.float32)
    base_profile = UserProfile(**scenarios[0]["profile"])
    for ti, top in enumerate(products["top"]):
        for hi, shoe in enumerate(products["shoes"]):
            top_shoe_color[ti, hi] = shoe_features(top, top, shoe, base_profile, engine)[2]
    for bi, bottom in enumerate(products["bottom"]):
        for hi, shoe in enumerate(products["shoes"]):
            features = shoe_features(bottom, bottom, shoe, base_profile, engine)
            shoe_formality[bi, hi], bottom_shoe_color[bi, hi] = features[1:]
    for si, scenario in enumerate(scenarios):
        profile, pose = UserProfile(**scenario["profile"]), PoseAnalysis(**scenario["pose"])
        for hi, shoe in enumerate(products["shoes"]):
            shoe_style[si, hi] = shoe_features(products["top"][0], products["bottom"][0], shoe, profile, engine)[0]
        base_scores = np.empty((nt, nb), dtype=np.float32)
        for ti, top in enumerate(products["top"]):
            for bi, bottom in enumerate(products["bottom"]):
                diagnostic = engine._diagnose_pair(engine._garment(top, "top", empty_outfit()),
                                                  engine._garment(bottom, "bottom", empty_outfit()), profile, pose)
                clothing[si, ti, bi] = [diagnostic["matrix"][c][k] for c in ("top", "bottom")
                                       for k in ("body_fit", "situation_fit", "style_fit")] + [diagnostic["harmony_score"]]
                base_scores[ti, bi] = diagnostic["overall_score"]
        scores[si] = (.85 * base_scores[:, :, None] + .05 * (
            shoe_style[si][None, None, :] + shoe_formality[None, :, :]
            + (top_shoe_color[:, None, :] + bottom_shoe_color[None, :, :]) / 2))
        print(f"indexed {scenario['id']}: {nt * nb * ns:,} combinations", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    identity = digest({"catalog": catalog, "scenarios": scenarios, "scorer": scoring_fingerprint()})
    scores_path = output / f"scores-{identity[:16]}.npy"
    temporary = scores_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, scores, allow_pickle=False)
    os.replace(temporary, scores_path)
    arrays_path = output / f"scores-{identity[:16]}.npz"
    temporary = arrays_path.with_suffix(".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, clothing=clothing, shoe_style=shoe_style,
                            shoe_formality=shoe_formality, top_shoe_color=top_shoe_color,
                            bottom_shoe_color=bottom_shoe_color)
    os.replace(temporary, arrays_path)
    metadata = {"version": VERSION, "identity": identity, "scorer": scoring_fingerprint(),
                "catalog": catalog, "groups": groups, "scenarios": scenarios,
                "features": FEATURE_NAMES, "arrays": arrays_path.name, "scores": scores_path.name,
                "combination_count": nt * nb * ns, "score_count": int(scores.size),
                "score_source": "rules_85pct_clothing_15pct_shoe_metadata_uncalibrated"}
    atomic_json(output / "index.json", metadata)
    return metadata


class OutfitLibrary:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.metadata = json.loads((self.path / "index.json").read_text(encoding="utf-8"))
        if self.metadata["version"] != VERSION or self.metadata["scorer"] != scoring_fingerprint():
            raise ValueError("Scoring code changed: rebuild the offline index")
        with np.load(self.path / self.metadata["arrays"], allow_pickle=False) as arrays:
            self.clothing = arrays["clothing"]
            self.shoe_style = arrays["shoe_style"]
            self.shoe_formality = arrays["shoe_formality"]
            self.top_shoe_color = arrays["top_shoe_color"]
            self.bottom_shoe_color = arrays["bottom_shoe_color"]
        self.groups = self.metadata["groups"]
        self.scores = np.load(self.path / self.metadata["scores"], mmap_mode="r", allow_pickle=False)
        self._render_state, self._verified_renders = None, []

    def _load_verified_renders(self):
        paths = sorted((self.path / "renders").rglob("*.json"))
        state = []
        for path in paths:
            stat = path.stat()
            output = path.with_suffix(".png")
            image_stat = output.stat() if output.exists() else None
            state.append((str(path), stat.st_mtime_ns, stat.st_size,
                          (image_stat.st_mtime_ns, image_stat.st_size) if image_stat else None))
        if state == self._render_state:
            return self._verified_renders
        accepted = []
        for path in paths:
            try:
                record = json.loads(path.read_text())
                if record.get("library_identity") != self.metadata["identity"] or not render_is_eligible(record):
                    continue
                output = path.with_suffix(".png")
                if output.exists() and hashlib.sha256(output.read_bytes()).hexdigest() == record["output_sha256"]:
                    accepted.append(record)
            except (OSError, ValueError, TypeError):
                continue  # A partial/corrupt record is never a verified candidate.
        self._render_state, self._verified_renders = state, accepted
        return accepted

    def query(self, scenario_id: str, kept: dict, *, limit=3, gender="", max_budget=None,
              available_ids=None, verified_only=False):
        if set(kept) - set(CATEGORIES) or limit < 0:
            raise ValueError("Invalid kept categories or limit")
        scenario_ids = [s["id"] for s in self.metadata["scenarios"]]
        if scenario_id not in scenario_ids:
            raise ValueError("Scenario was not precomputed")
        si = scenario_ids.index(scenario_id)
        selected = []
        for category in CATEGORIES:
            positions = []
            for i, record in enumerate(self.groups[category]):
                item = record["product"]
                attributes = {**item, **record.get("attributes", {})}
                if category not in kept and (not item["stock"] or record.get("reference_anchor")):
                    continue
                product_gender = item.get("gender", "")
                if gender and product_gender not in ("", "공용", "남녀공용") and gender not in product_gender:
                    continue
                if category not in kept and available_ids is not None and item["product_id"] not in available_ids:
                    continue
                if all(attributes.get(key) == value for key, value in kept.get(category, {}).items()):
                    positions.append(i)
            selected.append(positions)
        if not limit or any(not p for p in selected):
            return []
        selected_array = self.scores[si][np.ix_(*selected)].copy()
        render_records = {}
        for record in self._load_verified_renders():
            if (record.get("library_identity") == self.metadata["identity"]
                    and render_is_eligible(record)
                    and scenario_id in record.get("scores", {})
                    and (not gender or record.get("reference_gender", gender) == gender)):
                prior = render_records.get(record["combination_id"])
                if prior is None or record["scores"][scenario_id] > prior["scores"][scenario_id]:
                    render_records[record["combination_id"]] = record
        if verified_only:
            selected_array.fill(-np.inf)
            id_to_local = [{self.groups[c][original]["product"]["product_id"]: local
                            for local, original in enumerate(positions)}
                           for c, positions in zip(CATEGORIES, selected)]
            for record in render_records.values():
                ids = record["product_ids"]
                if all(pid in mapping for pid, mapping in zip(ids, id_to_local)):
                    rendered_score = record.get("scores", {}).get(scenario_id)
                    if rendered_score is not None:
                        selected_array[tuple(mapping[pid] for pid, mapping in zip(ids, id_to_local))] = rendered_score
        order = np.argsort(-selected_array.ravel(), kind="stable")
        results, seen_replacements = [], set()
        for flat in order:
            local = np.unravel_index(flat, selected_array.shape)
            if not np.isfinite(selected_array[local]):
                continue
            ti, bi, hi = (selected[axis][i] for axis, i in enumerate(local))
            records = [self.groups[c][i] for c, i in zip(CATEGORIES, (ti, bi, hi))]
            products = [r["product"] for r in records]
            specific_genders = {p.get("gender") for p in products if p.get("gender") in ("남성", "여성")}
            if len(specific_genders) > 1:
                continue
            replacements = [p for p in products if p["category"] not in kept]
            replacement_ids = tuple(p["product_id"] for p in replacements)
            if not replacement_ids or replacement_ids in seen_replacements:
                continue
            price = sum(p["price"] for p in replacements)
            if max_budget is not None and price > max_budget:
                continue
            combination_id = digest([p["product_id"] for p in products])[:20]
            render = render_records.get(combination_id)
            valid_render = bool(render)
            if verified_only and not valid_render:
                continue
            seen_replacements.add(replacement_ids)
            vector = np.concatenate((self.clothing[si, ti, bi], [self.shoe_style[si, hi],
                self.shoe_formality[bi, hi], (self.top_shoe_color[ti, hi] + self.bottom_shoe_color[bi, hi]) / 2]))
            results.append({"combination_id": combination_id, "scenario_id": scenario_id,
                            "product_ids": [p["product_id"] for p in products],
                            "replacement_ids": list(replacement_ids), "products": products,
                            "kept_categories": list(kept), "replacement_price": price,
                            "score": round(float(selected_array[local]), 3),
                            "predicted_score": round(float(self.scores[si, ti, bi, hi]), 3),
                            "ranking_source": "rendered" if verified_only else "predicted",
                            "score_source": self.metadata["score_source"],
                            "score_vector_source": "predicted_rule_components",
                            "score_vector": vector.tolist(), "feature_names": list(FEATURE_NAMES),
                            "render": render if valid_render else None})
            if len(results) == limit:
                break
        return results
