"""Evaluate body-shape-only score multipliers on one fixed candidate pool."""

from __future__ import annotations

import argparse
import csv
import json
import time
import sys
import tempfile
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterable

CONDITIONS = ("baseline_front", "arm_pose", "slight_angle")
MULTIPLIERS = (0.0, 1.0, 1.5, 2.0, 3.0)
EXPECTED_BODY_SHAPES = {
    "baseline_front": ("모래시계체형", "R-BOD-03"),
    "arm_pose": ("둥근체형", "R-BOD-07"),
    "slight_angle": ("둥근체형", "R-BOD-07"),
}
PROGRESS_LOG_PATH: Path | None = None


def _progress(message: str) -> None:
    line = message if message.startswith("[") else f"[{time.strftime('%H:%M:%S')}] {message}"
    print(line, flush=True)
    if PROGRESS_LOG_PATH is not None:
        PROGRESS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with PROGRESS_LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

RECORD_FIELDS = (
    "condition", "body_shape", "body_shape_rule_ids", "multiplier", "product_id",
    "product_name", "category", "rank", "base_score", "body_shape_score",
    "final_score", "matched_keywords", "body_shape_keywords", "rank_change_vs_1.0",
    "baseline_top3_overlap",
)


def overlap_at_k(first: Iterable[str], second: Iterable[str], k: int = 3) -> float:
    first_ids = list(dict.fromkeys(first))[:k]
    second_ids = set(list(dict.fromkeys(second))[:k])
    return len(set(first_ids) & second_ids) / k


def flatten_rule_ids(keyword_rules: dict[str, Any]) -> list[str]:
    return sorted({
        rule_id
        for keywords in (keyword_rules or {}).values()
        for rule_ids in keywords.values()
        for rule_id in rule_ids
        if rule_id.startswith("R-BOD-")
    })


def validate_body_shape_connections(records: dict[str, dict[str, Any]]) -> None:
    errors = []
    for condition, (expected_shape, expected_rule) in EXPECTED_BODY_SHAPES.items():
        record = records[condition]
        if record["body_shape"] != expected_shape:
            errors.append(
                f"{condition}: expected body_shape={expected_shape}, "
                f"got {record['body_shape']}"
            )
        if expected_rule not in record["body_shape_rule_ids"]:
            errors.append(
                f"{condition}: expected {expected_rule} in keyword_rules, "
                f"got {record['body_shape_rule_ids']}"
            )
    if errors:
        raise RuntimeError("Body-shape rule validation failed: " + "; ".join(errors))


def union_candidate_pools(pools: Iterable[dict[str, list[Any]]]) -> dict[str, list[Any]]:
    union: dict[str, dict[str, Any]] = {}
    for pool in pools:
        for category, products in pool.items():
            by_id = union.setdefault(category, {})
            for product in products:
                by_id.setdefault(product.product_id, product)
    return {
        category: list(products.values())
        for category, products in union.items()
    }


def interleave(products_by_category: dict[str, list[Any]], limit: int = 3) -> list[Any]:
    selected = []
    categories = list(products_by_category)
    depth = 0
    while len(selected) < limit and any(
        depth < len(products_by_category.get(category, [])) for category in categories
    ):
        for category in categories:
            products = products_by_category.get(category, [])
            if depth < len(products):
                selected.append(products[depth])
                if len(selected) >= limit:
                    break
        depth += 1
    return selected


def rank_fixed_candidates(
    candidate_pool: dict[str, list[Any]],
    targets: Any,
    searcher: Any,
    multiplier: float,
    limit: int = 3,
) -> list[Any]:
    ranked: dict[str, list[Any]] = {}
    for category, products in candidate_pool.items():
        attributes = targets.targets.get(category, {})
        rescored = []
        for rank, product in enumerate(products):
            item = {
                "goodsName": product.name,
                "brandName": product.brand,
                "finalPrice": product.price,
                "displayGenderText": product.gender,
                "reviewCount": product.review_count,
                "reviewScore": product.review_score,
            }
            base, body, final, matched, body_keywords = searcher.score_with_body_shape(
                item, attributes, rank, category, targets, multiplier
            )
            rescored.append(replace(
                product,
                base_score=round(base, 4),
                body_shape_score=round(body, 4),
                final_score=round(final, 4),
                retrieval_score=round(final, 4),
                matched_keywords=list(dict.fromkeys(matched)),
                body_shape_keywords=list(dict.fromkeys(body_keywords)),
                search_keywords=searcher._representative_keywords(matched, attributes),
            ))
        ranked[category] = sorted(
            rescored,
            key=lambda product: (-product.final_score, -product.review_count, product.product_id),
        )
    return interleave(ranked, limit)


def _run_condition(
    image_path: Path,
    work_dir: Path,
) -> tuple[dict[str, Any], dict[str, list[Any]]]:
    started = time.monotonic()
    _progress(f"[{time.strftime('%H:%M:%S')}] {image_path.stem}: pipeline analysis started")
    repo = Path(__file__).resolve().parents[2]
    web_dir = repo / "web"
    project_dir = repo / "ai_fashion_recommender"
    for path in (web_dir, project_dir, project_dir / "src"):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    from pipeline import get_engine, run_pipeline
    from schemas import UserProfile

    _progress("pipeline initialization started")
    engine = get_engine()
    _progress(f"pipeline initialization completed ({time.monotonic() - started:.1f}s)")
    searcher = engine.product_search
    original_search = engine.product_search
    original_keywords = engine.recommender.generate_target_keywords
    original_live_category = searcher._live_category
    candidate_pool: dict[str, list[Any]] = {}
    captured_keywords: Any = None

    def capture_keywords(*args: Any, **kwargs: Any) -> Any:
        nonlocal captured_keywords
        captured_keywords = original_keywords(*args, **kwargs)
        return captured_keywords

    def capture_candidates(
        category: str, attributes: dict[str, list[str]], profile: Any
    ) -> list[Any]:
        products = original_live_category(category, attributes, profile)
        candidate_pool[category] = products
        return products

    class CaptureSearch:
        def search(self, *args: Any, **kwargs: Any) -> list[Any]:
            return original_search.search(*args, **kwargs)

    engine.product_search = CaptureSearch()
    engine.recommender.generate_target_keywords = capture_keywords
    searcher._live_category = capture_candidates
    try:
        result = run_pipeline(
            image_path,
            UserProfile(),
            work_dir,
            lambda stage: _progress(f"[{time.strftime('%H:%M:%S')}] {image_path.stem}: stage={stage}"),
        )
    except Exception:
        _progress(f"[{time.strftime('%H:%M:%S')}] {image_path.stem}: pipeline failed")
        _progress(traceback.format_exc())
        raise
    finally:
        engine.product_search = original_search
        engine.recommender.generate_target_keywords = original_keywords
        searcher._live_category = original_live_category

    if captured_keywords is None:
        raise RuntimeError(f"Keyword generation was not captured for {image_path}")
    if not any(candidate_pool.values()):
        candidate_pool = searcher._local_fallback(
            captured_keywords, UserProfile(), engine.recommender.catalog.products
        )
    pose = result.payload["pose"]
    _progress(f"[{time.strftime('%H:%M:%S')}] {image_path.stem}: pipeline analysis completed ({time.monotonic() - started:.1f}s)")
    return {
        "body_shape": pose.get("body_shape", ""),
        "body_shape_rule_ids": flatten_rule_ids(captured_keywords.keyword_rules),
        "keyword_result": captured_keywords,
    }, candidate_pool


def _product_to_dict(product: Any) -> dict[str, Any]:
    return {
        "product_id": product.product_id,
        "name": product.name,
        "brand": product.brand,
        "price": product.price,
        "image_url": product.image_url,
        "url": product.url,
        "category": product.category,
        "gender": product.gender,
        "review_count": product.review_count,
        "review_score": product.review_score,
        "source": product.source,
        "search_keywords": product.search_keywords,
    }


def _product_from_dict(data: dict[str, Any]) -> Any:
    from musinsa_live_search import ShoppingProduct

    return ShoppingProduct(**data)


def _save_candidate_cache(path: Path, candidate_pool: dict[str, list[Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        category: [_product_to_dict(product) for product in products]
        for category, products in candidate_pool.items()
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _load_candidate_cache(path: Path) -> dict[str, list[Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {category: [_product_from_dict(product) for product in products]
            for category, products in payload.items()}


def _reuse_analysis(
    records_path: Path,
    keyword_generator: Any,
) -> dict[str, dict[str, Any]]:
    from schemas import OutfitAnalysis, PoseAnalysis, UserProfile

    records = {}
    with records_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in CONDITIONS:
                continue
            pose = PoseAnalysis(
                valid=True,
                full_body_score=1.0,
                body_shape=row["body_shape"],
                shoulder_hip_ratio=float(row["shoulder_hip_ratio"]),
                upper_lower_ratio=float(row["upper_lower_ratio"]),
                leg_ratio=float(row["leg_ratio"]),
                posture="재사용 분석",
                body_shape_confidence=float(row["body_shape_confidence"]),
            )
            outfit = OutfitAnalysis(
                parser_backend="reused_robustness_analysis",
                upper_color="",
                lower_color="",
                color_harmony="",
                detected_items=[],
                style="캐주얼",
            )
            targets = keyword_generator.generate(UserProfile(), pose, outfit)
            records[row["condition"]] = {
                "body_shape": pose.body_shape,
                "body_shape_rule_ids": flatten_rule_ids(targets.keyword_rules),
                "keyword_result": targets,
            }
            _progress(
                f"[{time.strftime('%H:%M:%S')}] {row['condition']}: reused analysis and generated keywords "
                f"(R-BOD={records[row['condition']]['body_shape_rule_ids']})"
            )
    missing = set(CONDITIONS) - set(records)
    if missing:
        raise RuntimeError(f"reuse analysis missing conditions: {sorted(missing)}")
    return records


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=RECORD_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], ensure_ascii=False)
                if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in RECORD_FIELDS
            })


def _collect_candidates(
    condition: str,
    targets: Any,
    searcher: Any,
    profile: Any,
    cache_path: Path,
) -> dict[str, list[Any]]:
    if cache_path.is_file():
        pool = _load_candidate_cache(cache_path)
        _progress(f"[{time.strftime('%H:%M:%S')}] {condition}: loaded candidate cache ({sum(map(len, pool.values()))})")
        return pool
    _progress(f"[{time.strftime('%H:%M:%S')}] {condition}: live candidate collection started")
    pool: dict[str, list[Any]] = {}
    try:
        for category, attributes in targets.targets.items():
            if category not in {"top", "bottom", "shoes"}:
                continue
            pool[category] = searcher._live_category(category, attributes, profile)
            _progress(
                f"[{time.strftime('%H:%M:%S')}] {condition}: {category} candidates={len(pool[category])}",
            )
    except Exception:
        _save_candidate_cache(cache_path, pool)
        _progress(f"[{time.strftime('%H:%M:%S')}] {condition}: candidate collection failed; partial cache={cache_path}")
        _progress(traceback.format_exc())
        raise
    _save_candidate_cache(cache_path, pool)
    _progress(f"[{time.strftime('%H:%M:%S')}] {condition}: live candidate collection completed ({sum(map(len, pool.values()))})")
    return pool


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# Body-shape weight experiment",
        "",
        f"- Fixed union candidate pool: {summary['fixed_candidate_count']}",
        "- Candidate collection: once per condition, then union reused for every multiplier",
        "",
        "| multiplier | baseline Top-3 (rank delta) | arm_pose Top-3 (rank delta) | slight_angle Top-3 (rank delta) | arm overlap | slight overlap | changed R-BOD products | pool |",
        "|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for multiplier in summary["multipliers"]:
        item = summary["comparison"][str(multiplier)]
        def top3_with_changes(condition: str) -> str:
            condition_item = item[condition]
            return " / ".join(
                f"{product_id} ({condition_item['rank_changes_vs_1.0'].get(product_id, 0):+d})"
                for product_id in condition_item["top3"]
            )
        lines.append(
            f"| {multiplier} | {top3_with_changes('baseline_front')} | "
            f"{top3_with_changes('arm_pose')} | {top3_with_changes('slight_angle')} | "
            f"{item['arm_pose']['baseline_overlap']:.3f} | {item['slight_angle']['baseline_overlap']:.3f} | "
            f"{item['changed_body_shape_product_count']} | {summary['fixed_candidate_count']} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def evaluate(
    data_dir: Path,
    output_dir: Path,
    *,
    reuse_analysis: bool = False,
    analysis_records: Path | None = None,
) -> None:
    global PROGRESS_LOG_PATH
    output_dir.mkdir(parents=True, exist_ok=True)
    PROGRESS_LOG_PATH = output_dir / "execution.log"
    PROGRESS_LOG_PATH.write_text("", encoding="utf-8")
    project_dir = Path(__file__).resolve().parents[1]
    src_dir = project_dir / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from musinsa_live_search import MusinsaLiveSearch
    from recommendation_keywords import RecommendationKeywordGenerator
    from fashion_rules import FashionRuleBook
    from schemas import UserProfile

    condition_records: dict[str, dict[str, Any]] = {}
    pools = []
    project_dir = Path(__file__).resolve().parents[1]
    records_path = analysis_records or (
        project_dir.parent.parent / "26-2_Modeling_Fashion-main" / "ai_fashion_recommender"
        / "outputs" / "recommendation_robustness" / "records.csv"
    )
    if reuse_analysis:
        rules = FashionRuleBook.from_markdown(project_dir / "FASHION_RULES_MASTER.md")
        keyword_generator = RecommendationKeywordGenerator(set(rules.rules))
        condition_records = _reuse_analysis(records_path, keyword_generator)
    else:
        for condition in CONDITIONS:
            image_path = data_dir / "person01" / f"{condition}.jpg"
            if not image_path.is_file():
                raise FileNotFoundError(f"Missing robustness image: {image_path}")
            with tempfile.TemporaryDirectory(prefix="body_shape_weight_") as temporary:
                record, pool = _run_condition(image_path, Path(temporary))
            condition_records[condition] = record
            pools.append(pool)
    validate_body_shape_connections(condition_records)
    searcher = MusinsaLiveSearch()
    profile = UserProfile()
    cache_dir = output_dir / "candidate_cache"
    for condition in CONDITIONS:
        pool = _collect_candidates(
            condition,
            condition_records[condition]["keyword_result"],
            searcher,
            profile,
            cache_dir / f"{condition}.json",
        )
        pools.append(pool)
    fixed_pool = union_candidate_pools(pools)
    _progress(f"[{time.strftime('%H:%M:%S')}] union candidate pool size={sum(map(len, fixed_pool.values()))}")
    rows = []
    top3_by_condition: dict[tuple[str, float], list[str]] = {}
    ranked_by_condition: dict[tuple[str, float], dict[str, int]] = {}
    ranked_products_by_condition: dict[tuple[str, float], list[Any]] = {}
    for condition in CONDITIONS:
        record = condition_records[condition]
        targets = record["keyword_result"]
        for multiplier in MULTIPLIERS:
            ranked = rank_fixed_candidates(fixed_pool, targets, searcher, multiplier, limit=sum(map(len, fixed_pool.values())))
            ranked_products_by_condition[(condition, multiplier)] = ranked
            top3 = [product.product_id for product in ranked[:3]]
            top3_by_condition[(condition, multiplier)] = top3
            ranked_by_condition[(condition, multiplier)] = {
                product.product_id: rank for rank, product in enumerate(ranked, 1)
            }
        baseline_ranks = ranked_by_condition[(condition, 1.0)]
        baseline_top3 = top3_by_condition[(condition, 1.0)]
        for multiplier in MULTIPLIERS:
            ranked = ranked_products_by_condition[(condition, multiplier)]
            top3 = top3_by_condition[(condition, multiplier)]
            for rank, product in enumerate(ranked, 1):
                rows.append({
                    "condition": condition,
                    "body_shape": record["body_shape"],
                    "body_shape_rule_ids": record["body_shape_rule_ids"],
                    "multiplier": multiplier,
                    "product_id": product.product_id,
                    "product_name": product.name,
                    "category": product.category,
                    "rank": rank,
                    "base_score": product.base_score,
                    "body_shape_score": product.body_shape_score,
                    "final_score": product.final_score,
                    "matched_keywords": product.matched_keywords,
                    "body_shape_keywords": product.body_shape_keywords,
                    "rank_change_vs_1.0": rank - baseline_ranks.get(product.product_id, rank),
                    "baseline_top3_overlap": overlap_at_k(baseline_top3, top3),
                })
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "scores.csv", rows)
    top3_rows = []
    comparison: dict[str, Any] = {}
    baseline_top3 = top3_by_condition[("baseline_front", 1.0)]
    for multiplier in MULTIPLIERS:
        item: dict[str, Any] = {}
        changed_products = 0
        for condition in CONDITIONS:
            top3 = top3_by_condition[(condition, multiplier)][:3]
            baseline_ranks = ranked_by_condition[(condition, 1.0)]
            current_ranks = ranked_by_condition[(condition, multiplier)]
            item[condition] = {
                "top3": top3,
                "rank_changes_vs_1.0": {
                    product_id: current_ranks.get(product_id, 0) - baseline_ranks.get(product_id, 0)
                    for product_id in top3
                },
                "baseline_overlap": overlap_at_k(baseline_top3, top3),
            }
            top3_rows.append({"multiplier": multiplier, "condition": condition, "top3": top3,
                              "rank_changes_vs_1.0": item[condition]["rank_changes_vs_1.0"],
                              "baseline_overlap": item[condition]["baseline_overlap"]})
        changed_products = len({
            product.product_id
            for condition in CONDITIONS
            for product in ranked_products_by_condition[(condition, multiplier)]
            if product.body_shape_score > 0 and multiplier != 1.0
        })
        item["changed_body_shape_product_count"] = changed_products
        comparison[str(multiplier)] = item
    (output_dir / "top3_summary.json").write_text(json.dumps(top3_rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output_dir / "top3_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("multiplier", "condition", "top3", "rank_changes_vs_1.0", "baseline_overlap"))
        writer.writeheader()
        for row in top3_rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value for key, value in row.items()})
    (output_dir / "fixed_candidate_pool.json").write_text(
        json.dumps({category: [product.product_id for product in products]
                    for category, products in fixed_pool.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "body_shape_connections.json").write_text(
        json.dumps({condition: {
            "body_shape": record["body_shape"],
            "body_shape_rule_ids": record["body_shape_rule_ids"],
            "keyword_rules": record["keyword_result"].keyword_rules,
        } for condition, record in condition_records.items()}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    summary = {
        "multipliers": MULTIPLIERS,
        "conditions": CONDITIONS,
        "fixed_candidate_count": sum(len(products) for products in fixed_pool.values()),
        "top3": {f"{condition}:{multiplier}": top3_by_condition[(condition, multiplier)]
                 for condition in CONDITIONS for multiplier in MULTIPLIERS},
        "baseline_top3_overlap": {
            f"{condition}:{multiplier}": rows_for_overlap(rows, condition, multiplier)
            for condition in CONDITIONS for multiplier in MULTIPLIERS
        },
        "comparison": comparison,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_summary_markdown(output_dir / "report.md", summary)
    _progress(f"result files saved under {output_dir}")


def rows_for_overlap(rows: list[dict[str, Any]], condition: str, multiplier: float) -> float:
    matches = [
        float(row["baseline_top3_overlap"])
        for row in rows
        if row["condition"] == condition and row["multiplier"] == multiplier
    ]
    return matches[0] if matches else 0.0


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    default_data = repo.parent / "26-2_Modeling_Fashion-main" / "ai_fashion_recommender" / "data" / "robustness"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=default_data)
    parser.add_argument("--output-dir", type=Path, default=repo / "outputs" / "recommendation_body_shape_weight")
    parser.add_argument("--reuse-analysis", action="store_true", help="Reuse prior robustness records.csv instead of running image analysis")
    parser.add_argument("--analysis-records", type=Path, help="Path to prior robustness records.csv")
    args = parser.parse_args()
    evaluate(args.data_dir, args.output_dir, reuse_analysis=args.reuse_analysis, analysis_records=args.analysis_records)


if __name__ == "__main__":
    main()
