"""Compare binary and semantic-group body-shape scoring on fixed artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

CONDITIONS = ("baseline_front", "arm_pose", "slight_angle")
MODES = ("binary", "semantic_group")
MULTIPLIERS = (0.0, 1.0, 1.5, 2.0, 3.0)
BASELINE_MULTIPLIER = 1.0


def _load_scored_rows(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    result = {condition: {} for condition in CONDITIONS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            condition = row["condition"]
            if condition not in result or float(row["multiplier"]) != BASELINE_MULTIPLIER:
                continue
            row["rank"] = int(row["rank"])
            row["base_score"] = float(row["base_score"])
            row["body_shape_score"] = float(row["body_shape_score"])
            row["matched_keywords"] = json.loads(row["matched_keywords"])
            row["body_shape_keywords"] = json.loads(row["body_shape_keywords"])
            result[condition][row["product_id"]] = row
    return result


def _load_candidates(path: Path) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for cache_path in sorted(path.glob("*.json")):
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        for products in payload.values():
            for product in products:
                candidates.setdefault(product["product_id"], product)
    return candidates


def _load_connections(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _active_keywords(connection: dict[str, Any], category: str) -> list[str]:
    rules = connection["keyword_rules"].get(category, {})
    return sorted({
        keyword
        for keyword, rule_ids in rules.items()
        if any(str(rule_id).startswith("R-BOD-") for rule_id in rule_ids)
    })


def _semantic_details(
    connection: dict[str, Any],
    category: str,
    matched_keywords: Iterable[str],
) -> tuple[list[str], list[str], float]:
    from body_shape_scoring import semantic_body_shape_score, semantic_group_for_keyword

    active_keywords = _active_keywords(connection, category)
    active_groups = sorted({
        group
        for keyword in active_keywords
        if (group := semantic_group_for_keyword(keyword)) is not None
    })
    matched_groups = sorted({
        group
        for keyword in matched_keywords
        if (group := semantic_group_for_keyword(keyword)) is not None
        and group in active_groups
    })
    score, _ = semantic_body_shape_score(active_keywords, matched_keywords)
    return active_groups, matched_groups, score


def _rank(
    rows: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    mode: str,
    multiplier: float,
    connection: dict[str, Any],
) -> list[dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        candidate = candidates[row["product_id"]]
        active_groups, matched_groups, semantic_score = _semantic_details(
            connection, row["category"], row["body_shape_keywords"]
        )
        body_score = row["body_shape_score"] if mode == "binary" else semantic_score
        enriched = dict(row)
        enriched["active_semantic_groups"] = active_groups
        enriched["matched_semantic_groups"] = matched_groups
        enriched["body_shape_score"] = body_score
        enriched["final_score"] = row["base_score"] + body_score * (multiplier - 1.0)
        enriched["review_count"] = int(candidate.get("review_count") or 0)
        by_category.setdefault(row["category"], []).append(enriched)

    ranked_categories = {
        category: sorted(
            category_rows,
            key=lambda item: (-item["final_score"], -item["review_count"], item["product_id"]),
        )
        for category, category_rows in by_category.items()
    }
    categories = list(ranked_categories)
    result: list[dict[str, Any]] = []
    depth = 0
    while len(result) < len(rows) and any(
        depth < len(ranked_categories[category]) for category in categories
    ):
        for category in categories:
            category_rows = ranked_categories[category]
            if depth < len(category_rows):
                result.append(category_rows[depth])
        depth += 1
    return result


def _overlap(first: Iterable[str], second: Iterable[str], k: int = 3) -> float:
    first_ids = set(list(first)[:k])
    second_ids = set(list(second)[:k])
    return len(first_ids & second_ids) / k


def _first_crossing(
    rows: dict[str, dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    mode: str,
    top_k: int | None,
    connection: dict[str, Any],
) -> float | None:
    products = list(rows.values())
    points: set[float] = set()
    for index, first in enumerate(products):
        first_score = first["body_shape_score"] if mode == "binary" else _semantic_details(
            connection, first["category"], first["body_shape_keywords"]
        )[2]
        for second in products[index + 1:]:
            second_score = second["body_shape_score"] if mode == "binary" else _semantic_details(
                connection, second["category"], second["body_shape_keywords"]
            )[2]
            delta = first_score - second_score
            if math.isclose(delta, 0.0):
                continue
            crossing = 1.0 + (second["base_score"] - first["base_score"]) / delta
            if math.isfinite(crossing) and crossing >= 0.0:
                points.add(crossing)
    baseline = [row["product_id"] for row in _rank(rows, candidates, mode, 1.0, connection)]
    for point in sorted(points):
        epsilon = max(1e-7, abs(point) * 1e-7)
        before = [row["product_id"] for row in _rank(rows, candidates, mode, max(0.0, point - epsilon), connection)]
        after = [row["product_id"] for row in _rank(rows, candidates, mode, point + epsilon, connection)]
        if top_k is None:
            changed = before != after
        else:
            changed = before[:top_k] != after[:top_k]
        if changed and (top_k is None or baseline[:top_k] != after[:top_k]):
            return point
    return None


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = (
        "condition", "body_shape", "mode", "multiplier", "product_id", "product_name",
        "category", "rank", "base_score", "body_shape_score", "final_score",
        "matched_keywords", "body_shape_keywords", "active_semantic_groups",
        "matched_semantic_groups", "rank_change_vs_baseline", "top10_changed",
        "top3_changed", "baseline_top3_overlap",
    )
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field], ensure_ascii=False)
                if isinstance(row.get(field), (list, dict)) else row.get(field, "")
                for field in fields
            })


def evaluate(input_dir: Path, output_dir: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    scores = _load_scored_rows(input_dir / "scores.csv")
    candidates = _load_candidates(input_dir / "candidate_cache")
    connections = _load_connections(input_dir / "body_shape_connections.json")
    if len(candidates) != 75:
        raise ValueError(f"Expected fixed candidate pool of 75, got {len(candidates)}")

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    rankings_by_key: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    report: list[str] = [
        "# Binary vs semantic-group body-shape comparison",
        "",
        "이 비교는 기존 `candidate_cache`, `scores.csv`, `body_shape_connections.json`만 읽었으며 live search와 이미지 분석을 실행하지 않았다.",
        "",
        f"- fixed candidate count: {len(candidates)}",
        f"- modes: {', '.join(MODES)}",
        f"- multipliers: {', '.join(str(value) for value in MULTIPLIERS)}",
        "",
    ]
    for condition in CONDITIONS:
        condition_rows = scores[condition]
        if len(condition_rows) != len(candidates):
            raise ValueError(f"{condition}: expected 75 scored products, got {len(condition_rows)}")
        connection = connections[condition]
        for mode in MODES:
            ranked_by_multiplier = {
                multiplier: _rank(condition_rows, candidates, mode, multiplier, connection)
                for multiplier in MULTIPLIERS
            }
            for multiplier, ranked in ranked_by_multiplier.items():
                rankings_by_key[(condition, mode, multiplier)] = ranked
            expected_ids = [
                row["product_id"]
                for row in sorted(condition_rows.values(), key=lambda item: item["rank"])
            ]
            baseline_ids = [row["product_id"] for row in ranked_by_multiplier[1.0]]
            if baseline_ids != expected_ids:
                raise AssertionError(f"{condition}/{mode}: multiplier=1.0 rank differs from scores.csv")
            if any(
                not math.isclose(row["final_score"], row["base_score"])
                for row in ranked_by_multiplier[1.0]
            ):
                raise AssertionError(f"{condition}/{mode}: multiplier=1.0 score differs from base_score")
            baseline = ranked_by_multiplier[1.0]
            baseline_ids = [row["product_id"] for row in baseline]
            baseline_positions = {product_id: index + 1 for index, product_id in enumerate(baseline_ids)}
            for multiplier, ranked in ranked_by_multiplier.items():
                current_positions = {row["product_id"]: index + 1 for index, row in enumerate(ranked)}
                top3_ids = [row["product_id"] for row in ranked[:3]]
                baseline_top3 = baseline_ids[:3]
                for rank, row in enumerate(ranked, 1):
                    rows.append({
                        "condition": condition,
                        "body_shape": connection["body_shape"],
                        "mode": mode,
                        "multiplier": multiplier,
                        "product_id": row["product_id"],
                        "product_name": row["product_name"],
                        "category": row["category"],
                        "rank": rank,
                        "base_score": row["base_score"],
                        "body_shape_score": row["body_shape_score"],
                        "final_score": row["final_score"],
                        "matched_keywords": row["matched_keywords"],
                        "body_shape_keywords": row["body_shape_keywords"],
                        "active_semantic_groups": row["active_semantic_groups"],
                        "matched_semantic_groups": row["matched_semantic_groups"],
                        "rank_change_vs_baseline": rank - baseline_positions[row["product_id"]],
                        "top10_changed": current_positions[row["product_id"]] != baseline_positions[row["product_id"]]
                        and (rank <= 10 or baseline_positions[row["product_id"]] <= 10),
                        "top3_changed": current_positions[row["product_id"]] != baseline_positions[row["product_id"]]
                        and (rank <= 3 or baseline_positions[row["product_id"]] <= 3),
                        "baseline_top3_overlap": _overlap(baseline_top3, top3_ids),
                    })
            active_groups = sorted({
                group
                for category in connection["keyword_rules"]
                for group in _semantic_details(connection, category, [])[0]
            })
            semantic_scores = [
                _semantic_details(connection, row["category"], row["body_shape_keywords"])[2]
                for row in condition_rows.values()
            ]
            matched_group_counts = Counter(
                group
                for row in condition_rows.values()
                for group in _semantic_details(connection, row["category"], row["body_shape_keywords"])[1]
            )
            group_rates: dict[str, dict[str, float | int]] = {}
            for group in active_groups:
                active_product_count = 0
                matched_product_count = 0
                for row in condition_rows.values():
                    active_for_product = group in _semantic_details(
                        connection, row["category"], row["body_shape_keywords"]
                    )[0]
                    matched_for_product = group in _semantic_details(
                        connection, row["category"], row["body_shape_keywords"]
                    )[1]
                    active_product_count += int(active_for_product)
                    matched_product_count += int(matched_for_product)
                group_rates[group] = {
                    "matched_products": matched_product_count,
                    "active_products": active_product_count,
                    "match_rate": matched_product_count / active_product_count
                    if active_product_count else 0.0,
                }
            summary = {
                "condition": condition,
                "body_shape": connection["body_shape"],
                "mode": mode,
                "active_semantic_groups": active_groups,
                "matched_semantic_groups": sorted(matched_group_counts),
                "body_shape_score_distribution": dict(sorted(Counter(semantic_scores if mode == "semantic_group" else [row["body_shape_score"] for row in condition_rows.values()]).items())),
                "unique_score_count": len(set(semantic_scores if mode == "semantic_group" else [row["body_shape_score"] for row in condition_rows.values()])),
                "semantic_group_match_rates": group_rates,
                "multipliers": {},
                "first_full_rank_crossing_multiplier": _first_crossing(condition_rows, candidates, mode, None, connection),
                "first_top10_crossing_multiplier": _first_crossing(condition_rows, candidates, mode, 10, connection),
                "first_top3_crossing_multiplier": _first_crossing(condition_rows, candidates, mode, 3, connection),
            }
            for multiplier, ranked in ranked_by_multiplier.items():
                baseline_ids = [row["product_id"] for row in baseline]
                current_ids = [row["product_id"] for row in ranked]
                summary["multipliers"][str(multiplier)] = {
                    "rank_changed_product_count": sum(left != right for left, right in zip(baseline_ids, current_ids)),
                    "top10_changed_product_count": len(set(current_ids[:10]) ^ set(baseline_ids[:10])),
                    "top3_changed_product_count": len(set(current_ids[:3]) ^ set(baseline_ids[:3])),
                    "top3": current_ids[:3],
                    "baseline_condition_top3_overlap": _overlap(
                        [row["product_id"] for row in rankings_by_key[("baseline_front", mode, multiplier)][:3]],
                        current_ids[:3],
                    ),
                }
            summaries.append(summary)

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "semantic_group_scores.csv", rows)
    summary_fields = (
        "condition", "body_shape", "mode", "active_semantic_groups", "matched_semantic_groups",
        "semantic_group_match_rates", "body_shape_score_distribution", "unique_score_count",
        "multipliers", "first_full_rank_crossing_multiplier", "first_top10_crossing_multiplier",
        "first_top3_crossing_multiplier",
    )
    with (output_dir / "semantic_group_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({field: json.dumps(summary.get(field, ""), ensure_ascii=False)
                             for field in summary_fields})
    for summary in summaries:
        report.extend([
            f"## {summary['condition']} / {summary['mode']}", "",
            f"- body shape: {summary['body_shape']}",
            f"- active semantic groups: {', '.join(summary['active_semantic_groups']) or '없음'}",
            f"- matched semantic groups: {', '.join(summary['matched_semantic_groups']) or '없음'}",
            f"- semantic group product match rates: {summary['semantic_group_match_rates']}",
            f"- body_shape_score distribution: {summary['body_shape_score_distribution']}",
            f"- unique score count: {summary['unique_score_count']}",
            f"- first full/Top-10/Top-3 crossing multiplier: {summary['first_full_rank_crossing_multiplier']} / {summary['first_top10_crossing_multiplier']} / {summary['first_top3_crossing_multiplier']}",
            "",
            "| multiplier | changed ranks | Top-10 changes | Top-3 changes | baseline Top-3 overlap | Top-3 |",
            "|---:|---:|---:|---:|---:|---|",
        ])
        for multiplier in MULTIPLIERS:
            item = summary["multipliers"][str(multiplier)]
            report.append(
                f"| {multiplier} | {item['rank_changed_product_count']} | {item['top10_changed_product_count']} | "
                f"{item['top3_changed_product_count']} | {item['baseline_condition_top3_overlap']:.3f} | "
                f"{', '.join(item['top3'])} |"
            )
        report.append("")
    report.extend([
        "## Validation",
        "",
        "- multiplier=1.0은 binary와 semantic_group 모두 base score를 사용하므로 기존 점수와 순위가 동일해야 한다.",
        "- baseline Top-3 overlap은 같은 mode와 multiplier의 baseline_front Top-3와 비교했다.",
        "- semantic group별 실제 상품 매칭률은 active semantic group이 적용되는 상품 중 matched 상품의 비율이다.",
    ])
    (output_dir / "semantic_group_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "outputs" / "recommendation_body_shape_weight"
    parser.add_argument("--input-dir", type=Path, default=default_dir)
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    args = parser.parse_args()
    evaluate(args.input_dir, args.output_dir)
    print(f"saved semantic-group comparison under {args.output_dir}")


if __name__ == "__main__":
    main()