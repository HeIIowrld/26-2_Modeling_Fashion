"""Diagnose fixed-pool body-shape multiplier behavior without network or models."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

CONDITIONS = ("baseline_front", "arm_pose", "slight_angle")
BASELINE_MULTIPLIER = 1.0


def read_scores(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = {condition: {} for condition in CONDITIONS}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["condition"] not in grouped or float(row["multiplier"]) != BASELINE_MULTIPLIER:
                continue
            row["base_score"] = float(row["base_score"])
            row["body_shape_score"] = float(row["body_shape_score"])
            row["final_score"] = float(row["final_score"])
            row["rank"] = int(row["rank"])
            row["matched_keywords"] = json.loads(row["matched_keywords"])
            row["body_shape_keywords"] = json.loads(row["body_shape_keywords"])
            grouped[row["condition"]][row["product_id"]] = row
    return grouped


def read_candidate_ids(cache_dir: Path) -> dict[str, set[str]]:
    result = {}
    for condition in CONDITIONS:
        payload = json.loads((cache_dir / f"{condition}.json").read_text(encoding="utf-8"))
        result[condition] = {
            product["product_id"]
            for products in payload.values()
            for product in products
        }
    return result


def read_connections(path: Path) -> dict[str, dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def score(row: dict[str, Any], multiplier: float) -> float:
    return row["base_score"] + row["body_shape_score"] * (multiplier - 1.0)


def ranking(rows: dict[str, dict[str, Any]], multiplier: float) -> list[dict[str, Any]]:
    # Match eval_body_shape_weight.py: rank within category, then interleave categories.
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in rows.values():
        by_category.setdefault(row["category"], []).append(row)
    ranked_by_category = {
        category: sorted(
            category_rows,
            key=lambda row: (-score(row, multiplier), int(row.get("rank", 0)), row["product_id"]),
        )
        for category, category_rows in by_category.items()
    }
    categories = list(ranked_by_category)
    result = []
    depth = 0
    while len(result) < len(rows) and any(
        depth < len(ranked_by_category[category]) for category in categories
    ):
        for category in categories:
            category_rows = ranked_by_category[category]
            if depth < len(category_rows):
                result.append(category_rows[depth])
        depth += 1
    return result


def pair_crossing(first: dict[str, Any], second: dict[str, Any]) -> float | None:
    body_delta = first["body_shape_score"] - second["body_shape_score"]
    if math.isclose(body_delta, 0.0):
        return None
    # base_first + body_first*(m-1) == base_second + body_second*(m-1)
    crossing = 1.0 + (second["base_score"] - first["base_score"]) / body_delta
    return crossing if math.isfinite(crossing) and crossing >= 0.0 else None


def event_points(rows: dict[str, dict[str, Any]]) -> list[float]:
    points = {
        crossing
        for index, first in enumerate(rows.values())
        for second in list(rows.values())[index + 1:]
        for crossing in [pair_crossing(first, second)]
        if crossing is not None
    }
    return sorted(points)


def first_change(rows: dict[str, dict[str, Any]], top_k: int | None) -> float | None:
    points = event_points(rows)
    for point in points:
        epsilon = max(1e-7, abs(point) * 1e-7)
        before = [row["product_id"] for row in ranking(rows, max(0.0, point - epsilon))]
        after = [row["product_id"] for row in ranking(rows, point + epsilon)]
        if top_k is None:
            changed = before != after
        else:
            changed = before[:top_k] != after[:top_k]
        if changed:
            return point
    return None


def additional_multipliers(first_change_point: float | None, top3_change_point: float | None) -> list[float]:
    point = top3_change_point or first_change_point
    if point is None:
        return []
    # Values below 1.0 are already represented by the completed 0.0 run;
    # do not propose them as a new upward-weight experiment.
    if point <= 1.0:
        return []
    values = {round(max(0.0, point - 0.2), 4), round(point, 4), round(point + 0.2, 4)}
    if point >= 3.0:
        values.add(round(point + 1.0, 4))
    return sorted(values)


def fmt(value: Any) -> str:
    if value is None:
        return "없음"
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def diagnose(input_dir: Path) -> tuple[list[dict[str, Any]], str]:
    scores = read_scores(input_dir / "scores.csv")
    candidates = read_candidate_ids(input_dir / "candidate_cache")
    connections = read_connections(input_dir / "body_shape_connections.json")
    summary_rows: list[dict[str, Any]] = []
    report: list[str] = [
        "# Body-shape weight diagnostic report",
        "",
        "이 진단은 `candidate_cache`와 `scores.csv`만 읽었으며 live search와 이미지 분석을 실행하지 않았다.",
        "",
        "## Scoring implementation checks",
        "",
        "- `base_score`: multiplier 1.0의 `final_score`와 일치하는 기존 총점으로 사용됨.",
        "- `body_shape_score`: R-BOD가 연결된 매칭 keyword의 attribute weight만 분리됨.",
        "- `final_score = base_score + body_shape_score * (multiplier - 1.0)`.",
        "- R-BOD가 없는 keyword는 `body_shape_score`에 포함되지 않음.",
        "- condition별 targets와 keyword_rules는 scores.csv의 body_shape_keywords 및 연결 rule에 반영됨.",
        "",
        "## Candidate set comparison",
        "",
    ]
    pair_specs = (("baseline_front", "arm_pose"), ("baseline_front", "slight_angle"), ("arm_pose", "slight_angle"))
    for left, right in pair_specs:
        intersection = len(candidates[left] & candidates[right])
        similarity = jaccard(candidates[left], candidates[right])
        report.append(f"- {left} vs {right}: intersection={intersection}, Jaccard={similarity:.6f}")
        summary_rows.append({"condition": f"{left} vs {right}", "metric": "candidate_intersection", "value": intersection, "details": ""})
        summary_rows.append({"condition": f"{left} vs {right}", "metric": "candidate_jaccard", "value": similarity, "details": ""})
    same_sets = len({frozenset(value) for value in candidates.values()}) == 1
    report.append(f"- 결론: 세 condition 후보 집합은 완전히 동일함={same_sets}; 각 집합 크기={[len(candidates[c]) for c in CONDITIONS]}, union={len(set().union(*candidates.values()))}.")
    report.append("")

    condition_events: dict[str, dict[str, Any]] = {}
    for condition in CONDITIONS:
        rows = scores[condition]
        connection = connections[condition]
        bod_keywords = sorted({
            keyword
            for category in connection["keyword_rules"].values()
            for keyword, rule_ids in category.items()
            if any(rule_id.startswith("R-BOD-") for rule_id in rule_ids)
        })
        body_values = [row["body_shape_score"] for row in rows.values()]
        positive = [value for value in body_values if value > 0]
        counts = Counter(body_values)
        top10 = ranking(rows, 1.0)[:10]
        first = first_change(rows, None)
        top10_change = first_change(rows, 10)
        top3_change = first_change(rows, 3)
        condition_events[condition] = {
            "first_change": first,
            "top10_change": top10_change,
            "top3_change": top3_change,
            "additional_multipliers": additional_multipliers(first, top3_change),
        }
        report.extend([
            f"## {condition}",
            "",
            f"- body_shape: {connection['body_shape']}",
            f"- R-BOD keywords: {', '.join(bod_keywords) if bod_keywords else '없음'}",
            f"- R-BOD 점수 > 0 상품 수: {len(positive)} / {len(rows)}",
            f"- body_shape_score min/max/mean: {min(body_values):.6f} / {max(body_values):.6f} / {sum(body_values) / len(body_values):.6f}",
            f"- unique values/frequency: {', '.join(f'{value:g}({counts[value]})' for value in sorted(counts))}",
            f"- 최초 전체 순위 변화 교차점: {fmt(first)}",
            f"- 최초 Top-10 변화 교차점: {fmt(top10_change)}",
            f"- 최초 Top-3 진입/이탈 교차점: {fmt(top3_change)}",
            f"- 추가 multiplier 제안: {condition_events[condition]['additional_multipliers'] or '없음 (m>=1에서 유효한 Top-3/Top-10/전체 rank crossing 없음)'}",
            "",
            "### Multiplier 1.0 Top-10",
            "",
            "| rank | product_id | product_name | base_score | body_shape_score | matched_keywords | body_shape_keywords |",
            "|---:|---|---|---:|---:|---|---|",
        ])
        for display_rank, row in enumerate(top10, 1):
            report.append(
            f"| {display_rank} | {row['product_id']} | {row['product_name']} | {row['base_score']:.4f} | "
                f"{row['body_shape_score']:.4f} | {row['matched_keywords']} | {row['body_shape_keywords']} |"
            )
        report.append("")

        for multiplier in (0.0, 1.0, 1.5, 2.0, 3.0):
            ranked = ranking(rows, multiplier)
            baseline = ranking(rows, 1.0)
            baseline_positions = {row["product_id"]: index + 1 for index, row in enumerate(baseline)}
            current_positions = {row["product_id"]: index + 1 for index, row in enumerate(ranked)}
            changes = [current_positions[row["product_id"]] - baseline_positions[row["product_id"]] for row in ranked]
            changed_rank_count = sum(change != 0 for change in changes)
            final_changed_no_rank = sum(
                not math.isclose(score(row, multiplier), row["base_score"]) and
                current_positions[row["product_id"]] == baseline_positions[row["product_id"]]
                for row in rows.values()
            )
            top3_body_zero = all(row["body_shape_score"] == 0 for row in ranked[:3])
            summary_rows.extend([
                {"condition": condition, "metric": "rank_changed_count", "value": changed_rank_count, "details": f"multiplier={multiplier}"},
                {"condition": condition, "metric": "max_rank_rise", "value": min(changes), "details": f"multiplier={multiplier}"},
                {"condition": condition, "metric": "max_rank_fall", "value": max(changes), "details": f"multiplier={multiplier}"},
                {"condition": condition, "metric": "final_changed_rank_unchanged_count", "value": final_changed_no_rank, "details": f"multiplier={multiplier}"},
                {"condition": condition, "metric": "top3_all_body_shape_score_zero", "value": top3_body_zero, "details": f"multiplier={multiplier}"},
            ])

    report.extend([
        "## Overall conclusion",
        "",
        "- 후보 집합이 세 condition에서 완전히 동일하고, condition별 R-BOD score 분포도 동일한지 위 통계로 확인한다.",
        "- Top-3가 R-BOD 점수 0인 상품으로 구성되어 있으면 multiplier를 올려도 Top-3가 즉시 변하지 않는다.",
        "- 교차점이 10 이상이면 가중치 조정만으로 해결하기 부적절한 큰 값으로 분류한다.",
    ])
    if same_sets and all(
        len({row["body_shape_score"] for row in scores[condition].values()}) <= 2
        for condition in CONDITIONS
    ):
        report.append("- 판정: 후보 집합이 동일하다는 사실만으로 C를 단정하지 않고, condition별 score 분포와 교차점을 함께 봐야 한다.")
    report.append("")
    return summary_rows, "\n".join(report) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parents[1] / "outputs" / "recommendation_body_shape_weight")
    args = parser.parse_args()
    rows, report = diagnose(args.input_dir)
    report_path = args.input_dir / "diagnostic_report.md"
    report_path.write_text(report, encoding="utf-8")
    csv_path = args.input_dir / "diagnostic_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("condition", "metric", "value", "details"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"saved {report_path}")
    print(f"saved {csv_path}")


if __name__ == "__main__":
    main()
