"""Explicitly separate AI reference labels from the human-ground-truth C-1 path."""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import benchmark_product_attributes as benchmark

SOURCE = "ai_visual_review_with_seller_evidence"
EXTRA_FIELDS = ["label_source", "ai_reviewer", "ai_reviewed_at", "seller_evidence_url",
                "seller_evidence_sha256", "review_artifact_sha256"]


def seller_item_type(category, path, mapping):
    if category == "shoes":
        for term, label in [("러닝화", "러닝화"), ("부츠", "부츠"), ("메리제인", "메리제인"),
                            ("로퍼", "로퍼"), ("스니커즈", "스니커즈"), ("슬리퍼", "슬리퍼"), ("샌들", "샌들")]:
            if term in path:
                return label
        return "other"
    for term, label in mapping:
        if term in path:
            return label
    return "other"


def build(args):
    if args.output.exists():
        raise ValueError("Reference CSV already exists; preserve the reviewed version")
    source = benchmark.read_csv(args.sample)
    review = json.loads(args.review.read_text(encoding="utf-8"))
    entries = review["rows"]
    if [r[0] for r in entries] != list(range(1, 101)) or len(source) != 100:
        raise ValueError("Require exactly one explicit review for each of the original 100 products")
    mapping = json.loads((benchmark.ROOT / "data/catalog_derivation.json").read_text(encoding="utf-8"))["musinsa_item_type"]
    sources = []
    for row, entry in zip(source, entries):
        index, cut, fit, length, material, evidence = entry
        detail_path = args.seller_details / f"{row['product_id']}.json"
        detail = json.loads(detail_path.read_text(encoding="utf-8"))
        data = detail["data"]
        if str(data["goodsNo"]) != row["product_id"].removeprefix("MS"):
            raise ValueError("Seller evidence product ID mismatch")
        item_type = seller_item_type(row["category"], data["baseCategoryFullPath"], mapping)
        # Explicitly reviewed exception: seller category is generic 기타 바지,
        # but its independent detail description identifies this as track pants.
        if row["product_id"] == "MS7036867":
            if "트랙 팬츠" not in data.get("goodsContents", ""):
                raise ValueError("Missing independent track-pants evidence")
            item_type = "트랙팬츠"
        row.update(cut_type=cut, cut_reviewer=review["reviewer"], cut_reviewed_at=review["reviewed_date"],
                   gold_item_type=item_type, gold_fit=fit, gold_length=length, gold_material=material,
                   human_reviewer="", reviewed_at="", label_evidence=evidence, label_source=SOURCE,
                   ai_reviewer=review["reviewer"], ai_reviewed_at=review["reviewed_date"],
                   seller_evidence_url=detail["url"], seller_evidence_sha256=benchmark.digest(detail_path),
                   review_artifact_sha256=benchmark.digest(args.review))
        sources.append({"product_id": row["product_id"], "url": detail["url"], "fetched_at": detail["fetched_at"],
                        "sha256": benchmark.digest(detail_path), "seller_category": data["baseCategoryFullPath"],
                        "canonical_item_type": item_type, "review_notes": evidence})
    validate_references(source)
    with args.output.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=benchmark.FIELDS + EXTRA_FIELDS)
        writer.writeheader()
        writer.writerows(source)
    benchmark.write_json(args.output.with_suffix(".sources.json"), {
        "review_sha256": benchmark.digest(args.review), "sample_sha256": benchmark.digest(args.sample),
        "category_mapping_sha256": benchmark.digest(benchmark.ROOT / "data/catalog_derivation.json"),
        "sources": sources})
    print(f"Saved 100 AI references; human_reviewer is empty: {args.output}")


def validate_references(rows):
    if len(rows) != 100 or len({r["product_id"] for r in rows}) != 100:
        raise ValueError("Require 100 distinct products")
    if Counter(r["category"] for r in rows) != benchmark.QUOTAS:
        raise ValueError("Category counts changed")
    for row in rows:
        if row.get("label_source") != SOURCE or row.get("human_reviewer") or row.get("reviewed_at"):
            raise ValueError("AI references must not impersonate human ground truth")
        for key in EXTRA_FIELDS + ["image_sha256", "cut_reviewer", "cut_reviewed_at", "label_evidence"]:
            if not row.get(key, "").strip():
                raise ValueError(f"Missing AI evidence: {row['product_id']}/{key}")
        if row.get("cut_type") not in (*benchmark.CUTS, "other"):
            raise ValueError("Missing cut review")
        for axis in benchmark.AXES:
            value = row.get(f"gold_{axis}", "")
            allowed = {v for v, _ in benchmark.vocabulary(row["category"], axis)} | {"other"}
            if value in benchmark.SKIP_LABELS:
                continue
            labels = value.split("|")
            if not value or any(v not in allowed for v in labels):
                raise ValueError(f"Invalid reference: {row['product_id']}/{axis}/{value}")
            if len(labels) > 1 and (axis != "material" or "other" in labels):
                raise ValueError("Invalid mixed labels")


def pair_predictions(rows, payload):
    validate_references(rows)
    records = {r["product_id"]: r for r in payload["results"]}
    if len(records) != len(payload["results"]):
        raise ValueError("Duplicate predictions")
    pairs = []
    for row in rows:
        record = records.get(row["product_id"])
        if record is None or record.get("error"):
            raise ValueError(f"Missing or failed inference: {row['product_id']}")
        for key in ("category", "name", "brand_name", "brand", "image_url", "image_sha256"):
            if row[key] != record.get(key):
                raise ValueError(f"Frozen input changed: {row['product_id']}/{key}")
        for axis in benchmark.AXES:
            gold = row[f"gold_{axis}"]
            name = record["title"][axis]["labels"]
            photo = record["photo"][axis]["labels"]
            hybrid = name or photo
            evaluable = gold not in benchmark.SKIP_LABELS
            equal = lambda labels: evaluable and set(labels) == set(gold.split("|"))
            pairs.append({"product_id": row["product_id"], "category": row["category"], "cut_type": row["cut_type"],
                          "axis": axis, "reference": gold, "label_source": row["label_source"], "evaluable": evaluable,
                          "title": "|".join(name), "photo": "|".join(photo), "hybrid_simulation": "|".join(hybrid),
                          "no_title_keyword": record["title"][axis]["no_title_keyword"],
                          "photo_status": record["photo"][axis]["status"],
                          "title_correct": equal(name), "photo_correct": equal(photo), "hybrid_correct": equal(hybrid),
                          "added_photo_label": not name and bool(photo)})
    return records, pairs


def summarize(pairs):
    result = {}
    for axis in benchmark.AXES:
        relevant = [p for p in pairs if p["axis"] == axis and p["evaluable"]]
        fallback = [p for p in relevant if p["added_photo_label"]]
        result[axis] = {"n": len(relevant), **{method: sum(p[f"{method}_correct"] for p in relevant)
                       for method in ("title", "photo", "hybrid")},
                       "fallback_correct": sum(p["photo_correct"] for p in fallback),
                       "fallback_count": len(fallback),
                       "fallback_wrong": sum(not p["photo_correct"] for p in fallback)}
    return result


def report(args):
    rows = benchmark.read_csv(args.labels)
    payload = json.loads(args.predictions.read_text(encoding="utf-8"))
    records, pairs = pair_predictions(rows, payload)
    summary = summarize(pairs)
    counts = Counter(r["cut_type"] for r in rows)
    lines = ["# C-1 자동 검수 비교", "", "**AI 참조 라벨과의 일치율이다. 사람 정답 기반 정확도가 아니다.**", "",
             f"같은 100개 상품: 상의 34·하의 34·신발 32. 착용컷 {counts['worn']}·평면컷 {counts['flat']}·기타 {counts['other']}.",
             "손으로 든 신발 1개는 기타 컷으로 남겨 전체 비교에 포함하고 착용/평면 표에서는 제외한다.",
             "사전에 정한 50:50 컷 할당의 사람 검수 표본이 아니며, 결과를 본 뒤 상품을 교체하지 않았다.", ""]
    for missing, title in ((False, "전체"), (True, "해당 축의 상품명 키워드가 없는 상품")):
        lines += [f"## {title}", "", "| 축 | 상품명 매칭 | 사진 판정 (착용컷) | 사진 판정 (평면컷) |",
                  "| --- | ---: | ---: | ---: |"]
        for axis in benchmark.AXES:
            values = [benchmark.metric(rows, records, axis, "title", missing=missing),
                      *[benchmark.metric(rows, records, axis, "photo", cut=cut, missing=missing) for cut in benchmark.CUTS]]
            lines.append(f"| {axis} | " + " | ".join(benchmark.cell(v) for v in values) + " |")
        lines += ["", "| 축 | 상품명 (착용컷) | 상품명 (평면컷) |", "| --- | ---: | ---: |"]
        for axis in benchmark.AXES:
            cells = [benchmark.cell(benchmark.metric(rows, records, axis, "title", cut, missing)) for cut in benchmark.CUTS]
            lines.append(f"| {axis} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += ["## 상품명 우선 보충 시뮬레이션", "",
              "실제 검색 배선 전, 같은 예측에서 상품명 라벨이 비어 있는 축에만 사진 라벨을 넣은 계산이다.", "",
              "| 축 | 상품명만 정답/평가 수 | 보충 후 정답/평가 수 | 사진으로 추가한 라벨의 정답/추가 수 | 추가 오답 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for axis, s in summary.items():
        lines.append(f"| {axis} | {benchmark.cell((s['title'], s['n'], 0))} | {benchmark.cell((s['hybrid'], s['n'], 0))} | "
                     f"{benchmark.cell((s['fallback_correct'], s['fallback_count'], 0))} | {s['fallback_wrong']} |")
    lines += ["", "unknown·not_applicable은 양쪽 모두 제외한다. 모델 판정 보류와 미지원은 오답이다. 소재는 라벨 집합 완전 일치를 쓴다.",
              "점수를 더 채워 일치율이 올라도 틀린 라벨을 함께 늘릴 수 있다. 따라서 보충 정밀도와 추가 오답을 반드시 함께 본다.", "",
              "| 축 | unknown | not_applicable |", "| --- | ---: | ---: |"]
    for axis in benchmark.AXES:
        lines.append(f"| {axis} | {sum(r[f'gold_{axis}']=='unknown' for r in rows)} | {sum(r[f'gold_{axis}']=='not_applicable' for r in rows)} |")
    lines += ["", f"참조 CSV SHA-256: `{benchmark.digest(args.labels)}`", "",
              f"예측 JSON SHA-256: `{benchmark.digest(args.predictions)}`", "",
              f"서버 헤드 SHA-256: `{payload['checkpoint_sha256']}`", "",
              "참조 라벨과 판매자 근거는 모델 예측과 독립된 파일에 보관했다. 사람 검수자는 빈칸으로 유지했다.", ""]
    args.output.write_text("\n".join(lines), encoding="utf-8")
    benchmark.write_json(args.output.with_suffix(".metrics.json"), {
        "label_source": SOURCE, "cut_counts": counts, "summary": summary, "pairs": pairs,
        "labels_sha256": benchmark.digest(args.labels), "predictions_sha256": benchmark.digest(args.predictions),
        "evaluation_code_sha256": benchmark.digest(__file__)})
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("build")
    p.add_argument("--sample", type=Path, required=True)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--seller-details", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        {"build": build, "report": report}[args.command](args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"AI reference evaluation failed: {exc}\n")


if __name__ == "__main__":
    main()
