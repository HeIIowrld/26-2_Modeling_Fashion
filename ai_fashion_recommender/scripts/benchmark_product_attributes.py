"""C-1: freeze products, collect human labels, and replay an attribute benchmark.

No catalog enrichment, training, or live-search behavior is changed here.
Run with --help; all commands except predict/download use the standard library.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fashion_attribute_schema import ATTRIBUTE_TASKS
from musinsa_live_search import MusinsaLiveSearch

AXES = ("item_type", "fit", "length", "material")
CATEGORIES = ("top", "bottom", "shoes")
QUOTAS = {"top": 34, "bottom": 34, "shoes": 32}
CUTS = ("worn", "flat")
SEED = "attribute-c1-v1"
TASKS = {
    "top": dict(zip(AXES, ("category", "upper_fit", "upper_length", "material"))),
    "bottom": dict(zip(AXES, ("lower_subtype", "lower_fit", "lower_length", "material"))),
    "shoes": {"item_type": None, "fit": None, "length": None, "material": "material"},
}
# Fixed before annotation. Each tuple is (gold/model label, query keyword).
# _score's aliases and first-match behavior remain untouched.
FIT = {
    "top": [("슬림핏", "슬림"), ("레귤러핏", "레귤러"),
            ("오버핏", "오버핏"), ("여유핏", "여유핏")],
    "bottom": [("슬림핏", "슬림"), ("스트레이트핏", "스트레이트"),
               ("테이퍼드핏", "테이퍼드"), ("와이드핏", "와이드"), ("플레어핏", "플레어")],
    "shoes": [],
}
LENGTH = {
    "top": [("크롭 기장", "허리선"), ("기본 기장", "기본 기장"), ("롱 기장", "롱")],
    "bottom": [("쇼츠·미니 기장", "쇼츠"), ("쇼츠·미니 기장", "반바지"),
               ("쇼츠·미니 기장", "미니"), ("무릎 기장", "무릎"),
               ("미디·7부 기장", "미디"), ("미디·7부 기장", "7부"),
               ("롱·긴바지 기장", "풀렝스"), ("롱·긴바지 기장", "긴바지")],
    "shoes": [],
}
SHOE_TYPES = ("스니커즈", "러닝화", "로퍼", "더비슈즈", "부츠", "샌들", "슬리퍼", "메리제인", "힐")
SKIP_LABELS = {"unknown", "not_applicable"}
FIELDS = ["product_id", "category", "name", "brand_name", "brand", "image_url", "url",
          "image_sha256", "cut_type", "cut_reviewer", "cut_reviewed_at",
          *[f"gold_{a}" for a in AXES], "human_reviewer", "reviewed_at", "label_evidence"]


def vocabulary(category, axis):
    if axis == "fit":
        return FIT[category]
    if axis == "length":
        return LENGTH[category]
    labels = (SHOE_TYPES if category == "shoes" and axis == "item_type"
              else ATTRIBUTE_TASKS[TASKS[category][axis]].labels)
    # Specific terms first, so e.g. 폴로 셔츠 is tested before 셔츠.
    return [(label, label) for label in sorted(labels, key=lambda s: (-len(s), s))]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    ids = [r["product_id"] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate product IDs")
    if any(r["category"] not in CATEGORIES for r in rows):
        raise ValueError("Unknown category")
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in FIELDS} for r in rows)


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def stable_order(row):
    return hashlib.sha256(f"{SEED}:{row['product_id']}".encode()).hexdigest()


def choose_sample(rows, provisional=False):
    selected = []
    for category, count in QUOTAS.items():
        pool = [r for r in rows if r["category"] == category and r.get("image_sha256")]
        groups = [(None, count)] if provisional else [(cut, count // 2) for cut in CUTS]
        for cut, size in groups:
            eligible = [r for r in pool if cut is None or (
                r.get("cut_type") == cut and r.get("cut_reviewer", "").strip()
                and r.get("cut_reviewed_at", "").strip())]
            eligible.sort(key=stable_order)
            if len(eligible) < size:
                raise ValueError(f"Insufficient reviewed images: {category}/{cut}: {len(eligible)} < {size}")
            selected.extend(eligible[:size])
    return selected


def title_prediction(row):
    search = MusinsaLiveSearch()
    item = {"goodsName": row["name"], "brandName": row["brand_name"], "brand": row["brand"]}
    axes = {}
    for axis in AXES:
        pairs = vocabulary(row["category"], axis)
        keywords = list(dict.fromkeys(keyword for _, keyword in pairs))
        score, matched = search._score(item, {axis: keywords}, rank=0)
        _, title_matched = search._score({"goodsName": row["name"]}, {axis: keywords}, rank=0)
        axes[axis] = {"labels": [dict((k, v) for v, k in pairs)[m] for m in matched],
                      "matched": matched, "score": score,
                      "no_title_keyword": not title_matched}
    return axes


def prepare(args):
    rows, sources = [], {}
    for category in CATEGORIES:
        path = args.snapshots / f"popular_{category}.json"
        snapshot = json.loads(path.read_text(encoding="utf-8"))
        sources[category] = {"sha256": digest(path), "url": snapshot.get("url"),
                             "fetched_at": snapshot.get("fetched_at")}
        for item in snapshot["items"]:
            pid = f"MS{item['goodsNo']}"
            rows.append(dict(product_id=pid, category=category, name=item["goodsName"],
                             brand_name=item.get("brandName", ""), brand=item.get("brand", ""),
                             image_url=item.get("thumbnail", ""),
                             url=f"https://www.musinsa.com/products/{item['goodsNo']}"))
    if len({r["product_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate snapshot IDs; resolve category provenance before sampling")
    if args.output.exists():
        raise ValueError("Refusing to replace existing human annotation pool")
    write_csv(args.output, rows)
    write_json(args.output.with_suffix(".sources.json"), sources)
    print(f"Prepared {len(rows)} candidates; human labels are empty.")


def download(args):
    from PIL import Image
    rows = read_csv(args.labels)
    args.images.mkdir(parents=True, exist_ok=True)

    def fetch(row):
        path = args.images / f"{row['product_id']}.jpg"
        try:
            if not path.exists():
                url = row["image_url"]
                if url.startswith("//"):
                    url = "https:" + url
                from urllib.parse import urlparse
                parsed = urlparse(url)
                if parsed.scheme != "https" or not (parsed.hostname or "").endswith(".msscdn.net"):
                    raise ValueError("Expected HTTPS Musinsa image URL")
                request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(request, timeout=20) as response:
                    data = response.read(12 * 1024 * 1024 + 1)
                if len(data) > 12 * 1024 * 1024:
                    raise ValueError("Image exceeds 12 MiB")
                import io
                with Image.open(io.BytesIO(data)) as image:
                    image.verify()
                path.write_bytes(data)
            with Image.open(path) as image:
                image.verify()
            actual = digest(path)
            if row.get("image_sha256") and row["image_sha256"] != actual:
                raise ValueError("Image changed since annotation")
            row["image_sha256"] = actual
            return {"product_id": row["product_id"], "sha256": actual}
        except Exception as exc:
            return {"product_id": row["product_id"], "error": str(exc)}

    with ThreadPoolExecutor(max_workers=6) as executor:
        results = list(executor.map(fetch, rows))
    write_csv(args.labels, rows)
    write_json(args.labels.with_suffix(".images.json"), results)
    print(f"Images: {sum('sha256' in r for r in results)}/{len(rows)}; failures recorded.")


def sample(args):
    if args.output.exists():
        raise ValueError("Output exists; choose a new filename to preserve annotations")
    rows = choose_sample(read_csv(args.pool), args.provisional)
    write_csv(args.output, rows)
    write_json(args.output.with_suffix(".sample.json"), {
        "status": "provisional_not_cut_stratified" if args.provisional else "cut_stratified",
        "seed": SEED, "pool_sha256": digest(args.pool), "labels_sha256": digest(args.output),
        "quotas": QUOTAS, "ids": [r["product_id"] for r in rows]})
    print(f"Selected {len(rows)} {'provisional' if args.provisional else 'stratified'} rows.")


def predict(args):
    actual = digest(args.checkpoint)
    if actual != args.expected_sha256:
        raise ValueError(f"Server checkpoint mismatch: {actual}")
    rows = read_csv(args.labels)
    from config import FASHION_SIGLIP_MODEL_ID
    from fashion_attribute_model import FashionAttributePredictor
    import open_clip
    import torch
    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    # Attribute heads only consume image embeddings; no text tokenizer is needed.
    # Use the server's explicit frozen snapshot rather than downloading unrelated
    # ONNX/tokenizer assets to satisfy snapshot_download's completeness check.
    backbone_dir = args.backbone_dir.resolve()
    if not (backbone_dir / "open_clip_config.json").is_file():
        raise ValueError("Expected the server's OpenCLIP backbone snapshot directory")
    model, _, preprocess = open_clip.create_model_and_transforms(f"local-dir:{backbone_dir}", device=device)
    model.eval()
    predictor = FashionAttributePredictor(args.checkpoint, image_encoder=model, preprocess=preprocess,
                                          model_id=FASHION_SIGLIP_MODEL_ID, device=device)
    results = []
    for row in rows:
        record = {"product_id": row["product_id"], "category": row["category"],
                  "name": row["name"], "brand_name": row["brand_name"], "brand": row["brand"],
                  "image_url": row["image_url"], "title": title_prediction(row), "photo": {}}
        path = args.images / f"{row['product_id']}.jpg"
        try:
            if not row.get("image_sha256") or digest(path) != row["image_sha256"]:
                raise ValueError("Missing or changed frozen image")
            record["image_sha256"] = digest(path)
            mapping = TASKS[row["category"]]
            from PIL import Image
            with Image.open(path) as image:
                # Match enrich_catalog -> predict_trained_attributes -> predict:
                # preserve checkpoint preprocessing and the predictor's defaults.
                out = predictor.predict(image, tasks=list(dict.fromkeys(t for t in mapping.values() if t)))
            record["raw_heads"] = {k: v.to_dict() for k, v in out.items()}
            for axis, task in mapping.items():
                pred = out.get(task)
                record["photo"][axis] = {
                    "labels": list(pred.labels) if pred and pred.accepted else [],
                    "status": ("unsupported" if task is None else "accepted" if pred.accepted else "abstained"),
                }
        except Exception as exc:
            record["error"] = f"{type(exc).__name__}: {exc}"
            record["photo"] = {a: {"labels": [], "status": "error"} for a in AXES}
        results.append(record)
        print(f"{len(results)}/{len(rows)} {row['product_id']} {record.get('error', 'ok')}", flush=True)
    source_files = [Path(__file__), *[ROOT / "src" / f for f in (
        "musinsa_live_search.py", "fashion_attribute_model.py", "fashion_attribute_schema.py", "config.py")]]
    backbone = {p.name: digest(p) for p in backbone_dir.iterdir() if p.is_file() and p.suffix in (".bin", ".safetensors", ".json")}
    write_json(args.output, {"created_at": now(), "checkpoint_sha256": actual,
        "backbone_id": FASHION_SIGLIP_MODEL_ID, "backbone_sha256": backbone,
        "device": device, "torch_version": torch.__version__,
        "prediction_call": "predict(image, tasks=tasks); geometry uses predictor default",
        "head_preprocessing": predictor.preprocessing, "head_geometry_dim": predictor.geometry_dim,
        "packages": {n: importlib.metadata.version(n) for n in ("open_clip_torch", "Pillow", "numpy")},
        "source_sha256": {p.name: digest(p) for p in source_files},
        "input_sha256": digest(args.labels), "vocabulary": {c: {a: vocabulary(c, a) for a in AXES} for c in CATEGORIES},
        "results": results})


def validate_labels(rows):
    if len(rows) != 100:
        raise ValueError(f"Expected 100 distinct products, got {len(rows)}")
    strata = Counter((r["category"], r.get("cut_type")) for r in rows)
    expected = {(c, cut): n // 2 for c, n in QUOTAS.items() for cut in CUTS}
    if dict(strata) != expected:
        raise ValueError(f"Cut quotas not met: {dict(strata)}; expected {expected}")
    for row in rows:
        for key in ("image_sha256", "cut_reviewer", "cut_reviewed_at", "human_reviewer", "reviewed_at", "label_evidence"):
            if not row.get(key, "").strip():
                raise ValueError(f"{row['product_id']}: missing {key}")
        for axis in AXES:
            value = row.get(f"gold_{axis}", "")
            labels = value.split("|")
            allowed = {label for label, _ in vocabulary(row["category"], axis)} | {"other"}
            if value in SKIP_LABELS:
                continue
            if not value or any(label not in allowed for label in labels):
                raise ValueError(f"{row['product_id']}/{axis}: invalid human label {value!r}")
            if len(labels) > 1 and axis != "material":
                raise ValueError(f"{row['product_id']}/{axis}: single label required")


def metric(rows, records, axis, method, cut=None, missing=False):
    correct = total = covered = 0
    for row in rows:
        if cut and row["cut_type"] != cut:
            continue
        gold = row[f"gold_{axis}"]
        if gold in SKIP_LABELS:
            continue
        record = records[row["product_id"]]
        if missing and not record["title"][axis]["no_title_keyword"]:
            continue
        predicted = record[method][axis]["labels"]
        total += 1
        covered += bool(predicted)
        correct += set(predicted) == set(gold.split("|"))
    return correct, total, covered


def cell(value):
    correct, total, _ = value
    return f"{correct / total:.1%} ({correct}/{total})" if total else "N/A (0개)"


def render_report(rows, payload, labels_hash, predictions_hash):
    validate_labels(rows)
    result_rows = payload["results"]
    records = {r["product_id"]: r for r in result_rows}
    if len(records) != len(result_rows):
        raise ValueError("Duplicate prediction IDs")
    for row in rows:
        if row["product_id"] not in records:
            raise ValueError(f"Missing prediction: {row['product_id']}")
        record = records[row["product_id"]]
        if record.get("error"):
            raise ValueError(f"Inference failed: {row['product_id']}: {record['error']}")
        for key in ("category", "name", "brand_name", "brand", "image_url", "image_sha256"):
            if row[key] != record.get(key):
                raise ValueError(f"Prediction input changed: {row['product_id']}/{key}")
    lines = ["# C-1 상품 속성 비교", "", "사람 검수 100개: 상의 34 · 하의 34 · 신발 32, 착용컷 50 · 평면컷 50.", ""]
    for missing, title in ((False, "전체"), (True, "해당 축의 상품명 키워드가 없는 상품")):
        lines += [f"## {title}", "", "| 축 | 상품명 매칭 | 사진 판정 (착용컷) | 사진 판정 (평면컷) |",
                  "| --- | ---: | ---: | ---: |"]
        for axis in AXES:
            values = [metric(rows, records, axis, "title", missing=missing),
                      *[metric(rows, records, axis, "photo", cut=cut, missing=missing) for cut in CUTS]]
            lines.append(f"| {axis} | " + " | ".join(cell(v) for v in values) + " |")
        lines += ["", "같은 컷끼리 비교한 상품명 정확도와 사진 판정률(비어 있지 않은 예측/평가 가능 정답):", "",
                  "| 축 | 상품명 (착용컷) | 상품명 (평면컷) | 사진 판정률 (착용컷) | 사진 판정률 (평면컷) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for axis in AXES:
            title_cells = [cell(metric(rows, records, axis, "title", cut, missing)) for cut in CUTS]
            photo_cells = []
            for cut in CUTS:
                _, n, covered = metric(rows, records, axis, "photo", cut, missing)
                photo_cells.append(cell((covered, n, covered)))
            lines.append(f"| {axis} | " + " | ".join(title_cells + photo_cells) + " |")
        lines.append("")
    lines += ["## 집계 기준과 제한", "", "- 분자는 정답과 라벨 집합이 완전히 같은 수, 분모는 사람이 판정 가능한 정답 수다. 소재도 집합 완전 일치를 사용한다.",
              "- 모델 판정 보류·미지원은 오답으로 센다. 실행 오류는 보고서 생성을 중단한다.",
              "- `unknown`·`not_applicable`은 두 방법에서 동일하게 제외한다. N/A는 정확도 0%와 다르다.",
              "- 상품명 수치는 고정 어휘를 입력한 `_score()`의 축별 첫 matched 값이다. 상품 분류기나 실제 검색 순위 정확도가 아니다.",
              "- 검색에는 브랜드도 들어가지만, 키워드 부재 여부는 goodsName만으로 판정한다.",
              "- 신발 종류·핏·기장은 의류 헤드 미지원이다. 신발 소재도 학습 분포 밖일 수 있다.", "",
              "| 축 | unknown | not_applicable | 사진 헤드 미지원 |", "| --- | ---: | ---: | ---: |"]
    for axis in AXES:
        unknown = sum(r[f"gold_{axis}"] == "unknown" for r in rows)
        na = sum(r[f"gold_{axis}"] == "not_applicable" for r in rows)
        unsupported = sum(records[r["product_id"]]["photo"][axis]["status"] == "unsupported" for r in rows)
        lines.append(f"| {axis} | {unknown} | {na} | {unsupported} |")
    lines += ["", "C-2 시작 여부: 이 표의 축별·컷별 결과를 검토한 뒤 결정한다. 이 도구는 C-2를 자동 활성화하지 않는다.", "",
              f"정답 CSV SHA-256: `{labels_hash}`", "", f"예측 JSON SHA-256: `{predictions_hash}`", "",
              f"헤드 SHA-256: `{payload['checkpoint_sha256']}`", "",
              "코드·백본·환경·원시 예측은 예측 JSON에 보존한다. 같은 CSV/JSON으로 집계하면 네트워크·GPU 없이 동일한 표가 나온다.", ""]
    return "\n".join(lines)


def report(args):
    text = render_report(read_csv(args.labels), json.loads(args.predictions.read_text(encoding="utf-8")),
                         digest(args.labels), digest(args.predictions))
    args.output.write_text(text, encoding="utf-8")
    print(args.output)


def review(args):
    rows = read_csv(args.labels)
    options = {c: {a: list(dict.fromkeys(v for v, _ in vocabulary(c, a))) + ["other", "unknown", "not_applicable"]
                   for a in AXES} for c in CATEGORIES}
    relative = os.path.relpath(args.images.resolve(), args.output.parent.resolve())
    data = json.dumps({"rows": rows, "options": options, "fields": FIELDS, "axes": AXES,
                       "images": relative, "filename": args.labels.name}, ensure_ascii=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("attribute_c1_review.html").read_text(encoding="utf-8")
    args.output.write_text(template.replace("__BENCHMARK_DATA__", data), encoding="utf-8")
    print(args.output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Create an empty human annotation pool from three frozen API snapshots")
    p.add_argument("--snapshots", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("download", help="Freeze and hash images; never infer labels")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p = sub.add_parser("sample", help="Choose 100 reviewed, cut-stratified images")
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--provisional", action="store_true", help="Draft only: ignore cut quotas, never a completed C-1 sample")
    p = sub.add_parser("predict", help="Run _score() and the deployed trained predictor without reading gold labels")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--backbone-dir", type=Path, required=True, help="Frozen server OpenCLIP snapshot (config and weights)")
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("report", help="Strictly validate human labels and produce the reproducible accuracy tables")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("review", help="Create a local image review/CSV labeling page; predictions stay hidden")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        globals()[args.command](args)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"C-1 incomplete: {exc}\n")


if __name__ == "__main__":
    main()
