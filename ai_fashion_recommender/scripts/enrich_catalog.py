from __future__ import annotations

"""크롤링한 상품 카탈로그에 규칙 엔진이 쓰는 속성을 채운다.

musinsa_crawler.py 가 만드는 products_musinsa.csv 에는 15개 칼럼밖에 없다.
products.csv 가 가진 27개 중 fit·material·formality 등 **16개가 빠져 있어서**,
그대로 추천에 쓰면 ProductCatalog 이 전부 기본값으로 채우고 그 속성을 보는
규칙이 조용히 잠든다. 그래서 크롤링본은 중간 산출물로 두고 여기서 채운 뒤 쓴다.

채우는 방법은 두 가지다.

1. **상품 사진에서 읽는다** — 학습된 17개 속성 헤드를 그대로 쓴다.
   item_type · fit · length · pattern · material · neckline · detail_level
2. **소재와 종류에서 유도한다** — 사진으로 판정할 수 없고 속성 태스크에도 없는 것.
   formality · warmth · breathability · water_resistant · visual_weight ·
   activity_tags · waistline · pattern_scale · pattern_contrast
   기준은 data/catalog_derivation.json 에 있고 **잠정값이다.**

사용법:
    python scripts/enrich_catalog.py                    # data/products_musinsa.csv -> _enriched.csv
    python scripts/enrich_catalog.py --limit 20         # 앞 20개만 (동작 확인용)
    python scripts/enrich_catalog.py --report           # 채우지 않고 규칙 발동 가능 여부만 점검
    python scripts/enrich_catalog.py --input a.csv --output b.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from config import (  # noqa: E402
    DATA_DIR,
    FASHION_ATTRIBUTE_HEADS_PATH,
    garment_image_path,
)

# ProductCatalog 이 읽는 전체 스키마. 순서를 products.csv 와 맞춰 두면 사람이 비교하기 쉽다.
OUTPUT_FIELDS = [
    "product_id", "name", "category", "color", "style", "purposes", "body_shapes",
    "price", "season", "stock", "url", "item_type", "fit", "length", "pattern",
    "material", "neckline", "formality", "activity_tags", "warmth", "breathability",
    "water_resistant", "visual_weight", "detail_level", "waistline",
    "pattern_scale", "pattern_contrast", "brand", "gender", "image_url", "image_path",
    # 출처 추적용. 무신사가 표기한 원본 값을 그대로 남겨 둔다.
    "detail_color", "detail_season", "detail_fit", "detail_thickness",
    "detail_sheer", "detail_category",
]

# 상·하의에 따라 같은 칼럼을 다른 태스크에서 가져온다.
TOP_TASKS = ["category", "upper_fit", "upper_length", "neckline", "collar",
             "sleeve_length", "pattern", "material", "detail"]
BOTTOM_TASKS = ["lower_subtype", "lower_fit", "lower_length", "lower_detail",
                "pattern", "material", "detail"]


def load_derivation() -> dict:
    path = DATA_DIR / "catalog_derivation.json"
    if not path.is_file():
        raise SystemExit(f"유도 표가 없습니다: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def first_label(prediction) -> str:
    """AttributePrediction 에서 채택된 라벨 하나를 꺼낸다. 확신이 없으면 빈 값."""
    if prediction is None or not getattr(prediction, "accepted", False):
        return ""
    labels = getattr(prediction, "labels", None) or []
    return labels[0] if labels else ""


def clamp(value: int, low: int = 1, high: int = 5) -> int:
    return max(low, min(high, int(value)))


def derive(row: dict, predicted: dict[str, str], table: dict) -> dict:
    """사진에서 읽지 못하는 칼럼을 소재·종류에서 유도한다."""
    is_top = row.get("category") == "top"
    item_type = predicted.get("item_type", "")
    material = predicted.get("material", "")
    pattern = predicted.get("pattern", "") or "무지"

    mat = table["material"].get(material, {})
    pat = table["pattern"].get(pattern, table["pattern"]["무지"])

    # 격식: 소재 기준을 종류가 덮어쓴다(블레이저는 코튼이어도 격식이 높다).
    formality = table["item_formality"].get(item_type, mat.get("formality", 3))

    warmth = mat.get("warmth", 3)
    breathability = mat.get("breathability", 3)
    # 소매·기장이 체감에 크게 작용하는 부분만 보정한다.
    if predicted.get("_sleeve") == "민소매":
        warmth, breathability = clamp(warmth - 1), clamp(breathability + 1)
    elif predicted.get("_sleeve") == "긴팔":
        warmth = clamp(warmth + 1)
    if predicted.get("length") in ("쇼츠·미니 기장",):
        warmth, breathability = clamp(warmth - 1), clamp(breathability + 1)

    activity = list(table["activity_tags"].get(item_type, []))
    for tag in table["activity_by_material"].get(material, []):
        if tag not in activity:
            activity.append(tag)
    if not activity:
        activity = ["일상"]

    detail_label = predicted.get("_detail", "")
    detail_level = table["detail_level"].get(detail_label, 1)

    waistline = ""
    if not is_top:
        waistline = table["waistline"].get(predicted.get("_lower_detail", ""), "")

    visual_weight = pat["visual_weight"]
    if detail_level >= 4:
        visual_weight = clamp(visual_weight + 1)

    return {
        "formality": clamp(formality),
        "warmth": clamp(warmth),
        "breathability": clamp(breathability),
        "water_resistant": "true" if mat.get("water_resistant") else "false",
        "visual_weight": clamp(visual_weight),
        "detail_level": clamp(detail_level),
        "activity_tags": "|".join(activity),
        "waistline": waistline,
        "pattern_scale": pat["scale"],
        "pattern_contrast": pat["contrast"],
    }


def normalize(predicted: dict[str, str], table: dict) -> dict[str, str]:
    """모델 라벨을 규칙 엔진이 비교하는 어휘로 옮긴다.

    recommendation_engine.py 는 손으로 만든 카탈로그 표기({"긴바지", "롱·맥시 기장"} 등)로
    비교한다. 모델의 스키마 라벨("롱·긴바지 기장")을 그대로 넣으면 값이 채워져 있는데도
    조건에 안 걸려서, 규칙이 오류 없이 빗나간다.
    """
    # 종류가 기장을 확정하는 경우는 모델 판정보다 우선한다. 평면 상품컷에서는
    # 몸 대비 기장을 볼 수 없어 upper_length 헤드가 "롱 기장"을 사실상 내지 않는다.
    by_item = table.get("length_by_item_type", {}).get(predicted.get("item_type", ""))
    if by_item:
        predicted["length"] = by_item
    else:
        length = predicted.get("length", "")
        if length:
            predicted["length"] = table["normalize_length"].get(length, length)
    neckline = predicted.get("neckline", "")
    if neckline in table["normalize_neckline"]:
        predicted["neckline"] = table["normalize_neckline"][neckline]
    return predicted


def derive_purposes(item_type: str, formality: int, table: dict) -> str:
    """목적을 종류와 격식에서 유도한다.

    크롤러의 상품명 키워드 추측은 데일리·출근·데이트·여행만 낼 수 있어
    **면접·결혼식 상품이 0개**가 된다. 그 목적을 고르면 추천 엔진이
    "조건을 만족하는 상품이 없습니다" ValueError 를 낸다(재현 확인).
    """
    purposes = ["데일리"]
    for value in table.get("purpose_by_item_type", {}).get(item_type, []):
        if value not in purposes:
            purposes.append(value)
    for value in table.get("purpose_by_formality", {}).get(str(formality), []):
        if value not in purposes:
            purposes.append(value)
    return "|".join(purposes)


def derive_style(item_type: str, formality: int, fallback: str, table: dict) -> str:
    """스타일을 종류에서 유도한다. 모르면 크롤러가 상품명에서 고른 값을 쓴다.

    상품명 키워드만 쓰면 480개 중 410개(85%)가 캐주얼 기본값으로 떨어져,
    면접·결혼식이 요구하는 포멀·미니멀 후보가 사실상 사라진다.
    """
    style = table.get("style_by_item_type", {}).get(item_type)
    if style:
        return style
    if formality >= 5:
        return "포멀"
    return fallback or "캐주얼"


def derive_body_shapes(is_top: bool, out: dict, table: dict) -> str:
    """이 상품이 어느 실루엣 목표에 도움이 되는지 적는다.

    크롤러는 전 체형 허용으로 뭉뚱그려 넣어 변별력이 없었다. 시선을 끄는 상품은
    그 옷이 놓인 쪽으로 시선을 모으므로 상의면 상체, 하의면 하체 강조에 쓰인다.
    """
    cfg = table.get("body_shape_eye_catching", {})
    def as_int(name, default=0):
        try:
            return int(out.get(name, default))
        except (TypeError, ValueError):
            return default

    eye_catching = (
        as_int("visual_weight") >= cfg.get("visual_weight_min", 4)
        or as_int("detail_level") >= cfg.get("detail_level_min", 3)
        or out.get("color", "") in cfg.get("bold_colors", [])
        or (out.get("pattern", "무지") not in ("", "무지"))
    )
    shapes = ["균형형"]
    if eye_catching:
        shapes.insert(0, "상체 강조형" if is_top else "하체 강조형")
    return "|".join(shapes)


def musinsa_item_type(row: dict, table: dict) -> str:
    """무신사 카테고리 경로에서 item_type 을 뽑는다. 없으면 빈 값(모델 판정을 쓴다).

    판매자가 등록한 값이라 사진 판정보다 정확하다. 실측에서 모델은 상의 248개 중
    72.2%만 맞혔고, 특히 **재킷을 코트로 43회** 오판했다. 평면 상품컷에는 몸이 없어
    기장을 잴 수 없기 때문이다. 그 오판은 "코트면 롱 기장" 규칙까지 오염시킨다.
    """
    path = (row.get("detail_category") or "").strip()
    if not path:
        return ""
    for keyword, label in table.get("musinsa_item_type", []):
        if keyword in path:
            return label
    return ""


def apply_musinsa_facts(row: dict, out: dict, table: dict) -> list[str]:
    """무신사가 직접 표기한 값으로 덮어쓴다. 모델 추측보다 우선한다.

    상품명 키워드 추측은 색상 38%·계절 85%가 기본값으로 떨어졌다. 상세 API의
    값은 판매자가 등록한 사실이므로 그쪽을 믿는다.

    소재는 여기서 채울 수 없다. specDesc 가 "상세페이지참조"이고 혼용률 표기는
    상세 이미지 안에 있어 OCR 없이는 못 읽는다. 소재는 그대로 모델 추정이다.
    """
    used = []
    color = (row.get("detail_color") or "").strip()
    if color:
        for palette, words in table["musinsa_color"].items():
            if any(word.lower() in color.lower() for word in words):
                out["color"] = palette
                used.append("color")
                break

    season = (row.get("detail_season") or "").strip()
    if season in table["musinsa_season"]:
        out["season"] = table["musinsa_season"][season]
        used.append("season")

    fit = (row.get("detail_fit") or "").strip()
    if fit:
        out["fit"] = fit
        used.append("fit")

    thickness = (row.get("detail_thickness") or "").strip()
    if thickness in table["musinsa_thickness_warmth"]:
        out["warmth"] = table["musinsa_thickness_warmth"][thickness]
        used.append("warmth")

    sheer = (row.get("detail_sheer") or "").strip()
    if sheer in table["musinsa_sheer_breathability"]:
        out["breathability"] = table["musinsa_sheer_breathability"][sheer]
        used.append("breathability")
    return used


def predict_for(classifier, image_path: Path, is_top: bool) -> dict[str, str]:
    """상품 사진 한 장에서 카탈로그 칼럼에 쓸 라벨을 뽑는다."""
    tasks = TOP_TASKS if is_top else BOTTOM_TASKS
    out = classifier.predict_trained_attributes(image_path, tasks=tasks)
    get = lambda name: first_label(out.get(name))  # noqa: E731

    result = {
        "pattern": get("pattern") or "무지",
        "material": get("material"),
        "_detail": get("detail"),
    }
    if is_top:
        result["item_type"] = get("category")
        result["fit"] = get("upper_fit")
        result["length"] = get("upper_length")
        # 넥라인이 안 잡히면 칼라로 대신한다. 칼라가 있는 옷은 넥라인이 가려진다.
        result["neckline"] = get("neckline") or get("collar")
        result["_sleeve"] = get("sleeve_length")
    else:
        result["item_type"] = get("lower_subtype")
        result["fit"] = get("lower_fit")
        result["length"] = get("lower_length")
        result["neckline"] = ""  # 하의에 넥라인을 넣으면 카탈로그 검사가 깨진다
        result["_lower_detail"] = get("lower_detail")
    return result


def coverage_report(rows: list[dict]) -> list[tuple[bool, str]]:
    """규칙이 잠들지 않을 조건을 tests/test_product_catalog_coverage.py 기준으로 본다."""
    def vals(name):
        return {r.get(name, "") for r in rows}

    def num(name, threshold):
        out = []
        for r in rows:
            try:
                out.append(int(r.get(name, 0)))
            except (TypeError, ValueError):
                pass
        return [v for v in out if v >= threshold]

    patterns = {r.get("pattern", "") for r in rows}
    tops = [r for r in rows if r.get("category") == "top"]
    bottoms = [r for r in rows if r.get("category") == "bottom"]

    # 목적별 후보 — 웹이 제공하는 목적인데 상품이 없으면 추천 엔진이 ValueError 를 낸다.
    # tests/test_product_catalog_coverage.py 의 test_every_purpose_has_stocked_tops_and_bottoms
    # 와 같은 조건이다. 이 검사를 빠뜨렸다가 면접·결혼식이 0개인 카탈로그를 통과시킨 적이 있다.
    purpose_rows = []
    for purpose in ("데일리", "데이트", "출근", "면접", "결혼식", "여행"):
        has_top = sum(1 for r in tops if purpose in (r.get("purposes") or ""))
        has_bottom = sum(1 for r in bottoms if purpose in (r.get("purposes") or ""))
        purpose_rows.append(
            (has_top >= 3 and has_bottom >= 3,
             f"목적 '{purpose}' 상의 {has_top} / 하의 {has_bottom}")
        )

    # 실루엣 목표별 후보 — body_shapes 가 전부 같은 값이면 목표를 골라도 변별이 안 된다.
    shape_rows = []
    for shape, pool, label in (("상체 강조형", tops, "상의"), ("하체 강조형", bottoms, "하의")):
        n = sum(1 for r in pool if shape in (r.get("body_shapes") or ""))
        shape_rows.append((n >= 3, f"'{shape}' {label} {n}개 (실루엣 목표)"))

    return purpose_rows + shape_rows + [
        (len(rows) > 0, f"상품 {len(rows)}개"),
        (len(tops) >= 3 and len(bottoms) >= 3, f"상의 {len(tops)} / 하의 {len(bottoms)}"),
        (any(r.get("water_resistant") == "true" for r in rows), "방수 상품 존재 (R-WEA)"),
        (bool(num("warmth", 4)), "warmth>=4 존재 (R-WEA)"),
        (bool(num("breathability", 4)), "breathability>=4 존재 (R-WEA)"),
        (any("운동" in r.get("activity_tags", "") for r in rows), "운동 태그 존재 (R-WEA)"),
        ("무지" in patterns and len(patterns) >= 4, f"패턴 {len(patterns)}종 (무지 포함)"),
        (any(r.get("pattern", "무지") != "무지" for r in tops)
         and any(r.get("pattern", "무지") != "무지" for r in bottoms), "패턴 충돌 규칙 발동 가능"),
        ("하이웨이스트" in vals("waistline"), "하이웨이스트 존재 (실루엣 규칙)"),
        (any(r.get("length") == "롱 기장" for r in tops), "롱 기장 상의 존재 (R-ACC-05)"),
        (not any(r.get("neckline") for r in rows if r.get("category") != "top"),
         "하의에 넥라인 없음"),
        (sum(1 for r in tops if r.get("neckline")) >= max(3, len(tops) // 2),
         f"넥라인 있는 상의 {sum(1 for r in tops if r.get('neckline'))}/{len(tops)} (R-ACC-04)"),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="", help="크롤링 CSV (기본: data/products_musinsa.csv)")
    parser.add_argument("--output", default="", help="결과 CSV (기본: data/products_musinsa_enriched.csv)")
    parser.add_argument("--limit", type=int, default=0, help="앞 N개만 처리(동작 확인용)")
    parser.add_argument("--report", action="store_true", help="채우지 않고 결과 CSV의 규칙 발동 가능 여부만 점검")
    parser.add_argument(
        "--rederive", action="store_true",
        help="이미 채운 CSV에서 목적·스타일·체형만 다시 계산한다. 사진과 모델을 쓰지 않아 "
             "몇 초면 끝나고, 유도 표(catalog_derivation.json)만 고쳤을 때 쓴다",
    )
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    src = Path(args.input) if args.input else DATA_DIR / "products_musinsa.csv"
    dest = Path(args.output) if args.output else DATA_DIR / "products_musinsa_enriched.csv"

    if args.report:
        target = dest if dest.is_file() else src
        if not target.is_file():
            raise SystemExit(f"점검할 CSV가 없습니다: {target}")
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"점검 대상: {target}  ({len(rows)}행)\n")
        failed = 0
        for ok, label in coverage_report(rows):
            print(f"  {'OK  ' if ok else '부족'} {label}")
            failed += 0 if ok else 1
        print(f"\n{'전부 통과' if not failed else f'{failed}건 부족 — 해당 규칙은 후보를 못 찾습니다'}")
        return 0 if not failed else 1

    if args.rederive:
        target = dest if dest.is_file() else src
        if not target.is_file():
            raise SystemExit(f"다시 계산할 CSV가 없습니다: {target}")
        table = load_derivation()
        with target.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        for row in rows:
            try:
                formality = int(row.get("formality", 3))
            except (TypeError, ValueError):
                formality = 3
            item_type = row.get("item_type", "")
            row["purposes"] = derive_purposes(item_type, formality, table)
            row["style"] = derive_style(item_type, formality, row.get("style", ""), table)
            row["body_shapes"] = derive_body_shapes(row.get("category") == "top", row, table)
        with dest.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
            writer.writeheader()
            writer.writerows({f: r.get(f, "") for f in OUTPUT_FIELDS} for r in rows)
        print(f"목적·스타일·체형만 다시 계산했습니다: {dest} ({len(rows)}행)" + "\n")
        failed = 0
        for ok, label in coverage_report(rows):
            print(f"  {'OK  ' if ok else '부족'} {label}")
            failed += 0 if ok else 1
        summary = "전부 통과" if not failed else f"{failed}건 부족"
        print("\n" + summary)
        return 0 if not failed else 1

    if not src.is_file():
        raise SystemExit(
            f"크롤링 CSV가 없습니다: {src}\n"
            "  scripts/musinsa_crawler.py 로 만들거나 팀원에게 받아 넣으세요."
        )

    table = load_derivation()
    with src.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if args.limit:
        rows = rows[: args.limit]

    print(f"입력 {src.name}  {len(rows)}행")
    print(f"속성 헤드 {FASHION_ATTRIBUTE_HEADS_PATH.name}")

    from fashion_model import FashionClassifier

    classifier = FashionClassifier(enabled=True, device=args.device,
                                   attribute_checkpoint=FASHION_ATTRIBUTE_HEADS_PATH)
    if not classifier.trained_attributes_enabled:  # property다
        raise SystemExit("학습된 속성 헤드를 불러오지 못했습니다. 체크포인트 경로를 확인하세요.")

    enriched, skipped = [], 0
    fact_counts: dict[str, int] = {}
    for index, row in enumerate(rows, 1):
        image_name = row.get("image_path", "")
        image = garment_image_path(image_name) if image_name else None
        out = dict(row)
        if image and image.is_file():
            predicted = predict_for(classifier, image, row.get("category") == "top")
            # 무신사 카테고리가 있으면 item_type 을 먼저 사실로 교체한다.
            # derive()와 normalize()가 item_type 을 보고 판단하므로(격식·활동 태그·
            # "코트면 롱 기장") 반드시 그 앞에 와야 한다.
            fact_type = musinsa_item_type(row, table)
            if fact_type:
                predicted["item_type"] = fact_type
                fact_counts["item_type"] = fact_counts.get("item_type", 0) + 1
            # derive()는 모델 라벨 기준으로 판단하므로 유도 뒤에 정규화한다.
            derived = derive(row, predicted, table)
            predicted = normalize(predicted, table)
            out.update({k: v for k, v in predicted.items() if not k.startswith("_")})
            out.update(derived)
            # 무신사 표기값이 마지막에 온다. 모델 추측을 덮어쓰기 위해서다.
            for column in apply_musinsa_facts(row, out, table):
                fact_counts[column] = fact_counts.get(column, 0) + 1

            # 목적·스타일·체형 적합도는 위에서 확정된 종류·격식·시각 무게에 기대므로
            # 무신사 표기값 반영까지 끝난 뒤에 계산한다.
            is_top = row.get("category") == "top"
            try:
                formality = int(out.get("formality", 3))
            except (TypeError, ValueError):
                formality = 3
            out["purposes"] = derive_purposes(out.get("item_type", ""), formality, table)
            out["style"] = derive_style(out.get("item_type", ""), formality,
                                        row.get("style", ""), table)
            out["body_shapes"] = derive_body_shapes(is_top, out, table)
        else:
            skipped += 1
            print(f"  [{index}] 사진 없음 -> 건너뜀: {row.get('product_id')} ({image_name or '경로 없음'})")
        enriched.append({field: out.get(field, "") for field in OUTPUT_FIELDS})
        if index % 25 == 0:
            print(f"  {index}/{len(rows)}", flush=True)

    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(enriched)

    print(f"\n저장 {dest}  ({len(enriched)}행, 사진 없어 건너뛴 것 {skipped}건)")
    if fact_counts:
        detail = ", ".join(f"{k} {v}" for k, v in sorted(fact_counts.items()))
        print(f"무신사 표기값으로 채운 칸: {detail}  (나머지는 모델 추정)")
    else:
        print("무신사 표기값 없음 — 크롤러를 --skip-details 없이 다시 돌리면 색상·계절·핏이 정확해집니다.")
    print("\n규칙 발동 가능 여부:")
    failed = 0
    for ok, label in coverage_report(enriched):
        print(f"  {'OK  ' if ok else '부족'} {label}")
        failed += 0 if ok else 1
    if failed:
        print(f"\n{failed}건 부족합니다. 해당 규칙은 후보를 못 찾으니 크롤링 범위를 넓히세요.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
