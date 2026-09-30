"""잘 맞는 옷의 실측과 상품 사이즈표를 비교한다. 신체 치수를 추정하지 않는다."""

from __future__ import annotations

from datetime import datetime, timezone

from product_measurements import positive_cm


REFERENCE_FIELDS = {
    "top": {"chest_width_cm": "가슴단면", "length_cm": "총장"},
    "bottom": {"waist_width_cm": "허리단면", "length_cm": "총장"},
}
PRIMARY_FIELD = {"top": "chest_width_cm", "bottom": "waist_width_cm"}


def size_score(comparison: dict) -> float | None:
    """0–100 기준 옷 유사도. None은 착용 불가가 아니라 평가 정보 부족이다.

    단면 차이와 기장 차이를 섞어 작은 허리의 불일치를 숨기지 않는다.
    1cm 오차는 허용하고 단면 6cm/총장 11cm 차이부터 0점으로 둔다.
    이는 초기 순위 휴리스틱이며 실제 착용 확률이나 신체 여유량이 아니다.
    """
    if comparison.get("status") != "compared":
        return None
    values = [max(0.0, 100 * (1 - max(0.0, abs(d["delta_cm"]) - 1)
                              / (10 if d["field"] == "length_cm" else 5)))
              for d in comparison["differences"]]
    return round(min(values), 2) if values else None


def blend_size_score(base: float, comparisons: list[dict]) -> tuple[float, float | None, float]:
    """사이즈 축 15%. 미확인 슬롯은 중립 50점, 평가 범위는 별도 반환한다."""
    if not comparisons:
        return base, None, 0.0
    scores = [size_score(value) for value in comparisons]
    score = sum(50.0 if value is None else value for value in scores) / len(scores)
    coverage = sum(value is not None for value in scores) / len(scores)
    return 0.85 * base + 0.15 * score, score, coverage


def validate_references(value: object) -> dict[str, dict[str, float]]:
    if value in (None, "", {}):
        return {}
    if not isinstance(value, dict):
        raise ValueError("기준 옷의 실측 정보를 확인해주세요.")
    result = {}
    for category, fields in value.items():
        if category not in REFERENCE_FIELDS or not isinstance(fields, dict):
            raise ValueError("기준 옷은 상의와 하의의 실측으로 입력해주세요.")
        cleaned = {}
        for key, raw in fields.items():
            if key not in REFERENCE_FIELDS[category]:
                raise ValueError("지원하지 않는 실측 항목입니다.")
            if raw in (None, ""):
                continue
            number = positive_cm(raw)
            if number is None:
                raise ValueError("옷의 실측은 0보다 크고 250 이하인 cm 값으로 입력해주세요.")
            cleaned[key] = number
        if cleaned:
            result[category] = cleaned
    return result


def compare_sizes(record: dict, category: str, reference: dict[str, float] | None = None) -> dict:
    """확인된 항목만 비교한다. 누락 정보에는 부적합 판정을 내리지 않는다."""
    if record.get("fetched_at"):
        try:
            age = (datetime.now(timezone.utc) - datetime.fromisoformat(record["fetched_at"])).total_seconds()
        except (ValueError, TypeError):
            age = float("inf")
        if not 0 <= age < 3600:
            # 장시간 실행되는 엔진에서도 오래된 품절 상태가 영구히 남지 않게 한다.
            record = {**record,
                      "sizes": [{**s, "available": None} for s in record.get("sizes", [])],
                      "variants": [{**v, "available": None} for v in record.get("variants", [])]}
    reference = reference or {}
    result = {
        "status": "missing_measurements", "summary": "상품 실측 정보가 없어 사이즈를 비교할 수 없어요.",
        "closest_size": None, "differences": [], "alternatives": [],
        "source_url": record.get("source_url", ""), "fetched_at": record.get("fetched_at", ""),
        "measurement_note": record.get("measurement_note", ""),
        "columns": REFERENCE_FIELDS.get(category, {}),
        "reference": reference,
        "size_options": [{"size": s["label"], "measurements": s.get("measurements", {}),
                          "available": s.get("available")} for s in record.get("sizes", [])],
        "ranking_bonus": 0.0,
    }
    if record.get("status") == "unavailable":
        result.update(status="unavailable", summary="상품 실측을 불러오지 못했어요. 상품 페이지에서 확인해주세요.")
        return result
    sizes = record.get("sizes") or []
    if not sizes or category not in REFERENCE_FIELDS:
        return result
    if all(size.get("available") is False for size in sizes):
        if any(v.get("available") is not False for v in record.get("variants", [])):
            result.update(status="insufficient_measurements", summary="비교 가능한 사이즈는 품절이며, 다른 판매 옵션에는 비교할 실측이 없어요.")
        else:
            result.update(status="no_available_sizes", summary="확인한 실측 사이즈의 판매 옵션이 모두 비활성화되었거나 품절이에요.")
        return result
    if not reference:
        result.update(status="needs_reference", summary="잘 맞는 옷의 실측을 입력하면 사이즈별로 비교할 수 있어요.")
        return result
    candidates = []
    for size in sizes:
        if size.get("available") is False:
            continue
        differences = []
        for key, label in REFERENCE_FIELDS[category].items():
            mine = positive_cm(reference.get(key))
            theirs = positive_cm(size.get("measurements", {}).get(key))
            if mine is not None and theirs is not None:
                differences.append({"field": key, "label": label, "reference_cm": mine,
                                    "product_cm": theirs, "delta_cm": round(theirs - mine, 2)})
        if not differences:
            continue
        missing = [key for key in REFERENCE_FIELDS[category] if key not in {d["field"] for d in differences}]
        primary = any(d["field"] == PRIMARY_FIELD[category] for d in differences)
        weights = [2 if d["field"] == PRIMARY_FIELD[category] else 1 for d in differences]
        distance = sum(abs(d["delta_cm"]) * w for d, w in zip(differences, weights)) / sum(weights)
        candidates.append({"size": size["label"], "differences": differences, "missing_fields": missing,
                           "available": size.get("available"), "distance": distance, "primary": primary})
    if not candidates:
        if all(size.get("available") is False for size in sizes):
            result.update(status="no_available_sizes", summary="확인한 실측 사이즈의 판매 옵션이 모두 비활성화되었거나 품절이에요.")
        else:
            result.update(status="insufficient_measurements", summary="입력한 항목과 비교할 수 있는 상품 실측이 없어요.")
        return result
    # 총장 하나만 우연히 일치하는 행보다 입력 항목을 모두 갖춘 행을 먼저 비교한다.
    candidates.sort(key=lambda c: (len(c["missing_fields"]), not c["primary"],
                                   -size_score({"status": "compared", "differences": c["differences"]}),
                                   c["distance"]))
    best = candidates[0]
    complete = not best["missing_fields"] and best["primary"]
    result.update(
        status="compared" if complete else "partial",
        summary=f"기준 옷의 실측과 가장 가까운 사이즈는 {best['size']}예요." if complete else "일부 실측만 비교했어요. 사이즈 선택에는 추가 확인이 필요해요.",
        closest_size=best["size"] if complete else None,
        compared_size=best["size"], differences=best["differences"], missing_fields=best["missing_fields"],
        availability=best["available"],
        alternatives=[{key: c[key] for key in ("size", "differences", "available", "missing_fields")} for c in candidates],
    )
    # ranking_bonus는 기존 호출자 호환용이다. 실제 순위에는 size_score만 사용한다.
    if complete:
        result["ranking_bonus"] = round(max(0.0, 1.0 - best["distance"] / 10.0), 4)
        result["score"] = size_score(result)
        if result["score"] < 50:
            result["summary"] = f"가장 가까운 {best['size']}도 기준 옷과 실측 차이가 커요. 구매 전 상세 치수를 확인해주세요."
    result["summary"] += " 기준 옷과의 비교이며 실제 착용을 보장하지 않아요."
    if best["available"] is not True or len(record.get("color_options") or []) > 1:
        result["summary"] += " 원하는 색상·사이즈 옵션의 재고는 상품 페이지에서 확인해주세요."
    return result
