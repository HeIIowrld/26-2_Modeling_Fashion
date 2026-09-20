"""잘 맞는 옷의 실측과 상품 사이즈표를 비교한다. 신체 치수를 추정하지 않는다."""

from __future__ import annotations

from product_measurements import positive_cm


REFERENCE_FIELDS = {
    "top": {"chest_width_cm": "가슴단면", "length_cm": "총장"},
    "bottom": {"waist_width_cm": "허리단면", "length_cm": "총장"},
}
PRIMARY_FIELD = {"top": "chest_width_cm", "bottom": "waist_width_cm"}


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
    candidates.sort(key=lambda c: (len(c["missing_fields"]), not c["primary"], c["distance"]))
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
    # 실측 유사도의 작은 보조점수일 뿐 착용 성공 확률은 아니다.
    # 미측정/부분 측정은 0점, 큰 차이도 감점 없이 두어 누락과 부적합을 구분한다.
    if complete:
        result["ranking_bonus"] = round(max(0.0, 1.0 - best["distance"] / 10.0), 4)
    return result
