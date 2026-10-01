"""상품이 표기한 사이즈 실측을 화면에 쓸 형태로 정리한다.

2026-10-01 팀 결정으로 **기준 옷 실측 입력 기능을 없앴다.** 사용자가 잘 맞는 옷의
단면·총장을 넣으면 상품 사이즈표와 비교해 주던 기능인데, 유의미하지 않다고 판단했다.

그래서 이 모듈에서 비교와 점수가 모두 빠졌다:
  · `validate_references` — 입력 검증. 입력받을 곳이 없어졌다.
  · `size_score` / `blend_size_score` — 기준 옷과의 유사도 0~100과 15% 배점.
    기준이 없으면 계산할 수 없다.
  · 가장 가까운 사이즈(`closest_size`)와 부위별 차이(`differences`).

남긴 것은 **상품이 스스로 표기한 실측표**다. 기준 옷과 무관하게, 구매 전에 치수를
확인하려는 사용자에게 그대로 쓸모가 있다. 신체 치수는 예나 지금이나 추정하지 않는다.
"""

from __future__ import annotations

# 화면 표의 열. 상의는 가슴단면·총장, 하의는 허리단면·총장만 보여 준다.
MEASUREMENT_FIELDS = {
    "top": {"chest_width_cm": "가슴단면", "length_cm": "총장"},
    "bottom": {"waist_width_cm": "허리단면", "length_cm": "총장"},
}


def size_table(record: dict | None, category: str) -> dict:
    """상품 실측표를 화면용으로 정리한다.

    실측이 없다고 해서 부적합 판정을 내리지 않는다 — 모르는 것과 안 맞는 것은 다르다.
    표를 만들 수 없으면 빈 `size_options` 를 돌려주고, 화면은 그 경우 아무것도 그리지 않는다.
    """
    record = record or {}
    columns = MEASUREMENT_FIELDS.get(category, {})
    sizes = record.get("sizes") or []
    if record.get("status") == "unavailable":
        status = "unavailable"
    elif sizes and columns:
        status = "available"
    else:
        status = "missing_measurements"
    return {
        "status": status,
        "columns": columns,
        "size_options": [
            {"size": size["label"], "measurements": size.get("measurements", {})}
            for size in sizes
        ],
        "measurement_note": record.get("measurement_note", ""),
        "source_url": record.get("source_url", ""),
    }
