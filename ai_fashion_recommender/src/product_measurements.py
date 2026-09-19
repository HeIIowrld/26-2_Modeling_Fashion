"""무신사 실측표와 판매 옵션을 수집한다. 추정 치수는 생성하지 않는다."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

from shopping_http import fetch_json


BASE_URL = "https://goods-detail.musinsa.com/api2/goods"
MEASUREMENT_FIELDS = {
    "총장": ("length_cm", "length"),
    "총기장": ("length_cm", "length"),
    "어깨너비": ("shoulder_cm", "width"),
    "어깨단면": ("shoulder_cm", "width"),
    "가슴단면": ("chest_width_cm", "flat_width"),
    "소매길이": ("sleeve_cm", "length"),
    "소매기장": ("sleeve_cm", "length"),
    "허리단면": ("waist_width_cm", "flat_width"),
    "엉덩이단면": ("hip_width_cm", "flat_width"),
    "허벅지단면": ("thigh_width_cm", "flat_width"),
    "밑위": ("rise_cm", "length"),
    "밑단단면": ("hem_width_cm", "flat_width"),
    "인심": ("inseam_cm", "length"),
}


def positive_cm(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or not 0 < number <= 250:
        return None
    rounded = round(number, 2)
    return rounded if rounded > 0 else None


def _label(value: object) -> str:
    return re.sub(r"\s+", "", str(value)).casefold()


def _rows(value: object) -> list[dict]:
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def normalize_size_table(product_id: str, actual: dict, options: dict | None = None) -> dict:
    """누락된 값은 보충하지 않고, 사이즈 이름이 정확히 일치할 때만 옵션을 연결한다."""
    data = actual.get("data") or {}
    if not isinstance(data, dict) or not isinstance(data.get("sizes", []), list):
        raise ValueError("실측표 형식 오류")
    # actual-size의 단위는 cm다. 다른 단위가 명시되면 자동 변환하지 않는다.
    unit = str(data.get("unit") or "cm").lower()
    issues = [] if unit == "cm" else ["unsupported_unit"]
    variants = []
    option_data = (options or {}).get("data") or {}
    for item in _rows(option_data.get("optionItems")) if isinstance(option_data, dict) else []:
        values = _rows(item.get("optionValues"))
        size_values = [v for v in values if str(v.get("optionName", "")).strip().lower() in {"사이즈", "size"}]
        size_label = str(size_values[0].get("name") or "") if len(size_values) == 1 else ""
        available = None
        if item.get("isDeleted") is True or item.get("activated") is False:
            available = False
        elif isinstance(item.get("isSoldOut"), bool):
            available = not item["isSoldOut"]
        elif isinstance(item.get("isOutOfStock"), bool):
            available = not item["isOutOfStock"]
        variants.append({
            "option_id": str(item.get("no") or ""),
            "size_label": size_label,
            "option_values": [{"name": str(v.get("optionName") or ""), "value": str(v.get("name") or "")} for v in values],
            "available": available,
        })
    sizes = []
    for row in data.get("sizes", []):
        if not isinstance(row, dict) or not str(row.get("name") or "").strip():
            continue
        measurements, kinds, conflicts = {}, {}, set()
        for item in _rows(row.get("items")):
            field = MEASUREMENT_FIELDS.get(str(item.get("name") or "").strip())
            value = positive_cm(item.get("value")) if unit == "cm" else None
            if field and value is not None:
                key, kind = field
                if key in conflicts:
                    continue
                if key in measurements and measurements[key] != value:
                    issues.append("conflicting_measurements")
                    conflicts.add(key)
                    measurements.pop(key, None)
                    kinds.pop(key, None)
                    continue
                measurements[key], kinds[key] = value, kind
        if not measurements:
            continue
        name = str(row["name"]).strip()
        matched = [v for v in variants if v["size_label"] and _label(v["size_label"]) == _label(name)]
        # 활성화된 옵션은 재고가 있다는 뜻이 아니다. 확인되지 않은 재고는 None.
        available = None
        if matched and all(v["available"] is False for v in matched):
            available = False
        elif any(v["available"] is True for v in matched):
            available = True
        sizes.append({"label": name, "measurements": measurements, "measurement_kinds": kinds,
                      "available": available, "option_ids": [v["option_id"] for v in matched]})
    return {
        "schema_version": 1, "product_id": product_id,
        "status": "ready" if sizes else "missing",
        "source": "musinsa_actual_size", "source_url": f"https://www.musinsa.com/products/{product_id.removeprefix('MS')}",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "unit": unit, "type_name": str(data.get("typeName") or ""),
        "measurement_note": str(data.get("description") or ""),
        "sizes": sizes, "variants": variants, "issues": list(dict.fromkeys(issues)),
    }


class ProductMeasurementClient:
    """원문과 정규화한 표를 상품별 JSON으로 캐시한다. 사용자 치수는 저장하지 않는다."""

    def __init__(self, cache_dir: Path | None = None, timeout: float = 2.0, cache_ttl: float = 3600.0):
        self.cache_dir = cache_dir
        self.timeout = timeout
        self.cache_ttl = cache_ttl
        self._cache: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def _fetch(self, number: str, resource: str) -> dict:
        return fetch_json(f"{BASE_URL}/{number}/{resource}", self.timeout)

    def get(self, product_id: str) -> dict:
        number = product_id.removeprefix("MS")
        if not re.fullmatch(r"[0-9]{1,12}", number):
            return {"status": "unavailable", "sizes": [], "issues": ["invalid_product_id"]}
        product_id = f"MS{number}"
        now = time.time()
        with self._lock:
            cached = self._cache.get(product_id)
        if cached and 0 <= now - cached[0] < (60 if cached[1]["status"] == "unavailable" else self.cache_ttl):
            return cached[1]
        path = self.cache_dir / f"{product_id}.json" if self.cache_dir else None
        if path:
            try:
                saved = json.loads(path.read_text(encoding="utf-8"))
                if (saved["record"]["schema_version"] == 1 and saved["record"]["product_id"] == product_id
                        and 0 <= now - saved["cached_at"] < self.cache_ttl):
                    return saved["record"]
            except (OSError, ValueError, KeyError, TypeError):
                pass
        actual, options = {}, None
        try:
            actual = self._fetch(number, "actual-size")
            try:
                options = self._fetch(number, "options")
            except (OSError, ValueError, TypeError):
                pass
            record = normalize_size_table(product_id, actual, options)
            if options is None:
                record["issues"].append("options_unavailable")
        except (OSError, ValueError, TypeError, KeyError):
            record = {"status": "unavailable", "sizes": [], "issues": ["fetch_failed"]}
        with self._lock:
            if len(self._cache) >= 512:
                self._cache.pop(next(iter(self._cache)))
            self._cache[product_id] = (now, record)
        if path and record["status"] != "unavailable":
            temporary = None
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as handle:
                    temporary = handle.name
                    json.dump({"cached_at": now, "record": record,
                               "raw": {"actual_size": actual.get("data"), "options": (options or {}).get("data")}},
                              handle, ensure_ascii=False, allow_nan=False)
                os.replace(temporary, path)
            except (OSError, ValueError):
                pass  # 읽기 전용 배포에서도 조회 결과는 사용할 수 있다.
            finally:
                if temporary:
                    try:
                        os.unlink(temporary)
                    except OSError:
                        pass
        return record
