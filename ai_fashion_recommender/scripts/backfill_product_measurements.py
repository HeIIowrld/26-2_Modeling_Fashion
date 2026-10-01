"""오프라인 카탈로그용 실측 캐시를 순차 수집한다. CSV/사용자 치수는 변경하지 않는다."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.error
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from config import PRODUCTS_CSV
from product_measurements import BASE_URL, ProductMeasurementClient
from shopping_http import fetch_json


class BackfillClient(ProductMeasurementClient):
    """실시간 요청과 달리 수집 작업은 429/일시 오류를 기다렸다 재시도한다."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rate_limited = False
        self.errors = []

    def _fetch(self, number, resource):
        for attempt in range(3):
            try:
                try:
                    return super()._fetch(number, resource)
                except ValueError as exc:
                    if str(exc) != "상품 응답 크기 제한 초과":
                        raise
                    # 옵션 조합이 많은 상품은 2MB를 넘는다. 배치에서만 8MB까지 허용한다.
                    return fetch_json(f"{BASE_URL}/{number}/{resource}", self.timeout, max_bytes=8_000_000)
            except (OSError, ValueError) as exc:
                code = getattr(exc, "code", None)
                retryable = code in (429, 500, 502, 503, 504) or isinstance(exc, urllib.error.URLError) and code is None
                if not retryable or attempt == 2:
                    self.rate_limited = code == 429
                    self.errors.append({"product_id": f"MS{number}", "resource": resource,
                                        "error": type(exc).__name__, "http_status": code})
                    raise
                wait = 30 * (attempt + 1)
                print(f"RETRY MS{number} {resource} HTTP={code} wait={wait}s", flush=True)
                time.sleep(wait)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--input", type=Path, default=PRODUCTS_CSV)
    cli.add_argument("--limit", type=int, default=20)
    cli.add_argument("--offset", type=int, default=0)
    cli.add_argument("--category", choices=("top", "bottom"))
    cli.add_argument("--delay", type=float, default=1.0)
    cli.add_argument("--report", type=Path, help="수집 결과 JSON 저장 위치")
    args = cli.parse_args()
    if args.limit < 1 or args.offset < 0 or args.delay < 1:
        cli.error("limit >= 1, offset >= 0, delay >= 1 이어야 합니다.")
    with args.input.open(encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle)
                if not args.category or row["category"] == args.category]
    rows = rows[args.offset:args.offset + args.limit]
    # 하루 안에 완료된 수집은 재사용한다. 런타임의 재고 유효기간(1시간)은 변경하지 않는다.
    client = BackfillClient(args.input.parent / "cache" / "product_measurements", timeout=5, cache_ttl=86400)
    ready = 0
    results = []
    for index, row in enumerate(rows):
        result = client.get(row["product_id"])
        results.append({"product_id": row["product_id"], "status": result["status"],
                        "issues": result.get("issues", []), "sizes": len(result.get("sizes", []))})
        print(f"{row['product_id']}: {result['status']}", flush=True)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            temporary = args.report.with_suffix(".tmp")
            temporary.write_text(json.dumps({"total": len(rows), "completed": len(results),
                                             "counts": dict(Counter(r["status"] for r in results)),
                                             "errors": client.errors, "results": results},
                                            ensure_ascii=False, indent=2), encoding="utf-8")
            temporary.replace(args.report)
        if client.rate_limited:
            print("429 재시도 한도 초과: 추가 수집을 중단합니다. 저장된 실측은 유지됩니다.", flush=True)
            return 1
        ready += result["status"] == "ready"
        if index + 1 < len(rows):
            time.sleep(args.delay)
    print(f"실측 확보 {ready}/{len(rows)}")
    return int(any(r["status"] == "unavailable" or "options_unavailable" in r["issues"] for r in results))


if __name__ == "__main__":
    raise SystemExit(main())
