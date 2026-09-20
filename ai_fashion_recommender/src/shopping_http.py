"""상품 조회의 응답 크기와 대기 시간을 제한하는 공용 함수."""

from __future__ import annotations

import json
import time
import urllib.request
from concurrent.futures import Executor, wait
from typing import Callable, Iterable, TypeVar


HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; FITTA/1.0; product-search)",
    "Accept": "application/json",
    "Referer": "https://www.musinsa.com/",
    "Accept-Language": "ko-KR,ko;q=0.9",
}
T = TypeVar("T")


def fetch_json(url: str, timeout: float = 3.0) -> dict:
    request = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read(2_000_001)
    if len(raw) > 2_000_000:
        raise ValueError("상품 응답 크기 제한 초과")
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise ValueError("상품 응답 형식 오류")
    meta = payload.get("meta") or {}
    if not isinstance(meta, dict) or meta.get("result", "SUCCESS") != "SUCCESS":
        raise ValueError("상품 조회 실패")
    return payload


def bounded_results(
    executor: Executor, calls: Iterable[Callable[[], T]], deadline: float,
) -> list[T | None]:
    """완료 순서와 무관하게 입력 순서를 유지한다. 제한 시간 뒤 작업은 취소한다."""
    futures = [executor.submit(call) for call in calls] if time.monotonic() < deadline else []
    if not futures:
        return []
    done, pending = wait(futures, timeout=max(0.0, deadline - time.monotonic()))
    for future in pending:
        future.cancel()
    results = []
    for future in futures:
        try:
            results.append(future.result() if future in done else None)
        except (OSError, ValueError, TypeError, KeyError):
            results.append(None)
    return results
