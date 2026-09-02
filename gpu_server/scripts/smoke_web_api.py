"""실행 중인 FITTA 웹 API의 인식·추천·VTON을 실제 이미지로 종단간 점검한다."""

from __future__ import annotations

import argparse
import json
import mimetypes
import time
import urllib.error
import urllib.request
from pathlib import Path


def _json_request(url: str, *, method: str = "GET", data: bytes | None = None, headers=None) -> dict:
    request = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"{method} {url}: HTTP {error.code} {detail}") from error


def _multipart(image_path: Path, profile: dict) -> tuple[bytes, str]:
    boundary = "----fitta-real-smoke-boundary"
    image = image_path.read_bytes()
    image_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    profile_bytes = json.dumps(profile, ensure_ascii=False).encode("utf-8")
    body = b"".join(
        (
            f"--{boundary}\r\n".encode(),
            f'Content-Disposition: form-data; name="image"; filename="{image_path.name}"\r\n'.encode(),
            f"Content-Type: {image_type}\r\n\r\n".encode(),
            image,
            b"\r\n",
            f"--{boundary}\r\n".encode(),
            b'Content-Disposition: form-data; name="profile"\r\n\r\n',
            profile_bytes,
            b"\r\n",
            f"--{boundary}--\r\n".encode(),
        )
    )
    return body, f"multipart/form-data; boundary={boundary}"


def _download(url: str) -> tuple[str, bytes]:
    with urllib.request.urlopen(url, timeout=120) as response:
        return response.headers.get_content_type(), response.read()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--keep", action="store_true", help="검증 뒤 서버의 임시 사진을 남긴다")
    args = parser.parse_args()
    base_url = args.base_url.rstrip("/")

    health = _json_request(f"{base_url}/api/health")
    print("health", json.dumps(health, ensure_ascii=False))
    if health.get("device") != "cuda" or not health.get("vton_enabled"):
        raise RuntimeError("CUDA와 VTON이 모두 활성화된 서버가 아닙니다.")
    if health.get("product_color_audits") != health.get("product_count"):
        raise RuntimeError("전체 상품 이미지 색상 점검 파일이 적용되지 않았습니다.")
    if not health.get("product_color_overrides"):
        raise RuntimeError("검증된 상품 색상 교정값이 적용되지 않았습니다.")

    profile = {
        "purpose": "데일리",
        "desired_style": "캐주얼",
        "change_scope": "전체 변경",
        "min_budget": 50_000,
        "max_budget": 250_000,
        "budget": 150_000,
        "season": "자동",
        "activity_level": "보통",
    }
    body, content_type = _multipart(args.image, profile)
    created = _json_request(
        f"{base_url}/api/analyze",
        method="POST",
        data=body,
        headers={"Content-Type": content_type},
    )
    job_id = created["job_id"]
    print("job", job_id)

    deadline = time.monotonic() + args.timeout
    seen = set()
    while time.monotonic() < deadline:
        state = _json_request(f"{base_url}/api/jobs/{job_id}")
        marker = (state.get("status"), state.get("stage"))
        if marker not in seen:
            print("progress", *marker)
            seen.add(marker)
        if state["status"] != "running":
            break
        time.sleep(1)
    else:
        raise TimeoutError(f"분석이 {args.timeout}초 안에 끝나지 않았습니다.")

    if state["status"] != "done":
        raise RuntimeError(f"분석 실패: {state.get('error')}")
    expected_stages = [
        "prepare", "wardrobe", "pose", "quality", "body", "segment",
        "attributes", "candidates", "scoring", "preview", "finalize",
    ]
    if state.get("stage_history") != expected_stages:
        raise RuntimeError(f"진행 단계 순서가 다릅니다: {state.get('stage_history')!r}")
    result = state["result"]
    shopping_results = result.get("shopping_results") or []
    if not shopping_results:
        raise RuntimeError("무신사 실시간 검색 결과가 없습니다.")
    echoed = result.get("request") or {}
    for key in ("purpose", "desired_style", "change_scope", "min_budget", "max_budget"):
        if echoed.get(key) != profile[key]:
            raise RuntimeError(f"요청 조건이 결과에 다르게 기록됐습니다: {key}={echoed.get(key)!r}")
    print(
        "analysis",
        json.dumps(
            {
                "pose_valid": result.get("pose", {}).get("valid"),
                "parser": result.get("engine", {}).get("parser_backend"),
                "shopping_results": len(shopping_results),
            },
            ensure_ascii=False,
        ),
    )

    for name in result.get("images", {}).values():
        content_type, image = _download(f"{base_url}/api/jobs/{job_id}/images/{name}")
        print("image", name, content_type, len(image))
        if not image:
            raise RuntimeError(f"결과 이미지가 비어 있습니다: {name}")

    available_by_category = {"top": [], "bottom": []}
    selected_by_category = {}
    for product in shopping_results:
        if "tryon_available" not in product or "tryon_reason" not in product:
            raise RuntimeError("무신사 상품의 합성 가능 여부가 응답에 없습니다.")
        if product.get("tryon_available") and product.get("category") in {"top", "bottom"}:
            available_by_category[product["category"]].append(product)
            selected_by_category.setdefault(product["category"], product)
    if set(selected_by_category) != {"top", "bottom"}:
        raise RuntimeError(f"합성 가능한 무신사 상의·하의가 모두 없습니다: {shopping_results!r}")

    expected_combinations = [
        [top["product_id"], bottom["product_id"]]
        for top in available_by_category["top"]
        for bottom in available_by_category["bottom"]
    ]
    shopping_batch = _json_request(
        f"{base_url}/api/jobs/{job_id}/shopping-tryon-batch",
        method="POST",
        data=b"",
    )
    while shopping_batch.get("status") in {"queued", "running"} and time.monotonic() < deadline:
        print(
            "shopping-tryon-batch",
            shopping_batch.get("status"),
            f"{shopping_batch.get('ready')}/{shopping_batch.get('total')}",
        )
        time.sleep(1)
        shopping_batch = _json_request(
            f"{base_url}/api/jobs/{job_id}/shopping-tryon-batch"
        )
    if shopping_batch.get("status") != "done":
        raise RuntimeError(f"무신사 전체 조합 배치가 완료되지 않았습니다: {shopping_batch!r}")
    actual_combinations = [
        item.get("product_ids") for item in shopping_batch.get("items") or []
    ]
    if actual_combinations != expected_combinations:
        raise RuntimeError(
            f"무신사 전체 조합이 일치하지 않습니다: expected={expected_combinations!r}, "
            f"actual={actual_combinations!r}"
        )
    for item in shopping_batch.get("items") or []:
        if item.get("status") != "done" or not item.get("image"):
            raise RuntimeError(f"완료되지 않은 무신사 조합이 있습니다: {item!r}")
        content_type, image = _download(
            f"{base_url}/api/jobs/{job_id}/images/{item['image']}"
        )
        if content_type != "image/jpeg" or not image.startswith(b"\xff\xd8\xff"):
            raise RuntimeError(f"무신사 조합 {item['index']} 결과가 유효한 JPEG가 아닙니다.")
        print(
            "shopping-tryon-item",
            item["index"],
            "+".join(item["product_ids"]),
            len(image),
        )

    selected_products = [selected_by_category["top"], selected_by_category["bottom"]]
    request_body = json.dumps(
        {"product_ids": [product["product_id"] for product in selected_products]}
    ).encode("utf-8")
    product_tryon = _json_request(
        f"{base_url}/api/jobs/{job_id}/tryon-products",
        method="POST",
        data=request_body,
        headers={"Content-Type": "application/json"},
    )
    content_type, image = _download(
        f"{base_url}/api/jobs/{job_id}/images/{product_tryon['image']}"
    )
    if content_type != "image/jpeg" or not image.startswith(b"\xff\xd8\xff"):
        raise RuntimeError("선택한 무신사 상·하의 합성 결과가 유효한 JPEG가 아닙니다.")
    if product_tryon.get("categories") != ["top", "bottom"]:
        raise RuntimeError(f"무신사 상품 합성 순서가 다릅니다: {product_tryon!r}")
    if not isinstance(product_tryon.get("warnings"), list):
        raise RuntimeError("무신사 상품 합성 품질 경고 목록이 없습니다.")
    print(
        "shopping-tryon",
        "+".join(product_tryon["product_ids"]),
        product_tryon["image"],
        len(image),
    )
    cached_product_tryon = _json_request(
        f"{base_url}/api/jobs/{job_id}/tryon-products",
        method="POST",
        data=request_body,
        headers={"Content-Type": "application/json"},
    )
    if not cached_product_tryon.get("cached") or cached_product_tryon["image"] != product_tryon["image"]:
        raise RuntimeError("같은 무신사 상품 조합이 캐시되지 않았습니다.")

    if not args.keep:
        deleted = _json_request(f"{base_url}/api/jobs/{job_id}", method="DELETE")
        print("delete", deleted.get("deleted"))
    print("SMOKE_OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
