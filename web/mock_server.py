"""FITTA 프론트엔드 시연용 경량 mock API 서버.

실제 모델 패키지가 없는 환경에서도 브라우저의 전체 흐름을 검토할 수 있도록
``web/app.py``와 같은 API 경로를 표준 라이브러리만으로 제공한다.

    python3 web/mock_server.py --host 0.0.0.0 --port 8000

업로드한 이미지는 메모리에만 보관하고 30분 뒤 또는 사용자가 삭제 버튼을 누를 때
제거한다. 분석 수치와 추천 결과는 시연용 데이터이며 실제 모델 추론값이 아니다.
"""

from __future__ import annotations

import argparse
import base64
import html
import json
import re
import threading
import time
import uuid
import warnings
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


warnings.filterwarnings("ignore", category=DeprecationWarning)
from cgi import FieldStorage, parse_header  # noqa: E402  # Python 3.12 표준 라이브러리

WEB_DIR = Path(__file__).resolve().parent
STATIC_DIR = WEB_DIR / "static"
OPTIONS = json.loads((STATIC_DIR / "fallback-options.json").read_text(encoding="utf-8"))
STAGES = [item["key"] for item in OPTIONS["stages"]]
SESSION_TTL_SECONDS = 30 * 60
MAX_REQUEST_BYTES = 40 * 1024 * 1024

JOBS: dict[str, dict] = {}
JOBS_LOCK = threading.Lock()
SAFE_JOB_ID = re.compile(r"^[0-9a-f]{32}$")
SAFE_IMAGE_NAME = re.compile(r"^[a-z0-9_-]+$")

COLOR_RGB = {
    "블랙": [25, 25, 25],
    "화이트": [235, 235, 235],
    "그레이": [130, 130, 130],
    "네이비": [35, 50, 90],
    "블루": [55, 110, 190],
    "브라운": [115, 75, 45],
    "베이지": [205, 185, 145],
    "레드": [185, 45, 45],
    "핑크": [220, 125, 155],
    "그린": [60, 130, 75],
    "카키": [105, 105, 55],
    "버건디": [115, 35, 55],
}

RULE_TITLES = {
    "R-CTX-01": "코디 목적과 격식 수준 일치",
    "R-CTX-02": "상·하의 격식도 조화",
    "R-COL-03": "상·하의 색상 관계",
    "R-SIL-01": "상·하의 실루엣 균형",
    "R-PAT-01": "패턴 시선 경쟁 방지",
    "R-MAT-02": "소재 질감 연결",
    "R-BUD-01": "사용자 예산 범위 준수",
    "R-ACT-01": "활동량에 맞는 착용감",
}


def _number(value, default: int) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _sniff_mime(data: bytes, declared: str) -> str:
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return declared if declared in {"image/jpeg", "image/png", "image/webp"} else "image/jpeg"


def _prune_jobs() -> None:
    deadline = time.monotonic() - SESSION_TTL_SECONDS
    with JOBS_LOCK:
        expired = [job_id for job_id, job in JOBS.items() if job["created"] < deadline]
        for job_id in expired:
            JOBS.pop(job_id, None)


MOCK_BOTTOM_LENGTHS = ("반바지", "무릎 기장", "7부 기장", "긴바지")
MOCK_LENGTH_HOLD = "하의 기장 비교를 보류했습니다. 밑단이 보이게 찍은 전신사진을 쓰거나, 지금 입은 하의 기장을 알려주시면 비교할 수 있어요."


MOCK_LENGTH_PROMPT = (
    "사진에서 하의 밑단이 잘려 지금 입은 옷의 기장을 확인하지 못했어요. "
    "기장을 알려주시면 기장 차이로 합성이 어색해질 조합을 미리 표시해요. "
    "밑단이 보이게 다시 찍으면 자동으로 판정해요."
)


# 조건 입력 전 사진 검사(/api/validate-photo). 파일 이름으로 판정을 고를 수 있게 해서
# 프런트에서 차단·경고·통과 세 경로를 모두 눌러볼 수 있게 한다.
MOCK_PREFLIGHT = {
    "skirt": (True, [], ["치마·원피스가 골반과 다리 윤곽을 가려 사진 기반 체형 분석은 보류합니다.",
                          "현재 사진으로 확인 가능한 착장 분석과 추천은 계속합니다."]),
    "loose": (True, [], ["하의 윤곽과 몸선을 구분하기 어렵습니다. 옷의 폭을 체형으로 사용하지 않습니다.",
                         "옷 때문에 체형 판정이 정확하지 않을 수 있습니다. 실제 둘레를 입력하면 그 값을 우선 사용합니다."]),
    "blur": (False, ["사진에서 사람의 정면 전신을 확인할 수 없습니다. 정면 전신사진을 사용하세요."], []),
}


def _mock_preflight(filename: str) -> dict:
    valid, issues, warnings = True, [], []
    for keyword, verdict in MOCK_PREFLIGHT.items():
        if keyword in filename.lower():
            valid, issues, warnings = verdict
            break
    visibility_status = (
        "occluded" if "skirt" in filename.lower()
        else "uncertain" if warnings else "no_obvious_occlusion"
    )
    return {"valid": valid, "issues": issues, "warnings": warnings,
            "quality": {"passed": valid, "issues": issues, "warnings": warnings,
                        "body_visibility": {"status": visibility_status,
                                            "passed": valid, "calibrated": False}}}


def _mock_length_warnings(job: dict, categories: list[str]) -> list[str]:
    """실제 서버의 기장 경고 흐름을 흉내 낸다.

    시연 상품의 하의는 모두 긴바지라 현재 기장이 무엇이든 '더 짧아지는' 경고는 없다.
    """
    if "bottom" not in categories or job.get("user_bottom_length"):
        return []
    return [MOCK_LENGTH_HOLD]


def _mock_shopping_tryon_batch(job: dict) -> dict:
    """추천 코디 하나에 합성 한 장. 실제 서버와 같은 단위로 배치를 재현한다."""
    by_id = {item["product_id"]: item for item in job["result"].get("shopping_results", [])}
    combinations = [
        [by_id[product_id] for product_id in outfit.get("product_ids", []) if product_id in by_id]
        for outfit in job["result"].get("shopping_outfits", [])
    ]
    combinations = [products for products in combinations if products]
    elapsed = max(0.0, time.monotonic() - job["created"] - 3.2)
    ready = min(len(combinations), int(elapsed / 0.9))
    items = []
    for index, products_for_look in enumerate(combinations, start=1):
        if index <= ready:
            status, image = "done", f"tryon-products-{index}"
        elif index == ready + 1:
            status, image = "running", None
        else:
            status, image = "queued", None
        items.append(
            {
                "index": index,
                "product_ids": [item["product_id"] for item in products_for_look],
                "categories": [item["category"] for item in products_for_look],
                "status": status,
                "image": image,
                "cached": False,
                "warnings": _mock_length_warnings(job, [item["category"] for item in products_for_look])
                if status == "done" else [],
                "error": None,
            }
        )
    total = len(items)
    return {
        "status": "done" if ready == total else "running",
        "reason": "",
        "total": total,
        "ready": ready,
        "finished": ready,
        "items": items,
    }


def _build_result(profile: dict, image_seed: int) -> dict:
    purpose = str(profile.get("purpose") or "데일리")
    style = str(profile.get("desired_style") or "미니멀")
    preferred = [str(value) for value in profile.get("preferred_colors") or []]
    palette = preferred[:2] + ["네이비", "그레이", "베이지", "블랙"]
    palette = list(dict.fromkeys(palette))
    while len(palette) < 4:
        palette.append("그레이")

    body_shapes = ["역삼각체형", "사각체형", "삼각체형"]
    body_shape = body_shapes[image_seed % len(body_shapes)]
    sources = {
        key: "trained_head"
        for key in (
            "upper_type", "layering_state", "sleeve_length", "sleeve_state", "sleeve_shape",
            "upper_length", "neckline", "collar", "fit", "pattern", "material", "silhouette",
            "details", "lower_type", "lower_subtype", "pant_leg_shape", "pant_length",
            "lower_fit", "lower_pattern", "lower_material", "lower_details",
        )
    }
    outfit = {
        "parser_backend": "mock-parser",
        "upper_color": "네이비",
        "lower_color": "그레이",
        "color_harmony": "안정적인 무채색 조합",
        "detected_items": ["셔츠", "팬츠"],
        "style": style,
        "upper_style": style,
        "lower_style": style,
        "upper_style_confidence": 0.89,
        "lower_style_confidence": 0.87,
        "upper_type": "셔츠",
        "lower_type": "팬츠",
        "lower_subtype": "슬랙스",
        "pant_leg_shape": "스트레이트",
        "pant_length": "발목 기장",
        "sleeve_length": "긴소매",
        "visible_sleeve_length": "긴소매",
        "sleeve_state": "정상 착용",
        "input_valid": True,
        "input_error_code": "",
        "input_error_message": "",
        "layering_state": "단일 상의",
        "upper_items": ["셔츠"],
        "inner_category": "해당 없음",
        "outer_category": "해당 없음",
        "wear_state_confidence": {"sleeve": 0.9},
        "upper_length": "기본 기장",
        "bottom_length": "발목 기장",
        "fit": "릴랙스드핏",
        "lower_fit": "세미와이드핏",
        "neckline": "칼라넥",
        "pattern": "무지",
        "material": "코튼",
        "lower_pattern": "무지",
        "lower_material": "코튼 혼방",
        "sleeve_shape": "기본 소매",
        "collar": "셔츠 칼라",
        "silhouette": "H라인",
        "details": ["단추"],
        "lower_details": ["원턱"],
        "attribute_sources": sources,
        "attribute_confidence": 0.88,
        "notes": ["목업 분석 결과입니다."],
    }
    result = {
        "mock": True,
        "input_quality": {"passed": True, "issues": [], "score": 0.94},
        "pose": {
            "valid": True,
            "full_body_score": 0.94,
            "body_shape": body_shape,
            "shoulder_hip_ratio": round(1.36 + (image_seed % 12) / 100, 2),
            "upper_lower_ratio": 0.96,
            "leg_ratio": 0.52,
            "posture": "정면에 가까움",
            "body_shape_confidence": 0.84,
            "warnings": [],
            "body_shape_basis": "사진 추정",
        },
        "outfit": outfit,
        "outfit_summary": {
            "상의": "네이비 셔츠 (긴소매)",
            "하의": "그레이 슬랙스 (스트레이트, 발목 기장)",
            "신발": "신발 인식 학습 준비 중 · 입력 조건으로 추천 가능",
        },
        "current_outfit_evaluation": {
            "total_score": 86.4,
            "verdict": "추천 코디로 보완할 수 있어요",
            "reliable": True,
            "keep_threshold": 85.0,
            "diagnostic_matrix": {
                "top": {"body_fit": 88.0, "situation_fit": 91.0, "style_fit": 87.0},
                "bottom": {"body_fit": 84.0, "situation_fit": 89.0, "style_fit": 86.0},
            },
            "pass_matrix": {
                "top": {"body_fit": True, "situation_fit": True, "style_fit": True},
                "bottom": {"body_fit": False, "situation_fit": True, "style_fit": True},
            },
            "harmony_score": 88.0,
            "summary_points": [
                "상의: 네이비 셔츠 — 체형 적합도 88점 · 상황 적합도 91점 · 스타일 적합도 87점.",
                "하의: 그레이 슬랙스 — 체형 적합도 84점 · 상황 적합도 89점 · 스타일 적합도 86점.",
                "상·하의 조화는 88점이며, 추천 코디로 보완할 수 있어요.",
            ],
        },
        "shopping_results": [
            {
                "product_id": "MSMOCKTOP1",
                "name": "오버핏 코튼 셔츠",
                "brand": "MUSINSA MOCK",
                "price": 59_000,
                "image_url": "https://image.msscdn.net/thumbnails/images/prd_img/202608/mock_top.jpg",
                "url": "https://www.musinsa.com/products/1000001",
                "category": "top",
                "gender": "공용",
                "review_count": 1240,
                "review_score": 94,
                "source": "mock",
                "search_keywords": ["여유핏", "코튼", "캐주얼"],
                "recommendation_reason": "추천 규칙에서 도출된 여유핏 조건을 충족해 추천했어요.",
                "recommendation_reason_source": "rules",
                "fit_evidence": ["추천 규칙에서 도출된 '여유핏' 핏 조건을 충족합니다.", "현재 착장에서 확인된 코튼 소재 조건에 맞습니다."],
                "fit_evidence_labels": ["핏", "소재"],
                "reason_rule_ids": ["R-SIL-01", "R-MAT-01"],
                "tryon_available": True,
                "tryon_reason": "",
            },
            {
                "product_id": "MSMOCKBOTTOM1",
                "name": "세미 와이드 데님 팬츠",
                "brand": "MUSINSA MOCK",
                "price": 69_000,
                "image_url": "https://image.msscdn.net/thumbnails/images/prd_img/202608/mock_bottom.jpg",
                "url": "https://www.musinsa.com/products/1000002",
                "category": "bottom",
                "gender": "공용",
                "review_count": 830,
                "review_score": 96,
                "source": "mock",
                "search_keywords": ["세미와이드", "데님", "풀렝스"],
                "recommendation_reason": "추천 규칙에서 확인된 세미와이드 핏 조건을 충족해 추천했어요.",
                "recommendation_reason_source": "rules",
                "fit_evidence": ["추천 규칙에서 도출된 '세미와이드' 핏 조건을 충족합니다.", "현재 착장에서 확인된 데님 소재 조건에 맞습니다."],
                "fit_evidence_labels": ["핏", "소재"],
                "reason_rule_ids": ["R-SIL-01", "R-MAT-01"],
                "tryon_available": True,
                "tryon_reason": "",
            },
        ],
        "rules": {
            "implemented": 43,
            "documented": 50,
            "scoring": 37,
            "unsupported": [
                {"id": "R-SIL-02", "reason": "상품 실측 사이즈 데이터가 필요합니다."},
                {"id": "R-ACC-01", "reason": "액세서리 카탈로그가 필요합니다."},
            ],
        },
        "engine": {
            "device": "mock-cpu",
            "trained_heads": True,
            "parser_backend": "mock-parser",
            "vton_enabled": True,
        },
        "tryon": {"available": True, "reason": "", "warnings": []},
        # 시연용: 사진에서 바지 밑단이 잘려 현재 기장을 확인하지 못한 상태를 보여준다.
        "length_check": {
            "status": "needs_input",
            "value": "",
            "options": list(MOCK_BOTTOM_LENGTHS),
            "message": MOCK_LENGTH_PROMPT,
        },
        "images": {
            "original": "original",
            "landmarks": "landmarks",
            "segmentation": "segmentation",
        },
    }
    categories = profile.get("change_categories")
    if categories is None:
        categories = {"상의만 변경": ["top"], "하의만 변경": ["bottom"],
                      "현재 유지": []}.get(profile.get("change_scope"), ["top", "bottom"])
    templates = {item["category"]: item for item in result["shopping_results"]}
    templates["shoes"] = {
        **templates["top"], "name": "스니커즈 UI 확인용 샘플", "category": "shoes",
        "search_keywords": ["스니커즈", style, purpose], "tryon_available": False,
        "tryon_reason": "신발은 가상 피팅 미지원",
        "recommendation_reason": "추천 조건에서 도출된 스니커즈 종류에 해당합니다.",
        "fit_evidence": ["추천 조건에서 도출된 '스니커즈' 종류에 해당합니다."],
        "fit_evidence_labels": ["종류"],
        "reason_rule_ids": ["R-CTX-01", "R-ACC-06"],
    }
    result["shopping_results"] = [
        {**templates[category], "product_id": f"MSMOCK{category.upper()}{index}",
         "name": f"{templates[category]['name']} {index}"}
        for category in ("top", "bottom", "shoes") if category in categories
        for index in range(1, 4)
    ]
    products_by_id = {item["product_id"]: item for item in result["shopping_results"]}
    current_labels = {
        "top": ("현재 상의", result["outfit_summary"]["상의"]),
        "bottom": ("현재 하의", result["outfit_summary"]["하의"]),
        "shoes": ("현재 신발", result["outfit_summary"]["신발"]),
    }
    current_items = [
        {"category": category, "label": current_labels[category][0],
         "description": current_labels[category][1]}
        for category in ("top", "bottom", "shoes") if category not in categories
    ]
    result["shopping_outfits"] = [
        {
            "combination_id": f"OUTFIT-{index}",
            "product_ids": [f"MSMOCK{category.upper()}{index}" for category in categories],
            "current_items": current_items,
            "reason": "현재 유지할 아이템의 실루엣과 선택한 스타일 키워드를 함께 맞춘 코디입니다.",
            "reason_source": "rules",
            "evidence": [
                "현재 착장에 남는 아이템과 교체 상품의 상·하의 조화를 확인했습니다.",
                f"선택한 {style} 스타일 키워드가 상품명에 실제로 포함된 상품을 묶었습니다.",
            ],
            "evidence_labels": ["현재 착장", "스타일"],
            "rule_ids": ["R-CMP-03", "R-CTX-01"],
            "products": [
                products_by_id[f"MSMOCK{category.upper()}{index}"] for category in categories
            ],
        }
        for index in range(1, 4)
    ]
    result["request"] = profile
    return result


def _svg_for(job: dict, name: str) -> bytes:
    mime = job["image_mime"]
    encoded = base64.b64encode(job["image"]).decode("ascii")
    data_url = f"data:{mime};base64,{encoded}"
    profile = job["profile"]
    purpose = html.escape(str(profile.get("purpose") or "데일리"))
    style = html.escape(str(profile.get("desired_style") or "미니멀"))

    common = f"""
      <defs>
        <clipPath id="photo"><rect x="36" y="36" width="696" height="952" rx="10"/></clipPath>
        <linearGradient id="shade" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stop-color="#101010" stop-opacity=".05"/>
          <stop offset="1" stop-color="#101010" stop-opacity=".42"/>
        </linearGradient>
      </defs>
      <rect width="768" height="1024" fill="#f3f0e8"/>
      <image href="{data_url}" x="36" y="36" width="696" height="952"
             preserveAspectRatio="xMidYMid slice" clip-path="url(#photo)"/>
    """

    if name == "landmarks":
        overlay = """
          <g stroke="#ff4f46" stroke-width="5" fill="#fff" fill-opacity=".92">
            <path d="M384 185 L305 305 L270 470 M384 185 L463 305 L498 470
                     M305 305 L345 520 L325 735 L300 925 M463 305 L423 520 L445 735 L470 925
                     M345 520 L423 520" fill="none"/>
            <g stroke-width="4">
              <circle cx="384" cy="185" r="10"/><circle cx="305" cy="305" r="10"/>
              <circle cx="463" cy="305" r="10"/><circle cx="270" cy="470" r="10"/>
              <circle cx="498" cy="470" r="10"/><circle cx="345" cy="520" r="10"/>
              <circle cx="423" cy="520" r="10"/><circle cx="325" cy="735" r="10"/>
              <circle cx="445" cy="735" r="10"/><circle cx="300" cy="925" r="10"/>
              <circle cx="470" cy="925" r="10"/>
            </g>
          </g>
          <rect x="55" y="55" width="166" height="38" rx="19" fill="#111"/>
          <text x="76" y="80" fill="#fff" font-size="18" font-family="sans-serif">POSE · MOCK</text>
        """
    elif name == "segmentation":
        overlay = """
          <g opacity=".43" style="mix-blend-mode:multiply">
            <path d="M275 270 Q384 210 493 270 L520 540 Q430 585 384 570 Q330 585 248 540Z" fill="#ff574e"/>
            <path d="M316 530 Q384 555 452 530 L500 930 Q410 968 384 955 Q350 968 275 930Z" fill="#536dfe"/>
          </g>
          <path d="M275 270 Q384 210 493 270 L520 540 Q430 585 384 570 Q330 585 248 540Z"
                fill="none" stroke="#ff574e" stroke-width="5" stroke-dasharray="12 9"/>
          <path d="M316 530 Q384 555 452 530 L500 930 Q410 968 384 955 Q350 968 275 930Z"
                fill="none" stroke="#536dfe" stroke-width="5" stroke-dasharray="12 9"/>
          <rect x="55" y="55" width="218" height="38" rx="19" fill="#111"/>
          <text x="76" y="80" fill="#fff" font-size="18" font-family="sans-serif">PARSER · MOCK</text>
        """
    else:
        overlay = f"""
          <rect x="36" y="36" width="696" height="952" rx="10" fill="url(#shade)"/>
          <rect x="72" y="702" width="624" height="238" rx="8" fill="#fff" fill-opacity=".94"/>
          <text x="102" y="752" fill="#e0332b" font-size="18" font-family="sans-serif" font-weight="700">FITTA · MOCK RESULT</text>
          <text x="102" y="802" fill="#111" font-size="34" font-family="sans-serif" font-weight="700">{purpose} / {style}</text>
          <line x1="102" y1="830" x2="666" y2="830" stroke="#d8d8d8"/>
          <circle cx="123" cy="875" r="20" fill="#23325a"/>
          <circle cx="177" cy="875" r="20" fill="#828282"/>
          <text x="222" y="883" fill="#333" font-size="20" font-family="sans-serif">규칙 기반 추천 보드</text>
          <text x="102" y="920" fill="#777" font-size="15" font-family="sans-serif">실제 모델 결과가 아닌 인터페이스 시연용 이미지입니다.</text>
        """

    svg = f"""<svg xmlns="http://www.w3.org/2000/svg" width="768" height="1024" viewBox="0 0 768 1024">
      {common}{overlay}
    </svg>"""
    return svg.encode("utf-8")


class MockHandler(SimpleHTTPRequestHandler):
    server_version = "FITTA-Mock/1.0"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def _json(self, payload: dict, status: int = HTTPStatus.OK) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _read_json(self) -> dict:
        length = _number(self.headers.get("Content-Length"), 0)
        if length <= 0 or length > MAX_REQUEST_BYTES:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def _job(self, job_id: str) -> dict | None:
        if not SAFE_JOB_ID.match(job_id):
            return None
        _prune_jobs()
        with JOBS_LOCK:
            return JOBS.get(job_id)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/options":
            self._json(OPTIONS)
            return
        if path == "/api/health":
            self._json(
                {
                    "mock": True,
                    "device": "mock-cpu",
                    "trained_heads": True,
                    "parser_backend": "mock-parser",
                    "vton_enabled": True,
                    "product_count": 583,
                    "rules_implemented": 43,
                    "rules_documented": 50,
                }
            )
            return
        if path == "/api/rules":
            self._json({"titles": RULE_TITLES})
            return
        if path == "/api/retention":
            self._json({"ttl_minutes": 30, "max_sessions": 20})
            return
        if path == "/api/tryon":
            self._json({"available": True, "reason": ""})
            return
        if path == "/api/mock/status":
            self._json({"mock": True, "jobs": len(JOBS), "message": "FITTA mock API is ready"})
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
        if match:
            job_id = match.group(1)
            job = self._job(job_id)
            if job is None:
                self._json({"detail": "분석 요청을 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            elapsed = time.monotonic() - job["created"]
            if elapsed < 3.2:
                stage_index = min(int(elapsed / 0.64), len(STAGES) - 1)
                self._json(
                    {"job_id": job_id, "status": "running", "stage": STAGES[stage_index], "error": None, "result": None}
                )
            else:
                self._json(
                    {
                        "job_id": job_id,
                        "status": "done",
                        "stage": None,
                        "error": None,
                        "result": job["result"],
                        "shopping_tryon_batch": _mock_shopping_tryon_batch(job),
                    }
                )
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/shopping-tryon-batch", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self._json(_mock_shopping_tryon_batch(job))
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/images/([a-z0-9_-]+)", path)
        if match:
            job_id, name = match.groups()
            job = self._job(job_id)
            if job is None or not SAFE_IMAGE_NAME.match(name):
                self._json({"detail": "이미지를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            if name == "original":
                data = job["image"]
                content_type = job["image_mime"]
            elif name in {"landmarks", "segmentation", "preview"} or name.startswith("tryon-"):
                data = _svg_for(job, name)
                content_type = "image/svg+xml; charset=utf-8"
            else:
                self._json({"detail": "이미지를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return

        super().do_GET()

    def _read_image_upload(self, form_out: dict | None = None):
        """multipart 업로드에서 전신사진을 꺼낸다. 실패하면 응답까지 보내고 None 을 준다."""
        content_type, params = parse_header(self.headers.get("Content-Type", ""))
        length = _number(self.headers.get("Content-Length"), 0)
        if content_type != "multipart/form-data" or "boundary" not in params:
            self._json({"detail": "multipart/form-data 요청이 필요합니다."}, HTTPStatus.BAD_REQUEST)
            return None
        if length <= 0 or length > MAX_REQUEST_BYTES:
            self._json({"detail": "업로드 요청이 너무 큽니다."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
            return None
        form = FieldStorage(
            fp=self.rfile,
            headers=self.headers,
            environ={
                "REQUEST_METHOD": "POST",
                "CONTENT_TYPE": self.headers["Content-Type"],
                "CONTENT_LENGTH": str(length),
            },
            keep_blank_values=True,
        )
        if form_out is not None:
            form_out["form"] = form
        if "image" not in form:
            self._json({"detail": "전신사진이 필요합니다."}, HTTPStatus.BAD_REQUEST)
            return None
        image_field = form["image"]
        if isinstance(image_field, list):
            image_field = image_field[0]
        image = image_field.file.read()
        if not image:
            self._json({"detail": "이미지 파일이 비어 있습니다."}, HTTPStatus.BAD_REQUEST)
            return None
        return image, image_field

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/validate-photo":
            # 실제 서버는 이 검사를 통과하지 않으면 2단계를 열지 않는다. 목에 이 경로가
            # 없으면 프런트 작업에서 1단계를 넘어갈 수 없다.
            upload = self._read_image_upload()
            if upload is None:
                return
            _image, field = upload
            self._json(_mock_preflight(getattr(field, "filename", "") or ""))
            return
        if path == "/api/analyze":
            parsed_form: dict = {}
            upload = self._read_image_upload(parsed_form)
            if upload is None:
                return
            image, image_field = upload
            try:
                profile = json.loads(parsed_form["form"].getvalue("profile", "{}"))
            except json.JSONDecodeError:
                self._json({"detail": "조건 값을 읽을 수 없습니다."}, HTTPStatus.BAD_REQUEST)
                return

            job_id = uuid.uuid4().hex
            image_seed = sum(image[:4096]) % 10_000
            job = {
                "created": time.monotonic(),
                "image": image,
                "image_mime": _sniff_mime(image, image_field.type or ""),
                "profile": profile,
                "result": _build_result(profile, image_seed),
                "feedback": [],
            }
            with JOBS_LOCK:
                JOBS[job_id] = job
            self._json({"job_id": job_id, "status": "running", "stage": STAGES[0]})
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/shopping-tryon-batch", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self._json(_mock_shopping_tryon_batch(job))
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/current-bottom-length", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            length = str(self._read_json().get("length") or "").strip()
            if length and length not in MOCK_BOTTOM_LENGTHS:
                self._json({"detail": "지원하지 않는 기장 값입니다."}, HTTPStatus.BAD_REQUEST)
                return
            job["user_bottom_length"] = length
            check = job["result"]["length_check"]
            check.update(value=length, status="user_input" if length else "needs_input",
                         message="입력한 현재 하의 기장으로 기장 차이를 확인해요." if length else MOCK_LENGTH_PROMPT)
            self._json({"length_check": check, "shopping_tryon_batch": _mock_shopping_tryon_batch(job)})
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/tryon-products", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            payload = self._read_json()
            product_ids = payload.get("product_ids") or []
            if not isinstance(product_ids, list) or not 1 <= len(product_ids) <= 2:
                self._json({"detail": "상품 번호 목록이 필요합니다."}, HTTPStatus.BAD_REQUEST)
                return
            self._json(
                {
                    "image": f"tryon-products-{len(product_ids)}",
                    "cached": False,
                    "mock": True,
                    "warnings": [],
                    "product_ids": product_ids,
                    "categories": ["top", "bottom"][: len(product_ids)],
                }
            )
            return

        if path == "/api/feedback":
            payload = self._read_json()
            self._json(
                {
                    "saved": True,
                    "mock": True,
                    "rank": _number(payload.get("rank"), 1),
                    "action": str(payload.get("action") or ""),
                }
            )
            return

        self._json({"detail": "API 경로를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)

    def do_DELETE(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})", path)
        if not match:
            self._json({"detail": "API 경로를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
            return
        with JOBS_LOCK:
            JOBS.pop(match.group(1), None)
        self._json({"deleted": True})

    def log_message(self, message: str, *args) -> None:
        # TLS 패킷이 일반 HTTP 포트로 들어오는 경우 긴 바이너리 문자열을 출력하지 않는다.
        rendered = message % args
        if "Bad request" in rendered:
            rendered = "HTTPS 요청을 HTTP mock 서버가 받았습니다"
        print(f"[{self.log_date_time_string()}] {self.client_address[0]} {rendered}")


def main() -> int:
    parser = argparse.ArgumentParser(description="FITTA mock API와 정적 화면을 함께 실행합니다.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), MockHandler)
    print(f"FITTA mock server: http://{args.host}:{args.port}")
    print("API: /api/mock/status · 종료: Ctrl+C")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nFITTA mock server stopped")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
