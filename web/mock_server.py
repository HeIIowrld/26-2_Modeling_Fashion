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


def _mock_tryon_batch(job: dict) -> dict:
    """실제 서버와 같은 배치 상태를 시간 경과로 재현한다."""
    elapsed = max(0.0, time.monotonic() - job["created"] - 3.2)
    ready = min(3, 1 + int(elapsed / 0.9))
    items = []
    for rank in range(1, 4):
        if rank <= ready:
            status, image = "done", f"tryon-{rank}"
        elif rank == ready + 1:
            status, image = "running", None
        else:
            status, image = "queued", None
        items.append(
            {
                "rank": rank,
                "status": status,
                "image": image,
                "warnings": [],
                "error": None,
            }
        )
    return {
        "status": "done" if ready == 3 else "running",
        "reason": "",
        "total": 3,
        "ready": ready,
        "finished": ready,
        "items": items,
    }


def _product(name: str, category: str, color: str, price: int, style: str, rank: int) -> dict:
    is_top = category == "top"
    return {
        "product_id": f"MOCK-{rank}-{category.upper()}",
        "name": name,
        "category": category,
        "color": color,
        "color_rgb": COLOR_RGB.get(color, [150, 150, 150]),
        "style": style,
        "price": max(price, 10_000),
        "season": "사계절",
        "url": "",
        "item_type": "셔츠" if is_top else "팬츠",
        "fit": "릴랙스드핏" if is_top else "세미와이드핏",
        "length": "기본 기장" if is_top else "발목 기장",
        "pattern": "무지",
        "material": "코튼" if is_top else "코튼 혼방",
        "neckline": "칼라넥" if is_top else "",
        "formality": 3,
    }


def _recommendation_names(purpose: str) -> list[tuple[str, str]]:
    if purpose in {"출근", "면접"}:
        return [
            ("클린 코튼 셔츠", "원턱 세미와이드 슬랙스"),
            ("텍스처드 미니멀 재킷", "스트레이트 슬랙스"),
            ("소프트 칼라 니트", "테이퍼드 팬츠"),
        ]
    if purpose == "데이트":
        return [
            ("소프트 니트 카디건", "클린 스트레이트 데님"),
            ("오픈칼라 셔츠", "세미와이드 코튼 팬츠"),
            ("라이트 크루넥 니트", "딥톤 데님 팬츠"),
        ]
    if purpose == "여행":
        return [
            ("라이트 유틸리티 셔츠", "이지 테이퍼드 팬츠"),
            ("코튼 오버셔츠", "스트레치 와이드 팬츠"),
            ("에어리 크루넥 탑", "라이트웨이트 카고 팬츠"),
        ]
    if purpose == "결혼식":
        return [
            ("텍스처드 싱글 블레이저", "클린 테이퍼드 슬랙스"),
            ("미니멀 칼라 재킷", "원턱 드레스 팬츠"),
            ("소프트 포멀 셔츠", "세미와이드 슬랙스"),
        ]
    return [
        ("릴랙스드 오픈칼라 셔츠", "세미와이드 코튼 팬츠"),
        ("미니멀 크루넥 니트", "클린 스트레이트 데님"),
        ("라이트 코튼 오버셔츠", "이지 테이퍼드 팬츠"),
    ]


def _build_result(profile: dict, image_seed: int) -> dict:
    purpose = str(profile.get("purpose") or "데일리")
    style = str(profile.get("desired_style") or "미니멀")
    preferred = [str(value) for value in profile.get("preferred_colors") or []]
    palette = preferred[:2] + ["네이비", "그레이", "베이지", "블랙"]
    palette = list(dict.fromkeys(palette))
    while len(palette) < 4:
        palette.append("그레이")

    minimum = max(_number(profile.get("min_budget"), 60_000), 30_000)
    maximum = max(_number(profile.get("max_budget"), 180_000), minimum)
    midpoint = (minimum + maximum) // 2
    names = _recommendation_names(purpose)
    recommendations = []
    scores = [93.6, 89.8, 86.4]

    for index, (top_name, bottom_name) in enumerate(names, start=1):
        factor = [1.0, 0.9, 1.06][index - 1]
        total = min(maximum, max(minimum, int(midpoint * factor)))
        top_price = max(10_000, int(total * 0.44))
        bottom_price = max(10_000, total - top_price)
        top_color = palette[(index - 1) % len(palette)]
        bottom_color = palette[index % len(palette)]
        recommendations.append(
            {
                "rank": index,
                "products": [
                    _product(top_name, "top", top_color, top_price, style, index),
                    _product(bottom_name, "bottom", bottom_color, bottom_price, style, index),
                ],
                "total_score": scores[index - 1],
                "score_breakdown": {
                    "purpose_tpo": 96.0 - index,
                    "weather_activity": 90.0 - index,
                    "silhouette": 94.0 - index * 2,
                    "color": 93.0 - index,
                    "pattern_material_complexity": 89.0 - index,
                    "preference": 95.0 - index * 2,
                },
                "reasons": [
                    f"{purpose} 상황의 격식과 활동량을 함께 고려한 조합입니다.",
                    f"{style} 분위기를 유지하면서 상·하의 볼륨 차이를 정돈했습니다.",
                    f"{top_color}와 {bottom_color}의 색상 관계가 안정적으로 이어집니다.",
                ],
                "applied_rules": ["R-CTX-01", "R-CTX-02", "R-COL-03", "R-SIL-01", "R-BUD-01"],
                "score_coverage": 88.0,
                "styling_tips": [
                    "신발과 가방은 하의와 비슷한 명도로 맞추면 전체가 길어 보여요.",
                    "액세서리는 한 가지 금속 톤으로 통일해 보세요.",
                ],
            }
        )

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
    return {
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
        },
        "recommendations": recommendations,
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
        "tryon": {"available": True, "reason": "", "warnings": [], "preview_kind": "tryon"},
        "images": {
            "original": "original",
            "landmarks": "landmarks",
            "segmentation": "segmentation",
            "preview": "preview",
        },
    }


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
                    {"job_id": job_id, "status": "done", "stage": None, "error": None, "result": job["result"]}
                )
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/tryon-batch", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self._json(_mock_tryon_batch(job))
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

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/analyze":
            content_type, params = parse_header(self.headers.get("Content-Type", ""))
            length = _number(self.headers.get("Content-Length"), 0)
            if content_type != "multipart/form-data" or "boundary" not in params:
                self._json({"detail": "multipart/form-data 요청이 필요합니다."}, HTTPStatus.BAD_REQUEST)
                return
            if length <= 0 or length > MAX_REQUEST_BYTES:
                self._json({"detail": "업로드 요청이 너무 큽니다."}, HTTPStatus.REQUEST_ENTITY_TOO_LARGE)
                return
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
            if "image" not in form:
                self._json({"detail": "전신사진이 필요합니다."}, HTTPStatus.BAD_REQUEST)
                return
            image_field = form["image"]
            if isinstance(image_field, list):
                image_field = image_field[0]
            image = image_field.file.read()
            if not image:
                self._json({"detail": "이미지 파일이 비어 있습니다."}, HTTPStatus.BAD_REQUEST)
                return
            try:
                profile = json.loads(form.getvalue("profile", "{}"))
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

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/tryon/(\d+)", path)
        if match:
            job_id, rank = match.groups()
            job = self._job(job_id)
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self._json({"image": f"tryon-{rank}", "cached": False, "mock": True, "warnings": []})
            return

        match = re.fullmatch(r"/api/jobs/([0-9a-f]{32})/tryon-batch", path)
        if match:
            job = self._job(match.group(1))
            if job is None:
                self._json({"detail": "분석 결과를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
                return
            self._json(_mock_tryon_batch(job))
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
