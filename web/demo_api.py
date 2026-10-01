"""GPU 호출 없이 input_person.jpg 시연을 처리하는 로컬 API.

분석 수치는 고정 스냅샷이며 모델 추론값이 아니다. 사진은 메모리에만 보관한다.
실제 서버와 같은 경로·진행 단계를 사용하고 결과는 설정한 시간이 지나야 공개한다.
"""

from __future__ import annotations

import asyncio
import copy
import io
import json
import os
import re
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from PIL import Image, ImageOps, UnidentifiedImageError
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException

SNAPSHOT = json.loads(Path(__file__).with_name("demo_input_person.json").read_text(encoding="utf-8"))
LOOK_DIR = Path(__file__).with_name("static") / "assets" / "looks"
LOOK_FILES = tuple(f"input-person-look-{index}.png" for index in range(1, 4))
STAGE_SECONDS = (
    ("prepare", 1), ("wardrobe", 0.5), ("pose", 1.2), ("quality", 1.3),
    ("segment", 6), ("attributes", 6), ("body", 1), ("candidates", 1),
    ("scoring", 8), ("preview", 2), ("finalize", 2),
)
JOB_ID = re.compile(r"^de000000[0-9a-f]{24}$")
MAX_UPLOAD_BYTES = 12 * 1024 * 1024
SESSION_TTL_SECONDS = 30 * 60
MAX_SESSIONS = 20


def is_demo_person(filename: str | None) -> bool:
    name = (filename or "").replace("\\", "/").rsplit("/", 1)[-1]
    return name.casefold() == "input_person.jpg"


def _normalize_image(raw: bytes) -> bytes:
    if not raw:
        raise HTTPException(400, "이미지 파일이 비어 있습니다.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "이미지 용량은 12MB 이하만 지원합니다.")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in {"JPEG", "MPO", "PNG", "WEBP"}:
                raise HTTPException(400, "JPG, PNG, WEBP 이미지만 지원합니다.")
            if opened.width * opened.height > 30_000_000:
                raise HTTPException(400, "이미지 해상도가 너무 큽니다.")
            output = io.BytesIO()
            ImageOps.exif_transpose(opened).convert("RGB").save(output, "JPEG", quality=95)
            return output.getvalue()
    except (OSError, ValueError, UnidentifiedImageError) as exc:
        raise HTTPException(400, "이미지 파일을 해석할 수 없습니다.") from exc


class DemoAPI:
    def __init__(self, *, duration_seconds: float | None = None, preflight_seconds: float = 2.0):
        self.duration_seconds = max(0.0, float(
            os.environ.get("FITTA_DEMO_SECONDS", "30") if duration_seconds is None else duration_seconds
        ))
        self.preflight_seconds = preflight_seconds
        self.clock = time.monotonic
        self.jobs: dict[str, dict] = {}
        self.expirations: dict[str, asyncio.TimerHandle] = {}

    def delete(self, job_id: str) -> None:
        self.jobs.pop(job_id, None)
        handle = self.expirations.pop(job_id, None)
        if handle:
            handle.cancel()

    def close(self) -> None:
        for job_id in list(self.jobs):
            self.delete(job_id)

    def _sweep(self) -> None:
        now = self.clock()
        for job_id, job in list(self.jobs.items()):
            if now - job["started_at"] >= SESSION_TTL_SECONDS:
                self.delete(job_id)

    def _status(self, job_id: str, job: dict) -> dict:
        elapsed = max(0.0, self.clock() - job["started_at"])
        history = []
        deadline = 0.0
        stage = None
        for name, seconds in STAGE_SECONDS:
            history.append(name)
            deadline += seconds * self.duration_seconds / 30.0
            if elapsed < deadline:
                stage = name
                break
        done = elapsed >= self.duration_seconds
        return {
            "job_id": job_id, "status": "done" if done else "running",
            "stage": None if done else stage, "stage_history": history, "error": None,
            "result": job["result"] if done else None,
            "shopping_tryon_batch": self._batch() if done else None,
        }

    @staticmethod
    def _batch(*, ready: bool = True) -> dict:
        if not ready:
            return {"status": "idle", "reason": "", "total": 3,
                    "ready": 0, "finished": 0, "items": []}
        products = {product["product_id"]: product for product in SNAPSHOT["shopping_results"]}
        items = [
            {
                "index": index,
                "product_ids": outfit["product_ids"],
                "categories": [products[product_id]["category"] for product_id in outfit["product_ids"]],
                "status": "done", "image": LOOK_FILES[index - 1],
                "cached": True, "warnings": [], "error": None,
            }
            for index, outfit in enumerate(SNAPSHOT["shopping_outfits"], start=1)
        ]
        return {"status": "done", "reason": "", "total": len(items),
                "ready": len(items), "finished": len(items), "items": items}

    async def _upload(self, request: Request) -> Response | None:
        if not request.headers.get("content-type", "").startswith("multipart/form-data"):
            return None
        # 캐시된 본문을 사용하므로 일반 사진을 파싱한 뒤에도 프록시로 그대로 전달한다.
        await request.body()
        try:
            form = await request.form()
        except HTTPException:
            return None  # 어떤 파일인지 확인할 수 없는 요청은 기존 API가 검증한다.
        try:
            image = form.get("image")
            if not isinstance(image, UploadFile) or not is_demo_person(image.filename):
                return None
            raw = await image.read(MAX_UPLOAD_BYTES + 1)
            normalized = _normalize_image(raw)
            if request.url.path == "/api/validate-photo":
                await asyncio.sleep(self.preflight_seconds)
                quality = {"passed": True, "issues": [], "warnings": []}
                return JSONResponse({"valid": True, "issues": [], "warnings": [], "quality": quality})
            try:
                profile = json.loads(form.get("profile") or "{}")
            except (ValueError, TypeError) as exc:
                raise HTTPException(400, "조건 값을 읽을 수 없습니다.") from exc
            if not isinstance(profile, dict):
                raise HTTPException(400, "조건 값은 객체여야 합니다.")
            self._sweep()
            if len(self.jobs) >= MAX_SESSIONS:
                raise HTTPException(429, "요청이 많습니다. 잠시 후 다시 시도해 주세요.",
                                    headers={"Retry-After": "60"})
            job_id = "de000000" + uuid.uuid4().hex[8:]
            result = copy.deepcopy(SNAPSHOT)
            products_by_id = {product["product_id"]: product for product in result["shopping_results"]}
            for outfit in result["shopping_outfits"]:
                outfit["products"] = [products_by_id[product_id] for product_id in outfit["product_ids"]]
            result["request"] = profile
            self.jobs[job_id] = {
                "started_at": self.clock(), "image": normalized, "result": result,
            }
            self.expirations[job_id] = asyncio.get_running_loop().call_later(
                SESSION_TTL_SECONDS, self.delete, job_id,
            )
            return JSONResponse({"job_id": job_id, "status": "running", "stage": "prepare"})
        finally:
            await form.close()

    async def handle(self, request: Request) -> Response | None:
        """None이면 기존 API로 넘긴다. 시연 사진의 모든 요청은 여기서 끝낸다."""
        try:
            path = request.url.path
            if request.method == "POST" and path in {"/api/validate-photo", "/api/analyze"}:
                return await self._upload(request)
            parts = path.split("/")
            if len(parts) < 4 or parts[1:3] != ["api", "jobs"] or not JOB_ID.fullmatch(parts[3]):
                return None
            job_id = parts[3]
            self._sweep()
            job = self.jobs.get(job_id)
            if job is None:
                raise HTTPException(404, "분석 요청을 찾을 수 없습니다.")
            suffix = parts[4:]
            if not suffix and request.method == "DELETE":
                self.delete(job_id)
                return JSONResponse({"deleted": True})
            if not suffix and request.method == "GET":
                return JSONResponse(self._status(job_id, job))
            ready = self.clock() - job["started_at"] >= self.duration_seconds
            if suffix == ["images", "original.jpg"] and request.method == "GET":
                return Response(job["image"], media_type="image/jpeg", headers={"Cache-Control": "no-store"})
            if len(suffix) == 2 and suffix[0] == "images" and suffix[1] in LOOK_FILES and request.method == "GET":
                if not ready:
                    raise HTTPException(404, "이미지를 찾을 수 없습니다.")
                return FileResponse(LOOK_DIR / suffix[1], media_type="image/png",
                                    headers={"Cache-Control": "no-store"})
            if suffix == ["shopping-tryon-batch"] and request.method in {"GET", "POST"}:
                if request.method == "POST" and not ready:
                    raise HTTPException(409, "추천 결과가 아직 준비되지 않았습니다.")
                return JSONResponse(self._batch(ready=ready))
            if suffix == ["tryon-products"] and request.method == "POST":
                if not ready:
                    raise HTTPException(409, "추천 결과가 아직 준비되지 않았습니다.")
                payload = await request.json()
                product_ids = payload.get("product_ids") if isinstance(payload, dict) else None
                item = next((item for item in self._batch()["items"]
                             if item["product_ids"] == product_ids), None)
                if item is None:
                    raise HTTPException(400, "추천된 상품 조합을 선택해 주세요.")
                return JSONResponse({key: item[key] for key in
                                     ("image", "cached", "warnings", "product_ids", "categories")})
            raise HTTPException(404, "요청을 찾을 수 없습니다.")
        except HTTPException as exc:
            return JSONResponse({"detail": exc.detail}, status_code=exc.status_code, headers=exc.headers)


def install_demo_api(application: FastAPI) -> DemoAPI:
    demo = DemoAPI()
    application.state.demo_api = demo

    @application.middleware("http")
    async def local_demo(request, call_next):
        response = await demo.handle(request)
        return response if response is not None else await call_next(request)

    return demo
