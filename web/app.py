"""AI 코디 추천 웹 서버.

저장소 루트에서 `python web/run_web.py`로 실행한다.
직접 띄우려면 `python -m uvicorn app:app --reload --app-dir web`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image, UnidentifiedImageError

WEB_DIR = Path(__file__).resolve().parent
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

from pipeline import (  # noqa: E402
    STAGES,
    PipelineError,
    TryOnNotReady,
    analyze_wardrobe_items,
    build_profile,
    form_options,
    generate_tryon_with_warnings,
    get_engine,
    run_pipeline,
    rule_titles,
    save_feedback,
    tryon_status,
)

# 업로드 사진은 프로젝트 폴더에 두지 않는다. 프로젝트가 OneDrive·Dropbox 같은
# 동기화 폴더 안에 있으면 사용자의 전신사진이 클라우드로 올라가기 때문이다.
# 기본값은 OS 임시 폴더이고, 필요하면 FASHION_WEB_SESSION_DIR로 바꾼다.
SESSION_ROOT = Path(
    os.environ.get("FASHION_WEB_SESSION_DIR")
    or Path(tempfile.gettempdir()) / "fitta_web_sessions"
)

# 사진을 프로젝트 안에 저장하던 이전 버전이 남긴 폴더. 시작할 때 비운다.
LEGACY_SESSION_ROOT = WEB_DIR.parent / "ai_fashion_recommender" / "outputs" / "web_sessions"

MAX_UPLOAD_BYTES = 12 * 1024 * 1024
MAX_WARDROBE_IMAGES = 8
ALLOWED_FORMATS = {"JPEG", "PNG", "WEBP"}
JOB_ID_PATTERN = re.compile(r"^[0-9a-f]{32}$")
IMAGE_NAME_PATTERN = re.compile(r"^[a-z_]+[0-9]*\.jpg$")

# 업로드한 전신사진은 개인정보다. 결과를 확인할 동안만 두고 곧바로 지운다.
SESSION_TTL = timedelta(minutes=30)
MAX_SESSIONS = 20
SWEEP_INTERVAL_SECONDS = 300
TRYON_BATCH_LIMIT = 3
TRYON_BATCH_ACTIVE_STATES = {"queued", "running"}

app = FastAPI(title="AI 코디 추천", docs_url=None, redoc_url=None)

# 화면을 이 서버가 직접 서빙하면 같은 출처라 CORS가 필요 없다. 화면만 GitHub Pages
# 같은 정적 호스팅에 올리고 연산만 이 서버로 보낼 때 필요하다.
# 업로드가 전신사진이라 아무 출처나 열지 않는다. 허용할 주소를 쉼표로 나열한다.
#   FASHION_ALLOWED_ORIGINS="https://heiiowrld.github.io"
_allowed_origins = [
    origin.strip()
    for origin in os.environ.get("FASHION_ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
]
if _allowed_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_allowed_origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
    )

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def prune_sessions(
    root: Path = SESSION_ROOT,
    ttl: timedelta = SESSION_TTL,
    max_sessions: int = MAX_SESSIONS,
    protected: frozenset[str] = frozenset(),
) -> list[str]:
    """보관 기한이 지났거나 개수를 넘은 세션 폴더를 지우고 지운 ID를 돌려준다.

    분석이 진행 중인 세션은 `protected`로 받아 건드리지 않는다.
    """
    if not root.is_dir():
        return []

    sessions = [path for path in root.iterdir() if path.is_dir() and path.name not in protected]
    sessions.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    deadline = datetime.now(timezone.utc) - ttl

    removed: list[str] = []
    for index, path in enumerate(sessions):
        modified = datetime.fromtimestamp(path.stat().st_mtime, timezone.utc)
        if (modified < deadline or index >= max_sessions) and purge_session(path):
            removed.append(path.name)
    return removed


def purge_session(path: Path) -> bool:
    """세션 폴더의 사진을 지운다. 사진이 하나도 남지 않았을 때만 참을 돌려준다.

    Windows에서 동기화 클라이언트나 바이러스 검사가 폴더 핸들을 잡고 있으면
    빈 폴더만 남을 수 있다. 사진이 지워졌다면 성공으로 보고 폴더는 다음 청소에
    다시 시도한다.
    """
    shutil.rmtree(path, ignore_errors=True)
    if not path.exists():
        return True
    return not any(item.is_file() for item in path.rglob("*"))


def _running_job_ids() -> frozenset[str]:
    with _jobs_lock:
        return frozenset(
            job_id
            for job_id, job in _jobs.items()
            if job["status"] == "running"
            or (job.get("tryon_batch") or {}).get("status") in TRYON_BATCH_ACTIVE_STATES
        )


def _forget_jobs(job_ids: list[str]) -> None:
    if not job_ids:
        return
    with _jobs_lock:
        for job_id in job_ids:
            _jobs.pop(job_id, None)


def sweep_now() -> list[str]:
    removed = prune_sessions(protected=_running_job_ids())
    _forget_jobs(removed)
    return removed


def _sweep_forever() -> None:
    while True:
        time.sleep(SWEEP_INTERVAL_SECONDS)
        try:
            sweep_now()
        except OSError:
            pass  # 다음 주기에 다시 시도한다.


def purge_legacy_sessions(root: Path = LEGACY_SESSION_ROOT) -> int:
    """사진을 프로젝트 폴더에 두던 이전 버전의 잔여물을 지운다.

    프로젝트가 OneDrive 안에 있으면 그 사진들이 계속 클라우드로 동기화되므로
    서버를 새로 켤 때마다 확인해서 비운다.
    """
    if not root.is_dir():
        return 0
    purged = sum(purge_session(path) for path in root.iterdir() if path.is_dir())
    shutil.rmtree(root, ignore_errors=True)
    return purged


@app.on_event("startup")
def _start_session_sweeper() -> None:
    # 이전 실행에서 남은 사진까지 정리한 뒤 주기 청소를 시작한다.
    purge_legacy_sessions()
    sweep_now()
    threading.Thread(target=_sweep_forever, daemon=True).start()


def _update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _record_job_stage(job_id: str, stage: str) -> None:
    """현재 단계와 실제 통과 순서를 함께 보존해 UI·운영 점검이 같은 값을 보게 한다."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        history = job.setdefault("stage_history", [])
        if not history or history[-1] != stage:
            history.append(stage)
        job["stage"] = stage


def _read_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


def _record_tryon_warnings(job_id: str, rank: int, warnings: list[str]) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.setdefault("tryon_warnings", {})[rank] = list(warnings)


def _tryon_batch_snapshot_locked(job: dict) -> dict:
    """잠금 안에서 내부 배치 상태를 JSON 응답 형태로 복사한다."""
    batch = job.get("tryon_batch") or {
        "status": "idle",
        "reason": "",
        "items": {},
    }
    items = [
        {
            "rank": int(rank),
            "status": item.get("status", "queued"),
            "image": item.get("image"),
            "warnings": list(item.get("warnings") or []),
            "error": item.get("error"),
        }
        for rank, item in sorted(
            (batch.get("items") or {}).items(), key=lambda pair: int(pair[0])
        )
    ]
    ready = sum(item["status"] == "done" for item in items)
    finished = sum(item["status"] in {"done", "failed"} for item in items)
    return {
        "status": batch.get("status", "idle"),
        "reason": batch.get("reason", ""),
        "total": len(items),
        "ready": ready,
        "finished": finished,
        "items": items,
    }


def _read_tryon_batch(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return _tryon_batch_snapshot_locked(job) if job is not None else None


def _initialize_tryon_batch(job_id: str) -> dict | None:
    """추천 결과에서 최대 세 개의 자동 합성 항목을 만든다."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        result = job.get("result") or {}
        recommendations = [
            item
            for item in (job.get("recommendations") or [])
            if item.products
        ][:TRYON_BATCH_LIMIT]
        capability = result.get("tryon") or {}
        available = bool(capability.get("available"))
        preview_kind = capability.get("preview_kind")
        preview_name = (result.get("images") or {}).get("preview")
        items: dict[int, dict] = {}
        for recommendation in recommendations:
            item = {
                "status": "queued" if available else "unavailable",
                "image": None,
                "warnings": [],
                "error": None,
            }
            if (
                available
                and recommendation.rank == 1
                and preview_kind == "tryon"
                and isinstance(preview_name, str)
                and IMAGE_NAME_PATTERN.match(preview_name)
                and (_session_dir(job_id) / preview_name).is_file()
            ):
                item.update(
                    status="done",
                    image=preview_name,
                    warnings=list(capability.get("warnings") or []),
                )
            items[recommendation.rank] = item

        if not recommendations:
            status, reason = "done", "생성할 추천 코디가 없습니다."
        elif not available:
            status, reason = "unavailable", str(capability.get("reason") or "")
        elif any(item["status"] == "queued" for item in items.values()):
            status, reason = "queued", ""
        else:
            status, reason = "done", ""
        job["tryon_batch"] = {
            "status": status,
            "reason": reason,
            "items": items,
        }
        return _tryon_batch_snapshot_locked(job)


def _set_tryon_batch_item(job_id: str, rank: int, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        item = ((job.get("tryon_batch") or {}).get("items") or {}).get(rank)
        if item is not None:
            item.update(fields)


def _finish_tryon_batch(job_id: str) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        batch = job.get("tryon_batch") or {}
        items = list((batch.get("items") or {}).values())
        ready = sum(item.get("status") == "done" for item in items)
        failed = sum(item.get("status") == "failed" for item in items)
        if failed == 0:
            batch["status"] = "done"
            batch["reason"] = ""
        elif ready:
            batch["status"] = "partial"
            batch["reason"] = "일부 착장샷을 만들지 못했습니다."
        else:
            batch["status"] = "failed"
            batch["reason"] = "착장샷을 만들지 못했습니다."


def _session_dir(job_id: str) -> Path:
    return SESSION_ROOT / job_id


def _generate_tryon_for_job(job_id: str, rank: int) -> dict:
    """한 순위의 합성을 캐시·직렬화하고 결과 메타데이터를 돌려준다."""
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None or job["status"] != "done" or job.get("cancelled"):
            raise LookupError("분석 결과를 찾을 수 없습니다.")
        work_lock = job.setdefault("work_lock", threading.Lock())

    # 자동 배치와 사용자의 수동 재시도가 같은 순위를 중복 생성하지 않게 한다.
    with work_lock:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None or job["status"] != "done" or job.get("cancelled"):
                raise LookupError("분석 결과를 찾을 수 없습니다.")
            recommendation = next(
                (item for item in job.get("recommendations") or [] if item.rank == rank),
                None,
            )
            if recommendation is None:
                raise LookupError("해당 순위의 추천을 찾을 수 없습니다.")
            result = job.get("result") or {}
            person_image = job["person_image"]
            tryon_context = job.get("tryon_context")

        # 실제 VTON으로 기록된 1순위 미리보기만 재사용한다. 재료 부족으로 만든
        # 추천 보드는 preview_kind=preview라 이 경로를 타지 않는다.
        capability = result.get("tryon") or {}
        preview_name = (result.get("images") or {}).get("preview")
        if (
            rank == 1
            and capability.get("preview_kind") == "tryon"
            and isinstance(preview_name, str)
            and IMAGE_NAME_PATTERN.match(preview_name)
        ):
            preview = _session_dir(job_id) / preview_name
            if preview.is_file():
                return {
                    "image": preview.name,
                    "cached": True,
                    "warnings": list(capability.get("warnings") or []),
                }

        output = _session_dir(job_id) / f"tryon_{rank}.jpg"
        if output.is_file():
            with _jobs_lock:
                current = _jobs.get(job_id) or {}
                warnings = list((current.get("tryon_warnings") or {}).get(rank) or [])
            return {"image": output.name, "cached": True, "warnings": warnings}

        generated, warnings = generate_tryon_with_warnings(
            person_image,
            recommendation,
            output,
            context=tryon_context,
        )
        warnings = list(warnings)
        _record_tryon_warnings(job_id, rank, warnings)
        return {"image": Path(generated).name, "cached": False, "warnings": warnings}


def _tryon_batch_worker(job_id: str) -> None:
    while True:
        with _jobs_lock:
            job = _jobs.get(job_id)
            if job is None or job.get("cancelled"):
                return
            batch = job.get("tryon_batch") or {}
            queued = [
                int(rank)
                for rank, item in sorted(
                    (batch.get("items") or {}).items(), key=lambda pair: int(pair[0])
                )
                if item.get("status") == "queued"
            ]
            if not queued:
                break
            rank = queued[0]
            batch["items"][rank].update(status="running", error=None)

        try:
            generated = _generate_tryon_for_job(job_id, rank)
        except TryOnNotReady as exc:
            _set_tryon_batch_item(job_id, rank, status="failed", error=str(exc))
        except Exception as exc:  # noqa: BLE001 - 항목별 실패를 나머지 순위와 격리한다.
            _set_tryon_batch_item(
                job_id,
                rank,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
        else:
            _set_tryon_batch_item(
                job_id,
                rank,
                status="done",
                image=generated["image"],
                warnings=list(generated.get("warnings") or []),
                error=None,
            )
    _finish_tryon_batch(job_id)


def _start_tryon_batch(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is None:
            return None
        batch = job.get("tryon_batch") or {}
        if batch.get("status") != "queued":
            return _tryon_batch_snapshot_locked(job)
        batch["status"] = "running"
        snapshot = _tryon_batch_snapshot_locked(job)
    threading.Thread(target=_tryon_batch_worker, args=(job_id,), daemon=True).start()
    return snapshot


def _worker(
    job_id: str,
    image_path: Path,
    profile,
    body_image_path: Path | None = None,
    wardrobe_image_paths: list[Path] | None = None,
) -> None:
    def on_stage(stage: str) -> None:
        _record_job_stage(job_id, stage)

    try:
        on_stage("prepare")
        # 최초 요청의 모델·카탈로그 로드 시간을 '사진 검사' 시간으로 표시하지 않는다.
        get_engine()
        on_stage("wardrobe")
        analyze_wardrobe_items(profile, wardrobe_image_paths or [])
        outcome = run_pipeline(image_path, profile, _session_dir(job_id), on_stage, body_image_path)
    except PipelineError as exc:
        # 분석에 실패하면 보여줄 결과가 없으므로 사진을 바로 지운다.
        _update_job(job_id, status="failed", stage=None, error=str(exc))
        purge_session(_session_dir(job_id))
    except Exception as exc:  # noqa: BLE001 - 사용자에게 사유를 그대로 전달한다.
        _update_job(job_id, status="failed", stage=None, error=f"{type(exc).__name__}: {exc}")
        purge_session(_session_dir(job_id))
    else:
        _update_job(
            job_id,
            status="done",
            stage=None,
            result=outcome.payload,
            recommendations=outcome.recommendations,
            person_image=outcome.person_image,
            tryon_context=outcome.tryon_context,
        )
        _initialize_tryon_batch(job_id)
        _start_tryon_batch(job_id)
    finally:
        # 새 분석마다 오래된 사진을 함께 정리한다.
        sweep_now()


@app.get("/api/options")
def options() -> dict:
    return form_options()


@app.get("/api/health")
def health() -> dict:
    """모델 적재 상태를 미리 알려 첫 분석의 대기 이유를 설명한다."""
    engine = get_engine()
    return {
        "device": engine.device,
        "trained_heads": engine.trained_heads,
        "parser_backend": engine.parser_backend,
        "vton_enabled": engine.tryon.enabled,
        "product_count": len(engine.recommender.catalog.products),
        "product_color_audits": len(engine.recommender.catalog.color_audits),
        "product_color_overrides": engine.recommender.catalog.color_override_count,
        "product_color_mismatches": engine.recommender.catalog.color_mismatch_count,
        "rules_implemented": len(engine.recommender.active_rule_ids),
        "rules_documented": len(engine.recommender.documented_rule_ids),
    }


@app.get("/api/rules")
def rules() -> dict:
    return {"titles": rule_titles()}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    profile: str = Form(...),
    body_image: UploadFile | None = File(None),
    wardrobe_images: list[UploadFile] = File(default=[]),
) -> JSONResponse:
    try:
        payload = json.loads(profile)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail=f"조건 값을 읽을 수 없습니다: {exc}") from exc

    owned_items = payload.get("owned_items") or []
    if len(wardrobe_images) > MAX_WARDROBE_IMAGES:
        raise HTTPException(status_code=400, detail=f"보유 옷 사진은 최대 {MAX_WARDROBE_IMAGES}장까지 지원합니다.")
    if len(wardrobe_images) != len(owned_items):
        raise HTTPException(status_code=400, detail="보유 옷 정보와 사진 수가 맞지 않습니다.")

    raw = await image.read()
    if not raw:
        raise HTTPException(status_code=400, detail="이미지 파일이 비어 있습니다.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="이미지 용량은 12MB 이하만 지원합니다.")

    job_id = uuid.uuid4().hex
    session = _session_dir(job_id)
    session.mkdir(parents=True, exist_ok=True)
    image_path = session / "original.jpg"
    try:
        with Image.open(BytesIO(raw)) as opened:
            if opened.format not in ALLOWED_FORMATS:
                raise HTTPException(status_code=400, detail="JPG, PNG, WEBP 이미지만 지원합니다.")
            opened.convert("RGB").save(image_path, "JPEG", quality=95)
    except UnidentifiedImageError as exc:
        shutil.rmtree(session, ignore_errors=True)
        raise HTTPException(status_code=400, detail="이미지 파일을 해석할 수 없습니다.") from exc
    except HTTPException:
        shutil.rmtree(session, ignore_errors=True)
        raise

    # 체형 파악용 사진은 선택이다. 없으면 코디 사진으로 추정한다.
    body_path = None
    if body_image is not None:
        body_raw = await body_image.read()
        if body_raw:
            if len(body_raw) > MAX_UPLOAD_BYTES:
                shutil.rmtree(session, ignore_errors=True)
                raise HTTPException(status_code=413, detail="이미지 용량은 12MB 이하만 지원합니다.")
            try:
                with Image.open(BytesIO(body_raw)) as opened:
                    if opened.format not in ALLOWED_FORMATS:
                        raise HTTPException(status_code=400, detail="JPG, PNG, WEBP 이미지만 지원합니다.")
                    body_path = session / "body.jpg"
                    opened.convert("RGB").save(body_path, "JPEG", quality=95)
            except UnidentifiedImageError as exc:
                shutil.rmtree(session, ignore_errors=True)
                raise HTTPException(status_code=400, detail="체형 사진을 해석할 수 없습니다.") from exc
            except HTTPException:
                shutil.rmtree(session, ignore_errors=True)
                raise

    wardrobe_paths: list[Path] = []
    for index, wardrobe_image in enumerate(wardrobe_images, start=1):
        wardrobe_raw = await wardrobe_image.read()
        if not wardrobe_raw:
            shutil.rmtree(session, ignore_errors=True)
            raise HTTPException(status_code=400, detail="비어 있는 보유 옷 사진이 있습니다.")
        if len(wardrobe_raw) > MAX_UPLOAD_BYTES:
            shutil.rmtree(session, ignore_errors=True)
            raise HTTPException(status_code=413, detail="보유 옷 사진은 장당 12MB 이하만 지원합니다.")
        wardrobe_path = session / f"wardrobe_{index}.jpg"
        try:
            with Image.open(BytesIO(wardrobe_raw)) as opened:
                if opened.format not in ALLOWED_FORMATS:
                    raise HTTPException(status_code=400, detail="보유 옷 사진은 JPG, PNG, WEBP만 지원합니다.")
                opened.convert("RGB").save(wardrobe_path, "JPEG", quality=95)
        except UnidentifiedImageError as exc:
            shutil.rmtree(session, ignore_errors=True)
            raise HTTPException(status_code=400, detail="보유 옷 사진을 해석할 수 없습니다.") from exc
        except HTTPException:
            shutil.rmtree(session, ignore_errors=True)
            raise
        wardrobe_paths.append(wardrobe_path)

    job = {
        "id": job_id,
        "status": "running",
        "stage": STAGES[0][0],
        "stage_history": [],
        "error": None,
        "result": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "work_lock": threading.Lock(),
        "cancelled": False,
    }
    with _jobs_lock:
        _jobs[job_id] = job

    profile_object = build_profile(payload)
    threading.Thread(
        target=_worker,
        args=(job_id, image_path, profile_object, body_path, wardrobe_paths),
        daemon=True,
    ).start()
    return JSONResponse({"job_id": job_id, "status": "running", "stage": job["stage"]})


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str) -> dict:
    job = _read_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="분석 요청을 찾을 수 없습니다.")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "stage": job["stage"],
        "stage_history": job.get("stage_history", []),
        "error": job["error"],
        "result": job["result"],
        "tryon_batch": _read_tryon_batch(job_id),
    }


@app.get("/api/jobs/{job_id}/images/{name}")
def job_image(job_id: str, name: str) -> FileResponse:
    if not JOB_ID_PATTERN.match(job_id) or not IMAGE_NAME_PATTERN.match(name):
        raise HTTPException(status_code=400, detail="잘못된 이미지 요청입니다.")
    path = _session_dir(job_id) / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="이미지를 찾을 수 없습니다.")
    return FileResponse(path, media_type="image/jpeg")


@app.get("/api/tryon")
def tryon_capability() -> dict:
    """화면이 '예상 착장샷' 자리를 어떻게 그릴지 결정하는 데 쓴다."""
    return tryon_status()


@app.get("/api/jobs/{job_id}/tryon-batch")
def tryon_batch_status(job_id: str) -> dict:
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="잘못된 요청입니다.")
    batch = _read_tryon_batch(job_id)
    if batch is None:
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    return batch


@app.post("/api/jobs/{job_id}/tryon-batch")
def start_tryon_batch(job_id: str) -> dict:
    """추천 결과 최대 세 개를 중복 없이 자동 합성 큐에 넣는다."""
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="잘못된 요청입니다.")
    job = _read_job(job_id)
    if job is None or job["status"] != "done":
        raise HTTPException(status_code=404, detail="분석 결과를 찾을 수 없습니다.")
    if job.get("tryon_batch") is None:
        _initialize_tryon_batch(job_id)
    return _start_tryon_batch(job_id) or {
        "status": "idle", "reason": "", "total": 0, "ready": 0, "finished": 0, "items": []
    }


@app.post("/api/jobs/{job_id}/tryon/{rank}")
def create_tryon(job_id: str, rank: int) -> dict:
    """추천 코디 하나를 입은 예상 착장샷을 생성한다."""
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="잘못된 요청입니다.")
    try:
        generated = _generate_tryon_for_job(job_id, rank)
    except LookupError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except TryOnNotReady as exc:
        # 501: 화면이 '준비 중' 안내를 그릴 수 있도록 실패와 구분한다.
        raise HTTPException(status_code=501, detail=str(exc)) from exc
    _set_tryon_batch_item(
        job_id,
        rank,
        status="done",
        image=generated["image"],
        warnings=list(generated.get("warnings") or []),
        error=None,
    )
    _finish_tryon_batch(job_id)
    return generated


@app.delete("/api/jobs/{job_id}")
def delete_job(job_id: str) -> dict:
    """사용자가 결과 화면에서 자기 사진을 즉시 지울 수 있게 한다."""
    if not JOB_ID_PATTERN.match(job_id):
        raise HTTPException(status_code=400, detail="잘못된 요청입니다.")
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job["cancelled"] = True
            work_lock = job.setdefault("work_lock", threading.Lock())
        else:
            work_lock = threading.Lock()
    # 진행 중인 한 장의 저장이 끝난 뒤 지워야 CatVTON이 삭제된 세션 폴더를
    # 다시 만들어 개인정보가 남는 경쟁 조건을 막을 수 있다.
    with work_lock:
        session = _session_dir(job_id)
        if session.is_dir() and not purge_session(session):
            raise HTTPException(status_code=500, detail="사진을 삭제하지 못했습니다. 잠시 후 다시 시도하세요.")
        _forget_jobs([job_id])
    return {"deleted": True}


@app.get("/api/retention")
def retention() -> dict:
    return {
        "ttl_minutes": int(SESSION_TTL.total_seconds() // 60),
        "max_sessions": MAX_SESSIONS,
    }


@app.post("/api/feedback")
def feedback(payload: dict) -> dict:
    action = str(payload.get("action") or "").strip()
    if not action:
        raise HTTPException(status_code=400, detail="피드백 종류가 필요합니다.")
    rank = int(payload.get("rank") or 1)
    note = str(payload.get("note") or "")
    return save_feedback(rank, action, note)


class NoCacheStaticFiles(StaticFiles):
    """화면 파일을 고쳤는데 브라우저가 옛 버전을 계속 쓰는 일을 막는다."""

    def is_not_modified(self, response_headers, request_headers) -> bool:  # noqa: D102
        return False

    async def get_response(self, path: str, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
        return response


app.mount("/", NoCacheStaticFiles(directory=WEB_DIR / "static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
