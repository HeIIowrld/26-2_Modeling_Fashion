from collections import deque
from types import SimpleNamespace
import asyncio

import pytest
from fastapi.testclient import TestClient
import web.app as server


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(server, "_jobs", {})
    monkeypatch.setattr(server, "_pending_analyzes", 0)
    monkeypatch.setattr(server, "_analysis_starts", deque())
    monkeypatch.setattr(server, "sweep_now", lambda: [])
    return TestClient(server.app)


def test_capacity_rejects_upload_without_evicting_results(admission):
    server._jobs.update({str(i): {"status": "done"} for i in range(server.MAX_SESSIONS)})
    response = admission.post("/api/analyze")
    assert response.status_code == 429
    assert response.headers["Retry-After"] == "60"
    assert len(server._jobs) == server.MAX_SESSIONS
    assert server._pending_analyzes == 0


def test_global_rate_limit_and_recovery(admission, monkeypatch):
    monkeypatch.setattr(server.time, "monotonic", lambda: 100)
    for _ in range(server.ANALYSES_PER_MINUTE):
        assert admission.post("/api/analyze").status_code == 422
    assert admission.post("/api/analyze").status_code == 429
    monkeypatch.setattr(server.time, "monotonic", lambda: 161)
    assert admission.post("/api/analyze").status_code == 422
    assert server._pending_analyzes == 0


def test_concurrent_uploads_reserve_capacity_and_release_on_failure(admission):
    async def run():
        request = SimpleNamespace(method="POST", url=SimpleNamespace(path="/api/analyze"))
        entered = asyncio.Event()
        release = asyncio.Event()
        async def blocked(_):
            entered.set()
            await release.wait()
            raise ValueError("upload failed")
        server._jobs["running"] = {"status": "running"}
        first = asyncio.create_task(server.limit_analysis(request, blocked))
        await entered.wait()
        response = await server.limit_analysis(request, blocked)
        assert response.status_code == 429
        release.set()
        with pytest.raises(ValueError):
            await first
        assert server._pending_analyzes == 0
    asyncio.run(run())


def test_revision_handles_utf8_and_missing_file(tmp_path):
    assert server.release_revision(tmp_path) is None
    (tmp_path / "REVISION").write_text("abc123\n", encoding="utf-8")
    assert server.release_revision(tmp_path) == "abc123"


def test_health_reports_backend_and_static_revisions(monkeypatch):
    catalog = SimpleNamespace(products=[], color_audits={}, color_override_count=0, color_mismatch_count=0)
    engine = SimpleNamespace(device="cpu", trained_heads=[], parser_backend="test",
                             tryon=SimpleNamespace(enabled=False, available=False),
                             recommender=SimpleNamespace(catalog=catalog, active_rule_ids=[], documented_rule_ids=[]))
    monkeypatch.setattr(server, "get_engine", lambda: engine)
    monkeypatch.setattr(server, "BACKEND_REVISION", "backend-old")
    monkeypatch.setattr(server, "release_revision", lambda root: "static-new")
    result = server.health()
    assert result["revision"] == "backend-old"
    assert result["static_revision"] == "static-new"


def test_catalog_quality_distinguishes_missing_cache_from_missing_measurements():
    products = [
        SimpleNamespace(measurement_record={}, color="", color_source="none"),
        SimpleNamespace(measurement_record={"status": "missing", "fetched_at": "2026-09-28T00:00:00+00:00"}, color="블랙", color_source="catalog"),
        SimpleNamespace(measurement_record={"status": "ready", "fetched_at": "2026-09-29T00:00:00+00:00"}, color="그린", color_source="photo"),
    ]
    quality = server.catalog_quality(products)
    assert quality["measurement_records"] == 2
    assert quality["measurements_ready"] == 1
    assert quality["measurements_missing"] == 1
    assert quality["unknown_colors"] == 1
    assert quality["photo_colors"] == 1
    assert quality["oldest_measurement_fetch"] == "2026-09-28T00:00:00+00:00"
