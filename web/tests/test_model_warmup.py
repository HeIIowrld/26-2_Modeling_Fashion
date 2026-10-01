from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
import web.app as server


@pytest.mark.parametrize("status", ["warming", "failed"])
def test_unready_health_does_not_wait_for_model_lock(monkeypatch, status):
    monkeypatch.setattr(server, "_model_readiness", {"status": status, "elapsed_seconds": None})
    engine = Mock(side_effect=AssertionError("must not wait for engine lock"))
    monkeypatch.setattr(server, "get_engine", engine)
    response = TestClient(server.app).get("/api/health")
    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.headers["Retry-After"] == "15"
    engine.assert_not_called()


@pytest.mark.parametrize("path", ["/api/analyze", "/api/validate-photo", "/api/jobs/abc/tryon-products"])
def test_warmup_rejects_model_requests_before_upload_validation(monkeypatch, path):
    monkeypatch.setattr(server, "_model_readiness", {"status": "warming"})
    response = TestClient(server.app).post(path)
    assert response.status_code == 503
    assert "준비 중" in response.json()["detail"]


def test_options_remain_available_during_warmup(monkeypatch):
    monkeypatch.setattr(server, "_model_readiness", {"status": "warming"})
    assert TestClient(server.app).get("/api/options").status_code == 200


def test_warmup_failure_is_not_reported_as_ready(monkeypatch):
    monkeypatch.setattr(server, "_model_readiness", {"status": "warming"})
    monkeypatch.setattr(server, "warmup_engine", Mock(side_effect=RuntimeError("missing checkpoint")))
    server._warm_models()
    assert server._model_readiness["status"] == "failed"
    assert server._model_readiness["elapsed_seconds"] >= 0


def test_successful_warmup_releases_admission(monkeypatch):
    monkeypatch.setattr(server, "_model_readiness", {"status": "warming"})
    monkeypatch.setattr(server, "warmup_engine", Mock())
    server._warm_models()
    assert server._model_readiness["status"] == "ready"
    # Missing upload is normal validation, rather than a warmup rejection.
    assert TestClient(server.app).post("/api/validate-photo").status_code == 422


def test_outfit_warmup_loads_clothes_and_shoes_without_generating():
    from shoe_tryon import OutfitTryOn
    clothes, shoes, parser = Mock(), Mock(), Mock()
    adapter = OutfitTryOn(clothes, shoes, parser)
    adapter.warmup()
    clothes.warmup.assert_called_once_with()
    shoes.warmup.assert_called_once_with()
    clothes.generate.assert_not_called()
    shoes.generate.assert_not_called()


def test_catvton_warmup_loads_shared_transition_editor(monkeypatch):
    from catvton_tryon import CatVTONTryOn
    adapter = CatVTONTryOn()
    load, parser, schedule, editor = Mock(), Mock(), Mock(), Mock()
    monkeypatch.setattr(adapter, "_load_pipeline", load)
    monkeypatch.setattr(adapter, "_get_garment_parser", parser)
    monkeypatch.setattr(adapter, "_apply_scheduler", schedule)
    adapter.transition_editor = editor
    adapter.warmup()
    load.assert_called_once_with()
    schedule.assert_called_once_with(load.return_value)
    parser.assert_called_once_with()
    editor.warmup.assert_called_once_with()


def test_offloaded_shoe_warmup_touches_weights_once(monkeypatch, tmp_path):
    import torch
    from shoe_tryon import ShoeTryOn
    adapter = ShoeTryOn(tmp_path)
    pipe = Mock()
    monkeypatch.setattr(adapter, "_load_pipeline", lambda: pipe)
    monkeypatch.setattr(torch, "Generator", Mock())
    adapter.warmup()
    adapter.warmup()
    pipe.assert_called_once()
    assert pipe.call_args.kwargs["num_inference_steps"] == 1
    assert pipe.call_args.kwargs["image"].size == (256, 256)


def test_failed_shoe_warmup_can_retry(monkeypatch, tmp_path):
    import torch
    from shoe_tryon import ShoeTryOn
    adapter = ShoeTryOn(tmp_path)
    pipe = Mock(side_effect=[RuntimeError("load failed"), None])
    monkeypatch.setattr(adapter, "_load_pipeline", lambda: pipe)
    monkeypatch.setattr(torch, "Generator", Mock())
    with pytest.raises(RuntimeError):
        adapter.warmup()
    assert not adapter._warmed_up
    adapter.warmup()
    assert adapter._warmed_up
