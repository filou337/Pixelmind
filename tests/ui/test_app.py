"""Streamlit pages render (no exception) with the API down and with a fake API."""

import sys
import types
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[2] / "app"
PAGES = ["Accueil.py", "pages/1_Nouvelle_analyse.py", "pages/2_Recherche_IA.py",
         "pages/3_Gouvernance.py"]  # fmt: skip

MODEL = {
    "name": "gbm",
    "kind": "ml",
    "version": "sha256:x",
    "threshold": 200,
    "test_metrics": {"f1": 0.43, "iou": 0.27, "precision": 0.38, "recall": 0.5, "ap": 0.4},
}
ANALYSIS = {
    "analysis_id": "pm_1",
    "created_at": "2026-11-27",
    "status": "completed",
    "quality_status": "PASS",
    "model_name": "gbm",
}
FAKE = {
    "/api/v1/health": {"status": "ok", "version": "0.0.1", "models": 1},
    "/api/v1/models": {"default": "gbm", "models": [MODEL]},
    "/api/v1/analyses": [ANALYSIS],
    "/api/v1/workflows": [{"workflow_id": "wf_1", "analysis_id": "pm_1", "created_at": "x",
                           "updated_at": "x", "status": "awaiting_review"}],
    "/api/v1/rag/index": {"n_chunks": 3, "corpus_sha256": "abcdef123456789", "created_at": "x",
                          "embedder": "e", "documents": [{"doc_id": "d", "doc_type": "adr",
                                                          "path": "p", "licence": "l"}]},
    "/api/v1/monitoring": {
        "data_quality": {"analyses": 2, "quality_status": {"PASS": {"count": 2, "share": 1.0}},
                         "rejected_reasons": {}},
        "model": {"usage": {"gbm": 2}, "changed_pixel_ratio": {"n": 2, "p50": 0.1, "p95": 0.2},
                  "empty_mask_rate": 0.0, "over_half_changed_rate": 0.0,
                  "f1_vs_reference": {"n": 0, "p50": None, "p95": None}},
        "service": {"analysis_latency_s": {"n": 2, "p50": 1.0, "p95": 2.0},
                    "inference_latency_s": {"n": 2, "p50": 0.5, "p95": 1.0}},
        "review": {"workflows": 1, "status": {}, "decisions": {}, "review_rate": 0.0},
        "rag": {"questions": 0, "abstention_rate": None, "llm_calls": 0,
                "tokens": {"input": 0, "output": 0}, "estimated_cost_usd": 0.0,
                "latency_s": {"n": 0, "p50": None, "p95": None}},
        "drift": {"status": "ok", "n": 30, "alert": False, "action": None,
                  "features": {"t2_brightness": {"label": "l", "psi": 0.4, "level": "stable",
                                                 "reference_mean": 82.0, "current_mean": 120.0}}},
    },
}  # fmt: skip


@pytest.fixture(autouse=True)
def app_path(monkeypatch):
    monkeypatch.syspath_prepend(str(APP))
    for mod in ("api_client", "ui"):
        sys.modules.pop(mod, None)


@pytest.mark.parametrize("page", PAGES)
def test_pages_explain_when_api_is_down(page, monkeypatch):
    monkeypatch.setenv("PIXELMIND_API_URL", "http://127.0.0.1:9")  # nothing listens there
    at = AppTest.from_file(str(APP / page), default_timeout=30).run()
    assert not at.exception
    assert any("API injoignable" in e.value for e in at.error)


@pytest.mark.parametrize("page", PAGES)
def test_pages_render_with_fake_api(page):
    import api_client

    class ApiError(RuntimeError):
        def __init__(self, status, detail):
            super().__init__(detail)
            self.status, self.detail = status, detail

    fake = types.ModuleType("api_client")
    fake.ApiError = ApiError
    fake.get = lambda path, **_: FAKE[path]
    fake.post = lambda *a, **k: {}
    fake.artifact = lambda uri: b""
    sys.modules["api_client"] = fake
    try:
        at = AppTest.from_file(str(APP / page), default_timeout=30).run()
        assert not at.exception, at.exception
        assert not at.error
    finally:
        sys.modules["api_client"] = api_client
