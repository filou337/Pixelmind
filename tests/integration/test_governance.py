import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from pixelmind.api.main import create_app
from pixelmind.models.registry import ModelRegistry
from pixelmind.monitoring.drift import FEATURES, build_reference, drift_report, image_stats, psi
from pixelmind.monitoring.report import build_monitoring
from pixelmind.services.analysis import AnalysisService
from pixelmind.services.store import AnalysisStore
from pixelmind.settings import Settings, parse_api_keys
from tests.integration.test_api import TEST_METRICS, pair, png

KEYS = {"alice": "a" * 20, "bob": "b" * 20}


def make_client(tmp_path, api_keys=None, reference=None):
    results = tmp_path / "experiments" / "levircd"
    results.mkdir(parents=True, exist_ok=True)
    (results / "absdiff.json").write_text(json.dumps({"threshold": 60, "test": TEST_METRICS}))
    settings = Settings(artifacts_dir=tmp_path, min_region_px=4, api_keys=api_keys or {},
                        reference_stats=reference or tmp_path / "none.json")  # fmt: skip
    svc = AnalysisService(settings, ModelRegistry.discover(tmp_path),
                          AnalysisStore(settings.analyses_dir / "analyses.db"))  # fmt: skip
    return TestClient(create_app(svc))


def upload(client, headers=None, seed=0, t1_mode=None):
    t1, t2, _ = pair(seed=seed)
    files = {
        "t1": ("t1.png", png(t1, t1_mode), "image/png"),
        "t2": ("t2.png", png(t2), "image/png"),
    }
    return client.post("/api/v1/analyses", files=files, headers=headers or {})


# --- authentication and audit trail ----------------------------------------


def test_api_keys_protect_everything_but_health(tmp_path):
    with make_client(tmp_path, KEYS) as c:
        assert c.get("/api/v1/health").status_code == 200
        r = c.get("/api/v1/models")
        assert r.status_code == 401 and r.headers["www-authenticate"] == "ApiKey"
        assert c.get("/api/v1/models", headers={"X-API-Key": "wrong" * 5}).status_code == 401
        assert upload(c).status_code == 401
        ok = upload(c, {"X-API-Key": KEYS["alice"]})
        assert ok.status_code == 201 and ok.json()["requested_by"] == "alice"


def test_audit_trail_records_who_did_what(tmp_path):
    with make_client(tmp_path, KEYS) as c:
        alice, bob = {"X-API-Key": KEYS["alice"]}, {"X-API-Key": KEYS["bob"]}
        aid = upload(c, alice, t1_mode="RGBA").json()["analysis_id"]  # WARN -> review
        wf = c.post(f"/api/v1/analyses/{aid}/workflows", json={}, headers=alice).json()
        assert wf["requested_by"] == "alice"
        c.post(f"/api/v1/workflows/{wf['workflow_id']}/review", headers=bob,
               json={"decision": "approved", "reviewer": "Bob B."})  # fmt: skip
        audit = c.get(f"/api/v1/analyses/{aid}/audit", headers=alice).json()
        assert audit["analysis"]["requested_by"] == "alice"
        review = audit["workflows"][0]["review"]
        assert review["submitted_by"] == "bob" and review["reviewer"] == "Bob B."
        assert audit["questions"] == []
        assert c.get("/api/v1/analyses/pm_x/audit", headers=alice).status_code == 404


def test_open_api_without_keys_and_request_ids(tmp_path):
    with make_client(tmp_path) as c:
        r = upload(c)
        assert r.json()["requested_by"] == "anonymous"
        assert len(r.headers["x-request-id"]) == 16
        echoed = c.get("/api/v1/health", headers={"X-Request-ID": "my-trace-01"})
        assert echoed.headers["x-request-id"] == "my-trace-01"
        bad = c.get("/api/v1/health", headers={"X-Request-ID": "bad id\\n"})
        assert bad.headers["x-request-id"] != "bad id\\n"


def test_parse_api_keys():
    assert parse_api_keys("") == {}
    assert parse_api_keys(f"alice:{'a' * 20}, bob:{'b' * 20}") == KEYS
    for bad in ("alice", "alice:short", ":" + "k" * 20):
        with pytest.raises(ValueError):
            parse_api_keys(bad)


# --- drift ------------------------------------------------------------------


def samples(n, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    return [{f: float(rng.normal(10 + shift, 1)) for f in FEATURES} for _ in range(n)]


def test_psi_detects_a_shift_but_not_noise():
    ref = build_reference(samples(500), "test")
    same = drift_report(ref, samples(100, seed=1))
    # a "modérée" level can occur by chance (p < 0.05 on 6 features); an alert must not
    assert same["status"] == "ok" and not same["alert"]
    shifted = drift_report(ref, samples(100, shift=2.0, seed=2))
    assert shifted["alert"] and shifted["action"]
    assert psi(ref["features"]["t2_brightness"], [100.0] * 50) > 1


def test_calibration_false_alarms_and_detection():
    """Reference of 445 images like LEVIR-CD train, windows of 30 analyses."""
    ref = build_reference(samples(445), "test")
    false_alarms = sum(drift_report(ref, samples(30, seed=s))["alert"] for s in range(100))
    detected = sum(drift_report(ref, samples(30, shift=1.0, seed=s))["alert"] for s in range(100))
    assert false_alarms <= 8 and detected >= 95


def test_drift_needs_enough_data_and_a_reference():
    ref = build_reference(samples(100), "test")
    assert drift_report(ref, samples(5))["status"] == "insufficient_data"
    assert drift_report(None, samples(50))["status"] == "no_reference"


def test_image_stats():
    t1 = np.full((4, 4, 3), 100, np.uint8)
    t2 = np.full((4, 4, 3), (150, 120, 90), np.uint8)
    s = image_stats(t1, t2)
    assert s["t2_brightness"] == 120 and s["pair_brightness_shift"] == 20
    assert s["pair_abs_diff"] == 20 and s["t2_red_share"] == round(150 / 360, 4)


# --- monitoring --------------------------------------------------------------


def test_monitoring_endpoint(tmp_path):
    ref_path = tmp_path / "ref.json"
    with make_client(tmp_path, reference=ref_path) as c:
        for seed in range(3):
            upload(c, seed=seed)
        stats = [a["input_stats"] for a in
                 (c.get(f"/api/v1/analyses/{x['analysis_id']}").json()
                  for x in c.get("/api/v1/analyses").json())]  # fmt: skip
        ref_path.write_text(json.dumps(build_reference(stats * 10, "synthetic")))
        m = c.get("/api/v1/monitoring").json()
    assert m["data_quality"]["analyses"] == 3
    assert m["data_quality"]["quality_status"]["PASS"]["count"] == 3
    assert m["model"]["usage"] == {"absdiff": 3}
    assert m["service"]["analysis_latency_s"]["n"] == 3
    assert m["drift"]["status"] == "insufficient_data"  # 3 < 20 analyses
    assert m["rag"]["questions"] == 0


def test_monitoring_cost_estimate():
    usage = {"input_tokens": 1_000_000, "output_tokens": 100_000}
    q = {
        "abstained": False,
        "latency_s": 1.0,
        "generation": {"llm_called": True, "model": "claude-opus-5-5", "usage": usage},
    }
    m = build_monitoring([], [], [q], None)
    assert m["rag"]["estimated_cost_usd"] == pytest.approx(4.0 + 2.0)
    unknown = {**q, "generation": {**q["generation"], "model": "some-other-model"}}
    assert build_monitoring([], [], [unknown], None)["rag"]["estimated_cost_usd"] is None
