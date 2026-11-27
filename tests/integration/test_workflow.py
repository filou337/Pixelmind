import json

import pytest
from fastapi.testclient import TestClient

from pixelmind.agents.workflow import ReviewWorkflow, quality_gate_rules
from pixelmind.api.main import create_app
from pixelmind.models.registry import ModelRegistry
from pixelmind.rag.generator import GenerationError
from pixelmind.services.analysis import AnalysisService
from pixelmind.services.store import AnalysisStore
from pixelmind.settings import Settings
from tests.integration.test_api import TEST_METRICS, pair, png


class FakeQA:
    def __init__(self, error=None):
        self.error, self.questions = error, []

    def ask(self, question, analysis=None, **_):
        self.questions.append((question, analysis["analysis_id"]))
        if self.error:
            raise self.error
        return {
            "question_id": "q_1", "question": question, "abstained": False,
            "answer": "Le modèle est une baseline : beaucoup de faux positifs [1].",
            "citations": [{"n": 1, "title": "Model card", "source": "docs/model_cards/x.md"}],
            "generation": {"model": "claude-opus-5-5", "provider": "claude"},
        }  # fmt: skip


@pytest.fixture
def env(tmp_path):
    results = tmp_path / "experiments" / "levircd"
    results.mkdir(parents=True)
    (results / "absdiff.json").write_text(json.dumps({"threshold": 60, "test": TEST_METRICS}))
    settings = Settings(artifacts_dir=tmp_path, min_region_px=4)
    svc = AnalysisService(settings, ModelRegistry.discover(tmp_path),
                          AnalysisStore(settings.analyses_dir / "analyses.db"))  # fmt: skip
    qa = FakeQA()
    with TestClient(create_app(svc, qa)) as client:
        yield client, svc, qa


def analyze(client, t1_mode=None, t2=None):
    t1, t2_, _ = pair()
    files = {"t1": ("t1.png", png(t1, t1_mode), "image/png"),
             "t2": ("t2.png", png(t2_ if t2 is None else t2), "image/png")}  # fmt: skip
    return client.post("/api/v1/analyses", files=files).json()


def test_gate_rules():
    base = {"status": "completed", "quality_status": "PASS", "quality_issues": [],
            "human_review_required": False}  # fmt: skip
    assert quality_gate_rules(base)["decision"] == "pass"
    assert quality_gate_rules({**base, "human_review_required": True})["reasons"] == [
        "résultat hors de la distribution d'entraînement"]  # fmt: skip
    warn = {**base, "quality_status": "WARN",
            "quality_issues": [{"status": "WARN", "code": "c", "message": "converti"}]}  # fmt: skip
    assert quality_gate_rules(warn) == {
        "decision": "review",
        "reasons": ["qualité des données WARN: converti"],
    }
    assert quality_gate_rules({**base, "status": "rejected", "warnings": ["x"]})["decision"] == (
        "rejected")  # fmt: skip


def test_clean_analysis_completes_without_review(env):
    client, _, qa = env
    a = analyze(client)
    r = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows", json={})
    assert r.status_code == 201, r.text
    wf = r.json()
    assert wf["status"] == "completed" and wf["gate"]["decision"] == "pass"
    assert [s["node"] for s in wf["trace"]] == ["load", "quality_gate", "report", "persist"]
    assert qa.questions == []  # explain not requested: no LLM call
    report = client.get(wf["report_uri"])
    assert report.status_code == 200 and report.headers["content-type"].startswith("text/markdown")
    assert "# Rapport d'analyse" in report.text and "validé automatiquement" in report.text


def test_review_pauses_then_resumes_with_feedback(env):
    client, _, qa = env
    a = analyze(client, t1_mode="RGBA")  # WARN -> human review
    wf = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows",
                     json={"explain": True}).json()  # fmt: skip
    assert wf["status"] == "awaiting_review"
    assert wf["pending_review"]["reasons"][0].startswith("qualité des données WARN")
    assert wf["research"]["answer"].endswith("[1].") and len(qa.questions) == 1
    assert wf["report_uri"] is None

    wid = wf["workflow_id"]
    assert client.get(f"/api/v1/analyses/{a['analysis_id']}/workflows").json()[0]["status"] == (
        "awaiting_review")  # fmt: skip
    review = {
        "decision": "corrected",
        "reviewer": "analyste",
        "comment": "un faux positif",
        "region_feedback": [{"bbox": [20, 10, 44, 30], "label": "false_positive"}],
    }
    done = client.post(f"/api/v1/workflows/{wid}/review", json=review).json()
    assert done["status"] == "completed" and done["review"]["decision"] == "corrected"
    assert [s["node"] for s in done["trace"]][-4:] == ["quality_gate", "human_review", "report",
                                                      "persist"]  # fmt: skip
    text = client.get(done["report_uri"]).text
    assert "corrected" in text and "false_positive" in text and "Explication documentaire" in text
    # a second review is refused
    assert client.post(f"/api/v1/workflows/{wid}/review", json=review).status_code == 409


def test_rejected_by_reviewer(env):
    client, _, _ = env
    a = analyze(client, t1_mode="RGBA")
    wid = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows", json={}).json()[
        "workflow_id"]  # fmt: skip
    done = client.post(f"/api/v1/workflows/{wid}/review",
                       json={"decision": "rejected", "reviewer": "r"}).json()  # fmt: skip
    assert done["status"] == "rejected"


def test_rejected_analysis_skips_research_and_review(env):
    client, _, qa = env
    t1, t2, _ = pair()
    a = analyze(client, t2=t2[:32])  # size mismatch -> rejected analysis
    wf = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows",
                     json={"explain": True}).json()  # fmt: skip
    assert wf["status"] == "rejected" and wf["gate"]["decision"] == "rejected"
    assert qa.questions == [] and wf["review"] is None
    assert "Aucune prédiction" in client.get(wf["report_uri"]).text


def test_llm_failure_degrades_gracefully(env):
    client, _, qa = env
    qa.error = GenerationError(503, "LLM non configuré")
    a = analyze(client)
    wf = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows",
                     json={"explain": True}).json()  # fmt: skip
    assert wf["status"] == "completed" and wf["research"] is None
    assert wf["warnings"] == ["explication documentaire indisponible : LLM non configuré"]


def test_review_survives_restart(env, tmp_path):
    client, svc, _ = env
    a = analyze(client, t1_mode="RGBA")
    wid = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows", json={}).json()[
        "workflow_id"]  # fmt: skip
    # a new process: fresh workflow object on the same checkpoint database
    db = svc.settings.artifacts_dir / "workflows" / "workflows.db"
    wf2 = ReviewWorkflow(svc.store, svc.settings.analyses_dir, lambda: None, db)
    assert wf2.get(wid)["status"] == "awaiting_review"
    assert wf2.review(wid, {"decision": "approved", "reviewer": "r"})["status"] == "completed"
    wf2.close()


def test_api_validation(env):
    client, _, _ = env
    assert client.post("/api/v1/analyses/pm_nope/workflows", json={}).status_code == 404
    assert client.get("/api/v1/workflows/wf_nope").status_code == 404
    assert (
        client.post(
            "/api/v1/workflows/wf_nope/review", json={"decision": "approved", "reviewer": "r"}
        ).status_code
        == 404
    )
    a = analyze(client)
    bad = {"decision": "maybe", "reviewer": "r"}
    wid = client.post(f"/api/v1/analyses/{a['analysis_id']}/workflows", json={}).json()[
        "workflow_id"]  # fmt: skip
    assert client.post(f"/api/v1/workflows/{wid}/review", json=bad).status_code == 422
    assert wid in {w["workflow_id"] for w in client.get("/api/v1/workflows").json()}
