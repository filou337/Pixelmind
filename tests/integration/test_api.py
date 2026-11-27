import io
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from pixelmind import __version__
from pixelmind.api.main import create_app
from pixelmind.models.registry import ModelRegistry
from pixelmind.services import postprocess
from pixelmind.services.analysis import AnalysisService
from pixelmind.services.store import AnalysisStore
from pixelmind.settings import Settings

TEST_METRICS = {"f1": 0.13, "iou": 0.07, "precision": 0.1, "recall": 0.17, "ap": 0.07}


def png(arr: np.ndarray, mode: str | None = None) -> bytes:
    buf = io.BytesIO()
    img = Image.fromarray(arr)
    (img.convert(mode) if mode else img).save(buf, format="PNG")
    return buf.getvalue()


def pair(size=64, seed=0):
    t1 = np.random.default_rng(seed).integers(60, 120, (size, size, 3), dtype=np.uint8)
    t2 = t1.copy()
    mask = np.zeros((size, size), np.uint8)
    t2[10:30, 20:44] = 250
    mask[10:30, 20:44] = 255
    return t1, t2, mask


@pytest.fixture
def settings(tmp_path):
    results = tmp_path / "experiments" / "levircd"
    results.mkdir(parents=True)
    (results / "absdiff.json").write_text(json.dumps({"threshold": 60, "test": TEST_METRICS}))
    # a DL result whose checkpoint is missing must not be served
    (results / "ghost.json").write_text(
        json.dumps({"threshold": 128, "test": {"f1": 0.9}, "extra": {"checkpoint": "nope.pt"}})
    )
    return Settings(artifacts_dir=tmp_path, max_upload_mb=0.5, max_pixels=200 * 200,
                    min_region_px=4)  # fmt: skip


@pytest.fixture
def client(settings):
    svc = AnalysisService(
        settings,
        ModelRegistry.discover(settings.artifacts_dir),
        AnalysisStore(settings.analyses_dir / "analyses.db"),
    )
    with TestClient(create_app(svc)) as c:
        yield c


def post(client, t1, t2, mask=None, **form):
    files = {"t1": ("t1.png", t1, "image/png"), "t2": ("t2.png", t2, "image/png")}
    if mask is not None:
        files["reference_mask"] = ("mask.png", mask, "image/png")
    return client.post("/api/v1/analyses", files=files, data=form)


def test_health(client):
    assert client.get("/api/v1/health").json() == {"status": "ok", "version": __version__,
                                                   "models": 1}  # fmt: skip


def test_models_lists_only_servable_models(client):
    body = client.get("/api/v1/models").json()
    assert body["default"] == "absdiff"
    assert [m["name"] for m in body["models"]] == ["absdiff"]
    assert body["models"][0]["threshold"] == 60


def test_analysis_end_to_end(client):
    t1, t2, mask = pair()
    r = post(client, png(t1), png(t2), png(mask), gsd_m="0.5")
    assert r.status_code == 201, r.text
    res = r.json()
    assert res["analysis_id"].startswith("pm_")
    assert res["quality_status"] == "PASS"
    assert res["status"] == "completed_with_warnings"  # baseline model warning
    assert res["model"] == {"name": "absdiff", "kind": "baseline", "threshold": 60,
                            "version": f"pixelmind-{__version__}"}  # fmt: skip
    pred = res["prediction"]
    assert pred["n_regions"] == 1
    assert pred["regions"][0]["bbox"] == [20, 10, 44, 30]
    assert pred["changed_pixel_ratio"] == pytest.approx(20 * 24 / 64**2, abs=1e-6)
    assert pred["changed_area_m2_estimate"] == pytest.approx(20 * 24 * 0.25)
    assert res["metrics"]["f1"] == pytest.approx(1.0)
    assert res["inputs"]["t1"]["width"] == 64
    assert res["limitations"]

    # persisted: same result, listed, artifacts downloadable
    assert client.get(f"/api/v1/analyses/{res['analysis_id']}").json() == res
    assert client.get("/api/v1/analyses").json()[0]["analysis_id"] == res["analysis_id"]
    for name, url in res["artifacts"].items():
        a = client.get(url)
        assert a.status_code == 200 and a.headers["content-type"] == "image/png", name
    mask_png = Image.open(io.BytesIO(client.get(res["artifacts"]["mask.png"]).content))
    assert np.array_equal(np.asarray(mask_png) > 0, mask > 0)


def test_no_gsd_means_no_area(client):
    t1, t2, _ = pair()
    res = post(client, png(t1), png(t2)).json()
    assert res["prediction"]["changed_area_m2_estimate"] is None
    assert res["metrics"] is None
    assert any("gsd_m" in w for w in res["warnings"])


def test_rgba_input_is_converted_with_warning(client):
    t1, t2, _ = pair()
    res = post(client, png(t1, "RGBA"), png(t2)).json()
    assert res["quality_status"] == "WARN"
    assert res["human_review_required"] is True
    assert {i["code"] for i in res["quality_issues"]} == {"t1_converted"}


def test_size_mismatch_is_rejected_and_stored(client):
    t1, t2, _ = pair()
    r = post(client, png(t1), png(t2[:32]))
    assert r.status_code == 422
    res = r.json()
    assert res["status"] == "rejected" and res["prediction"] is None
    assert "size_mismatch" in {i["code"] for i in res["quality_issues"]}
    assert client.get(f"/api/v1/analyses/{res['analysis_id']}").json()["status"] == "rejected"


@pytest.mark.parametrize(
    ("t1_bytes", "code"),
    [(b"not an image", "t1_unreadable"), (png(np.zeros((300, 300, 3), np.uint8)), "t1_too_large")],
)
def test_invalid_inputs_are_explained(client, t1_bytes, code):
    _, t2, _ = pair()
    r = post(client, t1_bytes, png(t2))
    assert r.status_code == 422
    assert code in {i["code"] for i in r.json()["quality_issues"]}


def test_upload_size_limit(client):
    _, t2, _ = pair()
    assert post(client, b"0" * (600 * 1024), png(t2)).status_code == 413


def test_unknown_model_and_bad_gsd(client):
    t1, t2, _ = pair()
    assert post(client, png(t1), png(t2), model="nope").status_code == 400
    assert post(client, png(t1), png(t2), gsd_m="-1").status_code == 400


@pytest.mark.parametrize(
    "path",
    [
        "/api/v1/analyses/pm_20260101_00000000/artifacts/mask.png",
        "/api/v1/analyses/..%2F..%2Fexperiments/artifacts/mask.png",
        "/api/v1/analyses/pm_20260101_00000000/artifacts/..%2Fanalyses.db",
        "/api/v1/analyses/unknown",
    ],
)
def test_not_found_and_no_path_traversal(client, path):
    assert client.get(path).status_code == 404


# --- post-processing --------------------------------------------------------


def test_to_mask_drops_small_regions():
    scores = np.zeros((20, 20), np.uint8)
    scores[2:8, 2:8] = 200  # 36 px
    scores[15, 15] = 200  # 1 px, noise
    mask = postprocess.to_mask(scores, threshold=100, min_region_px=4)
    assert mask.sum() == 36 and not mask[15, 15]
    n, regions = postprocess.regions(mask, scores)
    assert n == 1 and regions[0] == {"bbox": [2, 2, 8, 8], "area_px": 36, "mean_score": 0.784}
