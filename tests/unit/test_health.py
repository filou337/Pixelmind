from fastapi.testclient import TestClient

from pixelmind import __version__
from pixelmind.api.main import app


def test_health_returns_ok():
    response = TestClient(app).get("/api/v1/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}
