"""Thin HTTP client for the PIXELMIND API. The UI holds no business logic (dossier §13.1)."""

import os

import requests

API_URL = os.environ.get("PIXELMIND_API_URL", "http://localhost:8000").rstrip("/")
TIMEOUT = 300  # analyses and LLM answers can take a while on CPU


class ApiError(RuntimeError):
    def __init__(self, status: int, detail) -> None:
        super().__init__(f"{status}: {detail}")
        self.status, self.detail = status, detail


def _call(method: str, path: str, **kwargs):
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
    except requests.RequestException as exc:
        raise ApiError(0, f"API injoignable ({API_URL}) : {exc.__class__.__name__}") from None
    if r.status_code == 422 and r.headers.get("content-type", "").startswith("application/json"):
        body = r.json()
        if isinstance(body, dict) and body.get("status") == "rejected":
            return body  # rejected analysis: a normal, explained result
    if r.status_code >= 400:
        try:
            detail = r.json().get("detail", r.text)
        except ValueError:
            detail = r.text
        raise ApiError(r.status_code, detail)
    return r


def get(path: str, **params):
    return _call("GET", path, params=params).json()


def post(path: str, json=None, files=None, data=None):
    r = _call("POST", path, json=json, files=files, data=data)
    return r if isinstance(r, dict) else r.json()


def artifact(uri: str) -> bytes:
    return _call("GET", uri).content
