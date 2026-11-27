"""Persistence of analyses: one SQLite row per analysis (audit trail) + files on disk.

SQLite (stdlib) for the MVP; the interface is small enough to move to PostgreSQL later.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analyses (
    analysis_id    TEXT PRIMARY KEY,
    created_at     TEXT NOT NULL,
    status         TEXT NOT NULL,
    quality_status TEXT NOT NULL,
    model_name     TEXT,
    model_version  TEXT,
    result         TEXT NOT NULL
)
"""


class AnalysisStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as con, con:
            con.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)  # one connection per call: thread-safe

    def save(self, result: dict) -> None:
        model = result.get("model") or {}
        with closing(self._connect()) as con, con:
            con.execute(
                "INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    result["analysis_id"], result["created_at"], result["status"],
                    result["quality_status"], model.get("name"), model.get("version"),
                    json.dumps(result),
                ),
            )  # fmt: skip

    def get(self, analysis_id: str) -> dict | None:
        with closing(self._connect()) as con:
            row = con.execute(
                "SELECT result FROM analyses WHERE analysis_id = ?", (analysis_id,)
            ).fetchone()
        return json.loads(row[0]) if row else None

    def recent(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT analysis_id, created_at, status, quality_status, model_name "
                "FROM analyses ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        keys = ("analysis_id", "created_at", "status", "quality_status", "model_name")
        return [dict(zip(keys, r, strict=True)) for r in rows]

    def results(self, limit: int = 1000) -> list[dict]:
        """Most recent full results (monitoring)."""
        with closing(self._connect()) as con:
            rows = con.execute(
                "SELECT result FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
