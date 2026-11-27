"""Runtime settings, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = Path("artifacts")
    max_upload_mb: float = 25.0
    max_pixels: int = 2048 * 2048  # synchronous analysis: bounded CPU time and memory
    min_region_px: int = 16  # connected regions smaller than this are dropped (4 m² at 0.5 m)
    # LEVIR-CD train max is ~25 % changed pixels: above this the scene is out of distribution
    review_changed_ratio: float = 0.30
    # RAG (Sprint 5)
    llm_provider: str = "claude"  # claude | extractive
    llm_model: str = "claude-opus-5-5"
    llm_effort: str = "medium"
    llm_fallbacks: bool = True  # server-side fallback model on a safety decline
    rag_mode: str = "hybrid"
    rag_k: int = 5
    # below this best dense cosine: abstain without calling the LLM. Deliberately under the
    # 0.45 measured in rag_eval.md (thin margin): borderline cases go to the LLM, which is
    # instructed to abstain when the passages do not answer.
    rag_min_score: float = 0.40
    # Sprint 7: API keys {name: key}; empty = open API (local demo only)
    api_keys: dict[str, str] = field(default_factory=dict)
    reference_stats: Path = Path("docs/data/levircd_reference_stats.json")

    @property
    def analyses_dir(self) -> Path:
        return self.artifacts_dir / "analyses"

    @property
    def rag_dir(self) -> Path:
        return self.artifacts_dir / "rag"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            artifacts_dir=Path(env("PIXELMIND_ARTIFACTS_DIR", "artifacts")),
            max_upload_mb=float(env("PIXELMIND_MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_pixels=int(env("PIXELMIND_MAX_PIXELS", cls.max_pixels)),
            min_region_px=int(env("PIXELMIND_MIN_REGION_PX", cls.min_region_px)),
            llm_provider=env("PIXELMIND_LLM_PROVIDER", cls.llm_provider),
            llm_model=env("PIXELMIND_LLM_MODEL", cls.llm_model),
            llm_effort=env("PIXELMIND_LLM_EFFORT", cls.llm_effort),
            llm_fallbacks=env("PIXELMIND_LLM_FALLBACKS", "1") not in ("0", "false", "no"),
            rag_mode=env("PIXELMIND_RAG_MODE", cls.rag_mode),
            rag_k=int(env("PIXELMIND_RAG_K", cls.rag_k)),
            rag_min_score=float(env("PIXELMIND_RAG_MIN_SCORE", cls.rag_min_score)),
            api_keys=parse_api_keys(env("PIXELMIND_API_KEYS", "")),
            reference_stats=Path(env("PIXELMIND_REFERENCE_STATS", str(cls.reference_stats))),
        )


def parse_api_keys(raw: str) -> dict[str, str]:
    """'alice:key1,bob:key2' -> {'alice': 'key1', 'bob': 'key2'}."""
    keys = {}
    for item in filter(None, (x.strip() for x in raw.split(","))):
        name, sep, key = item.partition(":")
        if not sep or not name or len(key) < 16:
            raise ValueError("PIXELMIND_API_KEYS: expected name:key pairs, keys >= 16 chars")
        keys[name] = key
    return keys
