"""Runtime settings, read from environment variables (see .env.example)."""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    artifacts_dir: Path = Path("artifacts")
    max_upload_mb: float = 25.0
    max_pixels: int = 2048 * 2048  # synchronous analysis: bounded CPU time and memory
    min_region_px: int = 16  # connected regions smaller than this are dropped (4 m² at 0.5 m)
    # LEVIR-CD train max is ~25 % changed pixels: above this the scene is out of distribution
    review_changed_ratio: float = 0.30

    @property
    def analyses_dir(self) -> Path:
        return self.artifacts_dir / "analyses"

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            artifacts_dir=Path(env("PIXELMIND_ARTIFACTS_DIR", "artifacts")),
            max_upload_mb=float(env("PIXELMIND_MAX_UPLOAD_MB", cls.max_upload_mb)),
            max_pixels=int(env("PIXELMIND_MAX_PIXELS", cls.max_pixels)),
            min_region_px=int(env("PIXELMIND_MIN_REGION_PX", cls.min_region_px)),
        )
