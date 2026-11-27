"""Analysis use case: validate a T1/T2 upload, run a model, persist a traceable result.

Deterministic pipeline (dossier §2.1): input validation -> data quality -> inference ->
post-processing -> optional metrics against a reference mask -> artifacts + audit row.
"""

import hashlib
import io
import re
import secrets
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from pixelmind import __version__
from pixelmind.data.quality import (
    Issue,
    Status,
    binarize_mask,
    check_pair,
    normalize_mask,
    overall_status,
)
from pixelmind.evaluation.metrics import counts_at, metrics_from_counts
from pixelmind.models.registry import ModelRegistry
from pixelmind.services import postprocess
from pixelmind.services.store import AnalysisStore
from pixelmind.settings import Settings

ALLOWED_FORMATS = {"PNG", "JPEG", "TIFF"}
ARTIFACT_NAMES = {"t1.png", "t2.png", "reference_mask.png", "probability.png", "mask.png",
                  "overlay.png", "report.md"}  # fmt: skip
_ID_RE = re.compile(r"^pm_\d{8}_[0-9a-f]{8}$")

LIMITATIONS = [
    "Résultat d'aide à l'analyse : il signale des changements visuels candidats, "
    "principalement de bâtiments, et doit être vérifié par un humain.",
    "Ne prouve ni la légalité, ni la date, ni la nature d'une construction.",
    "Ombres, saisons, nuages ou décalage géométrique entre T1 et T2 peuvent créer "
    "des faux positifs.",
]


class InputError(ValueError):
    """Request-level error (HTTP 4xx), not stored."""

    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(message)
        self.status_code = status_code


class AnalysisRejected(Exception):
    """Blocking data-quality failure: the analysis is stored with status 'rejected'."""

    def __init__(self, result: dict) -> None:
        super().__init__(result["analysis_id"])
        self.result = result


@dataclass
class Upload:
    filename: str
    data: bytes


def new_analysis_id() -> str:
    return f"pm_{datetime.now(UTC):%Y%m%d}_{secrets.token_hex(4)}"


class AnalysisService:
    def __init__(self, settings: Settings, registry: ModelRegistry, store: AnalysisStore) -> None:
        self.settings, self.registry, self.store = settings, registry, store

    @classmethod
    def from_settings(cls, settings: Settings) -> "AnalysisService":
        return cls(
            settings,
            ModelRegistry.discover(settings.artifacts_dir),
            AnalysisStore(settings.analyses_dir / "analyses.db"),
        )

    # --- inputs ------------------------------------------------------------

    def _decode(self, upload: Upload, role: str) -> tuple[np.ndarray | None, dict, list[Issue]]:
        max_bytes = self.settings.max_upload_mb * 1024 * 1024
        if len(upload.data) > max_bytes:
            raise InputError(413, f"{role}: file larger than {self.settings.max_upload_mb} MB")
        info = {
            "filename": upload.filename,
            "sha256": hashlib.sha256(upload.data).hexdigest(),
            "bytes": len(upload.data),
        }
        try:
            with Image.open(io.BytesIO(upload.data)) as img:
                info |= {"format": img.format, "mode": img.mode, "width": img.width,
                         "height": img.height}  # fmt: skip
                if img.format not in ALLOWED_FORMATS:
                    msg = f"{role}: format {img.format} not accepted ({sorted(ALLOWED_FORMATS)})"
                    return None, info, [Issue(Status.FAIL, f"{role}_format", msg)]
                # size check on the header, before decoding: no decompression bombs
                if img.width * img.height > self.settings.max_pixels:
                    msg = (f"{role}: {img.width}x{img.height} exceeds "
                           f"{self.settings.max_pixels:,} pixels")  # fmt: skip
                    return None, info, [Issue(Status.FAIL, f"{role}_too_large", msg)]
                issues = []
                if role != "reference_mask" and img.mode in ("RGBA", "LA", "P", "CMYK"):
                    issues.append(Issue(Status.WARN, f"{role}_converted",
                                        f"{role}: mode {img.mode} converted to RGB"))  # fmt: skip
                    img = img.convert("RGB")
                elif role == "reference_mask" and img.mode in ("1", "P"):
                    img = img.convert("L")
                return np.asarray(img), info, issues
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            return None, info, [Issue(Status.FAIL, f"{role}_unreadable", f"{role}: {exc}")]

    # --- use case ----------------------------------------------------------

    def analyze(
        self, t1: Upload, t2: Upload, reference_mask: Upload | None = None,
        model: str | None = None, gsd_m: float | None = None,
    ) -> dict:  # fmt: skip
        t_start = time.perf_counter()
        try:
            entry = self.registry.get(model)
        except KeyError as exc:
            raise InputError(400, str(exc)) from None
        if gsd_m is not None and not 0 < gsd_m <= 1000:
            raise InputError(400, "gsd_m must be in ]0, 1000] metres per pixel")

        uploads = {"t1": t1, "t2": t2, "reference_mask": reference_mask}
        arrays, inputs, issues = {}, {}, []
        for role, up in uploads.items():
            if up is None:
                continue
            arrays[role], inputs[role], role_issues = self._decode(up, role)
            issues += role_issues
        if all(a is not None for a in arrays.values()):
            issues += check_pair(arrays["t1"], arrays["t2"], arrays.get("reference_mask"))
        quality = overall_status(issues)

        aid = new_analysis_id()
        result = {
            "analysis_id": aid,
            "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
            "pixelmind_version": __version__,
            "inputs": inputs,
            "quality_status": quality.name,
            "quality_issues": [i.to_dict() for i in issues],
            "model": {k: v for k, v in entry.summary().items() if k != "test_metrics"},
            "model_test_metrics": entry.summary()["test_metrics"],
            "parameters": {"min_region_px": self.settings.min_region_px, "gsd_m": gsd_m},
            "limitations": LIMITATIONS,
        }
        if quality is Status.FAIL:
            result |= {
                "status": "rejected",
                "prediction": None,
                "metrics": None,
                "warnings": [i.message for i in issues if i.status is Status.FAIL],
                "human_review_required": False,
                "artifacts": {},
                "timings": {"total_s": round(time.perf_counter() - t_start, 3)},
            }
            self.store.save(result)
            raise AnalysisRejected(result)

        a, b = arrays["t1"], arrays["t2"]
        t_inf = time.perf_counter()
        scores = entry.scorer()(a, b)
        inference_s = time.perf_counter() - t_inf
        mask = postprocess.to_mask(scores, entry.threshold, self.settings.min_region_px)
        n_regions, regions = postprocess.regions(mask, scores)
        ratio = float(mask.mean())

        metrics = None
        ref = arrays.get("reference_mask")
        if ref is not None:
            ref = binarize_mask(normalize_mask(ref)[0])
            metrics = metrics_from_counts(*counts_at(mask.astype(np.uint8), ref, 1))

        warnings = [i.message for i in issues if i.status is not Status.PASS]
        area_m2 = None
        if gsd_m is None:
            warnings.append("gsd_m (résolution au sol) non fourni : surfaces en m² non calculées")
        else:
            area_m2 = round(float(mask.sum()) * gsd_m**2, 1)
        if entry.kind != "dl":
            f1 = entry.test_metrics.get("f1") or 0
            warnings.append(
                f"modèle {entry.name} ({entry.kind}, F1 test {f1:.0%}) : "
                "beaucoup de faux positifs attendus"
            )
        limit = self.settings.review_changed_ratio
        ood = ratio > limit
        if ood:
            warnings.append(
                f"{ratio:.0%} de pixels changés, au-delà de {limit:.0%} "
                "(hors de la distribution d'entraînement)"
            )

        out_dir = self.settings.analyses_dir / aid
        out_dir.mkdir(parents=True, exist_ok=True)
        files = {
            "t1.png": a, "t2.png": b, "probability.png": scores,
            "mask.png": mask.astype(np.uint8) * 255,
            "overlay.png": postprocess.overlay(b, mask),
        }  # fmt: skip
        if ref is not None:
            files["reference_mask.png"] = ref.astype(np.uint8) * 255
        for name, arr in files.items():
            Image.fromarray(arr).save(out_dir / name)

        result |= {
            "status": "completed_with_warnings" if warnings else "completed",
            "prediction": {
                "changed_pixel_ratio": round(ratio, 6),
                "changed_area_m2_estimate": area_m2,
                "n_regions": n_regions,
                "regions": regions,
            },
            "metrics": metrics,
            "warnings": warnings,
            "human_review_required": quality is not Status.PASS or ood,
            "artifacts": {n: f"/api/v1/analyses/{aid}/artifacts/{n}" for n in files},
            "timings": {
                "inference_s": round(inference_s, 3),
                "total_s": round(time.perf_counter() - t_start, 3),
            },
        }
        self.store.save(result)
        return result

    def artifact_path(self, analysis_id: str, name: str) -> Path | None:
        """Path of an analysis artifact; None for anything outside the allowlist."""
        if not _ID_RE.match(analysis_id) or name not in ARTIFACT_NAMES:
            return None
        path = self.settings.analyses_dir / analysis_id / name
        return path if path.is_file() else None
