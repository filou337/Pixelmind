"""API contracts (dossier §12.3). Every analysis response follows AnalysisResult."""

from typing import Literal

from pydantic import BaseModel


class QualityIssue(BaseModel):
    status: Literal["PASS", "WARN", "REVIEW", "FAIL"]
    code: str
    message: str


class InputInfo(BaseModel):
    filename: str
    sha256: str
    bytes: int
    format: str | None = None
    mode: str | None = None
    width: int | None = None
    height: int | None = None


class TestMetrics(BaseModel):
    f1: float | None
    iou: float | None
    precision: float | None
    recall: float | None
    ap: float | None


class ModelInfo(BaseModel):
    name: str
    kind: Literal["baseline", "ml", "dl"]
    version: str
    threshold: int


class ModelSummary(ModelInfo):
    test_metrics: TestMetrics


class ModelList(BaseModel):
    default: str | None
    models: list[ModelSummary]


class Region(BaseModel):
    bbox: list[int]  # [x0, y0, x1, y1] in pixels, x1/y1 exclusive
    area_px: int
    mean_score: float


class Prediction(BaseModel):
    changed_pixel_ratio: float  # share of pixels, NOT a geographic area
    changed_area_m2_estimate: float | None  # only when gsd_m is given
    n_regions: int
    regions: list[Region]  # largest first, at most 100


class PixelMetrics(BaseModel):
    """Against the reference mask, when one was uploaded."""

    tp: int
    fp: int
    fn: int
    tn: int
    precision: float | None
    recall: float | None
    f1: float | None
    iou: float | None
    fpr: float | None


class Parameters(BaseModel):
    min_region_px: int
    gsd_m: float | None


class AnalysisResult(BaseModel):
    analysis_id: str
    created_at: str
    pixelmind_version: str
    status: Literal["completed", "completed_with_warnings", "rejected"]
    inputs: dict[str, InputInfo]
    quality_status: Literal["PASS", "WARN", "REVIEW", "FAIL"]
    quality_issues: list[QualityIssue]
    model: ModelInfo
    model_test_metrics: TestMetrics
    parameters: Parameters
    prediction: Prediction | None
    metrics: PixelMetrics | None
    warnings: list[str]
    human_review_required: bool
    limitations: list[str]
    artifacts: dict[str, str]
    timings: dict[str, float]


class AnalysisSummary(BaseModel):
    analysis_id: str
    created_at: str
    status: str
    quality_status: str
    model_name: str | None


class Health(BaseModel):
    status: Literal["ok"]
    version: str
    models: int
