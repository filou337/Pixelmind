"""API contracts (dossier §12.3). Every analysis response follows AnalysisResult."""

from typing import Literal

from pydantic import BaseModel, Field


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
    requested_by: str = "anonymous"
    input_stats: dict[str, float] | None = None  # drift monitoring


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


class QuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    mode: Literal["dense", "sparse", "hybrid"] | None = None
    k: int | None = Field(default=None, ge=1, le=20)
    doc_types: list[str] | None = None


class AnalysisQuestionRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)


class Citation(BaseModel):
    n: int  # the [n] marker in the answer
    source: str
    title: str
    cited_text: str
    chunk_id: str | None = None
    doc_id: str | None = None
    doc_type: str | None = None
    section: str | None = None
    page: int | None = None
    analysis_id: str | None = None


class RetrievedHit(BaseModel):
    rank: int
    score: float
    chunk_id: str | None
    doc_id: str | None
    section: str | None
    page: int | None


class RetrievalInfo(BaseModel):
    mode: str
    k: int
    doc_types: list[str] | None
    best_dense_score: float
    min_score: float
    relevant: bool
    hits: list[RetrievedHit]


class GenerationInfo(BaseModel):
    provider: str
    model: str | None
    prompt_version: str
    stop_reason: str | None
    usage: dict[str, int]
    llm_called: bool


class AnswerResult(BaseModel):
    question_id: str
    created_at: str
    question: str
    asked_by: str = "anonymous"
    analysis_id: str | None
    answer: str
    abstained: bool
    refused: bool
    citations: list[Citation]
    retrieval: RetrievalInfo
    generation: GenerationInfo
    index: dict[str, str | int | None]
    latency_s: float


class WorkflowRequest(BaseModel):
    explain: bool = False  # ask the RAG for a sourced explanation (LLM call)
    question: str | None = Field(default=None, min_length=3, max_length=1000)


class RegionFeedback(BaseModel):
    bbox: list[int] = Field(min_length=4, max_length=4)
    label: Literal["confirmed", "false_positive", "missed_change"]
    note: str | None = Field(default=None, max_length=500)


class ReviewRequest(BaseModel):
    decision: Literal["approved", "corrected", "rejected"]
    reviewer: str = Field(min_length=1, max_length=100)
    comment: str | None = Field(default=None, max_length=2000)
    region_feedback: list[RegionFeedback] = Field(default_factory=list, max_length=100)


class WorkflowSummary(BaseModel):
    workflow_id: str
    analysis_id: str
    created_at: str
    updated_at: str
    status: str
    review_decision: str | None = None


class WorkflowView(BaseModel):
    workflow_id: str
    analysis_id: str
    requested_by: str = "anonymous"
    status: Literal["running", "awaiting_review", "completed", "rejected"]
    pending_review: dict | None
    gate: dict | None
    review: dict | None
    research: dict | None
    report_uri: str | None
    warnings: list[str]
    trace: list[dict]


class Monitoring(BaseModel):
    data_quality: dict
    model: dict
    service: dict
    review: dict
    rag: dict
    drift: dict


class AuditTrail(BaseModel):
    analysis: AnalysisResult
    workflows: list[WorkflowView]
    questions: list[AnswerResult]
