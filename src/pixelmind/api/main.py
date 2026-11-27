"""PIXELMIND REST API.

uvicorn pixelmind.api.main:app
"""

import json
import logging
import re
import secrets
import threading
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from pixelmind import __version__
from pixelmind.agents.workflow import ReviewWorkflow
from pixelmind.api.schemas import (
    AnalysisQuestionRequest,
    AnalysisResult,
    AnalysisSummary,
    AnswerResult,
    AuditTrail,
    Health,
    ModelList,
    Monitoring,
    QuestionRequest,
    ReviewRequest,
    WorkflowRequest,
    WorkflowSummary,
    WorkflowView,
)
from pixelmind.monitoring.report import build_monitoring
from pixelmind.rag.generator import GenerationError
from pixelmind.rag.service import QALog
from pixelmind.services.analysis import AnalysisRejected, AnalysisService, InputError, Upload
from pixelmind.settings import Settings

log = logging.getLogger("pixelmind.access")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
PUBLIC_PATHS = {"/api/v1/health"}


def qa_for(state):
    """QA service of the app, built once on first use (loads the embedding model)."""
    with state.qa_lock:
        if state.qa is None:
            from pixelmind.rag.service import build_qa_service

            state.qa = build_qa_service(state.service.settings)
    return state.qa


def create_app(service: AnalysisService | None = None, qa_service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or AnalysisService.from_settings(Settings.from_env())
        app.state.qa = qa_service  # built on first question: loads the embedding model
        app.state.qa_lock = threading.Lock()
        svc = app.state.service
        if not svc.settings.api_keys:
            logging.getLogger("pixelmind").warning(
                "PIXELMIND_API_KEYS is not set: the API is open (local demo only)"
            )
        app.state.qa_log = QALog(svc.settings.rag_dir / "qa_log.db")
        app.state.workflow = ReviewWorkflow(
            svc.store,
            svc.settings.analyses_dir,
            lambda: qa_for(app.state),
            svc.settings.artifacts_dir / "workflows" / "workflows.db",
        )
        yield
        app.state.workflow.close()

    def authenticate(request: Request) -> None:
        """X-API-Key -> actor name, recorded in the audit trail (dossier §10.4–10.5)."""
        keys = request.app.state.service.settings.api_keys
        if not keys:
            request.state.actor = "anonymous"
            return
        if request.url.path in PUBLIC_PATHS:
            request.state.actor = "public"
            return
        given = request.headers.get("x-api-key", "")
        for name, key in keys.items():
            if secrets.compare_digest(given.encode(), key.encode()):
                request.state.actor = name
                return
        raise HTTPException(401, "missing or invalid X-API-Key",
                            headers={"WWW-Authenticate": "ApiKey"})  # fmt: skip

    app = FastAPI(
        title="PIXELMIND API",
        version=__version__,
        description="Détection de changements de bâtiments entre deux images (T1/T2). "
        "Les résultats sont des signaux d'aide à l'analyse, à vérifier par un humain.",
        lifespan=lifespan,
        dependencies=[Depends(authenticate)],
    )

    @app.middleware("http")
    async def access_log(request: Request, call_next):
        """One JSON log line per request, with a request id echoed in X-Request-ID."""
        rid = request.headers.get("x-request-id", "")
        rid = rid if _REQUEST_ID.match(rid) else secrets.token_hex(8)
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        log.info(json.dumps({
            "request_id": rid, "method": request.method, "path": request.url.path,
            "status": response.status_code,
            "duration_ms": round((time.perf_counter() - t0) * 1000, 1),
            "actor": getattr(request.state, "actor", None),
        }))  # fmt: skip
        return response

    def get_actor(request: Request) -> str:
        return request.state.actor

    Actor = Annotated[str, Depends(get_actor)]

    def get_service(request: Request) -> AnalysisService:
        return request.app.state.service

    Service = Annotated[AnalysisService, Depends(get_service)]

    def get_qa(request: Request):
        try:
            return qa_for(request.app.state)
        except FileNotFoundError as exc:
            raise HTTPException(503, str(exc)) from None

    def get_workflow(request: Request) -> ReviewWorkflow:
        return request.app.state.workflow

    Workflow = Annotated[ReviewWorkflow, Depends(get_workflow)]

    def ask(qa, question: str, **kwargs) -> dict:
        try:
            return qa.ask(question, **kwargs)
        except GenerationError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None

    async def read_upload(f: UploadFile, svc: AnalysisService) -> Upload:
        limit = int(svc.settings.max_upload_mb * 1024 * 1024)
        data = await f.read(limit + 1)  # never read more than the limit
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename}: larger than {svc.settings.max_upload_mb} MB")
        return Upload(f.filename or "", data)

    @app.get("/api/v1/health", response_model=Health)
    def health(svc: Service) -> dict:
        return {"status": "ok", "version": __version__, "models": len(svc.registry.entries)}

    @app.get("/api/v1/models", response_model=ModelList)
    def models(svc: Service) -> dict:
        default = svc.registry.default()
        return {
            "default": default.name if default else None,
            "models": [e.summary() for e in svc.registry.entries.values()],
        }

    @app.post(
        "/api/v1/analyses",
        response_model=AnalysisResult,
        status_code=201,
        responses={422: {"model": AnalysisResult, "description": "Rejected by quality checks"}},
    )
    async def create_analysis(
        t1: Annotated[UploadFile, File(description="Image at date T1 (PNG, JPEG or TIFF, RGB)")],
        t2: Annotated[UploadFile, File(description="Image at date T2, same size and area")],
        svc: Service,
        actor: Actor,
        reference_mask: Annotated[UploadFile | None, File(description="Ground truth")] = None,
        model: Annotated[str | None, Form(description="Model (default: best on test)")] = None,
        gsd_m: Annotated[float | None, Form(description="Ground sample distance, m/px")] = None,
    ):
        uploads = [await read_upload(f, svc) for f in (t1, t2)]
        ref = await read_upload(reference_mask, svc) if reference_mask else None
        try:
            return await run_in_threadpool(svc.analyze, *uploads, ref, model, gsd_m, actor)
        except InputError as exc:
            raise HTTPException(exc.status_code, str(exc)) from None
        except AnalysisRejected as exc:
            return JSONResponse(AnalysisResult(**exc.result).model_dump(), status_code=422)

    @app.get("/api/v1/analyses", response_model=list[AnalysisSummary])
    def list_analyses(svc: Service, limit: int = 50) -> list:
        return svc.store.recent(min(max(limit, 1), 500))

    @app.get("/api/v1/analyses/{analysis_id}", response_model=AnalysisResult)
    def get_analysis(analysis_id: str, svc: Service) -> dict:
        result = svc.store.get(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found")
        return result

    @app.get("/api/v1/analyses/{analysis_id}/artifacts/{name}", response_class=FileResponse)
    def get_artifact(analysis_id: str, name: str, svc: Service):
        path = svc.artifact_path(analysis_id, name)
        if path is None:
            raise HTTPException(404, "artifact not found")
        media = "text/markdown; charset=utf-8" if name.endswith(".md") else "image/png"
        return FileResponse(path, media_type=media)

    @app.post("/api/v1/questions", response_model=AnswerResult)
    def ask_docs(body: QuestionRequest, request: Request, actor: Actor) -> dict:
        """Question on the project documentation, answered with citations."""
        return ask(get_qa(request), body.question, mode=body.mode, k=body.k,
                   doc_types=body.doc_types, asked_by=actor)  # fmt: skip

    @app.post("/api/v1/analyses/{analysis_id}/questions", response_model=AnswerResult)
    def ask_analysis(
        analysis_id: str,
        body: AnalysisQuestionRequest,
        request: Request,
        svc: Service,
        actor: Actor,
    ) -> dict:
        """Question on one analysis: its measured results plus the documentation."""
        result = svc.store.get(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found")
        return ask(get_qa(request), body.question, analysis=result, asked_by=actor)

    # --- agents workflow (Sprint 6) -----------------------------------------

    @app.post("/api/v1/analyses/{analysis_id}/workflows", response_model=WorkflowView,
              status_code=201)  # fmt: skip
    def start_workflow(
        analysis_id: str, body: WorkflowRequest, svc: Service, wf: Workflow, actor: Actor
    ) -> dict:
        """Quality gate, optional sourced explanation, human review if needed, report."""
        if svc.store.get(analysis_id) is None:
            raise HTTPException(404, "analysis not found")
        return wf.start(analysis_id, body.explain, body.question, requested_by=actor)

    @app.get("/api/v1/analyses/{analysis_id}/workflows", response_model=list[WorkflowSummary])
    def analysis_workflows(analysis_id: str, wf: Workflow) -> list:
        return wf.index.list(analysis_id)

    @app.get("/api/v1/workflows", response_model=list[WorkflowSummary])
    def list_workflows(wf: Workflow, limit: int = 50) -> list:
        return wf.index.list(limit=min(max(limit, 1), 500))

    @app.get("/api/v1/workflows/{workflow_id}", response_model=WorkflowView)
    def get_workflow_view(workflow_id: str, wf: Workflow) -> dict:
        view = wf.get(workflow_id)
        if view is None:
            raise HTTPException(404, "workflow not found")
        return view

    @app.post("/api/v1/workflows/{workflow_id}/review", response_model=WorkflowView)
    def submit_review(workflow_id: str, body: ReviewRequest, wf: Workflow, actor: Actor) -> dict:
        try:
            return wf.review(workflow_id, {**body.model_dump(), "submitted_by": actor})
        except KeyError:
            raise HTTPException(404, "workflow not found") from None
        except ValueError as exc:
            raise HTTPException(409, str(exc)) from None

    @app.get("/api/v1/rag/index")
    def rag_index(svc: Service) -> dict:
        """Manifest of the documentary index (corpus version, embeddings, documents)."""
        path = svc.settings.rag_dir / "index_manifest.json"
        if not path.exists():
            raise HTTPException(404, "index RAG absent")
        return json.loads(path.read_text())

    # --- governance (Sprint 7) ----------------------------------------------

    @app.get("/api/v1/monitoring", response_model=Monitoring)
    def monitoring(request: Request, svc: Service, wf: Workflow) -> dict:
        """Data quality, model outputs, latency, reviews, RAG usage and cost, input drift."""
        return build_monitoring(
            svc.store.results(),
            wf.index.list(limit=5000),
            request.app.state.qa_log.results(),
            svc.settings.reference_stats,
        )

    @app.get("/api/v1/analyses/{analysis_id}/audit", response_model=AuditTrail)
    def audit(analysis_id: str, request: Request, svc: Service, wf: Workflow) -> dict:
        """Everything needed to rebuild a result without the conversation (dossier §10.5)."""
        result = svc.store.get(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found")
        flows = [wf.get(w["workflow_id"]) for w in wf.index.list(analysis_id)]
        questions = request.app.state.qa_log.results(analysis_id=analysis_id)
        return {"analysis": result, "workflows": [f for f in flows if f], "questions": questions}

    return app


app = create_app()
