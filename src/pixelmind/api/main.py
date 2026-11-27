"""PIXELMIND REST API.

uvicorn pixelmind.api.main:app
"""

import threading
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse

from pixelmind import __version__
from pixelmind.api.schemas import (
    AnalysisQuestionRequest,
    AnalysisResult,
    AnalysisSummary,
    AnswerResult,
    Health,
    ModelList,
    QuestionRequest,
)
from pixelmind.rag.generator import GenerationError
from pixelmind.services.analysis import AnalysisRejected, AnalysisService, InputError, Upload
from pixelmind.settings import Settings


def create_app(service: AnalysisService | None = None, qa_service=None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service or AnalysisService.from_settings(Settings.from_env())
        app.state.qa = qa_service  # built on first question: loads the embedding model
        app.state.qa_lock = threading.Lock()
        yield

    app = FastAPI(
        title="PIXELMIND API",
        version=__version__,
        description="Détection de changements de bâtiments entre deux images (T1/T2). "
        "Les résultats sont des signaux d'aide à l'analyse, à vérifier par un humain.",
        lifespan=lifespan,
    )

    def get_service(request: Request) -> AnalysisService:
        return request.app.state.service

    Service = Annotated[AnalysisService, Depends(get_service)]

    def get_qa(request: Request):
        state = request.app.state
        with state.qa_lock:
            if state.qa is None:
                from pixelmind.rag.service import build_qa_service

                try:
                    state.qa = build_qa_service(state.service.settings)
                except FileNotFoundError as exc:
                    raise HTTPException(503, str(exc)) from None
        return state.qa

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
        reference_mask: Annotated[UploadFile | None, File(description="Ground truth")] = None,
        model: Annotated[str | None, Form(description="Model (default: best on test)")] = None,
        gsd_m: Annotated[float | None, Form(description="Ground sample distance, m/px")] = None,
    ):
        uploads = [await read_upload(f, svc) for f in (t1, t2)]
        ref = await read_upload(reference_mask, svc) if reference_mask else None
        try:
            return await run_in_threadpool(svc.analyze, *uploads, ref, model, gsd_m)
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
        return FileResponse(path, media_type="image/png")

    @app.post("/api/v1/questions", response_model=AnswerResult)
    def ask_docs(body: QuestionRequest, request: Request) -> dict:
        """Question on the project documentation, answered with citations."""
        return ask(get_qa(request), body.question, mode=body.mode, k=body.k,
                   doc_types=body.doc_types)  # fmt: skip

    @app.post("/api/v1/analyses/{analysis_id}/questions", response_model=AnswerResult)
    def ask_analysis(
        analysis_id: str, body: AnalysisQuestionRequest, request: Request, svc: Service
    ) -> dict:
        """Question on one analysis: its measured results plus the documentation."""
        result = svc.store.get(analysis_id)
        if result is None:
            raise HTTPException(404, "analysis not found")
        return ask(get_qa(request), body.question, analysis=result)

    return app


app = create_app()
