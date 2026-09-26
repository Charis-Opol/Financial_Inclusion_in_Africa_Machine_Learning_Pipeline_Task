"""FastAPI app: POST /predict, GET /health.

Status-code contract:
    422  the client sent something invalid (schema/type/range/NaN/extra field)
    503  the service is up but not ready: no usable model, or no writable
         prediction log (serving unrecorded predictions would silently break
         monitoring and auditability) -- retryable
    500  a valid request hit a server-side failure (model/encoder error)

Run locally:  uvicorn fin_inclusion.serving.app:app --port 8000
"""
from __future__ import annotations

import logging
import re
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from fin_inclusion.serving.config import ServiceConfig
from fin_inclusion.serving.prediction_log import PredictionLogger
from fin_inclusion.serving.predictor import InferenceError, Prediction, Predictor
from fin_inclusion.serving.schemas import (
    ErrorResponse,
    HealthResponse,
    PredictionResponse,
    SurveyRecord,
)

logger = logging.getLogger("fin_inclusion.serving")

REQUEST_ID_HEADER = "X-Request-ID"
# Client-supplied IDs are echoed into logs, so only accept a safe charset
# (no newlines/control chars that could forge log lines); otherwise mint one.
_SAFE_REQUEST_ID = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class ModelNotReadyError(RuntimeError):
    pass


def _error(status_code: int, error: str, detail: object, request: Request) -> JSONResponse:
    body = ErrorResponse(error=error, detail=detail, request_id=getattr(request.state, "request_id", None))
    return JSONResponse(status_code=status_code, content=body.model_dump())


def _require_predictor(request: Request) -> Predictor:
    predictor: Predictor | None = request.app.state.predictor
    if predictor is None:
        raise ModelNotReadyError(request.app.state.load_error or "model not loaded")
    return predictor


def _prediction_event(
    request: Request, predictor: Predictor, record: SurveyRecord, prediction: Prediction, inference_ms: float
) -> dict:
    """One JSON-lines log record. Plain dict: encoding happens on the writer thread."""
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
        "request_id": request.state.request_id,
        "model_version": predictor.model_version,
        "features": record.model_dump(),
        "probability": prediction.probability,
        "prediction": prediction.label,
        "confidence": prediction.confidence,
        "decision_threshold": predictor.threshold,
        # Server-side time from request arrival to a ready prediction:
        # validation + threadpool hop + encoding + model. Excludes response
        # serialization and network time.
        "latency_ms": round((time.perf_counter_ns() - request.state.start_ns) / 1e6, 3),
        # Threadpool dispatch + encoding + model call. Under concurrent load
        # this is dominated by waiting for a worker thread / CPU share, not by
        # the model itself (~0.5 ms, see reports/inference_benchmark*.md):
        # a rising inference_ms at flat traffic means CPU starvation.
        "inference_ms": round(inference_ms, 3),
        "warnings": prediction.warnings,
    }


def create_app(config: ServiceConfig | None = None) -> FastAPI:
    config = config or ServiceConfig.from_env()
    logging.basicConfig(
        level=config.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # Load exactly once per process. A failed load does not crash the
        # process: it's logged with a traceback and reported by /health as
        # 503, so an orchestrator sees an unhealthy container with a reason
        # instead of a crash loop that hides the message.
        app.state.predictor = None
        app.state.load_error = None
        prediction_log = PredictionLogger(config.prediction_log_dir, config.prediction_log_queue_size)
        app.state.prediction_log = prediction_log
        try:
            prediction_log.start()
            predictor = Predictor.load(config.model_dir, config.inference_threads)
        except Exception as exc:
            app.state.load_error = f"{type(exc).__name__}: {exc}"
            logger.exception("Startup failed (model dir %s, log dir %s); serving 503 until fixed",
                             config.model_dir, config.prediction_log_dir)
        else:
            app.state.predictor = predictor
        yield
        prediction_log.close()

    app = FastAPI(
        title="Financial Inclusion — bank account prediction",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def assign_request_id(request: Request, call_next):
        request.state.start_ns = time.perf_counter_ns()
        incoming = request.headers.get(REQUEST_ID_HEADER, "")
        request.state.request_id = incoming if _SAFE_REQUEST_ID.match(incoming) else uuid.uuid4().hex
        response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request.state.request_id
        return response

    @app.exception_handler(RequestValidationError)
    async def on_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Drop pydantic's echoed `input`/`ctx`: a rejected NaN/Infinity would
        # otherwise be echoed back and crash JSON serialization (the response
        # encoder forbids NaN), turning a clean 422 into a 500.
        errors = [{"loc": e["loc"], "msg": e["msg"], "type": e["type"]} for e in exc.errors()]
        return _error(422, "invalid_request", errors, request)

    @app.exception_handler(ModelNotReadyError)
    async def on_not_ready(request: Request, exc: ModelNotReadyError) -> JSONResponse:
        response = _error(503, "model_unavailable", str(exc), request)
        response.headers["Retry-After"] = "30"
        return response

    @app.exception_handler(InferenceError)
    async def on_inference_error(request: Request, exc: InferenceError) -> JSONResponse:
        logger.error("Inference failed [request_id=%s]: %s", request.state.request_id, exc, exc_info=exc)
        return _error(500, "inference_failed", "The model could not score this request", request)

    @app.exception_handler(Exception)
    async def on_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error [request_id=%s]", getattr(request.state, "request_id", None))
        return _error(500, "internal_error", "Unexpected server error", request)

    @app.get(
        "/health",
        response_model=HealthResponse,
        responses={503: {"model": HealthResponse}},
    )
    async def health(request: Request):
        predictor: Predictor | None = request.app.state.predictor
        log_stats = request.app.state.prediction_log.stats()
        if predictor is None:
            body = HealthResponse(status="unavailable", detail=request.app.state.load_error, prediction_log=log_stats)
            return JSONResponse(status_code=503, content=body.model_dump())
        return HealthResponse(status="ok", model_version=predictor.model_version, prediction_log=log_stats)

    @app.post(
        "/predict",
        response_model=PredictionResponse,
        responses={422: {"model": ErrorResponse}, 500: {"model": ErrorResponse}, 503: {"model": ErrorResponse}},
    )
    async def predict(record: SurveyRecord, request: Request) -> PredictionResponse:
        predictor = _require_predictor(request)
        # Scoring is CPU-bound; running it on the threadpool keeps the event
        # loop free to accept other requests. XGBoost's C API releases the
        # GIL, so concurrent predictions genuinely run in parallel.
        inference_start = time.perf_counter_ns()
        prediction = await run_in_threadpool(predictor.predict, record)
        inference_ms = (time.perf_counter_ns() - inference_start) / 1e6
        request.app.state.prediction_log.log(
            _prediction_event(request, predictor, record, prediction, inference_ms)
        )
        return PredictionResponse(
            request_id=request.state.request_id,
            model_version=predictor.model_version,
            probability_bank_account=prediction.probability,
            predicted_bank_account=prediction.label,
            confidence=prediction.confidence,
            decision_threshold=predictor.threshold,
            warnings=prediction.warnings,
        )

    return app


app = create_app()
