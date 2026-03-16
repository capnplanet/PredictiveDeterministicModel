from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import agent, demo, features, health, ingest, predict, query, train
from app.core.performance import correlation_scope, emit_performance_event

app = FastAPI(title="Deterministic Multimodal Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def performance_metrics_middleware(request: Request, call_next):
    from time import perf_counter

    incoming_correlation = (request.headers.get("x-correlation-id") or "").strip()
    correlation_id = incoming_correlation or uuid4().hex
    started = perf_counter()
    with correlation_scope(correlation_id):
        try:
            response = await call_next(request)
        except Exception as exc:  # noqa: BLE001
            emit_performance_event(
                "api.request",
                status="error",
                duration_ms=(perf_counter() - started) * 1000.0,
                method=request.method,
                path=request.url.path,
                error_type=type(exc).__name__,
                correlation_source="header" if incoming_correlation else "generated",
            )
            raise

        response.headers["x-correlation-id"] = correlation_id
        emit_performance_event(
            "api.request",
            status="ok",
            duration_ms=(perf_counter() - started) * 1000.0,
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            correlation_source="header" if incoming_correlation else "generated",
        )
        return response

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(demo.router)
app.include_router(features.router)
app.include_router(train.router)
app.include_router(predict.router)
app.include_router(query.router)
app.include_router(agent.router)
