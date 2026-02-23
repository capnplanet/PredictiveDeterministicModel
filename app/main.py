from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import health, ingest, features, train, predict
from app.core.performance import emit_performance_event

app = FastAPI(title="Deterministic Multimodal Analytics")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def performance_metrics_middleware(request, call_next):
    from time import perf_counter

    started = perf_counter()
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
        )
        raise

    emit_performance_event(
        "api.request",
        status="ok",
        duration_ms=(perf_counter() - started) * 1000.0,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
    )
    return response

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(features.router)
app.include_router(train.router)
app.include_router(predict.router)
