from uuid import uuid4

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import agent, demo, features, health, ingest, predict, query, train
from app.core.authorization import AuthorizationError, authorize_request
from app.core.config import get_settings
from app.core.performance import correlation_scope, emit_performance_event
from app.core.security import AuthenticationError, Principal, authenticate_request, is_public_path

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
    settings = get_settings()

    principal = Principal(
        subject="anonymous",
        principal_type="anonymous",
        scopes=[],
        roles=[],
        claims={"auth_method": "none"},
    )

    with correlation_scope(correlation_id):
        if settings.auth_enabled and not is_public_path(request.url.path, settings):
            try:
                principal = authenticate_request(request, settings)
            except AuthenticationError as exc:
                mode = settings.auth_enforcement_mode.strip().lower()
                emit_performance_event(
                    "security.authn",
                    status="error",
                    mode=mode,
                    path=request.url.path,
                    method=request.method,
                    reason=exc.detail,
                    principal_type="anonymous",
                )

                if mode == "observe":
                    request.state.principal = principal
                else:
                    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            else:
                emit_performance_event(
                    "security.authn",
                    status="ok",
                    mode=settings.auth_enforcement_mode.strip().lower(),
                    path=request.url.path,
                    method=request.method,
                    principal_type=principal.principal_type,
                )

            try:
                authorize_request(request.url.path, request.method, principal)
            except AuthorizationError as exc:
                mode = settings.auth_enforcement_mode.strip().lower()
                emit_performance_event(
                    "security.authz",
                    status="error",
                    mode=mode,
                    path=request.url.path,
                    method=request.method,
                    reason=exc.detail,
                    principal_type=principal.principal_type,
                    principal_subject=principal.subject,
                )

                if mode != "observe":
                    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
            else:
                emit_performance_event(
                    "security.authz",
                    status="ok",
                    mode=settings.auth_enforcement_mode.strip().lower(),
                    path=request.url.path,
                    method=request.method,
                    principal_type=principal.principal_type,
                    principal_subject=principal.subject,
                )

        request.state.principal = principal

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
