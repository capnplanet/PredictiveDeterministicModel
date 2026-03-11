from fastapi import APIRouter, HTTPException
from sqlalchemy import text

from app.db.session import session_scope

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/")
async def health_check() -> dict[str, str]:
    try:
        with session_scope() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail={"status": "degraded", "database": "error"}) from exc
    return {"status": "ok", "database": "ok"}
