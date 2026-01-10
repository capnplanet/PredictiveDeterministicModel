from __future__ import annotations

from fastapi import APIRouter

from app.db.session import session_scope
from app.services.feature_extraction import extract_features_for_pending

router = APIRouter(prefix="/features", tags=["features"])


@router.post("/extract", response_model=dict)
async def extract_pending_features() -> dict:
    with session_scope() as session:
        count = extract_features_for_pending(session)
    return {"updated_artifacts": count}
