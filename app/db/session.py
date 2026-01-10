from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import get_settings

_settings = get_settings()
_engine = create_engine(_settings.database_url, echo=False, future=True)
_SessionLocal = sessionmaker(bind=_engine, autoflush=False, autocommit=False, expire_on_commit=False)


def get_engine():  # type: ignore[no-untyped-def]
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    return _SessionLocal


@contextmanager
def session_scope() -> Iterator[Session]:
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
