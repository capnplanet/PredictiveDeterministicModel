from __future__ import annotations

import jwt
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


def _clear_settings_cache() -> None:
    get_settings.cache_clear()


def test_auth_enforced_blocks_protected_path_without_token(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ENFORCEMENT_MODE", "enforce")
    monkeypatch.setenv("AUTH_JWT_HS256_SECRET", "test-secret")
    _clear_settings_cache()

    client = TestClient(app)
    response = client.get("/_auth_probe")

    assert response.status_code == 401
    assert "Authorization" in response.json()["detail"]

    _clear_settings_cache()


def test_auth_enforced_allows_valid_hs256_bearer_token_on_protected_path(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ENFORCEMENT_MODE", "enforce")
    monkeypatch.setenv("AUTH_JWT_HS256_SECRET", "test-secret")
    _clear_settings_cache()

    token = jwt.encode({"sub": "user-123", "roles": ["analyst"]}, "test-secret", algorithm="HS256")

    client = TestClient(app)
    response = client.get("/_auth_probe", headers={"Authorization": f"Bearer {token}"})

    # Route does not exist, but auth must pass so the request reaches router resolution.
    assert response.status_code == 404

    _clear_settings_cache()


def test_public_health_path_remains_accessible_without_token_when_auth_enabled(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ENFORCEMENT_MODE", "enforce")
    monkeypatch.setenv("AUTH_JWT_HS256_SECRET", "test-secret")
    _clear_settings_cache()

    client = TestClient(app)
    response = client.get("/health/")

    assert response.status_code in {200, 503}

    _clear_settings_cache()


def test_authz_denies_training_without_operator_role(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ENFORCEMENT_MODE", "enforce")
    monkeypatch.setenv("AUTH_JWT_HS256_SECRET", "test-secret")
    _clear_settings_cache()

    token = jwt.encode({"sub": "user-123", "roles": ["analyst"]}, "test-secret", algorithm="HS256")

    client = TestClient(app)
    response = client.post("/train", headers={"Authorization": f"Bearer {token}"}, json={})

    assert response.status_code == 403
    assert "required role" in response.json()["detail"]

    _clear_settings_cache()


def test_authz_allows_training_with_operator_role_to_reach_validation(monkeypatch) -> None:
    monkeypatch.setenv("AUTH_ENABLED", "true")
    monkeypatch.setenv("AUTH_ENFORCEMENT_MODE", "enforce")
    monkeypatch.setenv("AUTH_JWT_HS256_SECRET", "test-secret")
    _clear_settings_cache()

    token = jwt.encode({"sub": "user-123", "roles": ["ml_operator"]}, "test-secret", algorithm="HS256")

    client = TestClient(app)
    response = client.post(
        "/train",
        headers={"Authorization": f"Bearer {token}"},
        json={"config": {"epochs": "bad-type"}},
    )

    # AuthZ passed; request reached body validation.
    assert response.status_code == 422

    _clear_settings_cache()
