from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import jwt
from fastapi import Request

from app.core.config import Settings


class AuthenticationError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 401) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class Principal:
    subject: str
    principal_type: str
    scopes: List[str]
    roles: List[str]
    claims: Dict[str, Any]


def principal_to_context(principal: Principal) -> Dict[str, Any]:
    return {
        "subject": principal.subject,
        "principal_type": principal.principal_type,
        "roles": list(principal.roles),
        "scopes": list(principal.scopes),
    }


def principal_from_request(request: Request) -> Principal:
    value = getattr(request.state, "principal", None)
    if isinstance(value, Principal):
        return value
    return Principal(
        subject="anonymous",
        principal_type="anonymous",
        scopes=[],
        roles=[],
        claims={"auth_method": "none"},
    )


def _csv_values(raw: str) -> List[str]:
    return [value.strip() for value in raw.split(",") if value.strip()]


def is_public_path(path: str, settings: Settings) -> bool:
    for prefix in _csv_values(settings.auth_public_paths):
        if path == prefix or path.startswith(prefix):
            return True
    return False


def _decode_hs256_jwt(token: str, settings: Settings) -> Dict[str, Any]:
    if not settings.auth_jwt_hs256_secret:
        raise AuthenticationError(
            "AUTH_JWT_HS256_SECRET is required for HS256 token validation when auth is enabled.",
            status_code=500,
        )

    audience = settings.auth_oidc_audience.strip() or None
    issuer = settings.auth_oidc_issuer.strip() or None
    return jwt.decode(
        token,
        settings.auth_jwt_hs256_secret,
        algorithms=["HS256"],
        audience=audience,
        issuer=issuer,
        options={
            "verify_aud": audience is not None,
            "verify_iss": issuer is not None,
        },
    )


def _decode_oidc_jwt(token: str, settings: Settings) -> Dict[str, Any]:
    if not settings.auth_oidc_jwks_url:
        raise AuthenticationError(
            "Either AUTH_JWT_HS256_SECRET or AUTH_OIDC_JWKS_URL must be set when auth is enabled.",
            status_code=500,
        )

    jwks_client = jwt.PyJWKClient(settings.auth_oidc_jwks_url)
    signing_key = jwks_client.get_signing_key_from_jwt(token)
    audience = settings.auth_oidc_audience.strip() or None
    issuer = settings.auth_oidc_issuer.strip() or None
    algorithms = _csv_values(settings.auth_jwt_algorithms) or ["RS256"]

    return jwt.decode(
        token,
        signing_key.key,
        algorithms=algorithms,
        audience=audience,
        issuer=issuer,
        options={
            "verify_aud": audience is not None,
            "verify_iss": issuer is not None,
        },
    )


def _principal_from_claims(claims: Dict[str, Any]) -> Principal:
    subject = str(claims.get("sub", "")).strip()
    if not subject:
        raise AuthenticationError("Token is missing required 'sub' claim.")

    raw_scope = claims.get("scope", "")
    scopes = [scope.strip() for scope in str(raw_scope).split(" ") if scope.strip()]

    roles_claim = claims.get("roles", [])
    roles: List[str]
    if isinstance(roles_claim, list):
        roles = [str(role).strip() for role in roles_claim if str(role).strip()]
    else:
        roles = [role.strip() for role in str(roles_claim).split(",") if role.strip()]

    return Principal(
        subject=subject,
        principal_type="user",
        scopes=scopes,
        roles=roles,
        claims=claims,
    )


def authenticate_request(request: Request, settings: Settings) -> Principal:
    service_token = (request.headers.get("x-service-token") or "").strip()
    if service_token:
        allowed_service_tokens = set(_csv_values(settings.auth_service_tokens))
        if service_token in allowed_service_tokens:
            return Principal(
                subject="service",
                principal_type="service",
                scopes=["service"],
                roles=["service"],
                claims={"auth_method": "service_token"},
            )
        raise AuthenticationError("Invalid service token.", status_code=403)

    authorization = (request.headers.get("authorization") or "").strip()
    if not authorization:
        raise AuthenticationError("Missing Authorization bearer token.", status_code=401)

    if not authorization.lower().startswith("bearer "):
        raise AuthenticationError("Authorization header must be a Bearer token.", status_code=401)

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise AuthenticationError("Bearer token is empty.", status_code=401)

    try:
        if settings.auth_jwt_hs256_secret:
            claims = _decode_hs256_jwt(token, settings)
        else:
            claims = _decode_oidc_jwt(token, settings)
    except AuthenticationError:
        raise
    except jwt.PyJWTError as exc:
        raise AuthenticationError(f"Token validation failed: {exc}", status_code=401) from exc

    return _principal_from_claims(claims)
