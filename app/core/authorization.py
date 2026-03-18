from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

from app.core.security import Principal


class AuthorizationError(RuntimeError):
    def __init__(self, detail: str, status_code: int = 403) -> None:
        super().__init__(detail)
        self.detail = detail
        self.status_code = status_code


@dataclass(frozen=True)
class AuthorizationRule:
    prefix: str
    methods: Optional[List[str]]
    any_role: List[str]


RULES: List[AuthorizationRule] = [
    AuthorizationRule(prefix="/agents", methods=None, any_role=["admin"]),
    AuthorizationRule(prefix="/train", methods=["POST"], any_role=["ml_operator"]),
    AuthorizationRule(prefix="/features", methods=["POST"], any_role=["ml_operator"]),
    AuthorizationRule(prefix="/ingest", methods=["POST"], any_role=["data_ingest"]),
    AuthorizationRule(prefix="/predict", methods=["POST"], any_role=["analyst", "ml_operator"]),
    AuthorizationRule(prefix="/query", methods=["POST"], any_role=["analyst", "ml_operator"]),
]


def _match_rule(path: str, method: str) -> Optional[AuthorizationRule]:
    upper_method = method.upper()
    for rule in RULES:
        if not path.startswith(rule.prefix):
            continue
        if rule.methods is None or upper_method in rule.methods:
            return rule
    return None


def authorize_request(path: str, method: str, principal: Principal) -> None:
    if principal.principal_type == "service":
        return

    rule = _match_rule(path, method)
    if rule is None:
        return

    role_set = set(principal.roles)
    if any(role in role_set for role in rule.any_role):
        return

    required = ", ".join(rule.any_role)
    present = ", ".join(sorted(role_set)) if role_set else "none"
    raise AuthorizationError(
        f"Forbidden: required role one of [{required}], present roles [{present}] for {method.upper()} {path}.",
        status_code=403,
    )
