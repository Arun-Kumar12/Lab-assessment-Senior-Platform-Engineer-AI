from dataclasses import dataclass

from fastapi import Depends, Header, HTTPException, Request

ROLE_RANK = {"viewer": 1, "operator": 2}


@dataclass
class Principal:
    key: str
    role: str


def authenticate(request: Request, x_api_key: str | None = Header(default=None)) -> Principal:
    if not x_api_key:
        raise HTTPException(status_code=401, detail="missing X-API-Key header")
    role = request.app.state.api_keys.get(x_api_key)
    if role is None:
        raise HTTPException(status_code=401, detail="invalid API key")
    return Principal(key=x_api_key, role=role)


def require_role(role: str):
    def dependency(principal: Principal = Depends(authenticate)) -> Principal:
        if ROLE_RANK.get(principal.role, 0) < ROLE_RANK[role]:
            raise HTTPException(status_code=403, detail=f"{role} role required")
        return principal

    return dependency
