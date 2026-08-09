"""Password hashing, JWT issuance, and the role dependencies.

Argon2id, not MD5. The original prototype hashed passwords with unsalted MD5,
which is reversible via rainbow table in seconds; for a service holding health
records that is the single most consequential thing to get right.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Annotated

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from .config import get_settings
from .db import get_db
from .models import Seller, User

settings = get_settings()
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

_hasher = PasswordHasher()  # argon2id defaults: 3 iterations, 64 MiB, 4 lanes


def hash_password(plain: str) -> str:
    return _hasher.hash(plain)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        _hasher.verify(hashed, plain)
        return True
    except VerifyMismatchError:
        return False


def create_access_token(user_id: uuid.UUID, role: str) -> str:
    now = datetime.now(timezone.utc)
    return jwt.encode(
        {
            "sub": str(user_id),
            "role": role,
            "iat": now,
            "exp": now + timedelta(minutes=settings.access_token_ttl_minutes),
            "typ": "access",
        },
        settings.jwt_secret,
        algorithm=settings.jwt_algorithm,
    )


CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[Session, Depends(get_db)],
) -> User:
    try:
        payload = jwt.decode(
            token, settings.jwt_secret, algorithms=[settings.jwt_algorithm]
        )
        if payload.get("typ") != "access":
            raise CREDENTIALS_ERROR
        user_id = uuid.UUID(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise CREDENTIALS_ERROR

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise CREDENTIALS_ERROR
    return user


CurrentUser = Annotated[User, Depends(get_current_user)]


def get_current_seller(
    user: CurrentUser, db: Annotated[Session, Depends(get_db)]
) -> Seller:
    """Resolve the caller's seller profile, enforcing the licence gate.

    Verification and licence expiry are checked here rather than at listing
    creation so that no future endpoint can accidentally skip the check.
    """
    seller = db.query(Seller).filter(Seller.user_id == user.id).one_or_none()
    if seller is None:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Not registered as a seller")
    if not seller.can_list:
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "Seller licence is unverified or expired; listing is not permitted",
        )
    return seller


CurrentSeller = Annotated[Seller, Depends(get_current_seller)]


def require_admin(user: CurrentUser) -> User:
    if user.role != "admin":
        raise HTTPException(status.HTTP_403_FORBIDDEN, "Admin access required")
    return user
