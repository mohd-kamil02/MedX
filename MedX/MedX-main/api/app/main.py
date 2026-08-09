import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from .config import get_settings
from .db import engine
from .routers import forecast

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
settings = get_settings()

app = FastAPI(
    title="MedX API",
    description="Marketplace for near-expiry pharmaceuticals.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,   # never "*" — requests carry credentials
    allow_credentials=True,
    allow_methods=["GET", "POST", "PATCH", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

app.include_router(forecast.router)


@app.get("/health")
def health():
    """Liveness + database reachability. Used by the container healthcheck."""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_ok = True
    except Exception:
        logging.exception("health check: database unreachable")
        db_ok = False

    return {"status": "ok" if db_ok else "degraded", "database": db_ok}
