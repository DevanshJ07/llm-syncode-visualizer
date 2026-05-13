"""
FastAPI application entry point.

Run with:
    uvicorn main:app --reload --port 8000

The lifespan handler loads the model once at startup so the first
/generate request doesn't pay the cold-start penalty.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import generate as generate_router
from app.api.routes import experiments as experiments_router
from app.core.config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle hook."""
    # Attempt model load on startup.
    # Comment this out during UI-only development to skip the GPU requirement.
    # from app.services.llm_service import llm_service
    # llm_service.load_model()
    print(f"[Startup] {settings.app_name} ready.")
    yield
    print("[Shutdown] Cleaning up.")


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Research API for visualizing token-level LLM generation with Syncode.",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# CORS — allow the Next.js dev server and any origins in settings
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
app.include_router(generate_router.router, tags=["Generation"])
app.include_router(experiments_router.router, tags=["Experiments"])


@app.get("/health", tags=["Meta"])
async def health():
    return {"status": "ok", "app": settings.app_name}
