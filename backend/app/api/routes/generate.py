"""
POST /generate

Accepts a prompt + settings, runs Llama 3B (with or without Syncode),
stores the full experiment JSON to disk, and returns the experiment ID.

The heavy work happens in llm_service / syncode_service; this route is
only responsible for request validation, orchestration, and error handling.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.experiment_store import store
from app.services.llm_service import llm_service
from app.services.syncode_service import syncode_service
from app.core.config import settings

router = APIRouter()


@router.post("/generate", response_model=GenerateResponse, status_code=status.HTTP_201_CREATED)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run constrained or unconstrained generation and persist the experiment.

    Body:
        prompt       – natural language or partial C code prompt
        use_syncode  – whether to apply Syncode grammar constraints
        top_k        – number of top-k candidates to log per step
        max_new_tokens
        temperature

    Returns:
        experiment_id – use this to poll GET /experiment/{id}
    """

    # Guard: refuse if the model isn't loaded yet (server is still warming up)
    if not llm_service.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded yet. Try again in a moment.",
        )

    # If use_syncode requested but Syncode is unavailable, warn and fall back.
    effective_mode = "syncode" if request.use_syncode else "raw"
    if request.use_syncode and not syncode_service.is_available:
        effective_mode = "raw"

    # Create a new experiment record
    experiment = store.create_empty(
        prompt=request.prompt,
        mode=effective_mode,
        model_name=settings.model_name,
    )

    try:
        # TODO: replace with real generation in Phase 2
        generated_code, steps = llm_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            temperature=request.temperature,
        )
        experiment.generated_code = generated_code
        experiment.steps = steps
        experiment.total_steps = len(steps)
        store.save(experiment)

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    return GenerateResponse(experiment_id=experiment.experiment_id)
