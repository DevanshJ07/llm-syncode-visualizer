"""
POST /generate

Accepts a prompt + generation settings, runs TinyLlama token-by-token,
stores the full experiment JSON to disk, and returns the experiment ID.

The CPU-blocking inference runs inside a thread pool executor (inside
llm_service.generate) so this async route never stalls the event loop.

Model loads lazily on the first request — expect ~20-60 s cold start as
HuggingFace downloads and initialises TinyLlama.  Subsequent requests are
fast (model is cached in memory for the lifetime of the process).
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.experiment_store import store
from app.services.llm_service import llm_service
from app.core.config import settings

router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run greedy autoregressive generation and persist the experiment.

    Body:
        prompt         – text prompt fed to TinyLlama
        use_syncode    – ignored for now (Syncode not yet implemented)
        top_k          – how many top-probability candidates to log per step
        max_new_tokens – maximum tokens to generate
        temperature    – softmax temperature (affects logged probabilities;
                         token selection is always greedy / argmax)

    Returns:
        experiment_id  – use with GET /experiment/{id} to retrieve results
    """
    # Syncode is not implemented yet; all runs are "raw" mode.
    effective_mode = "raw"

    # Create a new experiment record (written to disk after generation)
    experiment = store.create_empty(
        prompt=request.prompt,
        mode=effective_mode,
        model_name=settings.model_name,
    )

    try:
        # llm_service.generate() is async — it dispatches the blocking
        # forward passes to a ThreadPoolExecutor so the event loop stays free.
        generated_code, steps = await llm_service.generate(
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
