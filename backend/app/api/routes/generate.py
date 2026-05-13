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
    Run Qwen2.5-Coder greedy generation and return the complete decoding trace.

    When request.use_syncode=True the backend applies Syncode C-grammar
    masking at every step and populates the following fields per DecodingStep:
        top_tokens_before_syncode  — raw top-k with is_masked annotation
        masked_tokens              — token IDs masked by the grammar (top-k only)
        valid_tokens_after_syncode — constrained top-k after masking
        entropy_after              — Shannon entropy of the constrained distribution
        num_masked                 — total masked tokens across the full vocabulary

    The response always includes:
        generated_text / steps / total_steps / mode ("raw" | "syncode")
        experiment_id — persisted to disk; retrieve later via GET /experiment/{id}
    """
    mode = "syncode" if request.use_syncode else "raw"

    experiment = store.create_empty(
        prompt=request.prompt,
        mode=mode,
        model_name=settings.model_name,
    )

    try:
        generated_text, steps = await llm_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            temperature=request.temperature,
            use_syncode=request.use_syncode,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {exc}",
        ) from exc

    experiment.generated_code = generated_text
    experiment.steps = steps
    experiment.total_steps = len(steps)
    store.save(experiment)

    return GenerateResponse(
        experiment_id=experiment.experiment_id,
        status="completed",
        generated_text=generated_text,
        model_name=settings.model_name,
        mode=mode,
        prompt=request.prompt,
        total_steps=len(steps),
        steps=steps,
    )
