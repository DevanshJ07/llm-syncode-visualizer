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
    Run TinyLlama greedy generation and return the complete decoding trace.

    The response contains:
      - generated_text       : the full decoded output
      - steps                : one entry per token, each with
            selected_token / selected_token_id
            top_tokens        (top-k candidates + probabilities)
            entropy_before    (Shannon entropy of the full vocab distribution)
            top_tokens_before_syncode / masked_tokens /
            valid_tokens_after_syncode / entropy_after / num_masked
                              (Syncode placeholder fields, empty until Phase 3)
      - experiment_id        : persisted to disk; retrieve later via
                               GET /experiment/{id}
    """
    experiment = store.create_empty(
        prompt=request.prompt,
        mode="raw",
        model_name=settings.model_name,
    )

    try:
        generated_text, steps = await llm_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            temperature=request.temperature,
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
        # --- generated output ---
        generated_text=generated_text,
        model_name=settings.model_name,
        mode="raw",
        prompt=request.prompt,
        total_steps=len(steps),
        # --- full decoding trace ---
        steps=steps,
    )
