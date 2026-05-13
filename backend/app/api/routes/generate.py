from __future__ import annotations

import logging

from fastapi import APIRouter, status

from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.experiment_store import store
from app.services.llm_service import llm_service
from app.core.config import settings

log = logging.getLogger(__name__)
router = APIRouter()


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run Qwen2.5-Coder greedy generation and return the complete decoding trace.

    This endpoint NEVER returns HTTP 500.  If the generation loop encounters
    any error (Syncode parser exception, grammar failure, unexpected model
    error) the response will have status="error" and message will describe
    what went wrong, but generated_text and steps will contain whatever was
    produced before the failure.

    When request.use_syncode=True the backend applies Syncode C-grammar
    masking at every step and populates the following fields per DecodingStep:
        top_tokens_before_syncode  — raw top-k with is_masked annotation
        masked_tokens              — rejected tokens with raw probabilities
        valid_tokens_after_syncode — constrained top-k after masking
        entropy_after              — Shannon entropy of the constrained distribution
        num_masked / masked_percentage / probability_mass_removed / vocab_size
        parser_error / parser_error_message / fallback_used  — recovery metadata
    """
    mode = "syncode" if request.use_syncode else "raw"

    experiment = store.create_empty(
        prompt=request.prompt,
        mode=mode,
        model_name=settings.model_name,
    )

    generated_text: str = ""
    steps = []
    response_status = "completed"
    response_message = ""

    try:
        generated_text, steps = await llm_service.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            top_k=request.top_k,
            temperature=request.temperature,
            use_syncode=request.use_syncode,
        )
    except Exception as exc:
        # The generation loop itself has a broad try-except that returns
        # partial results, so this outer handler should only be reached in
        # very unusual circumstances (e.g. model not loaded, OOM on first
        # forward pass).  We log the error and return a graceful response
        # instead of raising HTTPException(500).
        log.error("Outer generation catch: %s", exc, exc_info=True)
        response_status = "error"
        response_message = f"Generation failed: {type(exc).__name__}: {exc}"

    experiment.generated_code = generated_text
    experiment.steps = steps
    experiment.total_steps = len(steps)
    store.save(experiment)

    return GenerateResponse(
        experiment_id=experiment.experiment_id,
        status=response_status,
        message=response_message,
        generated_text=generated_text,
        model_name=settings.model_name,
        mode=mode,
        prompt=request.prompt,
        total_steps=len(steps),
        steps=steps,
    )
