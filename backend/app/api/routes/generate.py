from __future__ import annotations

import json
import logging
import traceback

from fastapi import APIRouter, HTTPException, status

from app.models.schemas import GenerateRequest, GenerateResponse
from app.services.experiment_store import store
from app.services.generation_validation import GenerationFailedError
from app.services.llm_service import llm_service
from app.core.config import settings

log = logging.getLogger(__name__)
router = APIRouter()


def _http_500_from_generation_error(exc: GenerationFailedError) -> HTTPException:
    detail = exc.to_detail()
    log.error(
        "[GEN validation failed] %s | detail=%s",
        exc,
        json.dumps(detail, default=str),
        exc_info=True,
    )
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail=detail,
    )


@router.post(
    "/generate",
    response_model=GenerateResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate(request: GenerateRequest) -> GenerateResponse:
    """
    Run generation and return the complete decoding trace.

    Returns HTTP 201 only when generation produces a valid non-empty trace.
    Any failure (empty steps, empty trace, exception, validation error) raises
    HTTP 500 with a structured detail payload — never a silent empty success.
    """
    mode = "syncode" if request.use_syncode else "raw"

    log.info(
        "[API /generate request] mode=%s prompt_len=%d max_new_tokens=%d "
        "top_k=%d T=%.2f do_sample=%s use_syncode=%s",
        mode,
        len(request.prompt),
        request.max_new_tokens,
        request.top_k,
        request.temperature,
        request.do_sample,
        request.use_syncode,
    )

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
            do_sample=request.do_sample,
            top_p=request.top_p,
            repetition_penalty=request.repetition_penalty,
        )
    except GenerationFailedError as exc:
        raise _http_500_from_generation_error(exc) from exc
    except Exception as exc:
        log.error(
            "[API /generate exception] %s: %s\n%s",
            type(exc).__name__,
            exc,
            traceback.format_exc(),
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "generation_exception",
                "message": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            },
        ) from exc

    # Belt-and-suspenders validation at the route layer.
    if len(steps) == 0:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "generation_failed",
                "message": "Route validation: zero decoding steps after generate()",
                "reasons": ["len(steps) == 0 at route boundary"],
            },
        )
    if not generated_text or not generated_text.strip():
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "generation_failed",
                "message": "Route validation: empty generated_text",
                "reasons": ["generated_text is empty at route boundary"],
            },
        )

    experiment.generated_code = generated_text
    experiment.steps = steps
    experiment.total_steps = len(steps)

    try:
        store.save(experiment)
        log.info(
            "[API experiment save] success experiment_id=%s steps=%d",
            experiment.experiment_id,
            len(steps),
        )
    except Exception as save_exc:
        log.error(
            "[API experiment save] FAILED experiment_id=%s: %s",
            experiment.experiment_id,
            save_exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "experiment_save_failed",
                "message": f"Generation succeeded but experiment save failed: {save_exc}",
                "experiment_id": experiment.experiment_id,
                "step_count": len(steps),
            },
        ) from save_exc

    response = GenerateResponse(
        experiment_id=experiment.experiment_id,
        status="completed",
        message="",
        generated_text=generated_text,
        model_name=settings.model_name,
        mode=mode,
        prompt=request.prompt,
        total_steps=len(steps),
        steps=steps,
    )

    try:
        payload_json = response.model_dump_json()
        payload_bytes = len(payload_json.encode("utf-8"))
        log.info(
            "[API /generate response] experiment_id=%s status=completed "
            "total_steps=%d payload_bytes=%d generated_text_len=%d",
            experiment.experiment_id,
            len(steps),
            payload_bytes,
            len(generated_text),
        )
        if log.isEnabledFor(logging.DEBUG):
            preview = payload_json[:2000]
            log.debug(
                "[API /generate response JSON preview] %s%s",
                preview,
                "…" if len(payload_json) > 2000 else "",
            )
    except Exception as ser_exc:
        log.error(
            "[API trace serialization] FAILED: %s",
            ser_exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "trace_serialization_failed",
                "message": f"Failed to serialize response: {ser_exc}",
            },
        ) from ser_exc

    return response
