"""
Debug / diagnostic endpoints.

GET /debug/last-trace
    Returns the full step-by-step diagnostic trace from the most recent
    generation.  Every step carries:
      • selection_source         – which distribution was used
      • raw_argmax / constrained_argmax – greedy choices from each dist
      • grammar_masked_count     – tokens masked by Syncode grammar
      • logits_diverge           – whether constrained ≠ raw logits
      • whitespace_tokens_masked – did grammar mask whitespace tokens?
      • raw_top3 / constrained_top3 – top candidates from each dist
      • entropy_before / entropy_after
      • fallback_used / parser_error

GET /debug/syncode-status
    Returns Syncode availability, whitespace token count, and the
    current settings.syncode_enabled flag.

These endpoints are read-only and never modify generation state.
"""

from __future__ import annotations

from fastapi import APIRouter

from app.core.config import settings
from app.services.llm_service import _last_trace, _trace_lock, llm_service

router = APIRouter()


@router.get("/debug/last-trace", tags=["Debug"])
async def get_last_trace() -> dict:
    """
    Return the full diagnostic trace from the last completed generation.

    Summary fields (top-level):
        generation_id, prompt, mode, effective_syncode
        summary.total_steps
        summary.syncode_active_steps   – steps where grammar mask changed logits
        summary.fallback_steps         – steps where Syncode fell back to raw
        summary.logits_diverge_steps   – steps where masked ≠ raw logits
        summary.grammar_masked_any_steps
        summary.whitespace_tokens_masked_steps
        summary.whitespace_stall_steps
        summary.whitespace_stall_step_num
        summary.generated_text_preview

    Per-step fields (in ``steps`` array):
        step, selected_token, selected_token_id, selection_source
        raw_argmax_token, raw_argmax_token_id
        constrained_argmax_token, constrained_argmax_token_id
        syncode_active, logits_diverge
        grammar_masked_count, num_masked_total, masked_percentage
        whitespace_tokens_masked, whitespace_tokens_accepted
        entropy_before, entropy_after
        fallback_used, parser_error, parser_error_message
        consecutive_whitespace_count
        raw_top3, constrained_top3
    """
    with _trace_lock:
        return dict(_last_trace)


@router.get("/debug/syncode-status", tags=["Debug"])
async def get_syncode_status() -> dict:
    """
    Report Syncode availability and configuration.

    Fields:
        syncode_enabled        – settings.SYNCODE_ENABLED flag
        model_loaded           – whether the model has been loaded
        syncode_available      – whether _SyncodeConstraint initialised OK
        whitespace_token_count – vocabulary IDs that decode to whitespace-only
    """
    syncode_obj = llm_service._syncode  # type: ignore[attr-defined]
    ws_count = len(syncode_obj._whitespace_ids) if syncode_obj is not None else 0
    return {
        "syncode_enabled": settings.syncode_enabled,
        "model_loaded": llm_service.is_loaded,
        "syncode_available": syncode_obj.available if syncode_obj is not None else False,
        "whitespace_token_count": ws_count,
    }
