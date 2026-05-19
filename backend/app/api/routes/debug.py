"""
Debug / diagnostic endpoints.

GET /debug/last-trace         – full step-by-step generation trace
GET /debug/syncode-status     – Syncode availability and config
GET /debug/forensic-summary   – per-step forensic trace of grammar_engine.mask_scores
GET /debug/forensic-full      – raw forensic log (all steps, all fields)

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
    """
    with _trace_lock:
        return dict(_last_trace)


@router.get("/debug/syncode-status", tags=["Debug"])
async def get_syncode_status() -> dict:
    """
    Report Syncode availability and configuration.
    """
    syncode_obj = llm_service._syncode  # type: ignore[attr-defined]
    ws_count = len(syncode_obj._whitespace_ids) if syncode_obj is not None else 0
    patch_status = getattr(syncode_obj, "_patch_status", {"installed": "unknown"}) if syncode_obj else {}
    mask_calls = getattr(syncode_obj, "_mask_call_count", 0) if syncode_obj else 0
    # Verify patch is actually in ge.__dict__ right now
    ge_has_patch = False
    if syncode_obj is not None and syncode_obj._processor is not None:
        ge = getattr(syncode_obj._processor, "grammar_engine", None)
        if ge is not None:
            ge_has_patch = "mask_scores" in ge.__dict__
    return {
        "syncode_enabled": settings.syncode_enabled,
        "model_loaded": llm_service.is_loaded,
        "syncode_available": syncode_obj.available if syncode_obj is not None else False,
        "whitespace_token_count": ws_count,
        "patch_status": patch_status,
        "mask_call_count": mask_calls,
        "ge_dict_has_patch_live": ge_has_patch,
    }


@router.get("/debug/forensic-summary", tags=["Debug"])
async def get_forensic_summary() -> dict:
    """
    Return a high-level summary of the forensic log gathered by the
    monkey-patched grammar_engine.mask_scores during the last generation.

    Key fields:
        total_steps          – how many steps ran through the forensic patch
        skip_steps           – steps where _parse_partial_output returned skip=True
                               (grammar parsing failed → scores returned unchanged)
        all_valid_mask_steps – steps where accept_mask was all-ones
                               (grammar accepted every vocab token → no masking)
        all_invalid_mask_steps – steps where accept_mask was all-zeros
                               (no valid tokens found → scores returned unchanged)
        masking_applied_steps – steps where ≥1 token was newly set to -inf
        unique_diagnoses     – set of root-cause labels seen across all steps
        first_step / last_step – detailed record for first and last step
    """
    syncode_obj = llm_service._syncode  # type: ignore[attr-defined]
    if syncode_obj is None:
        return {"error": "Syncode not initialised"}
    return syncode_obj.forensic_summary()


@router.get("/debug/forensic-full", tags=["Debug"])
async def get_forensic_full() -> dict:
    """
    Return every step record from the last generation's forensic log.

    Each record contains:
        step, partial_output, ge_start_from, ge_parse_failed,
        ge_ignore_whitespace, skip, accept_seqs, remainder_state,
        mask_stats (n_accepted, vocab_len, pct, all_valid, all_invalid,
                    ws_valid, ws_invalid, first_valid_ids),
        n_changed, n_newly_inf, diagnosis
    """
    syncode_obj = llm_service._syncode  # type: ignore[attr-defined]
    if syncode_obj is None:
        return {"error": "Syncode not initialised"}
    entries = syncode_obj.forensic_log
    return {"total_steps": len(entries), "steps": entries}
