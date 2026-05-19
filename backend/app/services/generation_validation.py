"""
Strict validation for generation results.

Any violation raises GenerationFailedError — callers must convert this to
HTTP 500 and must NOT return HTTP 201 with empty/partial payloads.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.models.schemas import DecodingStep

# UI placeholder strings that must never be returned as successful output.
_PLACEHOLDER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^\s*//\s*\(start\)\s*$", re.IGNORECASE),
    re.compile(r"^\s*//\s*start\s*$", re.IGNORECASE),
    re.compile(r"^\s*$"),
)


class GenerationFailedError(Exception):
    """Raised when generation produced no usable decoding trace."""

    def __init__(
        self,
        message: str,
        *,
        reasons: list[str] | None = None,
        generation_id: str | None = None,
        early_termination: str | None = None,
        step_count: int = 0,
        trace_step_count: int = 0,
        generated_text_preview: str = "",
    ) -> None:
        self.reasons = reasons or []
        self.generation_id = generation_id
        self.early_termination = early_termination
        self.step_count = step_count
        self.trace_step_count = trace_step_count
        self.generated_text_preview = generated_text_preview
        super().__init__(message)

    def to_detail(self) -> dict:
        return {
            "error": "generation_failed",
            "message": str(self),
            "reasons": self.reasons,
            "generation_id": self.generation_id,
            "early_termination": self.early_termination,
            "step_count": self.step_count,
            "trace_step_count": self.trace_step_count,
            "generated_text_preview": self.generated_text_preview,
        }


def is_placeholder_output(text: str) -> bool:
    """True when text is empty or matches known UI placeholder patterns."""
    if not text or not text.strip():
        return True
    for pattern in _PLACEHOLDER_PATTERNS:
        if pattern.match(text):
            return True
    return False


def validate_generation_result(
    generated_text: str,
    steps: list[DecodingStep],
    trace_steps: list[dict],
    *,
    generation_id: str | None = None,
    early_termination: str | None = None,
) -> None:
    """
    Assert generation produced a non-empty trace. Raises GenerationFailedError
    with a detailed message on any violation.
    """
    reasons: list[str] = []

    if len(steps) == 0:
        reasons.append("zero decoding steps (len(steps) == 0)")
    if not trace_steps:
        reasons.append("empty trace (_trace_steps has no entries)")
    if not generated_ids_nonempty(steps):
        reasons.append("empty token list (no selected_token_id in steps)")
    if not generated_text or not generated_text.strip():
        reasons.append("generated_text is empty or whitespace-only")
    elif is_placeholder_output(generated_text):
        reasons.append(
            f"placeholder-only output: {generated_text[:80]!r}"
        )
    if len(steps) > 0 and len(trace_steps) == 0:
        reasons.append("steps populated but trace is empty (serialization mismatch)")
    if len(steps) != len(trace_steps) and len(trace_steps) > 0:
        reasons.append(
            f"step/trace count mismatch: len(steps)={len(steps)} "
            f"len(trace)={len(trace_steps)}"
        )

    if not reasons:
        return

    msg = (
        f"Generation failed validation ({len(reasons)} issue(s)): "
        + "; ".join(reasons)
    )
    if early_termination:
        msg += f" [early_termination={early_termination}]"

    raise GenerationFailedError(
        msg,
        reasons=reasons,
        generation_id=generation_id,
        early_termination=early_termination,
        step_count=len(steps),
        trace_step_count=len(trace_steps),
        generated_text_preview=(generated_text or "")[:120],
    )


def generated_ids_nonempty(steps: list[DecodingStep]) -> bool:
    """True when at least one step has a selected token id."""
    return any(
        getattr(s, "selected_token_id", None) is not None
        for s in steps
    )
