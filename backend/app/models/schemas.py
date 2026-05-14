"""
Pydantic schemas for all API request / response bodies and the core
JSON logging format described in PROJECT_SPEC.md.

These schemas are the contract between the backend and the frontend —
keep them stable and version them explicitly if they change.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Core decoding data model (matches the JSON logging format in PROJECT_SPEC)
# ---------------------------------------------------------------------------

class TopToken(BaseModel):
    """
    One candidate token at a decoding step — the primary logging unit.

    Matches the JSON format from PROJECT_SPEC:
        { "token": "main", "probability": 0.42, "token_id": 1234 }
    """
    token: str          # human-readable decoded string (may contain whitespace/special chars)
    probability: float  # softmax probability AFTER temperature scaling, range [0, 1]
    token_id: int       # vocabulary index


class TokenCandidate(BaseModel):
    """
    Extended candidate model used for Syncode before/after distributions.
    Tracks masking state alongside probability.
    """
    token_id: int
    token_str: str
    probability: float
    is_masked: bool = False    # True when Syncode marked this token grammar-invalid
    is_selected: bool = False  # True for the finally chosen token


class MaskedTokenEntry(BaseModel):
    """
    A token that was rejected by Syncode grammar masking.
    Carries its raw (pre-mask) probability for visualisation.
    """
    token: str          # decoded string
    token_id: int       # vocabulary index
    raw_prob: float     # softmax probability BEFORE Syncode masking


class DecodingStep(BaseModel):
    """
    Full snapshot of one autoregressive decoding step.

    Core fields (populated by real generation):
        step, context, top_tokens, selected_token, selected_token_id, entropy_before

    Syncode fields (populated when use_syncode=True):
        masked_tokens, valid_tokens_after_syncode, top_tokens_before_syncode,
        entropy_after, num_masked, vocab_size, valid_token_count, masked_token_count,
        masked_percentage, probability_mass_removed
    """
    step: int
    context: str  # decoded text up to (but not including) this step's token

    # --- Real generation fields -------------------------------------------
    # Top-k candidates ranked by probability (after temperature scaling)
    top_tokens: list[TopToken] = Field(default_factory=list)

    # The token selected by greedy decoding (argmax of softmax probabilities)
    selected_token: str = ""
    selected_token_id: int = 0

    # Shannon entropy of the full vocabulary probability distribution
    entropy_before: Optional[float] = None

    # --- Syncode fields ---------------------------------------------------
    # Raw top-k before masking (with is_masked annotation)
    top_tokens_before_syncode: list[TokenCandidate] = Field(default_factory=list)
    # Rejected tokens — full objects with raw_prob for visualisation
    masked_tokens: list[MaskedTokenEntry] = Field(default_factory=list)
    # Top-k from the constrained (post-mask) distribution
    valid_tokens_after_syncode: list[TokenCandidate] = Field(default_factory=list)
    entropy_after: Optional[float] = None
    num_masked: int = 0

    # --- Syncode masking statistics per step ------------------------------
    vocab_size: int = 0
    valid_token_count: int = 0          # tokens that survived grammar masking
    masked_token_count: int = 0         # = num_masked (alias, kept for API clarity)
    masked_percentage: float = 0.0      # masked_token_count / vocab_size * 100
    probability_mass_removed: float = 0.0  # Σ raw_prob of all masked tokens

    # --- Parser recovery metadata -----------------------------------------
    # True if the Syncode grammar parser threw an exception at this step.
    # The generation is never aborted — raw/fallback logits are used instead.
    parser_error: bool = False
    parser_error_message: str = ""
    # True when Syncode masking was requested but the raw distribution was
    # used at this step (either because the processor returned None or because
    # the grammar parser failed).
    fallback_used: bool = False

    # --- Whitespace stall detection ----------------------------------------
    # Number of consecutive whitespace/newline-only tokens up to this step.
    consecutive_whitespace_count: int = 0
    # True when the stall threshold was exceeded at this step — generation
    # is stopped gracefully when this fires.
    whitespace_stall_detected: bool = False
    # Step number (1-indexed) where the first stall was detected, or None.
    whitespace_stall_step: Optional[int] = None

    # --- Pipeline integrity diagnostics ------------------------------------
    # True when Syncode's grammar mask was applied AND actually changed logits
    # (not a fallback / identity pass-through).
    syncode_active: bool = False
    # True when the masked logit tensor differs from the raw logit tensor at
    # at least one position (includes both grammar masking AND special-token
    # suppression applied during the Syncode fallback path).
    logits_diverge: bool = False
    # Raw argmax token — what greedy selection on the unmasked distribution
    # would have chosen.  Always populated.
    raw_argmax_token_id: int = 0
    raw_argmax_token: str = ""
    # Constrained argmax — what greedy selection on the masked distribution
    # would choose.  Only meaningful when syncode_active=True.
    constrained_argmax_token_id: int = 0
    constrained_argmax_token: str = ""
    # Describes which distribution was actually used for selection:
    #   "constrained_sampled"  – nucleus sample from grammar-masked distribution
    #   "constrained_greedy"   – argmax from grammar-masked distribution
    #   "raw_sampled"          – nucleus sample from unmasked distribution
    #   "raw_greedy"           – argmax from unmasked distribution
    #   "fallback_sampled"     – Syncode failed; nucleus sample from raw
    #   "fallback_greedy"      – Syncode failed; argmax from raw
    selection_source: str = "raw_greedy"
    # Number of tokens Syncode's grammar mask newly set to -inf (excludes
    # tokens that were already -inf in the raw logits and the special-token
    # suppression layer).
    grammar_masked_count: int = 0
    # True when Syncode masked ≥1 whitespace-only token at this step.
    whitespace_tokens_masked: bool = False


# ---------------------------------------------------------------------------
# Experiment container
# ---------------------------------------------------------------------------

class GenerationMode(str):
    RAW = "raw"
    SYNCODE = "syncode"


class ExperimentResult(BaseModel):
    """Top-level object stored to disk and returned by GET /experiment/{id}."""
    experiment_id: str
    prompt: str
    mode: str            # "raw" | "syncode"
    generated_code: str = ""
    steps: list[DecodingStep] = Field(default_factory=list)
    total_steps: int = 0
    model_name: str = ""
    created_at: str = ""  # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """POST /generate request body."""
    prompt: str = Field(..., min_length=1, max_length=4096)
    use_syncode: bool = False
    top_k: int = Field(default=20, ge=1, le=200)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.01, le=2.0)
    # Sampling parameters — enable richer, less deterministic generation so
    # Syncode constraint effects are clearly visible in the decoding trace.
    do_sample: bool = True
    top_p: float = Field(default=0.95, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.1, ge=1.0, le=2.0)


class GenerateResponse(BaseModel):
    """
    POST /generate response body.

    Returns the full experiment inline so the frontend can render the
    visualization immediately without a follow-up GET /experiment/{id}.
    The experiment is still persisted to disk under logs/experiments/.
    """
    # --- Experiment identity ---
    experiment_id: str
    status: str = "completed"   # "completed" | "error"
    message: str = ""

    # --- Generated output ---
    generated_text: str = ""    # decoded text of the newly generated tokens
    model_name: str = ""
    mode: str = "raw"           # "raw" | "syncode"
    prompt: str = ""
    total_steps: int = 0

    # --- Full decoding trace ---
    # One DecodingStep per generated token.  Each step contains:
    #   selected_token / selected_token_id  — the greedy-chosen token
    #   top_tokens                          — top-k candidates with probabilities
    #   entropy_before                      — Shannon entropy of the full vocab dist
    #   top_tokens_before_syncode           — placeholder for Phase 3 Syncode data
    #   masked_tokens / valid_tokens_after_syncode / entropy_after / num_masked
    #                                       — Syncode mask info (empty until Phase 3)
    steps: list[DecodingStep] = Field(default_factory=list)


class StepResponse(BaseModel):
    """GET /experiment/{id}/steps/{step} response body."""
    step: DecodingStep
    total_steps: int
