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
    Legacy / extended candidate model kept for Syncode phase compatibility.
    Used when we need to track masking state alongside probability.
    """
    token_id: int
    token_str: str
    probability: float
    is_masked: bool = False    # True when Syncode marked this token grammar-invalid
    is_selected: bool = False  # True for the finally chosen token


class DecodingStep(BaseModel):
    """
    Full snapshot of one autoregressive decoding step.

    Core fields (populated by real generation):
        step, context, top_tokens, selected_token, selected_token_id, entropy_before

    Syncode fields (populated in a future phase):
        masked_tokens, valid_tokens_after_syncode, top_tokens_before_syncode,
        entropy_after, num_masked
    """
    step: int
    context: str  # decoded text up to (but not including) this step's token

    # --- Real generation fields (Phase 2) ---------------------------------
    # Top-k candidates ranked by probability (after temperature scaling)
    top_tokens: list[TopToken] = Field(default_factory=list)

    # The token selected by greedy decoding (argmax of softmax probabilities)
    selected_token: str = ""
    selected_token_id: int = 0

    # Shannon entropy of the full vocabulary probability distribution
    entropy_before: Optional[float] = None

    # --- Syncode fields (Phase 3) -----------------------------------------
    # These will be populated when Syncode grammar masking is active
    top_tokens_before_syncode: list[TokenCandidate] = Field(default_factory=list)
    masked_tokens: list[int] = Field(default_factory=list)
    valid_tokens_after_syncode: list[TokenCandidate] = Field(default_factory=list)
    entropy_after: Optional[float] = None
    num_masked: int = 0


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
    use_syncode: bool = False   # Syncode not yet implemented; always falls back to raw
    top_k: int = Field(default=10, ge=1, le=200)
    max_new_tokens: int = Field(default=64, ge=1, le=512)
    temperature: float = Field(default=1.0, ge=0.01, le=2.0)


class GenerateResponse(BaseModel):
    """POST /generate response body."""
    experiment_id: str
    status: str = "completed"  # "completed" | "error"
    message: str = ""


class StepResponse(BaseModel):
    """GET /experiment/{id}/steps/{step} response body."""
    step: DecodingStep
    total_steps: int
