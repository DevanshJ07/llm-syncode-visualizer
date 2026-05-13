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

class TokenCandidate(BaseModel):
    """A single candidate token and its probability at one decoding step."""
    token_id: int
    token_str: str
    probability: float
    is_masked: bool = False          # True when Syncode marked this token invalid
    is_selected: bool = False        # True for the finally chosen token


class DecodingStep(BaseModel):
    """Full snapshot of one autoregressive decoding step."""
    step: int
    context: str                     # Partial generated text up to this step

    # Top-k candidates BEFORE Syncode masking is applied
    top_tokens_before_syncode: list[TokenCandidate] = Field(default_factory=list)

    # Token IDs that Syncode masked as grammar-invalid
    masked_tokens: list[int] = Field(default_factory=list)

    # Top-k candidates AFTER Syncode masking (re-normalised probabilities)
    valid_tokens_after_syncode: list[TokenCandidate] = Field(default_factory=list)

    # The token ultimately selected (greedy / sampling)
    selected_token: str = ""

    # Summary statistics
    entropy_before: Optional[float] = None
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
    mode: str                        # "raw" | "syncode"
    generated_code: str = ""
    steps: list[DecodingStep] = Field(default_factory=list)
    total_steps: int = 0
    model_name: str = ""
    created_at: str = ""             # ISO-8601 timestamp


# ---------------------------------------------------------------------------
# API request / response schemas
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """POST /generate request body."""
    prompt: str = Field(..., min_length=1, max_length=4096)
    use_syncode: bool = True
    top_k: int = Field(default=50, ge=1, le=200)
    max_new_tokens: int = Field(default=256, ge=1, le=1024)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)


class GenerateResponse(BaseModel):
    """POST /generate response body."""
    experiment_id: str
    status: str = "completed"        # "completed" | "error"
    message: str = ""


class StepResponse(BaseModel):
    """GET /experiment/{id}/steps/{step} response body."""
    step: DecodingStep
    total_steps: int
