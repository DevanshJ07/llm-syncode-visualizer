"""
LLM inference service.

Wraps the HuggingFace Transformers pipeline for Llama 3B and exposes a
single `generate()` method that returns structured DecodingStep data.

This module is intentionally decoupled from the FastAPI layer so it can be
tested independently and swapped for a different model without touching routes.

NOTE: Model loading is deferred to first use (lazy) so the server starts fast.
      For production, call `llm_service.load_model()` during the lifespan hook.
"""

from __future__ import annotations

import math
from typing import Generator

import torch

from app.core.config import settings
from app.models.schemas import DecodingStep, TokenCandidate


def _softmax(logits: list[float]) -> list[float]:
    max_l = max(logits)
    exps = [math.exp(l - max_l) for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def _entropy(probs: list[float]) -> float:
    return -sum(p * math.log(p + 1e-12) for p in probs)


class LLMService:
    """Thin wrapper around a HuggingFace causal-LM model."""

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False

    # ------------------------------------------------------------------
    def load_model(self) -> None:
        """Load the model and tokenizer onto the configured device.
        Call once during application startup."""
        from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy import

        print(f"[LLMService] Loading {settings.model_name} on {settings.device}...")
        self._tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
        self._model = AutoModelForCausalLM.from_pretrained(
            settings.model_name,
            torch_dtype=torch.float16 if settings.device == "cuda" else torch.float32,
            device_map=settings.device,
        )
        self._model.eval()
        self._loaded = True
        print("[LLMService] Model loaded.")

    # ------------------------------------------------------------------
    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """Run greedy generation and capture per-step decoding information.

        Returns:
            generated_code: final decoded string
            steps: list of DecodingStep objects (one per generated token)

        This is a STUB — real logit capture and Syncode hook-in will be
        implemented in the next phase.
        """
        if not self._loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # TODO: implement real token-by-token generation loop with logit capture
        steps: list[DecodingStep] = []
        generated_code = "/* [LLMService stub] model inference not yet wired */"
        return generated_code, steps


# Module-level singleton
llm_service = LLMService()
