"""
Syncode constrained decoding service.

Syncode wraps a HuggingFace model and injects a grammar-based logit processor
that masks tokens violating the target grammar (C in our case).

This module provides:
  - SyncodeService.apply_mask(logits, grammar_state) → masked_logits, masked_ids
  - SyncodeService.wrap_model(model, tokenizer) → syncode-wrapped model

The actual Syncode library must be installed separately:
    pip install git+https://github.com/uiuc-focal-lab/syncode.git

If Syncode is not installed, this service degrades gracefully (returns the
raw logits unchanged) so the backend stays runnable during development
without GPU hardware.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from app.core.config import settings

if TYPE_CHECKING:
    import torch


class SyncodeService:
    """Thin adapter around the Syncode library."""

    def __init__(self) -> None:
        self._available = False
        self._syncode = None
        self._try_import()

    # ------------------------------------------------------------------
    def _try_import(self) -> None:
        try:
            import syncode  # noqa: F401
            self._syncode = syncode
            self._available = True
        except ImportError:
            print(
                "[SyncodeService] syncode package not found — "
                "constrained decoding disabled. "
                "Install: pip install git+https://github.com/uiuc-focal-lab/syncode.git"
            )

    # ------------------------------------------------------------------
    @property
    def is_available(self) -> bool:
        return self._available and settings.syncode_enabled

    # ------------------------------------------------------------------
    def apply_mask(
        self,
        logits: "torch.Tensor",
        vocab_size: int,
    ) -> tuple["torch.Tensor", list[int]]:
        """Apply grammar mask to raw logits.

        Returns:
            masked_logits: logits with invalid token positions set to -inf
            masked_token_ids: list of token IDs that were masked

        STUB — will integrate Syncode's DFA-based logit processor.
        """
        if not self.is_available:
            return logits, []

        # TODO: plug in syncode.GrammarConstrainedLogitsProcessor for C grammar
        masked_token_ids: list[int] = []
        return logits, masked_token_ids

    # ------------------------------------------------------------------
    def wrap_model(self, model, tokenizer):
        """Return a Syncode-wrapped version of the model if available.

        STUB — will call syncode.Syncode(model=model, ...) in next phase.
        """
        if not self.is_available:
            return model
        # TODO: return syncode.Syncode(model=model, tokenizer=tokenizer, grammar="c")
        return model


# Module-level singleton
syncode_service = SyncodeService()
