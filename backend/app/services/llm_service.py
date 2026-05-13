"""
LLM inference service — real token-by-token autoregressive generation.

Model: TinyLlama/TinyLlama-1.1B-Chat-v1.0
Runtime: CPU-only (fp32), no GPU assumptions.

Design:
  - Model and tokenizer are loaded lazily on the first generate() call and
    cached globally for the lifetime of the process (one load, many uses).
  - The blocking CPU inference runs inside a ThreadPoolExecutor so it never
    blocks the FastAPI asyncio event loop.
  - We use HuggingFace's KV-cache (past_key_values) for O(1) per-step cost
    instead of re-running attention over the full context at every step.

Generation algorithm (per step):
  1. Forward pass  → logits[last_position]  (shape: [vocab_size])
  2. Temperature scaling  → logits / T
  3. Softmax  → probability distribution over full vocabulary
  4. Entropy  → -Σ p·log(p)  over full vocabulary
  5. Top-k extraction  → keep k highest-prob tokens for logging
  6. Greedy selection  → argmax(probs) is the chosen token
  7. Append to sequence and repeat
"""

from __future__ import annotations

import asyncio
import math
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import DecodingStep, TopToken

# Single-worker executor: model is not thread-safe for concurrent inference.
_executor = ThreadPoolExecutor(max_workers=1)


class LLMService:
    """
    Thin wrapper around a HuggingFace causal-LM.

    Public interface:
        llm_service.generate(prompt, max_new_tokens, top_k, temperature)
        → (generated_text: str, steps: list[DecodingStep])
    """

    def __init__(self) -> None:
        self._model = None
        self._tokenizer = None
        self._loaded = False
        self._loading = False  # guard against concurrent load attempts

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load the tokenizer and model weights from HuggingFace Hub (or local
        cache if already downloaded).  This is called lazily inside the
        thread executor, so it never blocks the async event loop.
        """
        if self._loaded or self._loading:
            return

        self._loading = True
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[LLMService] Loading '{settings.model_name}' on {settings.device} …")

            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_fast=True,
            )

            # fp32 on CPU — no half-precision to avoid BFloat16 issues on x86
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self._model.eval()
            self._loaded = True
            print(f"[LLMService] Model ready. "
                  f"Parameters: {sum(p.numel() for p in self._model.parameters()) / 1e6:.0f}M")
        finally:
            self._loading = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # Public async entry point
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Async wrapper: dispatches the CPU-blocking work to the thread pool
        so the FastAPI event loop stays responsive.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            _executor,
            self._run_generate_sync,
            prompt,
            max_new_tokens,
            top_k,
            temperature,
        )

    # ------------------------------------------------------------------
    # Core generation loop (runs in thread pool, NOT in async context)
    # ------------------------------------------------------------------

    @torch.no_grad()  # disable gradient tracking for inference — saves memory
    def _run_generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Token-by-token greedy generation with full per-step logging.

        Returns:
            generated_text  — only the newly generated tokens (not the prompt)
            steps           — one DecodingStep per generated token
        """
        # Lazy-load on first call (cached for subsequent calls)
        if not self._loaded:
            self.load_model()

        # ----------------------------------------------------------------
        # 1. Tokenise the prompt
        # ----------------------------------------------------------------
        # return_tensors="pt" gives us a [1, prompt_len] int64 tensor
        encoded = self._tokenizer(prompt, return_tensors="pt")
        input_ids: torch.Tensor = encoded["input_ids"]       # [1, prompt_len]
        prompt_len = input_ids.shape[1]

        steps: list[DecodingStep] = []
        generated_token_ids: list[int] = []

        # ----------------------------------------------------------------
        # 2. First forward pass over the full prompt to prime the KV cache
        # ----------------------------------------------------------------
        # past_key_values is the KV cache: a tuple of per-layer (K, V) tensors.
        # On subsequent steps we only pass the single new token — the model
        # attends to the full context via the cache, giving O(1) per step.
        outputs = self._model(input_ids=input_ids, use_cache=True)
        past_key_values = outputs.past_key_values

        # logits shape after prompt: [1, prompt_len, vocab_size]
        # We only need the last position's logits to predict the next token.
        last_logits: torch.Tensor = outputs.logits[0, -1, :]  # [vocab_size]

        # ----------------------------------------------------------------
        # 3. Autoregressive generation loop
        # ----------------------------------------------------------------
        for step_idx in range(max_new_tokens):
            # -- 3a. Temperature scaling ----------------------------------
            # Dividing logits by T < 1 sharpens the distribution (more greedy);
            # T > 1 flattens it (more uniform / exploratory).
            # We always pick the argmax (greedy) but still apply temperature
            # so the logged probabilities reflect the scaled distribution.
            scaled_logits = last_logits / temperature  # [vocab_size]

            # -- 3b. Softmax over full vocabulary -------------------------
            # probs[i] = probability of generating token i as the next token
            probs: torch.Tensor = F.softmax(scaled_logits, dim=-1)  # [vocab_size]

            # -- 3c. Shannon entropy of the distribution ------------------
            # H = -Σ p·log(p).  High entropy = uncertain / flat distribution.
            # We clamp to avoid log(0) = -inf.
            entropy: float = (
                -(probs * torch.log(probs.clamp(min=1e-12)))
                .sum()
                .item()
            )

            # -- 3d. Top-k candidate extraction ---------------------------
            # torch.topk returns (values, indices) sorted descending by value.
            k = min(top_k, probs.size(0))
            topk_probs, topk_ids = torch.topk(probs, k=k)

            top_tokens: list[TopToken] = []
            for rank in range(k):
                tid = int(topk_ids[rank].item())
                prob = float(topk_probs[rank].item())
                # decode a single token ID to its string representation
                tok_str = self._tokenizer.decode(
                    [tid],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                top_tokens.append(TopToken(token=tok_str, probability=prob, token_id=tid))

            # -- 3e. Greedy selection: pick the token with highest probability --
            selected_id: int = int(torch.argmax(probs).item())
            selected_str: str = self._tokenizer.decode(
                [selected_id],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

            # -- 3f. Build the context string (everything generated so far) --
            # Decode only the tokens generated *before* this step (not current)
            context: str = self._tokenizer.decode(
                generated_token_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )

            # -- 3g. Record the decoding step -----------------------------
            steps.append(
                DecodingStep(
                    step=step_idx + 1,
                    context=context,
                    top_tokens=top_tokens,
                    selected_token=selected_str,
                    selected_token_id=selected_id,
                    entropy_before=round(entropy, 4),
                )
            )

            generated_token_ids.append(selected_id)

            # -- 3h. Stop at EOS ------------------------------------------
            if selected_id == self._tokenizer.eos_token_id:
                break

            # -- 3i. KV-cache forward pass with the single new token ------
            # Passing only the new token ID keeps the per-step cost O(1)
            # (the model uses past_key_values to attend to previous context).
            next_input = torch.tensor([[selected_id]])  # [1, 1]
            outputs = self._model(
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        # ----------------------------------------------------------------
        # 4. Decode the full generated sequence (without the prompt)
        # ----------------------------------------------------------------
        generated_text: str = self._tokenizer.decode(
            generated_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_text, steps


# Module-level singleton — import `llm_service` in route handlers.
llm_service = LLMService()
