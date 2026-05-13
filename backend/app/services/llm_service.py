"""
LLM inference service — TinyLlama token-by-token autoregressive generation.

Model  : TinyLlama/TinyLlama-1.1B-Chat-v1.0
Runtime: CPU-only, fp32 (no GPU assumptions)

Public interface
----------------
llm_service.load_model()
    Eagerly loads the tokenizer and model weights. Safe to call multiple
    times — subsequent calls are no-ops.

llm_service.generate_step(logits, step_idx, context_ids, top_k, temperature)
    Core per-step computation: temperature scaling → softmax → entropy →
    top-k extraction → greedy selection.  Returns a DecodingStep and the
    selected token ID.  Decoupled from the generation loop so Syncode can
    intercept logits before calling this in Phase 3.

llm_service.generate(prompt, max_new_tokens, top_k, temperature)  [async]
    Full async generation loop. Dispatches blocking work to a thread
    pool so the FastAPI event loop is never stalled.

llm_service.tokenizer   — the loaded PreTrainedTokenizer (or None)
llm_service.model       — the loaded PreTrainedModel (or None)
llm_service.is_loaded   — True once load_model() has completed

Lazy loading
------------
load_model() is called automatically on the first generate() request.
The model stays in memory for the lifetime of the process — one download,
one load, many generation calls.

KV-cache
--------
We use past_key_values so each generation step only processes the single
new token (O(1) attention) instead of re-running the full context (O(n²)).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import DecodingStep, TopToken

if TYPE_CHECKING:
    # Only imported at type-check time to avoid slow top-level imports
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

# One worker: the model is not thread-safe for concurrent forward passes.
_executor = ThreadPoolExecutor(max_workers=1)


class LLMService:
    """Singleton service that owns the TinyLlama model and tokenizer."""

    def __init__(self) -> None:
        # Both are None until load_model() completes.
        self._model: "PreTrainedModel | None" = None
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._loaded: bool = False
        self._loading: bool = False  # guard against overlapping load calls

    # ------------------------------------------------------------------
    # Properties — expose raw HuggingFace objects for callers that need them
    # ------------------------------------------------------------------

    @property
    def model(self) -> "PreTrainedModel | None":
        """The loaded AutoModelForCausalLM instance, or None before load."""
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase | None":
        """The loaded AutoTokenizer instance, or None before load."""
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        """True once load_model() has finished successfully."""
        return self._loaded

    # ------------------------------------------------------------------
    # load_model — eager or lazy model initialisation
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Download (first run) and load TinyLlama into CPU memory.

        Uses AutoTokenizer and AutoModelForCausalLM from HuggingFace
        Transformers.  fp32 is used instead of fp16/bf16 to avoid
        precision issues on CPUs that lack native bfloat16 support.

        Thread-safe: a _loading guard prevents duplicate loads if two
        requests arrive before the first load completes.

        The loaded objects are cached in self._model and self._tokenizer
        for the lifetime of the process — subsequent calls return immediately.
        """
        # Already loaded or a load is in progress — skip.
        if self._loaded or self._loading:
            return

        self._loading = True
        try:
            # Lazy import keeps server startup fast when no generation
            # has been requested yet.
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"[LLMService] Loading '{settings.model_name}' …")

            # --- Tokenizer -----------------------------------------------
            # use_fast=True selects the Rust-backed tokenizer for TinyLlama,
            # which is significantly faster for encoding/decoding.
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_fast=True,
            )

            # --- Model ---------------------------------------------------
            # torch_dtype=float32 ensures compatibility with all x86 CPUs.
            # low_cpu_mem_usage=True streams weights shard-by-shard so peak
            # RAM during loading stays close to the model's final footprint
            # (~2.2 GB for TinyLlama 1.1B) rather than spiking to 2x.
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )

            # Switch to eval mode: disables dropout and batch-norm tracking,
            # which are only needed during training.
            self._model.eval()

            param_count = sum(p.numel() for p in self._model.parameters())
            print(
                f"[LLMService] Ready — {param_count / 1e6:.0f}M parameters, "
                f"device={settings.device}"
            )
            self._loaded = True

        finally:
            # Always clear the guard even if an exception was raised,
            # so a retry is possible without restarting the server.
            self._loading = False

    # ------------------------------------------------------------------
    # generate_step — core per-step computation (decoupled from the loop)
    # ------------------------------------------------------------------

    def generate_step(
        self,
        logits: torch.Tensor,    # raw logits for the current position, shape [vocab_size]
        step_idx: int,           # 0-based index used for the 1-based step number in the log
        context_ids: list[int],  # token IDs generated so far (decoded to build context string)
        top_k: int,
        temperature: float,
    ) -> tuple[DecodingStep, int]:
        """
        Process raw logits for one decoding step.

        This method is intentionally separated from the generation loop so
        that Phase 3 (Syncode) can inject grammar masking between the forward
        pass and this function without duplicating any probability math.

        Steps performed:
            1. Temperature scaling  — logits / T
            2. Softmax              — converts to probability distribution
            3. Shannon entropy      — H = -Σ p·log(p)
            4. Top-k extraction     — topk_probs, topk_ids via torch.topk
            5. Token decoding       — vocab IDs → human-readable strings
            6. Greedy selection     — argmax(probs)
            7. Context decoding     — context_ids → text prefix

        Returns:
            step         : fully populated DecodingStep ready for JSON logging
            selected_id  : vocabulary index of the chosen token
        """
        # Step 1 — Temperature scaling.
        # Lower T → sharper distribution (model is more decisive).
        # Higher T → flatter distribution (model is more exploratory).
        # Selection is always greedy (argmax); temperature only affects
        # the probability values that are logged and displayed.
        scaled_logits: torch.Tensor = logits / temperature  # [vocab_size]

        # Step 2 — Softmax to convert to proper probabilities.
        # probs[i] is the probability that token i is the next token.
        probs: torch.Tensor = F.softmax(scaled_logits, dim=-1)  # [vocab_size]

        # Step 3 — Shannon entropy H = -Σ p·log(p) over the full vocabulary.
        # High value (e.g. > 5) → model is uncertain, distribution is flat.
        # Low value (e.g. < 1) → model is confident, distribution is peaked.
        # Clamp before log to avoid log(0) = -inf.
        entropy: float = (
            -(probs * torch.log(probs.clamp(min=1e-12))).sum().item()
        )

        # Step 4 — Extract the top-k highest-probability token IDs.
        # torch.topk returns tensors sorted in descending probability order.
        k = min(top_k, probs.size(0))
        topk_probs, topk_ids = torch.topk(probs, k=k)  # both shape [k]

        # Step 5 — Decode each top-k token ID to its string representation.
        # skip_special_tokens=False ensures <eos>, <unk> etc. are visible.
        # clean_up_tokenization_spaces=False preserves leading spaces that
        # are part of the token (e.g. " int" is different from "int").
        top_tokens: list[TopToken] = []
        for rank in range(k):
            tid = int(topk_ids[rank].item())
            prob = float(topk_probs[rank].item())
            tok_str = self._tokenizer.decode(  # type: ignore[union-attr]
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            top_tokens.append(TopToken(token=tok_str, probability=prob, token_id=tid))

        # Step 6 — Greedy selection: the token with the highest probability.
        # argmax over the full vocabulary, not just the logged top-k subset.
        selected_id: int = int(torch.argmax(probs).item())
        selected_str: str = self._tokenizer.decode(  # type: ignore[union-attr]
            [selected_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # Step 7 — Decode the context (all tokens generated before this step).
        # This is what the model "sees" as the generated prefix so far.
        context: str = self._tokenizer.decode(  # type: ignore[union-attr]
            context_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        step = DecodingStep(
            step=step_idx + 1,          # 1-indexed for the JSON log
            context=context,
            top_tokens=top_tokens,
            selected_token=selected_str,
            selected_token_id=selected_id,
            entropy_before=round(entropy, 4),
        )

        return step, selected_id

    # ------------------------------------------------------------------
    # generate — async entry point for the FastAPI route
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Async wrapper around _run_generate_sync.

        The blocking CPU work (model forward passes) runs inside the single-
        worker ThreadPoolExecutor so the FastAPI asyncio event loop is free
        to handle other requests while generation is in progress.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            _executor,
            self._run_generate_sync,
            prompt,
            max_new_tokens,
            top_k,
            temperature,
        )

    # ------------------------------------------------------------------
    # _run_generate_sync — the blocking generation loop
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _run_generate_sync(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Token-by-token greedy generation loop (synchronous, blocking).

        Runs in the ThreadPoolExecutor — never call from async code directly.

        Algorithm:
            1. Lazy-load the model if not yet loaded.
            2. Tokenise the prompt → input_ids [1, prompt_len].
            3. One full forward pass over the prompt to prime the KV cache.
               This is the only O(prompt_len) operation; all subsequent steps
               are O(1) thanks to the KV cache.
            4. For each new token:
               a. Call generate_step() to get probabilities and select a token.
               b. Stop if EOS is selected.
               c. Run a forward pass with the single new token + KV cache.
            5. Decode and return the generated text and the step log.

        Returns:
            generated_text : text of the newly generated tokens (no prompt)
            steps          : list of DecodingStep, one entry per token
        """
        # Lazy load — no-op if already loaded.
        if not self._loaded:
            self.load_model()

        # --- 1. Tokenise the prompt ------------------------------------
        # add_special_tokens=True prepends the BOS token automatically.
        encoded = self._tokenizer(  # type: ignore[misc]
            prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids: torch.Tensor = encoded["input_ids"]  # [1, prompt_len]

        steps: list[DecodingStep] = []
        generated_ids: list[int] = []   # token IDs produced so far

        # --- 2. Prime the KV cache with one full forward pass ----------
        # past_key_values holds the key/value attention tensors for every
        # transformer layer.  On the next forward call we pass just the
        # new single token and these cached tensors, so the attention
        # computation for the prompt tokens is never repeated.
        outputs = self._model(input_ids=input_ids, use_cache=True)  # type: ignore[misc]
        past_key_values = outputs.past_key_values

        # logits: [1, prompt_len, vocab_size]
        # Slice the last position — that's the distribution for the next token.
        last_logits: torch.Tensor = outputs.logits[0, -1, :]  # [vocab_size]

        # --- 3. Autoregressive loop ------------------------------------
        for step_idx in range(max_new_tokens):

            # generate_step() does: temperature → softmax → entropy →
            # top-k → decode → greedy select.
            # It returns the structured step log and the chosen token ID.
            step, selected_id = self.generate_step(
                logits=last_logits,
                step_idx=step_idx,
                context_ids=generated_ids,
                top_k=top_k,
                temperature=temperature,
            )
            steps.append(step)
            generated_ids.append(selected_id)

            # Stop when the model emits the end-of-sequence token.
            if selected_id == self._tokenizer.eos_token_id:  # type: ignore[union-attr]
                break

            # Incremental forward pass: one new token + the cached context.
            # output logits: [1, 1, vocab_size]
            next_input = torch.tensor([[selected_id]])  # [1, 1]
            outputs = self._model(  # type: ignore[misc]
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        # --- 4. Decode final generated text ----------------------------
        # Decode only the newly generated token IDs, not the prompt.
        generated_text: str = self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        return generated_text, steps


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
# Import this object in route handlers:
#   from app.services.llm_service import llm_service
#
# Never instantiate LLMService() directly elsewhere — the model weights
# must only be loaded once per process.
llm_service = LLMService()
