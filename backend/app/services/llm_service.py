from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import DecodingStep, TopToken

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# Single worker — LlamaForCausalLM is not safe for concurrent forward passes.
_executor = ThreadPoolExecutor(max_workers=1)


class LLMService:
    """
    Singleton wrapper around Qwen/Qwen2.5-Coder-1.5B-Instruct.

    Thread-safe lazy loading: the model is downloaded and initialised on the
    first generate() call, then reused for every subsequent request.
    """

    def __init__(self) -> None:
        self._model: "PreTrainedModel | None" = None
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._loaded: bool = False
        # Re-entrant lock so the same thread can call load_model() twice
        # (e.g. from _run_generate_sync which itself runs in the executor).
        self._lock: threading.Lock = threading.Lock()

    # ------------------------------------------------------------------
    # Read-only accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> "PreTrainedModel | None":
        return self._model

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase | None":
        return self._tokenizer

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # ------------------------------------------------------------------
    # load_model
    # ------------------------------------------------------------------

    def load_model(self) -> None:
        """
        Load AutoTokenizer and AutoModelForCausalLM for Qwen2.5-Coder-Instruct.

        Safe to call multiple times — a threading.Lock ensures only one
        thread performs the actual load; all others block until it finishes.

        Model is loaded in fp32 on CPU.  low_cpu_mem_usage=True streams
        weight shards so peak RAM stays near the final footprint (~3 GB).
        """
        # Fast path — already loaded, skip the lock entirely.
        if self._loaded:
            return

        with self._lock:
            # Second check inside the lock — another thread may have loaded
            # the model while this thread was waiting to acquire the lock.
            if self._loaded:
                return

            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            log.info("Loading tokenizer: %s", settings.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_fast=True,          # Rust-backed tokenizer, much faster encode/decode
            )

            # Qwen2.5 has no dedicated pad token; reuse EOS so that any
            # batching helpers don't raise a ValueError.  Single-sample
            # inference is unaffected.
            if self._tokenizer.pad_token_id is None:  # type: ignore[union-attr]
                self._tokenizer.pad_token_id = (  # type: ignore[union-attr]
                    self._tokenizer.eos_token_id  # type: ignore[union-attr]
                    if isinstance(self._tokenizer.eos_token_id, int)  # type: ignore[union-attr]
                    else self._tokenizer.eos_token_id[0]  # type: ignore[union-attr]
                )

            log.info("Loading model: %s (fp32, CPU)", settings.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float32,   # fp32 for x86 CPU compatibility
                low_cpu_mem_usage=True,      # stream shards, avoid 2× RAM spike
            )
            self._model.eval()  # disable dropout; not needed for inference

            n_params = sum(p.numel() for p in self._model.parameters())
            log.info("Model ready — %.0fM parameters", n_params / 1e6)
            self._loaded = True

    # ------------------------------------------------------------------
    # generate_step
    # ------------------------------------------------------------------

    def generate_step(
        self,
        logits: torch.Tensor,   # [vocab_size]  raw logits for the current position
        step_idx: int,          # 0-based; stored as step_idx+1 in the log
        context_ids: list[int], # token IDs generated so far → decoded to context string
        top_k: int,
        temperature: float,
    ) -> tuple[DecodingStep, int]:
        """
        Compute probabilities, entropy, top-k candidates, and greedy selection
        from a single set of raw logits.

        Decoupled from the generation loop so Phase 3 (Syncode) can mask
        logits before calling this without touching any probability math.

        Returns
        -------
        step        : DecodingStep populated and ready for JSON logging
        selected_id : vocabulary index of the greedy-chosen token
        """
        # Temperature scaling — divide before softmax so the distribution
        # sharpens (T<1) or flattens (T>1) before probabilities are computed.
        scaled: torch.Tensor = logits / temperature  # [vocab_size]

        # Softmax → proper probability distribution over the full vocabulary.
        probs: torch.Tensor = F.softmax(scaled, dim=-1)  # [vocab_size]

        # Shannon entropy H = -Σ p·log(p).
        # Clamp before log to avoid -inf at p=0.
        # High H → uncertain (flat distribution). Low H → confident (peaked).
        entropy: float = float(
            -(probs * torch.log(probs.clamp(min=1e-12))).sum()
        )

        # top-k extraction — torch.topk returns values and indices sorted
        # in descending order.
        k = min(top_k, probs.size(0))
        topk_probs, topk_ids = torch.topk(probs, k=k)

        # Decode each candidate token ID to a human-readable string.
        # skip_special_tokens=False keeps <eos> visible in the table.
        # clean_up_tokenization_spaces=False preserves leading spaces
        # that are semantically part of the token (e.g. " int" ≠ "int").
        top_tokens: list[TopToken] = [
            TopToken(
                token=self._tokenizer.decode(  # type: ignore[union-attr]
                    [int(topk_ids[i])],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                probability=float(topk_probs[i]),
                token_id=int(topk_ids[i]),
            )
            for i in range(k)
        ]

        # Greedy selection — argmax over the FULL vocabulary, not just top-k.
        selected_id: int = int(torch.argmax(probs))
        selected_str: str = self._tokenizer.decode(  # type: ignore[union-attr]
            [selected_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        # Decode the context produced so far (everything before this step).
        context: str = self._tokenizer.decode(  # type: ignore[union-attr]
            context_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        step = DecodingStep(
            step=step_idx + 1,
            context=context,
            top_tokens=top_tokens,
            selected_token=selected_str,
            selected_token_id=selected_id,
            entropy_before=round(entropy, 4),
        )
        return step, selected_id

    # ------------------------------------------------------------------
    # generate  (async public API)
    # ------------------------------------------------------------------

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        top_k: int,
        temperature: float,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Async entry point for FastAPI route handlers.

        Dispatches the blocking CPU work to the single-worker
        ThreadPoolExecutor so the asyncio event loop stays free.
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
    # _run_generate_sync  (blocking, runs inside executor)
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
        Token-by-token greedy generation.  Synchronous / blocking — always
        called through the ThreadPoolExecutor, never from async context.

        Steps
        -----
        1. Lazy-load model if needed.
        2. Tokenise prompt → input_ids [1, prompt_len].
        3. Full forward pass over the prompt to prime the KV cache.
           past_key_values lets every subsequent step run in O(1) attention
           instead of re-processing the full context.
        4. Loop up to max_new_tokens:
             a. Call generate_step() → DecodingStep + selected_id.
             b. Break on EOS.
             c. Forward pass with the single new token + cached KV state.
        5. Decode and return generated text + step list.
        """
        if not self._loaded:
            self.load_model()

        # Format the prompt with the chat template when available.
        # Qwen2.5-Coder-Instruct expects the <|im_start|>user … <|im_end|> format;
        # apply_chat_template produces that wrapper automatically.
        # Falls back to the raw prompt string for plain base models.
        if callable(getattr(self._tokenizer, "apply_chat_template", None)):
            formatted_prompt: str = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        # Tokenise — add_special_tokens ensures BOS / chat-header tokens are added.
        encoded = self._tokenizer(  # type: ignore[misc]
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids: torch.Tensor = encoded["input_ids"]  # [1, prompt_len]

        # Build the set of EOS token IDs to watch for.
        # Qwen2.5 tokenizer returns eos_token_id as a list [151645, 151643]
        # (<|im_end|> and <|endoftext|>); older models return a plain int.
        raw_eos = self._tokenizer.eos_token_id  # type: ignore[union-attr]
        eos_ids: set[int] = (
            set(raw_eos) if isinstance(raw_eos, list) else {raw_eos}
        )

        steps: list[DecodingStep] = []
        generated_ids: list[int] = []

        # Prime KV cache — one forward pass over the entire prompt.
        outputs = self._model(input_ids=input_ids, use_cache=True)  # type: ignore[misc]
        past_key_values = outputs.past_key_values
        # logits: [1, prompt_len, vocab_size] → slice last position → [vocab_size]
        last_logits: torch.Tensor = outputs.logits[0, -1, :]

        log.debug("Starting generation: max_new_tokens=%d top_k=%d T=%.2f",
                  max_new_tokens, top_k, temperature)

        for step_idx in range(max_new_tokens):
            step, selected_id = self.generate_step(
                logits=last_logits,
                step_idx=step_idx,
                context_ids=generated_ids,
                top_k=top_k,
                temperature=temperature,
            )
            steps.append(step)
            generated_ids.append(selected_id)

            if selected_id in eos_ids:
                log.debug("EOS reached at step %d", step_idx + 1)
                break

            # Incremental forward pass — one token [1,1] + cached KV tensors.
            next_input = torch.tensor([[selected_id]])  # [1, 1]
            outputs = self._model(  # type: ignore[misc]
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[0, -1, :]  # [vocab_size]

        generated_text: str = self._tokenizer.decode(  # type: ignore[union-attr]
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        log.debug("Generation complete: %d tokens", len(steps))
        return generated_text, steps


# Module-level singleton — import this in route handlers.
# Never instantiate LLMService() elsewhere.
llm_service = LLMService()
