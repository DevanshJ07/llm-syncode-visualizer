from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import DecodingStep, TokenCandidate, TopToken

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# Single worker — Qwen2ForCausalLM is not safe for concurrent forward passes.
_executor = ThreadPoolExecutor(max_workers=1)


# ---------------------------------------------------------------------------
# Syncode grammar constraint wrapper
# ---------------------------------------------------------------------------

class _SyncodeConstraint:
    """
    Wrapper around syncode 0.4.x SyncodeLogitsProcessor.

    Real API (confirmed against syncode 0.4.16):
        from syncode import Grammar, SyncodeLogitsProcessor
        grammar   = Grammar('c')          # Grammar object, not a plain string
        processor = SyncodeLogitsProcessor(
                        grammar, tokenizer,
                        use_cache=True,          # caches DFA mask store to disk
                        parse_output_only=True,  # skip prompt tokens in parse state
                        mode='grammar_mask',
                    )
        processor.reset()                 # MUST call before every new generation
        masked = processor(all_input_ids, logits.unsqueeze(0))  # [1, vocab]

    The DFA mask store is cached to disk by syncode, so only the first run
    for a given (grammar, tokenizer) pair is slow (~30 s for C grammar).
    """

    def __init__(self, tokenizer: "PreTrainedTokenizerBase", grammar: str = "c") -> None:
        self._processor = None
        self._available = False

        try:
            from syncode import Grammar, SyncodeLogitsProcessor  # noqa: PLC0415

            log.info(
                "Initializing Syncode %s-grammar processor "
                "(first run compiles DFA mask store — may take ~30 s).",
                grammar,
            )
            gram_obj = Grammar(grammar)
            self._processor = SyncodeLogitsProcessor(
                grammar=gram_obj,
                tokenizer=tokenizer,
                use_cache=True,
                parse_output_only=True,
                num_samples=1,
                mode="grammar_mask",
            )
            self._available = True
            log.info("Syncode ready (grammar=%s)", grammar)

        except ImportError:
            log.warning(
                "syncode package not found — install with: pip install syncode. "
                "use_syncode requests will fall back to raw mode."
            )
        except Exception as exc:
            log.warning(
                "Syncode initialization failed (%s). Falling back to raw mode.", exc
            )

    @property
    def available(self) -> bool:
        return self._available

    def reset(self) -> None:
        """Reset parse state — MUST be called before each new generation."""
        if self._processor is not None:
            try:
                self._processor.reset()
            except Exception as exc:
                log.debug("Syncode reset failed: %s", exc)

    def mask(
        self,
        all_input_ids: torch.Tensor,  # [1, seq_len] — full context (prompt + generated)
        logits: torch.Tensor,          # [vocab_size]
    ) -> "torch.Tensor | None":
        """
        Returns logits with -inf for grammar-invalid tokens.
        Returns None on any failure so the caller falls back to raw mode.

        Calls SyncodeLogitsProcessor.__call__(input_ids [1,T], scores [1,V])
        which returns [1, vocab_size] with -inf for invalid tokens.
        """
        if not self._available or self._processor is None:
            return None
        try:
            out: torch.Tensor = self._processor(
                input_ids=all_input_ids,
                scores=logits.unsqueeze(0),  # [1, vocab_size]
            )
            return out.squeeze(0)            # [vocab_size]
        except Exception as exc:
            log.debug("Syncode mask step failed: %s", exc)
            return None


# ---------------------------------------------------------------------------
# LLMService
# ---------------------------------------------------------------------------

class LLMService:
    """
    Singleton wrapper around Qwen/Qwen2.5-Coder-1.5B-Instruct.

    Thread-safe lazy loading: the model is downloaded and initialised on the
    first generate() call, then reused for every subsequent request.

    When settings.syncode_enabled is True the service also initialises a
    _SyncodeConstraint (C grammar) that can be activated per-request via
    the use_syncode flag.
    """

    def __init__(self) -> None:
        self._model: "PreTrainedModel | None" = None
        self._tokenizer: "PreTrainedTokenizerBase | None" = None
        self._loaded: bool = False
        self._syncode: "_SyncodeConstraint | None" = None
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

        If settings.syncode_enabled is True the Syncode C-grammar processor
        is also initialised here (once, cached globally).
        """
        if self._loaded:
            return

        with self._lock:
            if self._loaded:
                return

            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415

            log.info("Loading tokenizer: %s", settings.model_name)
            self._tokenizer = AutoTokenizer.from_pretrained(
                settings.model_name,
                use_fast=True,
            )

            # Qwen2.5 has no dedicated pad token; reuse EOS to satisfy any
            # batching helpers that require pad_token_id to be set.
            if self._tokenizer.pad_token_id is None:  # type: ignore[union-attr]
                raw_eos = self._tokenizer.eos_token_id  # type: ignore[union-attr]
                self._tokenizer.pad_token_id = (  # type: ignore[union-attr]
                    raw_eos if isinstance(raw_eos, int) else raw_eos[0]
                )

            log.info("Loading model: %s (fp32, CPU)", settings.model_name)
            self._model = AutoModelForCausalLM.from_pretrained(
                settings.model_name,
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
            )
            self._model.eval()

            n_params = sum(p.numel() for p in self._model.parameters())
            log.info("Model ready — %.0fM parameters", n_params / 1e6)

            # Optionally initialise Syncode grammar constraint.
            if settings.syncode_enabled:
                self._syncode = _SyncodeConstraint(self._tokenizer)
            else:
                log.info("Syncode disabled (SYNCODE_ENABLED=false). "
                         "Set to true and restart to enable grammar masking.")

            self._loaded = True

    # ------------------------------------------------------------------
    # generate_step
    # ------------------------------------------------------------------

    def generate_step(
        self,
        logits: torch.Tensor,           # [vocab_size] raw logits for the current position
        step_idx: int,                  # 0-based; stored as step_idx+1 in the log
        context_ids: list[int],         # generated IDs so far → decoded to context string
        top_k: int,
        temperature: float,
        masked_logits: "torch.Tensor | None" = None,  # [vocab_size] or None for raw mode
    ) -> tuple[DecodingStep, int]:
        """
        Compute per-step decoding data from raw (and optionally Syncode-masked) logits.

        Raw mode  (masked_logits is None):
            probs = softmax(logits / T)
            selected = argmax(probs)
            top_tokens filled; Syncode fields empty.

        Syncode mode  (masked_logits provided):
            probs_raw    = softmax(logits / T)         → entropy_before, top_tokens_before_syncode
            probs_masked = softmax(masked_logits / T)  → entropy_after, valid_tokens_after_syncode
            selected     = argmax(probs_masked)         (from constrained distribution)
            masked_tokens = top-k IDs that were -inf in masked_logits

        generate_step is intentionally decoupled from the generation loop so
        that Phase 3 Syncode can inject the masked_logits without touching
        any probability maths.
        """
        # ── RAW distribution (always computed) ─────────────────────────────
        scaled_raw: torch.Tensor = logits / temperature
        probs_raw: torch.Tensor = F.softmax(scaled_raw, dim=-1)

        entropy_before: float = float(
            -(probs_raw * torch.log(probs_raw.clamp(min=1e-12))).sum()
        )

        k = min(top_k, probs_raw.size(0))
        topk_raw_probs, topk_raw_ids = torch.topk(probs_raw, k=k)

        # ── Resolve masked distribution ─────────────────────────────────────
        use_syncode = masked_logits is not None

        if use_syncode:
            # Safety: if syncode masked every token fall back to raw selection.
            if (masked_logits == float("-inf")).all():  # type: ignore[operator]
                log.warning("Syncode masked ALL tokens at step %d — using raw selection", step_idx + 1)
                use_syncode = False
                masked_logits = None

        if use_syncode and masked_logits is not None:
            scaled_masked = masked_logits / temperature
            probs_masked: torch.Tensor = F.softmax(scaled_masked, dim=-1)

            entropy_after: float | None = float(
                -(probs_masked * torch.log(probs_masked.clamp(min=1e-12))).sum()
            )

            # Greedy selection from the CONSTRAINED distribution.
            selected_id: int = int(torch.argmax(probs_masked))

            # A token is "masked" if it was valid in raw logits but set to -inf
            # by Syncode.  We report only those appearing in the raw top-k so
            # the frontend payload stays small.
            masked_flag: torch.Tensor = (
                (masked_logits == float("-inf")) & (logits > float("-inf"))
            )
            masked_in_topk: list[int] = [
                int(topk_raw_ids[i])
                for i in range(k)
                if masked_flag[int(topk_raw_ids[i])]
            ]
            num_masked_total: int = int(masked_flag.sum())

            # top_tokens_before_syncode — raw top-k with is_masked annotation
            top_tokens_before_syncode: list[TokenCandidate] = [
                TokenCandidate(
                    token_id=int(topk_raw_ids[i]),
                    token_str=self._tokenizer.decode(  # type: ignore[union-attr]
                        [int(topk_raw_ids[i])],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    probability=float(topk_raw_probs[i]),
                    is_masked=bool(masked_flag[int(topk_raw_ids[i])]),
                    is_selected=int(topk_raw_ids[i]) == selected_id,
                )
                for i in range(k)
            ]

            # valid_tokens_after_syncode — top-k from the constrained distribution
            topk_masked_probs, topk_masked_ids = torch.topk(probs_masked, k=k)
            valid_tokens_after_syncode: list[TokenCandidate] = [
                TokenCandidate(
                    token_id=int(topk_masked_ids[i]),
                    token_str=self._tokenizer.decode(  # type: ignore[union-attr]
                        [int(topk_masked_ids[i])],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    probability=float(topk_masked_probs[i]),
                    is_masked=False,  # by definition — these survived masking
                    is_selected=int(topk_masked_ids[i]) == selected_id,
                )
                for i in range(k)
                if float(topk_masked_probs[i]) > 1e-12  # skip effectively zero tokens
            ]

        else:
            # Raw mode — greedy from the unmasked distribution
            selected_id = int(torch.argmax(probs_raw))
            entropy_after = None
            top_tokens_before_syncode = []
            valid_tokens_after_syncode = []
            masked_in_topk = []
            num_masked_total = 0

        # ── Decode strings ──────────────────────────────────────────────────
        selected_str: str = self._tokenizer.decode(  # type: ignore[union-attr]
            [selected_id],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )

        context: str = self._tokenizer.decode(  # type: ignore[union-attr]
            context_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # top_tokens — raw top-k always present (used by raw-mode charts)
        top_tokens: list[TopToken] = [
            TopToken(
                token=self._tokenizer.decode(  # type: ignore[union-attr]
                    [int(topk_raw_ids[i])],
                    skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                ),
                probability=float(topk_raw_probs[i]),
                token_id=int(topk_raw_ids[i]),
            )
            for i in range(k)
        ]

        step = DecodingStep(
            step=step_idx + 1,
            context=context,
            top_tokens=top_tokens,
            selected_token=selected_str,
            selected_token_id=selected_id,
            entropy_before=round(entropy_before, 4),
            # Syncode fields — empty in raw mode, populated in Syncode mode
            top_tokens_before_syncode=top_tokens_before_syncode,
            masked_tokens=masked_in_topk,
            valid_tokens_after_syncode=valid_tokens_after_syncode,
            entropy_after=round(entropy_after, 4) if entropy_after is not None else None,
            num_masked=num_masked_total,
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
        use_syncode: bool = False,
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
            use_syncode,
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
        use_syncode: bool = False,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Token-by-token greedy generation.  Synchronous / blocking — always
        called through the ThreadPoolExecutor, never from async context.

        Steps
        -----
        1. Lazy-load model (and Syncode if enabled) if needed.
        2. Apply Qwen chat template to the prompt.
        3. Tokenise → input_ids [1, prompt_len].
        4. Full forward pass over the prompt to prime the KV cache.
        5. Loop up to max_new_tokens:
             a. If use_syncode: get grammar mask from _SyncodeConstraint.
             b. Call generate_step() with optional masked_logits.
             c. Break on EOS.
             d. Incremental forward pass with new token + cached KV state.
             e. Update all_input_ids (for Syncode context tracking).
        6. Decode and return generated text + step list.
        """
        if not self._loaded:
            self.load_model()

        # Determine if Syncode is actually usable for this request.
        effective_syncode = (
            use_syncode
            and self._syncode is not None
            and self._syncode.available
        )
        if use_syncode and not effective_syncode:
            log.warning(
                "use_syncode=True but Syncode is unavailable — "
                "generating in raw mode."
            )

        # Reset Syncode parse state so this generation starts from a clean
        # grammar state (the processor is reused across requests).
        if effective_syncode and self._syncode is not None:
            self._syncode.reset()
            log.debug("Syncode parse state reset for new generation")

        # ── Format prompt with Qwen chat template ──────────────────────────
        if callable(getattr(self._tokenizer, "apply_chat_template", None)):
            formatted_prompt: str = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            formatted_prompt = prompt

        encoded = self._tokenizer(  # type: ignore[misc]
            formatted_prompt,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids: torch.Tensor = encoded["input_ids"]  # [1, prompt_len]

        # Track the full sequence (prompt + generated) for Syncode context.
        # The model uses past_key_values so only needs the new token each step,
        # but Syncode's LogitsProcessor needs the full sequence to parse state.
        all_input_ids: torch.Tensor = input_ids

        # Build EOS set — Qwen2.5 returns a list [151645, 151643].
        raw_eos = self._tokenizer.eos_token_id  # type: ignore[union-attr]
        eos_ids: set[int] = (
            set(raw_eos) if isinstance(raw_eos, list) else {raw_eos}
        )

        steps: list[DecodingStep] = []
        generated_ids: list[int] = []

        # ── Prime KV cache ─────────────────────────────────────────────────
        outputs = self._model(input_ids=input_ids, use_cache=True)  # type: ignore[misc]
        past_key_values = outputs.past_key_values
        last_logits: torch.Tensor = outputs.logits[0, -1, :]  # [vocab_size]

        log.debug(
            "Starting generation: max_new_tokens=%d top_k=%d T=%.2f syncode=%s",
            max_new_tokens, top_k, temperature, effective_syncode,
        )

        for step_idx in range(max_new_tokens):
            # ── Optional Syncode mask ───────────────────────────────────────
            masked_logits: torch.Tensor | None = None
            if effective_syncode and self._syncode is not None:
                masked_logits = self._syncode.mask(all_input_ids, last_logits)
                # If Syncode masked every token it's a bug — fall back to raw.
                if masked_logits is not None and (masked_logits == float("-inf")).all():
                    log.warning("Syncode masked all tokens at step %d — using raw", step_idx + 1)
                    masked_logits = None

            step, selected_id = self.generate_step(
                logits=last_logits,
                step_idx=step_idx,
                context_ids=generated_ids,
                top_k=top_k,
                temperature=temperature,
                masked_logits=masked_logits,
            )
            steps.append(step)
            generated_ids.append(selected_id)

            if selected_id in eos_ids:
                log.debug("EOS reached at step %d", step_idx + 1)
                break

            # ── Incremental forward pass ────────────────────────────────────
            next_input = torch.tensor([[selected_id]])  # [1, 1]
            outputs = self._model(  # type: ignore[misc]
                input_ids=next_input,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            last_logits = outputs.logits[0, -1, :]

            # Extend full sequence for Syncode context tracking.
            all_input_ids = torch.cat([all_input_ids, next_input], dim=1)

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
