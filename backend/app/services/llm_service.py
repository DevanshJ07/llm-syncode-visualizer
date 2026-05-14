from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from app.core.config import settings
from app.models.schemas import DecodingStep, MaskedTokenEntry, TokenCandidate, TopToken

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase

log = logging.getLogger(__name__)

# Single worker — Qwen2ForCausalLM is not safe for concurrent forward passes.
_executor = ThreadPoolExecutor(max_workers=1)

# Consecutive whitespace tokens before graceful stop.
_WHITESPACE_STALL_THRESHOLD: int = 10


# ---------------------------------------------------------------------------
# Sampling helpers
# ---------------------------------------------------------------------------

def _apply_repetition_penalty(
    logits: torch.Tensor,
    past_ids: list[int],
    penalty: float,
) -> torch.Tensor:
    """
    Apply repetition penalty (Keskar et al. 2019) to raw logits in-place.

    Tokens that already appeared in past_ids have their logit divided (if
    positive) or multiplied (if negative) by *penalty*, making them less
    likely to be re-selected.  penalty=1.0 is a no-op.
    """
    if penalty == 1.0 or not past_ids:
        return logits
    logits = logits.clone()
    for token_id in set(past_ids):
        if token_id < logits.size(0):
            val = logits[token_id]
            logits[token_id] = val / penalty if val >= 0 else val * penalty
    return logits


def _nucleus_sample(probs: torch.Tensor, top_p: float) -> int:
    """
    Top-p (nucleus) sampling (Holtzman et al. 2020).

    Keeps the smallest set of tokens whose cumulative probability mass exceeds
    *top_p*, then re-normalises and samples.  Degrades to uniform sampling
    over the entire distribution when top_p >= 1.0.
    """
    if top_p >= 1.0:
        return int(torch.multinomial(probs.clamp(min=0.0), num_samples=1).item())

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Shift cumulative probs right so the first token is always included,
    # then zero out tokens past the nucleus boundary.
    to_remove = (cumulative_probs - sorted_probs) > top_p
    filtered = sorted_probs.clone()
    filtered[to_remove] = 0.0

    total = filtered.sum()
    if total <= 0.0:
        # Degenerate: fall back to the highest-probability token.
        return int(sorted_indices[0].item())

    filtered = filtered / total
    sampled_local_idx = int(torch.multinomial(filtered, num_samples=1).item())
    return int(sorted_indices[sampled_local_idx].item())


def _is_whitespace_token(token_str: str) -> bool:
    """Return True when a decoded token contains only whitespace/newline chars."""
    return bool(token_str) and not token_str.strip()


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
    ) -> "tuple[torch.Tensor | None, str | None]":
        """
        Returns (masked_logits, error_message).

        masked_logits:
            torch.Tensor [vocab_size] with -inf for grammar-invalid tokens, or
            None if Syncode is unavailable / the processor raised an exception.
        error_message:
            None on success; a string describing what went wrong otherwise.

        The caller should ALWAYS fall back to raw logits when masked_logits
        is None — generation must never be aborted due to a parser error.
        """
        if not self._available or self._processor is None:
            return None, None
        try:
            out: torch.Tensor = self._processor(
                input_ids=all_input_ids,
                scores=logits.unsqueeze(0),  # [1, vocab_size]
            )
            return out.squeeze(0), None      # [vocab_size], no error
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            log.warning("Syncode mask step failed (%s) — falling back to raw logits", msg)
            return None, msg


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
        parser_error_msg: "str | None" = None,        # set when grammar parser threw
        fallback_used: bool = False,                  # True when raw was used as fallback
        # Sampling parameters
        do_sample: bool = True,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
        past_generated_ids: "list[int] | None" = None,
        # Whitespace stall metadata (tracked externally, attached here for the step log)
        consecutive_whitespace: int = 0,
        whitespace_stall_detected: bool = False,
        whitespace_stall_step: "int | None" = None,
    ) -> tuple[DecodingStep, int]:
        """
        Compute per-step decoding data from raw (and optionally Syncode-masked) logits.

        Raw mode  (masked_logits is None):
            probs = softmax(penalized_logits / T)
            selected = nucleus_sample(probs, top_p)  or argmax when do_sample=False
            top_tokens filled; Syncode fields empty.

        Syncode mode  (masked_logits provided):
            probs_raw    = softmax(logits / T)                → entropy_before, top_tokens_before_syncode
            probs_masked = softmax(masked_logits / T)         → entropy_after, valid_tokens_after_syncode
            selected     = nucleus_sample(penalized_masked_probs, top_p)
            masked_tokens = top-k IDs that were -inf in masked_logits

        Repetition penalty is applied to the logits used for *selection* only;
        visualisation distributions (entropy, top_tokens) use the raw/masked
        logits without penalty so the charts reflect the model's true output.

        generate_step is intentionally decoupled from the generation loop so
        Syncode can inject masked_logits without touching any probability maths.
        """
        _past = past_generated_ids or []

        # ── RAW distribution (always computed — used for visualisation) ─────
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

            # Sampling from the CONSTRAINED distribution.
            # Apply repetition penalty to masked logits before selection so
            # previously generated tokens are further suppressed; the
            # visualisation probs_masked is kept penalty-free for clarity.
            if do_sample:
                pen_masked = _apply_repetition_penalty(masked_logits, _past, repetition_penalty)
                sel_probs_masked = F.softmax(pen_masked / temperature, dim=-1)
                selected_id: int = _nucleus_sample(sel_probs_masked, top_p)
            else:
                selected_id = int(torch.argmax(probs_masked))

            # Boolean mask: True where Syncode set a token to -inf.
            # "grammar-invalid" = was finite in raw logits but -inf after masking.
            masked_flag: torch.Tensor = (
                (masked_logits == float("-inf")) & (logits > float("-inf"))
            )
            num_masked_total: int = int(masked_flag.sum())
            vocab_sz: int = int(logits.size(0))
            valid_cnt: int = vocab_sz - num_masked_total

            # Probability mass removed = Σ raw_prob of all masked tokens.
            prob_mass_removed: float = float(probs_raw[masked_flag].sum())

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

            # masked_tokens — top-k rejected tokens from raw distribution,
            # with their raw probabilities for the "Masked Tokens" panel.
            masked_in_topk: list[MaskedTokenEntry] = [
                MaskedTokenEntry(
                    token_id=int(topk_raw_ids[i]),
                    token=self._tokenizer.decode(  # type: ignore[union-attr]
                        [int(topk_raw_ids[i])],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    raw_prob=float(topk_raw_probs[i]),
                )
                for i in range(k)
                if masked_flag[int(topk_raw_ids[i])]
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
            # Raw mode — sample (or greedy) from the unmasked distribution.
            if do_sample:
                pen_raw = _apply_repetition_penalty(logits, _past, repetition_penalty)
                sel_probs_raw = F.softmax(pen_raw / temperature, dim=-1)
                selected_id = _nucleus_sample(sel_probs_raw, top_p)
            else:
                selected_id = int(torch.argmax(probs_raw))
            entropy_after = None
            top_tokens_before_syncode = []
            valid_tokens_after_syncode = []
            masked_in_topk = []
            num_masked_total = 0
            vocab_sz = int(logits.size(0))
            valid_cnt = vocab_sz
            prob_mass_removed = 0.0

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

        masked_pct: float = (
            round(num_masked_total / vocab_sz * 100, 2) if vocab_sz > 0 else 0.0
        )

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
            # Masking statistics
            vocab_size=vocab_sz,
            valid_token_count=valid_cnt,
            masked_token_count=num_masked_total,
            masked_percentage=masked_pct,
            probability_mass_removed=round(prob_mass_removed, 6),
            # Parser recovery metadata
            parser_error=parser_error_msg is not None,
            parser_error_message=parser_error_msg or "",
            fallback_used=fallback_used,
            # Whitespace stall metadata
            consecutive_whitespace_count=consecutive_whitespace,
            whitespace_stall_detected=whitespace_stall_detected,
            whitespace_stall_step=whitespace_stall_step,
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
        do_sample: bool = True,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Async entry point for FastAPI route handlers.

        Dispatches the blocking CPU work to the single-worker
        ThreadPoolExecutor so the asyncio event loop stays free.
        """
        import functools  # noqa: PLC0415
        loop = asyncio.get_running_loop()
        fn = functools.partial(
            self._run_generate_sync,
            prompt,
            max_new_tokens,
            top_k,
            temperature,
            use_syncode,
            do_sample,
            top_p,
            repetition_penalty,
        )
        return await loop.run_in_executor(_executor, fn)

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
        do_sample: bool = True,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ) -> tuple[str, list[DecodingStep]]:
        """
        Token-by-token generation with nucleus sampling and whitespace stall
        protection.  Synchronous / blocking — always called through the
        ThreadPoolExecutor, never from async context.

        Steps
        -----
        1. Lazy-load model (and Syncode if enabled) if needed.
        2. Apply Qwen chat template to the prompt.
        3. Tokenise → input_ids [1, prompt_len].
        4. Full forward pass over the prompt to prime the KV cache.
        5. Loop up to max_new_tokens:
             a. If use_syncode: get grammar mask from _SyncodeConstraint.
             b. Call generate_step() with sampling params and stall metadata.
             c. Break on EOS or whitespace stall (>_WHITESPACE_STALL_THRESHOLD
                consecutive whitespace-only tokens).
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

        # ── Format prompt ───────────────────────────────────────────────────
        # In Syncode/C-grammar mode we MUST NOT use the chat template.
        #
        # The chat template wraps the prompt in <|im_start|>user…<|im_end|>
        # <|im_start|>assistant\n.  The model then assigns high probability to
        # conversational preamble tokens ("Sure, here's a function…") that the
        # C LALR grammar cannot parse.  When the grammar parser fails it falls
        # back to unconstrained decoding, at which point the model's top-1
        # token is the chat EOS token <|im_end|> — producing the observed
        # "c<|im_end|>" output.
        #
        # A C-comment completion prompt sidesteps this:
        #   - The model is anchored to produce C code from token 0
        #   - The grammar starts from its root state (valid for any C file)
        #   - No chat control tokens appear in either direction
        if effective_syncode:
            formatted_prompt: str = f"// {prompt}\n"
        elif callable(getattr(self._tokenizer, "apply_chat_template", None)):
            formatted_prompt = self._tokenizer.apply_chat_template(  # type: ignore[union-attr]
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

        # Precompute the set of special/chat token IDs that must never appear
        # in grammar-constrained output.  This is a belt-and-suspenders guard:
        # Syncode already pads its accept_mask with False for IDs beyond
        # tokenizer.vocab_size (which covers the Qwen chat tokens <|im_end|>
        # etc.), but if the grammar parser falls back to unconstrained decoding
        # (exception in incremental LALR parse) it would skip the mask entirely.
        # By explicitly zeroing these IDs we make the suppression unconditional.
        syncode_suppress_ids: set[int] = set()
        if effective_syncode:
            vocab_dim = int(last_logits.size(0))
            syncode_suppress_ids = {
                sid for sid in self._tokenizer.all_special_ids  # type: ignore[union-attr]
                if sid < vocab_dim
            }
            log.debug(
                "Syncode special-token suppression covers %d IDs (vocab_dim=%d)",
                len(syncode_suppress_ids), vocab_dim,
            )

        log.debug(
            "Starting generation: max_new_tokens=%d top_k=%d T=%.2f "
            "do_sample=%s top_p=%.2f rep_penalty=%.2f syncode=%s",
            max_new_tokens, top_k, temperature,
            do_sample, top_p, repetition_penalty, effective_syncode,
        )

        # Whitespace stall state — reset per generation.
        consecutive_whitespace_count: int = 0
        whitespace_stall_step_num: int | None = None

        try:
            for step_idx in range(max_new_tokens):
                # ── Optional Syncode mask ───────────────────────────────────
                masked_logits: torch.Tensor | None = None
                step_parser_error: str | None = None
                step_fallback_used: bool = False

                if effective_syncode and self._syncode is not None:
                    masked_logits, step_parser_error = self._syncode.mask(
                        all_input_ids, last_logits
                    )

                    # Syncode processor returned None (parser exception or
                    # unavailable): clone raw logits so special-token suppression
                    # can still be applied without mutating last_logits.
                    if masked_logits is None:
                        step_fallback_used = True
                        masked_logits = last_logits.clone()

                    # Unconditionally suppress chat/special tokens regardless of
                    # whether the grammar mask succeeded or fell back.
                    for sid in syncode_suppress_ids:
                        masked_logits[sid] = float("-inf")

                    # If every token is masked (shouldn't happen), abandon the
                    # masked distribution and use raw.
                    if (masked_logits == float("-inf")).all():
                        log.warning(
                            "Syncode masked all tokens at step %d — using raw",
                            step_idx + 1,
                        )
                        masked_logits = None
                        step_fallback_used = True

                # Determine if this step is already in a stall (stall fired
                # on a previous step; we annotate this step too for clarity).
                step_stall_detected = (
                    whitespace_stall_step_num is not None
                )

                step, selected_id = self.generate_step(
                    logits=last_logits,
                    step_idx=step_idx,
                    context_ids=generated_ids,
                    top_k=top_k,
                    temperature=temperature,
                    masked_logits=masked_logits,
                    parser_error_msg=step_parser_error,
                    fallback_used=step_fallback_used,
                    do_sample=do_sample,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    past_generated_ids=generated_ids,
                    consecutive_whitespace=consecutive_whitespace_count,
                    whitespace_stall_detected=step_stall_detected,
                    whitespace_stall_step=whitespace_stall_step_num,
                )
                steps.append(step)
                generated_ids.append(selected_id)

                if selected_id in eos_ids:
                    log.debug("EOS reached at step %d", step_idx + 1)
                    break

                # ── Whitespace stall detection ──────────────────────────────
                # Count consecutive tokens that contain only whitespace/newlines.
                # If the count exceeds the threshold we stop gracefully to avoid
                # infinite grammar-valid whitespace loops in constrained mode.
                if _is_whitespace_token(step.selected_token):
                    consecutive_whitespace_count += 1
                    if (
                        consecutive_whitespace_count > _WHITESPACE_STALL_THRESHOLD
                        and whitespace_stall_step_num is None
                    ):
                        whitespace_stall_step_num = step_idx + 1
                        log.warning(
                            "Whitespace stall detected at step %d "
                            "(%d consecutive whitespace tokens) — stopping generation.",
                            whitespace_stall_step_num,
                            consecutive_whitespace_count,
                        )
                        break
                else:
                    consecutive_whitespace_count = 0

                # ── Incremental forward pass ────────────────────────────────
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

        except Exception as exc:
            # Any unexpected exception in the generation loop is caught here.
            # We return whatever tokens were generated before the failure so
            # the API always returns a valid (possibly partial) response.
            log.error(
                "Generation loop failed at step %d/%d: %s — returning %d partial tokens",
                step_idx + 1 if "step_idx" in dir() else 0,
                max_new_tokens,
                exc,
                len(steps),
                exc_info=True,
            )

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
