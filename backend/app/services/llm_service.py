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
# Debug trace buffer — stores the last completed generation's step-level
# diagnostic data.  Protected by a lock because the executor thread writes
# it while the event loop may read it concurrently.
# ---------------------------------------------------------------------------
_trace_lock = threading.Lock()
_last_trace: dict = {
    "generation_id": None,
    "prompt": "",
    "mode": "raw",
    "effective_syncode": False,
    "steps": [],
    "summary": {},
}


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
        self._tokenizer = tokenizer
        # Whitespace token IDs — precomputed once for fast per-step checking.
        self._whitespace_ids: frozenset[int] = self._build_whitespace_ids(tokenizer)
        log.info(
            "Syncode: %d whitespace-only token IDs in vocabulary",
            len(self._whitespace_ids),
        )

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

    @staticmethod
    def _build_whitespace_ids(tokenizer: "PreTrainedTokenizerBase") -> frozenset[int]:
        """
        Scan the vocabulary and collect IDs whose decoded string is pure
        whitespace (spaces, newlines, tabs).  Capped at vocab_size to avoid
        iterating into special-token ranges that may be sparse.
        """
        ws_ids: set[int] = set()
        vocab_size = getattr(tokenizer, "vocab_size", None) or 0
        for token_id in range(vocab_size):
            try:
                decoded = tokenizer.decode(
                    [token_id],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
                if decoded and not decoded.strip():
                    ws_ids.add(token_id)
            except Exception:
                pass
        return frozenset(ws_ids)

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
        step_idx: int = 0,
    ) -> "tuple[torch.Tensor | None, str | None, dict]":
        """
        Returns (masked_logits, error_message, diagnostics).

        masked_logits:
            torch.Tensor [vocab_size] with -inf for grammar-invalid tokens, or
            None if Syncode is unavailable / the processor raised an exception.
        error_message:
            None on success; a string describing what went wrong otherwise.
        diagnostics:
            {
                "grammar_masked_count":   int   # tokens newly set to -inf by Syncode
                "logits_changed":         bool  # any element differs from raw?
                "whitespace_tokens_masked": bool  # any whitespace-only token masked?
                "whitespace_tokens_accepted": int # whitespace IDs still valid after mask
                "valid_token_count":      int
            }

        The caller MUST always fall back to raw logits when masked_logits is
        None — generation must never be aborted due to a parser error.
        """
        diag: dict = {
            "grammar_masked_count": 0,
            "logits_changed": False,
            "whitespace_tokens_masked": False,
            "whitespace_tokens_accepted": 0,
            "valid_token_count": int(logits.size(0)),
        }
        if not self._available or self._processor is None:
            return None, None, diag
        try:
            out: torch.Tensor = self._processor(
                input_ids=all_input_ids,
                scores=logits.unsqueeze(0),  # [1, vocab_size]
            )
            out = out.squeeze(0)  # [vocab_size]

            # ── Diagnostics ────────────────────────────────────────────────
            # Tokens that were finite in raw but become -inf after masking.
            newly_masked: torch.Tensor = (
                (out == float("-inf")) & (logits > float("-inf"))
            )
            n_newly_masked = int(newly_masked.sum())
            logits_changed = bool((out != logits).any())

            # Whitespace coverage — are whitespace tokens masked or accepted?
            ws_ids_tensor = torch.tensor(
                list(self._whitespace_ids), dtype=torch.long
            ) if self._whitespace_ids else torch.tensor([], dtype=torch.long)

            ws_masked = False
            ws_accepted = 0
            if ws_ids_tensor.numel() > 0:
                valid_ws = ws_ids_tensor[ws_ids_tensor < out.size(0)]
                if valid_ws.numel() > 0:
                    ws_out = out[valid_ws]
                    ws_masked = bool((ws_out == float("-inf")).any())
                    ws_accepted = int((ws_out > float("-inf")).sum())

            valid_count = int((out > float("-inf")).sum())

            diag.update({
                "grammar_masked_count": n_newly_masked,
                "logits_changed": logits_changed,
                "whitespace_tokens_masked": ws_masked,
                "whitespace_tokens_accepted": ws_accepted,
                "valid_token_count": valid_count,
            })

            log.debug(
                "Syncode step %d: newly_masked=%d logits_changed=%s "
                "ws_masked=%s ws_accepted=%d valid=%d",
                step_idx + 1, n_newly_masked, logits_changed,
                ws_masked, ws_accepted, valid_count,
            )
            return out, None, diag

        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            log.warning("Syncode mask step failed (%s) — falling back to raw logits", msg)
            return None, msg, diag


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
        # Pipeline integrity diagnostics (passed from generation loop)
        syncode_grammar_changed: bool = False,   # Syncode's pure grammar mask changed logits
        syncode_diag: "dict | None" = None,      # diagnostics from _SyncodeConstraint.mask()
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

        Pipeline assertions (active at all log levels):
          • When do_sample=False AND Syncode masked 0 tokens AND logits are
            identical: selected_id must equal raw_argmax_id.
          • When fallback_used=True: selection_source must start with "fallback".

        generate_step is intentionally decoupled from the generation loop so
        Syncode can inject masked_logits without touching any probability maths.
        """
        _past = past_generated_ids or []
        _diag = syncode_diag or {}

        # ── RAW distribution (always computed — used for visualisation) ─────
        scaled_raw: torch.Tensor = logits / temperature
        probs_raw: torch.Tensor = F.softmax(scaled_raw, dim=-1)

        entropy_before: float = float(
            -(probs_raw * torch.log(probs_raw.clamp(min=1e-12))).sum()
        )

        k = min(top_k, probs_raw.size(0))
        topk_raw_probs, topk_raw_ids = torch.topk(probs_raw, k=k)
        _topk_raw_id_list: list[int] = topk_raw_ids.tolist()

        # Always compute raw greedy argmax for diagnostic comparison.
        raw_argmax_id: int = int(torch.argmax(probs_raw))

        # ── Resolve masked distribution ─────────────────────────────────────
        use_syncode = masked_logits is not None

        if use_syncode:
            # Safety: if syncode masked every token fall back to raw selection.
            if (masked_logits == float("-inf")).all():  # type: ignore[operator]
                log.warning("Syncode masked ALL tokens at step %d — using raw selection", step_idx + 1)
                use_syncode = False
                masked_logits = None

        if use_syncode and masked_logits is not None:
            # ── Constrained distribution ────────────────────────────────────
            # probs_masked is derived EXCLUSIVELY from masked_logits (Syncode
            # output), NOT from the raw logits.  Any divergence between
            # probs_masked and probs_raw proves the grammar mask is active.
            scaled_masked = masked_logits / temperature
            probs_masked: torch.Tensor = F.softmax(scaled_masked, dim=-1)

            entropy_after: float | None = float(
                -(probs_masked * torch.log(probs_masked.clamp(min=1e-12))).sum()
            )

            # Constrained greedy argmax — always computed even when sampling,
            # so we can assert the greedy invariant and report the rank.
            # NOTE: argmax(masked_logits) == argmax(probs_masked) because
            # softmax is a strictly monotone transform.
            constrained_argmax_id: int = int(torch.argmax(masked_logits))

            # Constrained top-k — extracted from probs_masked, NOT probs_raw.
            # This is what valid_tokens_after_syncode MUST be built from.
            topk_masked_probs, topk_masked_ids = torch.topk(probs_masked, k=k)
            _topk_con_id_list: list[int] = topk_masked_ids.tolist()

            # ── Selection ───────────────────────────────────────────────────
            if do_sample:
                pen_masked = _apply_repetition_penalty(masked_logits, _past, repetition_penalty)
                sel_probs_masked = F.softmax(pen_masked / temperature, dim=-1)
                selected_id: int = _nucleus_sample(sel_probs_masked, top_p)
                _sel_source = "fallback_sampled" if fallback_used else "constrained_sampled"
            else:
                # GREEDY PATH: selected_id MUST equal constrained_argmax_id.
                selected_id = constrained_argmax_id
                _sel_source = "fallback_greedy" if fallback_used else "constrained_greedy"

            # ── Hard greedy assertion ────────────────────────────────────────
            # assert selected_token_id == constrained_logits.argmax().item()
            # This fires regardless of whether Syncode changed anything — it
            # verifies the fundamental invariant that greedy always picks the
            # constrained argmax, not the raw argmax.
            if not do_sample:
                _expected_greedy = int(torch.argmax(masked_logits))
                if selected_id != _expected_greedy:
                    _amsg = (
                        f"[GREEDY ASSERTION FAILED] step={step_idx+1}: "
                        f"selected_id={selected_id} ({self._tokenizer.decode([selected_id], skip_special_tokens=False)!r}) "  # type: ignore[union-attr]
                        f"!= constrained_logits.argmax()={_expected_greedy} "
                        f"({self._tokenizer.decode([_expected_greedy], skip_special_tokens=False)!r}) — "  # type: ignore[union-attr]
                        f"greedy selection must always equal constrained_logits.argmax()!"
                    )
                    log.error(_amsg)
                    print(_amsg, flush=True)

            # ── Boolean mask ────────────────────────────────────────────────
            # grammar-invalid = finite in raw logits but -inf after masking.
            # Captures grammar masking AND the special-token suppression layer.
            masked_flag: torch.Tensor = (
                (masked_logits == float("-inf")) & (logits > float("-inf"))
            )
            num_masked_total: int = int(masked_flag.sum())
            vocab_sz: int = int(logits.size(0))
            valid_cnt: int = vocab_sz - num_masked_total

            # ── Pipeline diagnostics ────────────────────────────────────────
            _logits_diverge = bool((masked_logits != logits).any())
            _syncode_active = syncode_grammar_changed and not fallback_used
            _grammar_masked = _diag.get("grammar_masked_count", num_masked_total)
            _ws_masked = _diag.get("whitespace_tokens_masked", False)

            # ── Rank of selected token in each distribution ─────────────────
            _sel_rank_raw = (
                _topk_raw_id_list.index(selected_id)
                if selected_id in _topk_raw_id_list else -1
            )
            _sel_rank_con = (
                _topk_con_id_list.index(selected_id)
                if selected_id in _topk_con_id_list else -1
            )

            # ── Assertion: when fallback_used, source tag must say "fallback"
            if fallback_used and not _sel_source.startswith("fallback"):
                log.error(
                    "ASSERTION FAILED step %d: fallback_used=True but "
                    "selection_source=%r — source tagging inconsistency!",
                    step_idx + 1, _sel_source,
                )

            # ── Verify AFTER-SYNCODE visualization uses constrained logits ──
            # valid_tokens_after_syncode is built from topk_masked_probs which
            # comes from probs_masked = softmax(masked_logits / T).
            # Log top-1 of each distribution so the console makes it explicit.
            _raw_top1_prob = float(topk_raw_probs[0]) if k > 0 else 0.0
            _con_top1_prob = float(topk_masked_probs[0]) if k > 0 else 0.0
            _raw_top1_id = int(topk_raw_ids[0]) if k > 0 else -1
            _con_top1_id = int(topk_masked_ids[0]) if k > 0 else -1

            # Verify: if logits diverge, the constrained top-1 prob at the
            # constrained argmax must differ from raw prob at the same ID.
            _con_prob_at_argmax = float(probs_masked[constrained_argmax_id])
            _raw_prob_at_con_argmax = float(probs_raw[constrained_argmax_id])
            _after_syncode_uses_constrained = (
                not _logits_diverge  # identical logits → both probs the same
                or abs(_con_prob_at_argmax - _raw_prob_at_con_argmax) > 1e-9
            )
            if _logits_diverge and not _after_syncode_uses_constrained:
                _vmsg = (
                    f"[VISUALIZATION BUG] step={step_idx+1}: logits diverged but "
                    f"probs_masked[constrained_argmax]={_con_prob_at_argmax:.6f} "
                    f"== probs_raw[constrained_argmax]={_raw_prob_at_con_argmax:.6f} "
                    f"— valid_tokens_after_syncode may be using wrong logits!"
                )
                log.error(_vmsg)
                print(_vmsg, flush=True)

            # Probability mass removed = Σ raw_prob of all masked tokens.
            prob_mass_removed: float = float(probs_raw[masked_flag].sum())

            # top_tokens_before_syncode — raw top-k (BEFORE masking) with
            # is_masked annotation to show which tokens were masked.
            # Source: probs_raw  (topk_raw_probs, topk_raw_ids)
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

            # masked_tokens — top-k tokens rejected by grammar, with raw probs.
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

            # valid_tokens_after_syncode — top-k AFTER masking.
            # Source: probs_masked = softmax(masked_logits / T)
            # IMPORTANT: probabilities here are from the CONSTRAINED distribution,
            # NOT from probs_raw.  This is what the "AFTER SYNCODE" panel must use.
            valid_tokens_after_syncode: list[TokenCandidate] = [
                TokenCandidate(
                    token_id=int(topk_masked_ids[i]),
                    token_str=self._tokenizer.decode(  # type: ignore[union-attr]
                        [int(topk_masked_ids[i])],
                        skip_special_tokens=False,
                        clean_up_tokenization_spaces=False,
                    ),
                    probability=float(topk_masked_probs[i]),  # from probs_masked ✓
                    is_masked=False,   # survived masking by definition
                    is_selected=int(topk_masked_ids[i]) == selected_id,
                )
                for i in range(k)
                if float(topk_masked_probs[i]) > 1e-12
            ]

        else:
            # Raw mode — sample (or greedy) from the unmasked distribution.
            constrained_argmax_id = raw_argmax_id  # no constraint; same as raw
            _topk_con_id_list = _topk_raw_id_list  # same distribution
            if do_sample:
                pen_raw = _apply_repetition_penalty(logits, _past, repetition_penalty)
                sel_probs_raw = F.softmax(pen_raw / temperature, dim=-1)
                selected_id = _nucleus_sample(sel_probs_raw, top_p)
                _sel_source = "raw_sampled"
            else:
                selected_id = raw_argmax_id
                _sel_source = "raw_greedy"
            entropy_after = None
            top_tokens_before_syncode = []
            valid_tokens_after_syncode = []
            masked_in_topk = []
            num_masked_total = 0
            vocab_sz = int(logits.size(0))
            valid_cnt = vocab_sz
            prob_mass_removed = 0.0
            _logits_diverge = False
            _syncode_active = False
            _grammar_masked = 0
            _ws_masked = False
            _sel_rank_raw = (
                _topk_raw_id_list.index(selected_id)
                if selected_id in _topk_raw_id_list else -1
            )
            _sel_rank_con = _sel_rank_raw  # raw mode: same distribution
            _raw_top1_id = int(topk_raw_ids[0]) if k > 0 else -1
            _raw_top1_prob = float(topk_raw_probs[0]) if k > 0 else 0.0
            _con_top1_id = _raw_top1_id
            _con_top1_prob = _raw_top1_prob

        # ── Decode strings ──────────────────────────────────────────────────
        def _tok(tid: int) -> str:
            return self._tokenizer.decode(  # type: ignore[union-attr]
                [tid],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )

        selected_str: str = _tok(selected_id)
        raw_argmax_str: str = _tok(raw_argmax_id)
        constrained_argmax_str: str = _tok(constrained_argmax_id)

        context: str = self._tokenizer.decode(  # type: ignore[union-attr]
            context_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        # ── Side-by-side top-k verification print ───────────────────────────
        # Always printed to stdout so the console shows which logits are used
        # for each distribution without requiring DEBUG log level.
        _raw_top3_str = "  ".join(
            f"{_tok(int(topk_raw_ids[i]))!r}@{float(topk_raw_probs[i]):.4f}"
            for i in range(min(3, k))
        )
        if use_syncode:
            _con_top3_str = "  ".join(
                f"{_tok(int(topk_masked_ids[i]))!r}@{float(topk_masked_probs[i]):.4f}"  # type: ignore[possibly-undefined]
                for i in range(min(3, k))
            )
            _after_src = "probs_masked=softmax(masked_logits/T)"
        else:
            _con_top3_str = "(raw mode — no constraint)"
            _after_src = "probs_raw=softmax(logits/T)"

        print(
            f"[VERIFY step {step_idx+1:>3}]"
            f"  RAW  top3: {_raw_top3_str}"
            f"  |  CONSTRAINED top3 [{_after_src}]: {_con_top3_str}"
            f"  |  selected={selected_str!r}(id={selected_id}) src={_sel_source}"
            f"  rank_raw={_sel_rank_raw} rank_con={_sel_rank_con}"
            f"  |  grammar_masked={_grammar_masked}"
            f"  logits_diverge={_logits_diverge}"
            f"  raw_top1_prob={_raw_top1_prob:.4f}"
            f"  con_top1_prob={_con_top1_prob:.4f}"
            f"  con_top1{'==raw_top1' if _raw_top1_id == _con_top1_id else '!=raw_top1'}",
            flush=True,
        )

        # DEBUG-level structured log for log aggregators.
        log.debug(
            "Step %d | sel=%r (id=%d src=%s rank_raw=%d rank_con=%d) | "
            "raw_argmax=%r (id=%d) | constrained_argmax=%r (id=%d) | "
            "grammar_masked=%d logits_diverge=%s syncode_active=%s "
            "ws_masked=%s fallback=%s",
            step_idx + 1,
            selected_str, selected_id, _sel_source, _sel_rank_raw, _sel_rank_con,
            raw_argmax_str, raw_argmax_id,
            constrained_argmax_str, constrained_argmax_id,
            _grammar_masked, _logits_diverge,
            _syncode_active, _ws_masked, fallback_used,
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
            # Pipeline integrity diagnostics
            syncode_active=_syncode_active,
            logits_diverge=_logits_diverge,
            raw_argmax_token_id=raw_argmax_id,
            raw_argmax_token=raw_argmax_str,
            constrained_argmax_token_id=constrained_argmax_id,
            constrained_argmax_token=constrained_argmax_str,
            selection_source=_sel_source,
            grammar_masked_count=_grammar_masked,
            whitespace_tokens_masked=_ws_masked,
            # Rank of selected token in each distribution
            selected_rank_raw=_sel_rank_raw,
            selected_rank_constrained=_sel_rank_con,
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

        # Per-step trace records — written to _last_trace at the end.
        import uuid as _uuid  # noqa: PLC0415
        _generation_id = str(_uuid.uuid4())[:8]
        _trace_steps: list[dict] = []

        print(
            f"[TRACE start] gen_id={_generation_id} "
            f"mode={'syncode' if effective_syncode else 'raw'} "
            f"max_new_tokens={max_new_tokens} top_k={top_k} T={temperature} "
            f"do_sample={do_sample} top_p={top_p} rep_pen={repetition_penalty}",
            flush=True,
        )

        try:
            for step_idx in range(max_new_tokens):
                # ── Optional Syncode mask ───────────────────────────────────
                masked_logits: torch.Tensor | None = None
                step_parser_error: str | None = None
                step_fallback_used: bool = False

                # Diagnostics from _SyncodeConstraint.mask() — populated below.
                step_syncode_grammar_changed: bool = False
                step_syncode_diag: dict = {}

                if effective_syncode and self._syncode is not None:
                    masked_logits, step_parser_error, step_syncode_diag = (
                        self._syncode.mask(all_input_ids, last_logits, step_idx)
                    )

                    # True when Syncode's grammar mask (before special-token
                    # suppression) actually changed at least one logit value.
                    step_syncode_grammar_changed = step_syncode_diag.get(
                        "logits_changed", False
                    )

                    # Syncode processor returned None (parser exception or
                    # unavailable): clone raw logits so special-token suppression
                    # can still be applied without mutating last_logits.
                    if masked_logits is None:
                        step_fallback_used = True
                        masked_logits = last_logits.clone()
                        step_syncode_grammar_changed = False

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
                        step_syncode_grammar_changed = False

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
                    syncode_grammar_changed=step_syncode_grammar_changed,
                    syncode_diag=step_syncode_diag,
                )
                steps.append(step)
                generated_ids.append(selected_id)

                # ── Build per-step trace record ─────────────────────────────
                trace_step: dict = {  # noqa: RUF012
                    "step": step_idx + 1,
                    "selected_token": step.selected_token,
                    "selected_token_id": selected_id,
                    "selection_source": step.selection_source,
                    "raw_argmax_token": step.raw_argmax_token,
                    "raw_argmax_token_id": step.raw_argmax_token_id,
                    "constrained_argmax_token": step.constrained_argmax_token,
                    "constrained_argmax_token_id": step.constrained_argmax_token_id,
                    "syncode_active": step.syncode_active,
                    "logits_diverge": step.logits_diverge,
                    "grammar_masked_count": step.grammar_masked_count,
                    "num_masked_total": step.num_masked,
                    "masked_percentage": step.masked_percentage,
                    "whitespace_tokens_masked": step.whitespace_tokens_masked,
                    "whitespace_tokens_accepted": step_syncode_diag.get(
                        "whitespace_tokens_accepted", 0
                    ),
                    "entropy_before": step.entropy_before,
                    "entropy_after": step.entropy_after,
                    "fallback_used": step.fallback_used,
                    "parser_error": step.parser_error,
                    "parser_error_message": step.parser_error_message,
                    "consecutive_whitespace_count": step.consecutive_whitespace_count,
                    # Rank of selected token in each distribution — key invariants:
                    #   greedy+constrained → rank_constrained must be 0
                    #   greedy+raw         → rank_raw must be 0
                    "selected_rank_raw": step.selected_rank_raw,
                    "selected_rank_constrained": step.selected_rank_constrained,
                    # Top-3 from each distribution (source labelled)
                    "raw_top3": [
                        {"token": t.token, "prob": round(t.probability, 6), "id": t.token_id}
                        for t in step.top_tokens[:3]
                    ],
                    # valid_tokens_after_syncode uses probs_masked (constrained);
                    # top_tokens_before_syncode uses probs_raw (raw) with masking flags
                    "constrained_top3_after_syncode": [
                        {
                            "token": t.token_str,
                            "prob": round(t.probability, 6),
                            "id": t.token_id,
                            "source": "probs_masked",
                        }
                        for t in step.valid_tokens_after_syncode[:3]
                    ],
                    "raw_top3_before_syncode": [
                        {
                            "token": t.token_str,
                            "prob": round(t.probability, 6),
                            "id": t.token_id,
                            "is_masked": t.is_masked,
                            "source": "probs_raw",
                        }
                        for t in step.top_tokens_before_syncode[:3]
                    ],
                }
                _trace_steps.append(trace_step)
                # INFO-level print for immediate console visibility even when
                # the server log level is set above DEBUG.
                print(
                    f"[TRACE step {step_idx+1:>3}] "
                    f"sel={step.selected_token!r:>12} (id={selected_id}) "
                    f"src={step.selection_source:<22} "
                    f"raw_top1={step.raw_argmax_token!r} "
                    f"con_top1={step.constrained_argmax_token!r} "
                    f"grammar_masked={step.grammar_masked_count:>5} "
                    f"logits_diverge={step.logits_diverge} "
                    f"ws_masked={step.whitespace_tokens_masked} "
                    f"fallback={step.fallback_used}",
                    flush=True,
                )

                if selected_id in eos_ids:
                    log.debug("EOS reached at step %d", step_idx + 1)
                    print(f"[TRACE] EOS at step {step_idx + 1}", flush=True)
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

        # ── Write debug trace ────────────────────────────────────────────────
        n_steps = len(_trace_steps)
        n_syncode_active = sum(1 for s in _trace_steps if s["syncode_active"])
        n_fallback = sum(1 for s in _trace_steps if s["fallback_used"])
        n_logits_diverge = sum(1 for s in _trace_steps if s["logits_diverge"])
        n_grammar_masked_any = sum(
            1 for s in _trace_steps if s["grammar_masked_count"] > 0
        )
        n_ws_masked = sum(1 for s in _trace_steps if s["whitespace_tokens_masked"])
        n_ws_stall = sum(
            1 for s in _trace_steps if s["consecutive_whitespace_count"] > 0
        )

        summary = {
            "generation_id": _generation_id,
            "total_steps": n_steps,
            "syncode_active_steps": n_syncode_active,
            "fallback_steps": n_fallback,
            "logits_diverge_steps": n_logits_diverge,
            "grammar_masked_any_steps": n_grammar_masked_any,
            "whitespace_tokens_masked_steps": n_ws_masked,
            "whitespace_stall_steps": n_ws_stall,
            "whitespace_stall_step_num": whitespace_stall_step_num,
            "generated_text_preview": generated_text[:120],
        }

        print(
            f"[TRACE end] gen_id={_generation_id} steps={n_steps} "
            f"syncode_active={n_syncode_active}/{n_steps} "
            f"fallback={n_fallback}/{n_steps} "
            f"logits_diverge={n_logits_diverge}/{n_steps} "
            f"grammar_masked_any={n_grammar_masked_any}/{n_steps} "
            f"ws_masked={n_ws_masked}/{n_steps} "
            f"ws_stall={n_ws_stall}/{n_steps}",
            flush=True,
        )

        with _trace_lock:
            _last_trace.update({
                "generation_id": _generation_id,
                "prompt": prompt,
                "mode": "syncode" if effective_syncode else "raw",
                "effective_syncode": effective_syncode,
                "steps": _trace_steps,
                "summary": summary,
            })

        log.debug("Generation complete: %d tokens", len(steps))
        return generated_text, steps


# Module-level singleton — import this in route handlers.
# Never instantiate LLMService() elsewhere.
llm_service = LLMService()
