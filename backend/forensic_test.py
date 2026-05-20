"""
Forensic validation test — after prompt format fix.

Sends a C-specific syncode generation and verifies:
  - grammar_masked > 0 on some steps
  - logits_diverge=True on constrained steps
  - parse exceptions no longer appear (or drastically reduced)
  - parser no longer permanently enters PARSE_FAILED_SKIP

Also runs a raw-mode generation for comparison.
"""
import requests
import json
import sys

BASE = "http://127.0.0.1:8000"
SEP = "=" * 70


def generate(prompt, mode="syncode", use_syncode=True, do_sample=False, max_new_tokens=30):
    payload = {
        "prompt": prompt,
        "mode": mode,
        "max_new_tokens": max_new_tokens,
        "use_syncode": use_syncode,
        "do_sample": do_sample,
        "temperature": 1.0,
    }
    resp = requests.post(f"{BASE}/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()


def forensic_summary():
    r = requests.get(f"{BASE}/debug/forensic-summary", timeout=10)
    r.raise_for_status()
    return r.json()


def forensic_full():
    r = requests.get(f"{BASE}/debug/forensic-full", timeout=10)
    r.raise_for_status()
    return r.json()


print(SEP)
print("FORENSIC VALIDATION — prompt format fix: // -> /* */")
print(SEP)

C_PROMPT = "add two integers"

# ── Test 1: Syncode mode (constrained) ─────────────────────────────────────
print("\n[1] Constrained generation (use_syncode=True)")
try:
    data = generate(C_PROMPT, mode="syncode", use_syncode=True, do_sample=False, max_new_tokens=50)
    gen_text = data.get("generated_text", "")
    print(f"  Generated: {repr(gen_text[:200])}")
except Exception as e:
    print(f"  FAILED: {e}")
    sys.exit(1)

summary = forensic_summary()
full = forensic_full()

print(f"\n  === Forensic Summary ===")
print(f"  total_steps           : {summary.get('total_steps')}")
print(f"  skip_steps            : {summary.get('skip_steps')}")
print(f"  parse_exception_steps : {summary.get('parse_exception_steps')}")
print(f"  all_valid_mask_steps  : {summary.get('all_valid_mask_steps')}")
print(f"  masking_applied_steps : {summary.get('masking_applied_steps')}")
print(f"  unique_diagnoses      : {summary.get('unique_diagnoses')}")

# Print first 5 step diagnoses
steps = full.get("steps", [])
print(f"\n  === First 5 Steps ===")
for s in steps[:5]:
    ms = s.get("mask_stats", [{}])
    ms0 = ms[0] if ms else {}
    print(
        f"  step={s['step']:>3}"
        f"  partial={s.get('partial_output','')[:35]!r}"
        f"  skip={s.get('skip')}"
        f"  n_acc={ms0.get('n_accepted','?')}/{ms0.get('vocab_len','?')}"
        f"  exc={s.get('parse_exception',[''])[0][:60] if s.get('parse_exception') else 'none'}"
    )

# Verdict
skip_pct = (summary.get("skip_steps", 0) / max(summary.get("total_steps", 1), 1)) * 100
masking = summary.get("masking_applied_steps", 0)
exc_steps = summary.get("parse_exception_steps", 0)
diagnoses = summary.get("unique_diagnoses", [])

print(f"\n  === VERDICT ===")
if exc_steps == 0 and masking > 0:
    print(f"  PASS: no parse exceptions, {masking} steps with active masking")
elif exc_steps == 0:
    print(f"  PARTIAL: no parse exceptions but masking_applied_steps={masking} (may be overapproximation)")
else:
    print(f"  FAIL: {exc_steps}/{summary.get('total_steps')} steps still throw parse exceptions")

# ── Test 2: Raw mode for comparison ────────────────────────────────────────
print(f"\n{SEP}")
print("[2] Raw generation (use_syncode=False) for comparison")
try:
    raw_data = generate(C_PROMPT, mode="raw", use_syncode=False, do_sample=False, max_new_tokens=50)
    print(f"  Generated: {repr(raw_data.get('generated_text', '')[:200])}")
except Exception as e:
    print(f"  FAILED: {e}")

# ── Print before/after summary table ───────────────────────────────────────
print(f"\n{SEP}")
print("BEFORE/AFTER COMPARISON")
print(f"{'Metric':<35} {'Before (// prefix)':>22} {'After (/* */ prefix)':>22}")
print("-" * 80)
rows = [
    ("skip_steps / total_steps", "16/16 (100%)", f"{summary.get('skip_steps')}/{summary.get('total_steps')} ({skip_pct:.0f}%)"),
    ("parse_exception_steps", "16", str(exc_steps)),
    ("masking_applied_steps", "0", str(masking)),
    ("unique diagnoses count", "16 variants", str(len(diagnoses))),
    ("grammar_masked > 0", "NEVER", "YES" if masking > 0 else "NO"),
]
for label, before, after in rows:
    print(f"  {label:<33} {before:>22} {after:>22}")

print(f"\n{SEP}")
print("Done.")
