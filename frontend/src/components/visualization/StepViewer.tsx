"use client";

/**
 * StepViewer — full detail panel for a single decoding step.
 *
 * RAW MODE  (step.num_masked === 0 and no Syncode data):
 *   • entropy bar
 *   • raw top-k probability chart  (blue = selected, green = candidate)
 *   • probability table
 *
 * SYNCODE MODE  (step.top_tokens_before_syncode populated):
 *   ┌─────────────────────────────────────────────────────────────┐
 *   │  entropy before → entropy after  (delta badge)              │
 *   ├──────────────────────┬──────────────────────────────────────┤
 *   │  Before Syncode      │  After Syncode                       │
 *   │  raw top-k           │  constrained top-k                   │
 *   │  masked bars = red   │  only valid tokens shown             │
 *   │  ✗ num_masked        │  ✓ selected token highlighted        │
 *   └──────────────────────┴──────────────────────────────────────┘
 *   • full probability tables for both distributions
 */

import { Badge } from "@/components/ui/Badge";
import { TokenProbabilityChart } from "@/components/visualization/TokenProbabilityChart";
import { formatPct } from "@/lib/utils";
import type { DecodingStep, TopToken } from "@/types/decoding";

interface Props {
  step: DecodingStep;
}

// ── helpers ─────────────────────────────────────────────────────────────────

function cn(...cls: (string | boolean | undefined | null)[]): string {
  return cls.filter(Boolean).join(" ");
}

function entropyColor(h: number): string {
  if (h < 2) return "#3fb950";
  if (h < 4) return "#d29922";
  return "#f85149";
}

function EntropyBar({ value, label }: { value: number; label?: string }) {
  const MAX_H = 10; // ln(32000) ≈ 10.4 for Qwen vocab
  const pct = Math.min((value / MAX_H) * 100, 100);
  const col = entropyColor(value);
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <span className="text-[10px] uppercase tracking-wider text-[#484f58]">{label}</span>
      )}
      <div className="flex items-center gap-2">
        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-border">
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all"
            style={{ width: `${pct}%`, background: col }}
          />
        </div>
        <span className="w-14 text-right font-mono text-[11px]" style={{ color: col }}>
          {value.toFixed(3)}
        </span>
      </div>
    </div>
  );
}

/** Probability table — works for TopToken[] (raw) or a normalised set. */
function ProbTable({
  rows,
  selectedId,
  maskedIds,
}: {
  rows: { token_id: number; token: string; probability: number }[];
  selectedId: number;
  maskedIds?: Set<number>;
}) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full border-collapse font-mono text-[11px]">
        <thead>
          <tr className="border-b border-surface-border text-left text-[#484f58]">
            <th className="pb-1 pr-3 font-medium">#</th>
            <th className="pb-1 pr-3 font-medium">token</th>
            <th className="pb-1 pr-3 font-medium">id</th>
            <th className="pb-1 text-right font-medium">prob</th>
          </tr>
        </thead>
        <tbody>
          {rows.map((t, rank) => {
            const isSel = t.token_id === selectedId;
            const isMasked = maskedIds?.has(t.token_id) ?? false;
            return (
              <tr
                key={t.token_id}
                className={cn(
                  "border-b border-surface-border/40",
                  isSel && "text-accent-blue",
                  isMasked && !isSel && "text-[#f85149] opacity-50",
                  !isSel && !isMasked && "text-[#8b949e]",
                )}
              >
                <td className="py-0.5 pr-3 text-[#484f58]">{rank + 1}</td>
                <td className={cn("py-0.5 pr-3", isMasked && "line-through")}>
                  {JSON.stringify(t.token)}
                  {isSel && <span className="ml-1 text-[9px]">✓</span>}
                  {isMasked && <span className="ml-1 text-[9px]">✗</span>}
                </td>
                <td className="py-0.5 pr-3 text-[#484f58]">{t.token_id}</td>
                <td className="py-0.5 text-right">{formatPct(t.probability, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ── main component ───────────────────────────────────────────────────────────

export function StepViewer({ step }: Props) {
  const isSyncodeMode =
    step.top_tokens_before_syncode.length > 0 ||
    step.valid_tokens_after_syncode.length > 0 ||
    step.num_masked > 0;

  // Normalise valid_tokens_after_syncode to TopToken[] for the chart
  const validAsTopTokens: TopToken[] = step.valid_tokens_after_syncode.map((tc) => ({
    token: tc.token_str,
    probability: tc.probability,
    token_id: tc.token_id,
  }));

  // Convert masked_tokens list → Set for O(1) lookup
  const maskedSet = new Set<number>(step.masked_tokens);

  // Top-k from raw (for the "before" chart in Syncode mode)
  const beforeTopTokens: TopToken[] = step.top_tokens_before_syncode.map((tc) => ({
    token: tc.token_str,
    probability: tc.probability,
    token_id: tc.token_id,
  }));

  // Selected ID from the constrained distribution (same as step.selected_token_id)
  const selectedId = step.selected_token_id;

  return (
    <div className="flex flex-col gap-4">

      {/* ── Header ──────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
          Step {step.step}
        </span>
        <Badge variant="selected">{JSON.stringify(step.selected_token)}</Badge>
        <span className="font-mono text-[10px] text-[#484f58]">
          id&nbsp;{step.selected_token_id}
        </span>

        {isSyncodeMode && (
          <Badge variant="info">Syncode</Badge>
        )}
        {step.num_masked > 0 && (
          <Badge variant="masked">{step.num_masked.toLocaleString()} masked</Badge>
        )}

        <span className="ml-auto font-mono text-[10px] text-[#484f58]">
          top-1&nbsp;p&nbsp;=&nbsp;
          <span className="text-[#e6edf3]">
            {formatPct(step.top_tokens[0]?.probability ?? 0, 2)}
          </span>
        </span>
      </div>

      {/* ── SYNCODE MODE ─────────────────────────────────────────────────── */}
      {isSyncodeMode ? (
        <>
          {/* Entropy before / after with delta */}
          <div className="rounded-md border border-surface-border bg-surface p-3 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
                Entropy — H = −Σ p·log(p)
              </span>
              {step.entropy_before !== null && step.entropy_after !== null && (
                <span
                  className="ml-auto font-mono text-[10px]"
                  title="Entropy reduction from grammar masking"
                  style={{
                    color:
                      step.entropy_after < step.entropy_before ? "#3fb950" : "#f85149",
                  }}
                >
                  Δ {(step.entropy_after - step.entropy_before).toFixed(3)}
                </span>
              )}
            </div>
            {step.entropy_before !== null && (
              <EntropyBar value={step.entropy_before} label="Before Syncode" />
            )}
            {step.entropy_after !== null && (
              <EntropyBar value={step.entropy_after} label="After Syncode" />
            )}
          </div>

          {/* Side-by-side charts */}
          <div className="grid gap-4 lg:grid-cols-2">
            {/* Before Syncode */}
            <div className="flex flex-col gap-2 rounded-md border border-surface-border bg-surface p-3">
              <div className="flex items-center gap-2">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-[#8b949e]">
                  Before Syncode
                </p>
                <span className="ml-auto font-mono text-[10px] text-[#484f58]">
                  raw distribution
                </span>
              </div>
              <TokenProbabilityChart
                candidates={beforeTopTokens.length > 0 ? beforeTopTokens : step.top_tokens}
                selectedTokenId={selectedId}
                maskedIds={step.masked_tokens}
              />
              {/* Table */}
              <ProbTable
                rows={(beforeTopTokens.length > 0 ? beforeTopTokens : step.top_tokens).map(
                  (t) => ({ token_id: t.token_id, token: t.token, probability: t.probability }),
                )}
                selectedId={selectedId}
                maskedIds={maskedSet}
              />
            </div>

            {/* After Syncode */}
            <div className="flex flex-col gap-2 rounded-md border border-accent-blue/20 bg-surface p-3">
              <div className="flex items-center gap-2">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-accent-blue">
                  After Syncode
                </p>
                <span className="ml-auto font-mono text-[10px] text-[#484f58]">
                  constrained distribution
                </span>
              </div>
              <TokenProbabilityChart
                candidates={validAsTopTokens.length > 0 ? validAsTopTokens : step.top_tokens}
                selectedTokenId={selectedId}
              />
              {/* Table */}
              <ProbTable
                rows={(validAsTopTokens.length > 0 ? validAsTopTokens : step.top_tokens).map(
                  (t) => ({ token_id: t.token_id, token: t.token, probability: t.probability }),
                )}
                selectedId={selectedId}
              />
            </div>
          </div>

          {/* Masked count callout */}
          {step.num_masked > 0 && (
            <div className="flex items-center gap-2 rounded-md border border-[#f85149]/20 bg-red-900/10 px-3 py-2 text-xs text-[#f85149]">
              <span className="text-base">✗</span>
              <span>
                <strong>{step.num_masked.toLocaleString()}</strong> tokens suppressed by C-grammar
                constraint — the constrained distribution is renormalised over the remaining{" "}
                <strong>{(
                  (step.top_tokens[0]
                    ? Math.round(step.top_tokens[0].probability * 100) / 100
                    : 0) * 100
                ).toFixed(0)}
                %</strong> of vocabulary mass.
              </span>
            </div>
          )}
        </>
      ) : (
        /* ── RAW MODE ──────────────────────────────────────────────────── */
        <>
          {step.entropy_before !== null && (
            <EntropyBar value={step.entropy_before} label="Entropy H = −Σ p·log(p)" />
          )}

          <TokenProbabilityChart
            candidates={step.top_tokens}
            selectedTokenId={selectedId}
            title={`Top ${step.top_tokens.length} candidates`}
          />

          <ProbTable
            rows={step.top_tokens.map((t) => ({
              token_id: t.token_id,
              token: t.token,
              probability: t.probability,
            }))}
            selectedId={selectedId}
          />
        </>
      )}
    </div>
  );
}
