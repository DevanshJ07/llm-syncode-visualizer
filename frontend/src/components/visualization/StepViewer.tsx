"use client";

/**
 * StepViewer — full detail panel for a single decoding step.
 *
 * Shows:
 *   - Selected token (blue), top-k bar chart
 *   - When Syncode data present:
 *       valid tokens (green) / masked tokens (red count badge)
 *   - Exact probability table (rank, token string, id, probability)
 *   - Entropy value + visual indicator
 *   - Masked token count (0 until Syncode is enabled)
 */

import { Badge } from "@/components/ui/Badge";
import { TokenProbabilityChart } from "@/components/visualization/TokenProbabilityChart";
import { formatPct } from "@/lib/utils";
import type { DecodingStep } from "@/types/decoding";

interface Props {
  step: DecodingStep;
}

function EntropyBar({ value }: { value: number }) {
  // TinyLlama vocab ≈ 32k → theoretical max H ≈ ln(32000) ≈ 10.4
  const MAX_H = 10;
  const pct = Math.min((value / MAX_H) * 100, 100);
  const color =
    value < 2 ? "#3fb950" : value < 4 ? "#d29922" : "#f85149";

  return (
    <div className="flex items-center gap-2">
      <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-surface-border">
        <div
          className="absolute left-0 top-0 h-full rounded-full transition-all"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="w-12 text-right font-mono text-[11px]" style={{ color }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

export function StepViewer({ step }: Props) {
  const hasSyncode =
    step.masked_tokens.length > 0 ||
    step.valid_tokens_after_syncode.length > 0;

  return (
    <div className="flex flex-col gap-4">
      {/* ── Header ─────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
          Step {step.step}
        </span>
        <Badge variant="selected">{JSON.stringify(step.selected_token)}</Badge>
        <span className="font-mono text-[10px] text-[#484f58]">
          id&nbsp;{step.selected_token_id}
        </span>

        {step.num_masked > 0 && (
          <Badge variant="masked">{step.num_masked} masked</Badge>
        )}

        <span className="ml-auto font-mono text-[10px] text-[#484f58]">
          top-1&nbsp;p&nbsp;=&nbsp;
          <span className="text-[#e6edf3]">
            {formatPct(step.top_tokens[0]?.probability ?? 0, 2)}
          </span>
        </span>
      </div>

      {/* ── Entropy bar ─────────────────────────────────────────── */}
      {step.entropy_before !== null && (
        <div className="flex flex-col gap-1">
          <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
            Entropy H = −Σ p·log(p)
          </span>
          <EntropyBar value={step.entropy_before} />
        </div>
      )}

      {/* ── Probability bar chart ────────────────────────────────── */}
      <TokenProbabilityChart
        candidates={step.top_tokens}
        selectedTokenId={step.selected_token_id}
        title={hasSyncode ? "After Syncode masking" : `Top ${step.top_tokens.length} candidates`}
      />

      {/* ── Syncode section (visible when data is present) ──────── */}
      {hasSyncode && (
        <div className="flex flex-col gap-2 rounded-md border border-surface-border bg-surface p-3">
          <div className="flex items-center justify-between">
            <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
              Syncode masking
            </span>
            <Badge variant="masked">{step.num_masked} tokens masked</Badge>
          </div>
          {step.entropy_after !== null && (
            <div className="flex flex-col gap-1">
              <span className="text-[10px] text-[#484f58]">Entropy after masking</span>
              <EntropyBar value={step.entropy_after} />
            </div>
          )}
        </div>
      )}

      {/* ── Probability table ─────────────────────────────────────── */}
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
            {step.top_tokens.map((t, rank) => {
              const isSelected = t.token_id === step.selected_token_id;
              // In Syncode mode, mark tokens that were masked (not in valid list)
              const isMasked =
                hasSyncode &&
                step.masked_tokens.includes(t.token_id);

              return (
                <tr
                  key={t.token_id}
                  className={cn(
                    "border-b border-surface-border/40",
                    isSelected
                      ? "text-accent-blue"
                      : isMasked
                      ? "text-token-masked line-through opacity-60"
                      : "text-[#8b949e]"
                  )}
                >
                  <td className="py-0.5 pr-3 text-[#484f58]">{rank + 1}</td>
                  <td className="py-0.5 pr-3">
                    {JSON.stringify(t.token)}
                    {isSelected && (
                      <span className="ml-1 text-[9px] text-accent-blue">✓</span>
                    )}
                    {isMasked && (
                      <span className="ml-1 text-[9px] text-token-masked">✗</span>
                    )}
                  </td>
                  <td className="py-0.5 pr-3 text-[#484f58]">{t.token_id}</td>
                  <td className="py-0.5 text-right">{formatPct(t.probability, 3)}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// Local helper to avoid importing from @/lib/utils in a barrel
function cn(...classes: (string | boolean | undefined | null)[]): string {
  return classes.filter(Boolean).join(" ");
}
