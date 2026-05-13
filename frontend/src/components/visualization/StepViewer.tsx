"use client";

/**
 * StepViewer — full detail panel for a single decoding step.
 *
 * RAW MODE (no Syncode data):
 *   • entropy bar
 *   • raw top-k probability chart
 *   • probability table
 *
 * SYNCODE MODE (top_tokens_before_syncode populated):
 *   • Syncode impact metrics card
 *   • Entropy before → after (dual bars + delta badge)
 *   • Side-by-side: BEFORE SYNCODE | AFTER SYNCODE
 *   • MASKED TOKENS section (scrollable, red, shows raw_prob)
 */

import { useMemo } from "react";
import { Badge } from "@/components/ui/Badge";
import { TokenProbabilityChart } from "@/components/visualization/TokenProbabilityChart";
import { formatPct } from "@/lib/utils";
import type { DecodingStep, MaskedTokenEntry, TopToken } from "@/types/decoding";

interface Props {
  step: DecodingStep;
}

// ── helpers ──────────────────────────────────────────────────────────────────

function cn(...cls: (string | boolean | undefined | null)[]): string {
  return cls.filter(Boolean).join(" ");
}

function entropyColor(h: number): string {
  if (h < 2) return "#3fb950";
  if (h < 4) return "#d29922";
  return "#f85149";
}

function maskSeverityColor(pct: number): string {
  if (pct < 50) return "#3fb950";
  if (pct < 85) return "#d29922";
  return "#f85149";
}

// ── sub-components ────────────────────────────────────────────────────────────

function EntropyBar({ value, label }: { value: number; label?: string }) {
  const MAX_H = 11;
  const pct = Math.min((value / MAX_H) * 100, 100);
  const col = entropyColor(value);
  return (
    <div className="flex flex-col gap-1">
      {label && (
        <span className="text-[10px] uppercase tracking-wider text-[#484f58]">{label}</span>
      )}
      <div className="flex items-center gap-2">
        <div className="relative h-2 flex-1 overflow-hidden rounded-full bg-[#21262d]">
          <div
            className="absolute left-0 top-0 h-full rounded-full transition-all duration-300"
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

/** Scrollable probability table for both raw and constrained distributions. */
function ProbTable({
  rows,
  selectedId,
  maskedIdSet,
  maxRows = 20,
}: {
  rows: { token_id: number; token: string; probability: number }[];
  selectedId: number;
  maskedIdSet?: Set<number>;
  maxRows?: number;
}) {
  const display = rows.slice(0, maxRows);
  return (
    <div className="max-h-64 overflow-y-auto rounded border border-[#21262d]">
      <table className="w-full border-collapse font-mono text-[11px]">
        <thead className="sticky top-0 bg-[#0d1117]">
          <tr className="border-b border-[#21262d] text-left text-[#484f58]">
            <th className="px-2 py-1 font-medium">#</th>
            <th className="px-2 py-1 font-medium">token</th>
            <th className="px-2 py-1 font-medium">id</th>
            <th className="px-2 py-1 text-right font-medium">prob</th>
          </tr>
        </thead>
        <tbody>
          {display.map((t, rank) => {
            const isSel = t.token_id === selectedId;
            const isMasked = maskedIdSet?.has(t.token_id) ?? false;
            return (
              <tr
                key={`${t.token_id}-${rank}`}
                className={cn(
                  "border-b border-[#21262d]/50 transition-colors",
                  isSel && "bg-blue-900/20",
                  isMasked && !isSel && "opacity-50",
                )}
              >
                <td className="px-2 py-0.5 text-[#484f58]">{rank + 1}</td>
                <td className={cn("px-2 py-0.5", isSel && "text-[#58a6ff]", isMasked && !isSel && "text-[#f85149] line-through")}>
                  {JSON.stringify(t.token)}
                  {isSel && <span className="ml-1 text-[9px]">✓</span>}
                  {isMasked && !isSel && <span className="ml-1 text-[9px]">✗</span>}
                </td>
                <td className="px-2 py-0.5 text-[#484f58]">{t.token_id}</td>
                <td className="px-2 py-0.5 text-right text-[#8b949e]">{formatPct(t.probability, 3)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

/** Scrollable list of masked / rejected tokens. */
function MaskedTokensPanel({ tokens }: { tokens: MaskedTokenEntry[] }) {
  if (tokens.length === 0) return null;
  return (
    <div className="flex flex-col gap-2">
      <div className="flex items-center gap-2">
        <p className="text-[11px] font-semibold uppercase tracking-wider text-[#f85149]">
          ✗ Masked Tokens
        </p>
        <span className="font-mono text-[10px] text-[#484f58]">
          top-{tokens.length} rejected by C grammar
        </span>
      </div>
      <div className="max-h-48 overflow-y-auto rounded border border-[#f85149]/20 bg-red-900/5">
        <table className="w-full border-collapse font-mono text-[11px]">
          <thead className="sticky top-0 bg-[#160a0a]">
            <tr className="border-b border-[#f85149]/20 text-left text-[#f85149]/60">
              <th className="px-2 py-1 font-medium">#</th>
              <th className="px-2 py-1 font-medium">token</th>
              <th className="px-2 py-1 font-medium">id</th>
              <th className="px-2 py-1 text-right font-medium">raw prob</th>
              <th className="px-2 py-1 font-medium">reason</th>
            </tr>
          </thead>
          <tbody>
            {tokens.map((t, i) => (
              <tr
                key={`${t.token_id}-${i}`}
                className="border-b border-[#f85149]/10 text-[#f85149]/80"
              >
                <td className="px-2 py-0.5 text-[#f85149]/40">{i + 1}</td>
                <td className="px-2 py-0.5 line-through">{JSON.stringify(t.token)}</td>
                <td className="px-2 py-0.5 text-[#f85149]/40">{t.token_id}</td>
                <td className="px-2 py-0.5 text-right">{formatPct(t.raw_prob, 3)}</td>
                <td className="px-2 py-0.5 text-[10px] text-[#f85149]/50">invalid C</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

/** Syncode impact metrics card — vocab size, valid/masked counts, prob mass, entropy delta. */
function SyncodeMetricsCard({ step }: { step: DecodingStep }) {
  const maskCol = maskSeverityColor(step.masked_percentage);
  const entropyDelta =
    step.entropy_before !== null && step.entropy_after !== null
      ? step.entropy_after - step.entropy_before
      : null;

  const metrics: { label: string; value: string; color?: string; title?: string }[] = [
    {
      label: "Vocab",
      value: step.vocab_size.toLocaleString(),
      title: "Total vocabulary size",
    },
    {
      label: "Valid",
      value: step.valid_token_count.toLocaleString(),
      color: "#3fb950",
      title: "Tokens surviving grammar masking",
    },
    {
      label: "Masked",
      value: step.masked_token_count.toLocaleString(),
      color: "#f85149",
      title: "Tokens rejected by C grammar",
    },
    {
      label: "Masked %",
      value: `${step.masked_percentage.toFixed(1)}%`,
      color: maskCol,
      title: "Percentage of vocabulary rejected",
    },
    {
      label: "Mass removed",
      value: formatPct(step.probability_mass_removed, 2),
      color: "#d29922",
      title: "Raw probability mass of rejected tokens",
    },
    ...(entropyDelta !== null
      ? [
          {
            label: "ΔH",
            value: entropyDelta.toFixed(3),
            color: entropyDelta < 0 ? "#3fb950" : "#f85149",
            title: "Entropy change: negative = more focused distribution",
          },
        ]
      : []),
  ];

  return (
    <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3">
      <p className="mb-2 text-[10px] uppercase tracking-wider text-[#484f58]">
        Syncode Impact — Step {step.step}
      </p>
      <div className="grid grid-cols-3 gap-2 sm:grid-cols-6">
        {metrics.map((m) => (
          <div key={m.label} className="flex flex-col gap-0.5" title={m.title}>
            <span className="text-[9px] uppercase tracking-wider text-[#484f58]">{m.label}</span>
            <span
              className="font-mono text-[13px] font-semibold"
              style={{ color: m.color ?? "#e6edf3" }}
            >
              {m.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ── main component ────────────────────────────────────────────────────────────

export function StepViewer({ step }: Props) {
  const isSyncodeMode = useMemo(
    () =>
      step.top_tokens_before_syncode.length > 0 ||
      step.valid_tokens_after_syncode.length > 0 ||
      step.num_masked > 0,
    [step],
  );

  // Normalise valid_tokens_after_syncode → TopToken[] for the chart
  const validAsTopTokens: TopToken[] = useMemo(
    () =>
      step.valid_tokens_after_syncode.map((tc) => ({
        token: tc.token_str,
        probability: tc.probability,
        token_id: tc.token_id,
      })),
    [step.valid_tokens_after_syncode],
  );

  // Before-Syncode tokens for the chart
  const beforeTopTokens: TopToken[] = useMemo(
    () =>
      step.top_tokens_before_syncode.map((tc) => ({
        token: tc.token_str,
        probability: tc.probability,
        token_id: tc.token_id,
      })),
    [step.top_tokens_before_syncode],
  );

  // Set of masked token IDs for O(1) lookup in tables/charts
  const maskedIdSet = useMemo(
    () => new Set<number>(step.masked_tokens.map((m) => m.token_id)),
    [step.masked_tokens],
  );

  const selectedId = step.selected_token_id;

  return (
    <div className="flex flex-col gap-4">

      {/* ── Header ────────────────────────────────────────────────────────── */}
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[10px] uppercase tracking-wider text-[#484f58]">Step {step.step}</span>
        <Badge variant="selected">{JSON.stringify(step.selected_token)}</Badge>
        <span className="font-mono text-[10px] text-[#484f58]">id&nbsp;{step.selected_token_id}</span>
        {isSyncodeMode && <Badge variant="info">Syncode</Badge>}
        {step.num_masked > 0 && (
          <Badge variant="masked">{step.num_masked.toLocaleString()} masked</Badge>
        )}
        <span className="ml-auto font-mono text-[10px] text-[#484f58]">
          top-1&nbsp;p&nbsp;=&nbsp;
          <span className="text-[#e6edf3]">{formatPct(step.top_tokens[0]?.probability ?? 0, 2)}</span>
        </span>
      </div>

      {/* ── SYNCODE MODE ──────────────────────────────────────────────────── */}
      {isSyncodeMode ? (
        <>
          {/* Metrics card */}
          <SyncodeMetricsCard step={step} />

          {/* Entropy comparison */}
          <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3 flex flex-col gap-3">
            <div className="flex items-center gap-2">
              <span className="text-[10px] uppercase tracking-wider text-[#484f58]">
                Entropy — H = −Σ p·log(p)
              </span>
              {step.entropy_before !== null && step.entropy_after !== null && (
                <span
                  className="ml-auto font-mono text-[11px] font-semibold"
                  title="Entropy change from grammar masking (negative = more focused)"
                  style={{
                    color: step.entropy_after < step.entropy_before ? "#3fb950" : "#f85149",
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

          {/* Side-by-side: BEFORE | AFTER */}
          <div className="grid gap-4 lg:grid-cols-2">
            {/* BEFORE SYNCODE */}
            <div className="flex flex-col gap-3 rounded-md border border-[#21262d] bg-[#0d1117] p-3">
              <div className="flex items-center justify-between">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-[#8b949e]">
                  Before Syncode
                </p>
                <span className="font-mono text-[10px] text-[#484f58]">raw distribution</span>
              </div>
              <TokenProbabilityChart
                candidates={beforeTopTokens.length > 0 ? beforeTopTokens : step.top_tokens}
                selectedTokenId={selectedId}
                maskedIds={step.masked_tokens.map((m) => m.token_id)}
              />
              <ProbTable
                rows={(beforeTopTokens.length > 0 ? beforeTopTokens : step.top_tokens).map((t) => ({
                  token_id: t.token_id,
                  token: t.token,
                  probability: t.probability,
                }))}
                selectedId={selectedId}
                maskedIdSet={maskedIdSet}
              />
            </div>

            {/* AFTER SYNCODE */}
            <div className="flex flex-col gap-3 rounded-md border border-[#58a6ff]/20 bg-[#0d1117] p-3">
              <div className="flex items-center justify-between">
                <p className="text-[11px] font-semibold uppercase tracking-wider text-[#58a6ff]">
                  After Syncode
                </p>
                <span className="font-mono text-[10px] text-[#484f58]">constrained distribution</span>
              </div>
              <TokenProbabilityChart
                candidates={validAsTopTokens.length > 0 ? validAsTopTokens : step.top_tokens}
                selectedTokenId={selectedId}
              />
              <ProbTable
                rows={(validAsTopTokens.length > 0 ? validAsTopTokens : step.top_tokens).map((t) => ({
                  token_id: t.token_id,
                  token: t.token,
                  probability: t.probability,
                }))}
                selectedId={selectedId}
              />
            </div>
          </div>

          {/* MASKED TOKENS section */}
          <MaskedTokensPanel tokens={step.masked_tokens} />

          {/* Summary callout */}
          {step.num_masked > 0 && (
            <div className="flex items-start gap-2 rounded-md border border-[#f85149]/20 bg-red-900/10 px-3 py-2 text-xs text-[#f85149]">
              <span className="mt-0.5 text-sm">✗</span>
              <span>
                <strong>{step.num_masked.toLocaleString()}</strong> of{" "}
                <strong>{step.vocab_size.toLocaleString()}</strong> tokens (
                <strong>{step.masked_percentage.toFixed(1)}%</strong>) suppressed by C-grammar
                constraint, removing{" "}
                <strong>{formatPct(step.probability_mass_removed, 1)}</strong> of raw probability
                mass. The constrained distribution is renormalised over the remaining{" "}
                <strong>{step.valid_token_count.toLocaleString()}</strong> valid tokens.
              </span>
            </div>
          )}
        </>
      ) : (
        /* ── RAW MODE ────────────────────────────────────────────────────── */
        <>
          {step.entropy_before !== null && (
            <div className="rounded-md border border-[#21262d] bg-[#0d1117] p-3">
              <EntropyBar value={step.entropy_before} label="Entropy H = −Σ p·log(p)" />
            </div>
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
