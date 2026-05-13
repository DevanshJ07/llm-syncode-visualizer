"use client";

/**
 * TokenStep — collapsible card for one decoding step.
 *
 * Collapsed view (always visible):
 *   [step#]  [context snippet]  [selected token]  [entropy]  [▸/▾]
 *
 * Expanded view (click to toggle):
 *   Full TokenProbabilityChart with all top-k candidates.
 *   Blue bar = selected token.  Green bars = other candidates.
 *
 * A quick-peek "token probability table" is also shown below the chart
 * so researchers can read exact probabilities without hovering.
 */

import { useState } from "react";

import { Badge } from "@/components/ui/Badge";
import { TokenProbabilityChart } from "@/components/visualization/TokenProbabilityChart";
import { formatPct, truncate } from "@/lib/utils";
import type { DecodingStep } from "@/types/decoding";

interface Props {
  step: DecodingStep;
  isActive?: boolean;
  onClick?: () => void;
}

/** Returns a colour for the left border based on how aggressively Syncode masked at this step. */
function severityBorderColor(pct: number): string {
  if (pct <= 0) return "transparent";
  if (pct < 50) return "#3fb950"; // green — light masking
  if (pct < 85) return "#d29922"; // yellow — moderate masking
  return "#f85149";               // red — heavy masking
}

export function TokenStep({ step, isActive, onClick }: Props) {
  const [expanded, setExpanded] = useState(false);

  const topProb = step.top_tokens[0]?.probability ?? 0;
  const hasMasking = step.masked_percentage > 0;
  const severityColor = hasMasking ? severityBorderColor(step.masked_percentage) : "transparent";

  return (
    <div
      className={`rounded-md border transition-colors ${
        isActive
          ? "border-accent-blue/60 bg-blue-900/10"
          : "border-surface-border bg-surface-raised hover:border-[#30363d]"
      }`}
      style={hasMasking ? { borderLeftColor: severityColor, borderLeftWidth: 3 } : undefined}
    >
      {/* ------------------------------------------------------------------ */}
      {/* Collapsed header row                                                 */}
      {/* ------------------------------------------------------------------ */}
      <button
        type="button"
        onClick={() => {
          onClick?.();
          setExpanded((e) => !e);
        }}
        className="flex w-full items-center gap-3 px-3 py-2.5 text-left"
      >
        {/* Step number */}
        <span className="w-7 shrink-0 text-center font-mono text-xs text-[#484f58]">
          {step.step}
        </span>

        {/* Context snippet */}
        <span className="flex-1 truncate font-mono text-[11px] text-[#8b949e]">
          {step.context ? truncate(step.context, 55) : <em className="text-[#484f58]">start</em>}
        </span>

        {/* Selected token badge */}
        <Badge variant="selected" className="shrink-0 max-w-[100px] truncate">
          {JSON.stringify(step.selected_token)}
        </Badge>

        {/* Masking percentage (Syncode mode only) */}
        {hasMasking && (
          <span
            className="shrink-0 font-mono text-[10px]"
            title={`${step.masked_percentage.toFixed(1)}% of vocabulary masked by Syncode`}
            style={{ color: severityColor }}
          >
            {step.masked_percentage.toFixed(0)}%✗
          </span>
        )}

        {/* Entropy display */}
        {step.entropy_before !== null && (
          <span
            className="shrink-0 font-mono text-[10px]"
            title="Shannon entropy of the full vocabulary distribution"
            style={{
              color: step.entropy_before < 2 ? "#3fb950" : step.entropy_before < 4 ? "#d29922" : "#f85149",
            }}
          >
            H={step.entropy_before.toFixed(2)}
          </span>
        )}

        {/* Top probability */}
        <span className="shrink-0 font-mono text-[10px] text-[#484f58]">
          p={formatPct(topProb, 1)}
        </span>

        <span className="text-[#484f58] text-xs">{expanded ? "▾" : "▸"}</span>
      </button>

      {/* ------------------------------------------------------------------ */}
      {/* Expanded body: chart + probability table                             */}
      {/* ------------------------------------------------------------------ */}
      {expanded && (
        <div className="border-t border-surface-border px-4 pb-4 pt-3 flex flex-col gap-4">
          <TokenProbabilityChart
            candidates={step.top_tokens}
            selectedTokenId={step.selected_token_id}
            title={`Top ${step.top_tokens.length} candidates — step ${step.step}`}
          />

          {/* Quick-read probability table */}
          <div className="overflow-x-auto">
            <table className="w-full border-collapse text-[11px] font-mono">
              <thead>
                <tr className="border-b border-surface-border text-left text-[#484f58]">
                  <th className="pb-1 pr-3 font-medium">rank</th>
                  <th className="pb-1 pr-3 font-medium">token</th>
                  <th className="pb-1 pr-3 font-medium">id</th>
                  <th className="pb-1 font-medium text-right">prob</th>
                </tr>
              </thead>
              <tbody>
                {step.top_tokens.map((t, rank) => {
                  const isSelected = t.token_id === step.selected_token_id;
                  return (
                    <tr
                      key={t.token_id}
                      className={`border-b border-surface-border/40 ${
                        isSelected ? "text-accent-blue" : "text-[#8b949e]"
                      }`}
                    >
                      <td className="py-0.5 pr-3 text-[#484f58]">{rank + 1}</td>
                      <td className="py-0.5 pr-3">
                        {JSON.stringify(t.token)}
                        {isSelected && (
                          <span className="ml-1 text-[9px] text-accent-blue">✓</span>
                        )}
                      </td>
                      <td className="py-0.5 pr-3 text-[#484f58]">{t.token_id}</td>
                      <td className="py-0.5 text-right">
                        {formatPct(t.probability, 3)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
}
