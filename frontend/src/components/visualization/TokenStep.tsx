"use client";

/**
 * TokenStep — summary card for a single decoding step.
 *
 * Shows:
 *   - step number
 *   - current context (truncated)
 *   - selected token (highlighted)
 *   - number of masked tokens
 *   - entropy before / after Syncode
 *
 * Click to expand: renders TokenProbabilityChart for full candidate list.
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

export function TokenStep({ step, isActive, onClick }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      className={`rounded-md border transition-colors ${
        isActive
          ? "border-accent-blue/60 bg-blue-900/10"
          : "border-surface-border bg-surface-raised hover:border-[#30363d]"
      }`}
    >
      {/* Header row */}
      <button
        type="button"
        onClick={() => {
          onClick?.();
          setExpanded((e) => !e);
        }}
        className="flex w-full items-center gap-4 px-4 py-3 text-left"
      >
        <span className="w-8 shrink-0 text-center font-mono text-xs text-[#484f58]">
          {step.step}
        </span>

        <span className="flex-1 truncate font-mono text-xs text-[#8b949e]">
          {truncate(step.context, 60)}
        </span>

        <Badge variant="selected">{JSON.stringify(step.selected_token)}</Badge>

        {step.num_masked > 0 && (
          <Badge variant="masked">{step.num_masked} masked</Badge>
        )}

        {step.entropy_before !== null && (
          <span className="shrink-0 text-xs text-[#484f58]">
            H={step.entropy_before.toFixed(2)}
            {step.entropy_after !== null && (
              <> → {step.entropy_after.toFixed(2)}</>
            )}
          </span>
        )}

        <span className="text-[#484f58]">{expanded ? "▾" : "▸"}</span>
      </button>

      {/* Expanded chart */}
      {expanded && (
        <div className="border-t border-surface-border px-4 pb-4 pt-3 grid gap-4 sm:grid-cols-2">
          <TokenProbabilityChart
            candidates={step.top_tokens_before_syncode}
            title="Before Syncode"
          />
          <TokenProbabilityChart
            candidates={step.valid_tokens_after_syncode}
            title="After Syncode"
          />
        </div>
      )}
    </div>
  );
}
