"use client";

/**
 * TokenProbabilityChart
 *
 * Horizontal bar chart for the top-k token candidates at one decoding step.
 *
 * Colour coding:
 *   blue  (#58a6ff) — the selected token (greedy argmax)
 *   red   (#f85149) — a token masked by Syncode (grammar-invalid)
 *   green (#3fb950) — all other (unmasked, unselected) candidates
 *
 * Props
 * -----
 * candidates      TopToken[]     — sorted by probability before passing in
 * selectedTokenId number         — highlights the greedy-chosen bar in blue
 * maskedIds       number[]       — IDs to colour red (Syncode masked tokens)
 * title           string         — optional chart heading
 */

import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer,
} from "recharts";
import type { TopToken } from "@/types/decoding";
import { formatPct } from "@/lib/utils";

interface Props {
  candidates: TopToken[];
  selectedTokenId?: number;
  /** Token IDs that were masked by Syncode — rendered in red. */
  maskedIds?: number[];
  title?: string;
}

/** Escape non-printable chars so Recharts YAxis labels render safely. */
function safeLabel(token: string): string {
  return JSON.stringify(token);
}

function barColor(
  tokenId: number,
  selectedTokenId: number | undefined,
  maskedSet: Set<number>,
): string {
  if (tokenId === selectedTokenId) return "#58a6ff"; // blue — selected
  if (maskedSet.has(tokenId))      return "#f85149"; // red  — masked by Syncode
  return "#3fb950";                                   // green — valid candidate
}

export function TokenProbabilityChart({
  candidates,
  selectedTokenId,
  maskedIds,
  title,
}: Props) {
  if (candidates.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-sm text-[#484f58]">
        No token data
      </div>
    );
  }

  const maskedSet = new Set<number>(maskedIds ?? []);
  const hasMasked = maskedSet.size > 0;

  const data = [...candidates]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 20)
    .map((t) => ({
      ...t,
      label: safeLabel(t.token),
      pct: t.probability,
      isMasked: maskedSet.has(t.token_id),
      isSelected: t.token_id === selectedTokenId,
    }));

  // 18px per bar keeps 20 candidates in ~380px — avoids overflow
  const barHeight = 18;
  const chartHeight = data.length * barHeight + 24;

  return (
    <div className="flex flex-col gap-2">
      {title && (
        <p className="text-[11px] font-semibold uppercase tracking-wider text-[#8b949e]">
          {title}
        </p>
      )}

      <ResponsiveContainer width="100%" height={chartHeight}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 40, bottom: 0, left: 4 }}
        >
          <XAxis
            type="number"
            domain={[0, Math.max(...data.map((d) => d.pct)) * 1.1]}
            tickFormatter={(v) => formatPct(v, 0)}
            tick={{ fill: "#484f58", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={96}
            tick={{ fill: "#c9d1d9", fontSize: 11, fontFamily: "monospace" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.03)" }}
            contentStyle={{
              background: "#161b22",
              border: "1px solid #21262d",
              borderRadius: 6,
              fontSize: 12,
              color: "#e6edf3",
            }}
            formatter={(
              value: number,
              _name: string,
              props: { payload?: { token: string; isMasked?: boolean } },
            ) => [
              `${formatPct(value, 3)}${props.payload?.isMasked ? "  ✗ masked" : ""}`,
              props.payload?.token ?? "probability",
            ]}
          />
          <Bar dataKey="pct" radius={[0, 3, 3, 0]} maxBarSize={16}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={barColor(entry.token_id, selectedTokenId, maskedSet)}
                opacity={entry.isMasked ? 0.45 : entry.isSelected ? 1 : 0.75}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-[10px] text-[#484f58]">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#58a6ff]" />
          selected
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#3fb950] opacity-75" />
          valid candidate
        </span>
        {hasMasked && (
          <span className="flex items-center gap-1">
            <span className="inline-block h-2 w-2 rounded-sm bg-[#f85149] opacity-45" />
            masked by Syncode
          </span>
        )}
      </div>
    </div>
  );
}
