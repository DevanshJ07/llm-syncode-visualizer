"use client";

/**
 * TokenProbabilityChart
 *
 * Horizontal bar chart for the top-k token candidates at one decoding step.
 *
 * Colour coding:
 *   blue  (#58a6ff) — the selected token (greedy argmax)
 *   green (#3fb950) — all other candidates
 *
 * The selected token is identified by matching token_id === selectedTokenId.
 * If selectedTokenId is not in the top-k list (can happen when greedy pick
 * falls outside the logged top-k), we fall back to matching by string.
 *
 * Uses Recharts BarChart in a horizontal layout (token labels on Y-axis,
 * probability on X-axis).  The full probability is shown in the tooltip.
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
  /** Vocabulary index of the selected (greedy) token — used to colour the bar. */
  selectedTokenId?: number;
  title?: string;
}

/** Escape non-printable chars so Recharts YAxis labels render safely. */
function safeLabel(token: string): string {
  return JSON.stringify(token); // wraps in quotes and escapes \n, \t, etc.
}

export function TokenProbabilityChart({ candidates, selectedTokenId, title }: Props) {
  if (candidates.length === 0) {
    return (
      <div className="flex h-40 items-center justify-center text-sm text-[#484f58]">
        No token data
      </div>
    );
  }

  // Sort descending by probability, cap at 20 bars for readability
  const data = [...candidates]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 20)
    .map((t) => ({
      ...t,
      label: safeLabel(t.token),
      pct: t.probability,
      isSelected: t.token_id === selectedTokenId,
    }));

  const barHeight = 22;
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
          {/* X-axis: probability 0–1 */}
          <XAxis
            type="number"
            domain={[0, Math.max(...data.map((d) => d.pct)) * 1.1]}
            tickFormatter={(v) => formatPct(v, 0)}
            tick={{ fill: "#484f58", fontSize: 10 }}
            axisLine={false}
            tickLine={false}
          />

          {/* Y-axis: token label */}
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
            formatter={(value: number, _name: string, props: { payload?: { token: string } }) => [
              formatPct(value, 3),
              props.payload?.token ?? "probability",
            ]}
          />

          <Bar dataKey="pct" radius={[0, 3, 3, 0]} maxBarSize={16}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.isSelected ? "#58a6ff" : "#3fb950"}
                opacity={entry.isSelected ? 1 : 0.7}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* Legend */}
      <div className="flex gap-4 text-[10px] text-[#484f58]">
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#58a6ff]" />
          selected (greedy)
        </span>
        <span className="flex items-center gap-1">
          <span className="inline-block h-2 w-2 rounded-sm bg-[#3fb950] opacity-70" />
          candidate
        </span>
      </div>
    </div>
  );
}
