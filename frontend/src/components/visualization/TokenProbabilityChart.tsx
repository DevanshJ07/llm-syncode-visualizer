"use client";

/**
 * TokenProbabilityChart
 *
 * Horizontal bar chart showing the top-k token candidates for one decoding step.
 * Bars are colour-coded:
 *   - red   → masked by Syncode (grammar-invalid)
 *   - blue  → the selected token
 *   - green → valid, not selected
 *
 * Uses Recharts BarChart in a horizontal layout.
 *
 * TODO Phase 2: overlay before/after distributions as grouped bars.
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
import type { TokenCandidate } from "@/types/decoding";
import { formatPct } from "@/lib/utils";

interface Props {
  candidates: TokenCandidate[];
  title?: string;
}

function tokenColor(t: TokenCandidate): string {
  if (t.is_selected) return "#58a6ff";   // accent-blue
  if (t.is_masked) return "#f85149";     // accent-red
  return "#3fb950";                       // accent-green
}

export function TokenProbabilityChart({ candidates, title }: Props) {
  if (candidates.length === 0) {
    return (
      <div className="flex h-48 items-center justify-center text-sm text-[#484f58]">
        No token data
      </div>
    );
  }

  const data = [...candidates]
    .sort((a, b) => b.probability - a.probability)
    .slice(0, 20)
    .map((t) => ({
      ...t,
      label: JSON.stringify(t.token_str),
      pct: t.probability,
    }));

  return (
    <div className="flex flex-col gap-2">
      {title && <p className="text-xs font-medium text-[#8b949e] uppercase tracking-wider">{title}</p>}
      <ResponsiveContainer width="100%" height={Math.min(data.length * 28 + 20, 480)}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 24 }}>
          <XAxis
            type="number"
            domain={[0, 1]}
            tickFormatter={(v) => formatPct(v, 0)}
            tick={{ fill: "#8b949e", fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="label"
            width={90}
            tick={{ fill: "#e6edf3", fontSize: 11, fontFamily: "monospace" }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            cursor={{ fill: "rgba(255,255,255,0.04)" }}
            contentStyle={{
              background: "#161b22",
              border: "1px solid #21262d",
              borderRadius: 6,
              fontSize: 12,
            }}
            formatter={(value: number) => [formatPct(value), "probability"]}
          />
          <Bar dataKey="pct" radius={[0, 3, 3, 0]} maxBarSize={18}>
            {data.map((entry, i) => (
              <Cell key={i} fill={tokenColor(entry)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
