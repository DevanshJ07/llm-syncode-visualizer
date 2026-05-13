"use client";

/**
 * EntropyChart — line chart of Shannon entropy H across all decoding steps.
 *
 * - Each point = one generated token.
 * - Active step is highlighted with a vertical reference line + large dot.
 * - Colour: green when H < 2 (confident), yellow 2–4, red > 4 (uncertain).
 * - Click any point to jump to that step.
 */

import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from "recharts";
import type { DecodingStep } from "@/types/decoding";

interface Props {
  steps: DecodingStep[];
  activeStep: number;           // 0-indexed
  onStepClick?: (idx: number) => void;
}

export function EntropyChart({ steps, activeStep, onStepClick }: Props) {
  const data = steps.map((s, i) => ({
    idx: i,
    step: s.step,
    entropy: s.entropy_before ?? 0,
    token: s.selected_token,
  }));

  if (data.length === 0) return null;

  const maxH = Math.max(...data.map((d) => d.entropy));

  return (
    <div className="flex flex-col gap-1">
      <p className="text-[10px] uppercase tracking-wider text-[#484f58]">
        Entropy across steps — H = −Σ p·log(p)
      </p>
      <ResponsiveContainer width="100%" height={90}>
        <LineChart
          data={data}
          margin={{ top: 8, right: 8, bottom: 0, left: 0 }}
          onClick={(e) => {
            const idx = e?.activePayload?.[0]?.payload?.idx;
            if (typeof idx === "number") onStepClick?.(idx);
          }}
          style={{ cursor: "pointer" }}
        >
          <XAxis
            dataKey="step"
            tick={{ fill: "#484f58", fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            interval="preserveStartEnd"
          />
          <YAxis
            domain={[0, Math.ceil(maxH + 0.5)]}
            tick={{ fill: "#484f58", fontSize: 9 }}
            axisLine={false}
            tickLine={false}
            width={28}
          />
          <Tooltip
            contentStyle={{
              background: "#161b22",
              border: "1px solid #21262d",
              borderRadius: 6,
              fontSize: 11,
              color: "#e6edf3",
            }}
            formatter={(value: number, _: string, props: { payload?: { token: string } }) => [
              value.toFixed(4),
              `H  (token: ${JSON.stringify(props.payload?.token ?? "")})`,
            ]}
            labelFormatter={(label: number) => `Step ${label}`}
          />
          {/* Vertical line at the active step */}
          <ReferenceLine
            x={activeStep + 1}
            stroke="#58a6ff"
            strokeWidth={1}
            strokeDasharray="3 3"
          />
          <Line
            type="monotone"
            dataKey="entropy"
            stroke="#58a6ff"
            strokeWidth={1.5}
            dot={false}
            activeDot={{ r: 4, fill: "#58a6ff", stroke: "#0f1117", strokeWidth: 2 }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
