"use client";

/**
 * GenerationSettings — collapsible panel for top_k, max_new_tokens, temperature.
 * Receives value / onChange from PromptForm (controlled component pattern).
 */

import type { GenerateRequest } from "@/types/decoding";

type SettingsValue = Omit<GenerateRequest, "prompt" | "use_syncode">;

interface Props {
  value: SettingsValue;
  onChange: (v: SettingsValue) => void;
}

function NumberField({
  label,
  value,
  min,
  max,
  step,
  onChange,
  hint,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
  hint?: string;
}) {
  return (
    <div className="flex flex-col gap-1">
      <label className="flex items-center justify-between text-xs text-[#8b949e]">
        <span>{label}</span>
        <span className="font-mono text-[#e6edf3]">{value}</span>
      </label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className="w-full accent-accent-blue"
      />
      {hint && <p className="text-[10px] text-[#484f58]">{hint}</p>}
    </div>
  );
}

export function GenerationSettings({ value, onChange }: Props) {
  const set = (key: keyof SettingsValue, v: number) =>
    onChange({ ...value, [key]: v });

  return (
    <div className="grid gap-4 rounded-md border border-surface-border bg-surface p-4 sm:grid-cols-3">
      <NumberField
        label="Top-k"
        value={value.top_k}
        min={1}
        max={200}
        step={1}
        onChange={(v) => set("top_k", v)}
        hint="Candidates logged per step"
      />
      <NumberField
        label="Max new tokens"
        value={value.max_new_tokens}
        min={16}
        max={1024}
        step={16}
        onChange={(v) => set("max_new_tokens", v)}
      />
      <NumberField
        label="Temperature"
        value={value.temperature}
        min={0}
        max={2}
        step={0.05}
        onChange={(v) => set("temperature", v)}
        hint="0 = greedy, 1 = normal sampling"
      />
    </div>
  );
}
