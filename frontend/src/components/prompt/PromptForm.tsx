"use client";

/**
 * PromptForm — controlled input component.
 *
 * All generation state lives in the parent (page.tsx) via useGeneration.
 * This component only owns the prompt text, settings, and Syncode toggle.
 *
 * The Syncode toggle is now functional.  When enabled, POST /generate
 * will apply C-grammar masking at each decoding step and populate the
 * top_tokens_before_syncode / masked_tokens / valid_tokens_after_syncode
 * fields in the response.
 */

import { useState } from "react";
import { Button } from "@/components/ui/Button";
import { GenerationSettings } from "@/components/prompt/GenerationSettings";
import type { GenerateRequest } from "@/types/decoding";

interface Props {
  onSubmit: (req: GenerateRequest) => void;
  isLoading: boolean;
  error: string | null;
}

const DEFAULT_PROMPT = `Write a C function that reverses a null-terminated string in-place.`;

export function PromptForm({ onSubmit, isLoading, error }: Props) {
  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [useSyncode, setUseSyncode] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Omit<GenerateRequest, "prompt" | "use_syncode">>({
    top_k: 20,
    max_new_tokens: 64,
    temperature: 1.0,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ prompt, use_syncode: useSyncode, ...settings });
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      {/* Prompt */}
      <div className="flex flex-col gap-1.5">
        <label
          htmlFor="prompt"
          className="text-xs font-medium uppercase tracking-wider text-[#8b949e]"
        >
          Prompt
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={4}
          disabled={isLoading}
          placeholder="Enter a prompt…"
          className="code-block w-full resize-y rounded-md border border-surface-border bg-surface p-3 text-[#e6edf3] placeholder-[#484f58] focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue disabled:opacity-50"
        />
      </div>

      {/* Syncode toggle — now functional */}
      <label className="flex cursor-pointer items-center gap-3 rounded-md border border-surface-border bg-surface px-3 py-2 transition-colors hover:border-[#30363d]">
        {/* Custom toggle track */}
        <div className="relative shrink-0" onClick={() => !isLoading && setUseSyncode((v) => !v)}>
          <div
            className={`h-5 w-9 rounded-full transition-colors ${
              useSyncode ? "bg-accent-blue" : "bg-surface-border"
            }`}
          />
          <div
            className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform ${
              useSyncode ? "translate-x-4" : "translate-x-0.5"
            }`}
          />
        </div>
        <div className="flex flex-col gap-0.5">
          <span className={`text-sm ${useSyncode ? "text-accent-blue" : "text-[#8b949e]"}`}>
            Syncode constrained decoding
          </span>
          <span className="text-[10px] text-[#484f58]">
            {useSyncode
              ? "Grammar masking active — invalid C tokens will be suppressed"
              : "Off — raw greedy decoding, no grammar constraint"}
          </span>
        </div>
        {useSyncode && (
          <span className="ml-auto rounded border border-accent-blue/40 bg-accent-blue/10 px-1.5 py-0.5 text-[10px] text-accent-blue">
            C grammar
          </span>
        )}
      </label>

      {/* Advanced settings */}
      <button
        type="button"
        disabled={isLoading}
        onClick={() => setShowSettings((v) => !v)}
        className="self-start text-xs text-[#8b949e] transition-colors hover:text-accent-blue disabled:opacity-40"
      >
        {showSettings ? "▾ Hide settings" : "▸ Advanced settings"}
      </button>

      {showSettings && (
        <GenerationSettings value={settings} onChange={setSettings} />
      )}

      {error && (
        <p className="rounded-md border border-accent-red/30 bg-red-900/20 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}

      <p className="text-[11px] text-[#484f58]">
        Qwen2.5-Coder-1.5B · CPU
        {useSyncode && " · Syncode C grammar (DFA builds on first run ~30 s)"}
        {!useSyncode && " · ~30–90 s on first run (model downloads once)"}
      </p>

      <Button type="submit" loading={isLoading} size="lg" className="self-start">
        {isLoading ? "Generating…" : "Generate"}
      </Button>
    </form>
  );
}
