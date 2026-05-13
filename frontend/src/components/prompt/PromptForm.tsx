"use client";

/**
 * PromptForm — controlled input component.
 *
 * All generation state lives in the parent (page.tsx) via useGeneration.
 * This component only owns the prompt text and settings UI state.
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
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Omit<GenerateRequest, "prompt" | "use_syncode">>({
    top_k: 10,
    max_new_tokens: 64,
    temperature: 1.0,
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onSubmit({ prompt, use_syncode: false, ...settings });
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

      {/* Syncode — disabled until Phase 3 */}
      <div className="flex items-center gap-3 rounded-md border border-surface-border/40 bg-surface px-3 py-2 opacity-40">
        <div className="relative shrink-0">
          <div className="h-5 w-9 rounded-full bg-surface-border" />
          <div className="absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-[#484f58] shadow" />
        </div>
        <span className="text-sm text-[#8b949e]">Syncode constrained decoding</span>
        <span className="ml-auto rounded border border-surface-border px-1.5 py-0.5 text-[10px] text-[#484f58]">
          Phase 3
        </span>
      </div>

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
        TinyLlama-1.1B · CPU · ~30–90 s on first run (model downloads once)
      </p>

      <Button type="submit" loading={isLoading} size="lg" className="self-start">
        {isLoading ? "Generating…" : "Generate"}
      </Button>
    </form>
  );
}
