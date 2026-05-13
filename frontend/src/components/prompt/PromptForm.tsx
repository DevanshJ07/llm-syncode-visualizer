"use client";

/**
 * PromptForm — the primary input component on the home page.
 *
 * Renders:
 *   - multi-line prompt textarea
 *   - Syncode toggle
 *   - collapsible GenerationSettings
 *   - submit button wired to the useGeneration hook
 *
 * On successful generation it calls onSuccess(experimentId) so the
 * parent page can navigate to the experiment viewer.
 */

import { useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/Button";
import { GenerationSettings } from "@/components/prompt/GenerationSettings";
import { useGeneration } from "@/hooks/useGeneration";
import type { GenerateRequest } from "@/types/decoding";

const DEFAULT_PROMPT = `// Write a C function that reverses a null-terminated string in-place.`;

export function PromptForm() {
  const router = useRouter();
  const { status, error, generate } = useGeneration();

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [useSyncode, setUseSyncode] = useState(true);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Omit<GenerateRequest, "prompt" | "use_syncode">>({
    top_k: 50,
    max_new_tokens: 256,
    temperature: 1.0,
  });

  const isLoading = status === "generating" || status === "fetching";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const id = await generate({ prompt, use_syncode: useSyncode, ...settings });
    if (id) router.push(`/experiment/${id}`);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      {/* Prompt textarea */}
      <div className="flex flex-col gap-1.5">
        <label htmlFor="prompt" className="text-xs font-medium text-[#8b949e] uppercase tracking-wider">
          Prompt
        </label>
        <textarea
          id="prompt"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          rows={6}
          placeholder="Enter a C code prompt…"
          className="code-block w-full resize-y rounded-md border border-surface-border bg-surface p-3 text-[#e6edf3] placeholder-[#484f58] focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
        />
      </div>

      {/* Syncode toggle */}
      <label className="flex cursor-pointer items-center gap-3">
        <div className="relative">
          <input
            type="checkbox"
            className="sr-only peer"
            checked={useSyncode}
            onChange={(e) => setUseSyncode(e.target.checked)}
          />
          <div className="h-5 w-9 rounded-full bg-surface-border peer-checked:bg-accent-blue transition-colors" />
          <div className="absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform peer-checked:translate-x-4" />
        </div>
        <span className="text-sm text-[#e6edf3]">Syncode constrained decoding</span>
        <span className="text-xs text-[#8b949e]">
          {useSyncode ? "ON – grammar constraints applied" : "OFF – raw generation"}
        </span>
      </label>

      {/* Advanced settings toggle */}
      <button
        type="button"
        onClick={() => setShowSettings(!showSettings)}
        className="self-start text-xs text-[#8b949e] hover:text-accent-blue transition-colors"
      >
        {showSettings ? "▾ Hide settings" : "▸ Advanced settings"}
      </button>

      {showSettings && (
        <GenerationSettings value={settings} onChange={setSettings} />
      )}

      {/* Error */}
      {error && (
        <p className="rounded-md border border-accent-red/30 bg-red-900/20 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}

      <Button type="submit" loading={isLoading} size="lg" className="self-start">
        {isLoading
          ? status === "generating"
            ? "Generating…"
            : "Loading result…"
          : "Generate Code"}
      </Button>
    </form>
  );
}
