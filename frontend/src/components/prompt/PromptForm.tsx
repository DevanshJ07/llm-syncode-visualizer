"use client";

/**
 * PromptForm — primary input on the home page.
 *
 * Sends a POST /generate request and navigates to /experiment/{id} on success.
 * Syncode toggle is disabled (shown as "coming soon") until Phase 3.
 */

import { useState } from "react";
import { useRouter } from "next/navigation";

import { Button } from "@/components/ui/Button";
import { GenerationSettings } from "@/components/prompt/GenerationSettings";
import { useGeneration } from "@/hooks/useGeneration";
import type { GenerateRequest } from "@/types/decoding";

const DEFAULT_PROMPT = `Write a C function that reverses a null-terminated string in-place.`;

export function PromptForm() {
  const router = useRouter();
  const { status, error, generate } = useGeneration();

  const [prompt, setPrompt] = useState(DEFAULT_PROMPT);
  const [showSettings, setShowSettings] = useState(false);
  const [settings, setSettings] = useState<Omit<GenerateRequest, "prompt" | "use_syncode">>({
    top_k: 10,
    max_new_tokens: 64,
    temperature: 1.0,
  });

  const isLoading = status === "generating" || status === "fetching";

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    // use_syncode is always false until Phase 3
    const id = await generate({ prompt, use_syncode: false, ...settings });
    if (id) router.push(`/experiment/${id}`);
  };

  return (
    <form onSubmit={handleSubmit} className="flex flex-col gap-4">
      {/* Prompt textarea */}
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
          rows={5}
          placeholder="Enter a prompt…"
          className="code-block w-full resize-y rounded-md border border-surface-border bg-surface p-3 text-[#e6edf3] placeholder-[#484f58] focus:border-accent-blue focus:outline-none focus:ring-1 focus:ring-accent-blue"
        />
      </div>

      {/* Syncode toggle — disabled, coming soon */}
      <div className="flex items-center gap-3 rounded-md border border-surface-border/50 bg-surface px-3 py-2 opacity-50">
        <div className="relative shrink-0">
          {/* Always-off unchecked toggle */}
          <div className="h-5 w-9 rounded-full bg-surface-border" />
          <div className="absolute top-0.5 left-0.5 h-4 w-4 rounded-full bg-[#484f58] shadow" />
        </div>
        <span className="text-sm text-[#8b949e]">Syncode constrained decoding</span>
        <span className="ml-auto rounded border border-surface-border px-1.5 py-0.5 text-[10px] font-medium text-[#484f58]">
          coming soon
        </span>
      </div>

      {/* Advanced settings toggle */}
      <button
        type="button"
        onClick={() => setShowSettings(!showSettings)}
        className="self-start text-xs text-[#8b949e] transition-colors hover:text-accent-blue"
      >
        {showSettings ? "▾ Hide settings" : "▸ Advanced settings"}
      </button>

      {showSettings && (
        <GenerationSettings value={settings} onChange={setSettings} />
      )}

      {/* Error banner */}
      {error && (
        <p className="rounded-md border border-accent-red/30 bg-red-900/20 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}

      {/* Model info note */}
      <p className="text-[11px] text-[#484f58]">
        Model: TinyLlama-1.1B · CPU · generation may take 30–90 s on first run
        (model downloads and loads once).
      </p>

      <Button type="submit" loading={isLoading} size="lg" className="self-start">
        {isLoading
          ? status === "generating"
            ? "Generating…"
            : "Fetching result…"
          : "Generate"}
      </Button>
    </form>
  );
}
