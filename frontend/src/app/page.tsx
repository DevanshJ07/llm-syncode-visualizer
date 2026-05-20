"use client";

/**
 * Home page — Generate & Visualize
 *
 * Before generation: full-width prompt form.
 * After generation: visualization takes over the page.
 *
 * Layout (post-generation):
 * ┌─────────────────────────────────────────────────────┐
 * │ [Re-generate strip — collapsed PromptForm]          │
 * ├──────────────────────┬──────────────────────────────┤
 * │  Generated code      │  Step Viewer                 │
 * │  (grows token by     │  (bar chart + table for      │
 * │   token as slider    │   the current step)          │
 * │   is scrubbed)       │                              │
 * ├──────────────────────┴──────────────────────────────┤
 * │  StepPlayer  (slider + transport + speed)           │
 * ├─────────────────────────────────────────────────────┤
 * │  EntropyChart  (line chart across all steps)        │
 * ├─────────────────────────────────────────────────────┤
 * │  DecodingTimeline  (all steps, expandable cards)    │
 * └─────────────────────────────────────────────────────┘
 */

import { useCallback, useEffect, useMemo, useState } from "react";

import { PromptForm } from "@/components/prompt/PromptForm";
import { CodeViewer } from "@/components/output/CodeViewer";
import { StepViewer } from "@/components/visualization/StepViewer";
import { StepPlayer } from "@/components/visualization/StepPlayer";
import { EntropyChart } from "@/components/visualization/EntropyChart";
import { DecodingTimeline } from "@/components/visualization/DecodingTimeline";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { Card } from "@/components/ui/Card";
import { useGeneration } from "@/hooks/useGeneration";
import { formatDate, formatPct } from "@/lib/utils";
import type { GenerateRequest } from "@/types/decoding";

export default function HomePage() {
  const { status, experiment, error, generate, reset } = useGeneration();

  // Which step is the "camera" pointing at (0-indexed)
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playIntervalMs, setPlayIntervalMs] = useState(1000);
  const [showForm, setShowForm] = useState(true);

  const isLoading = status === "generating";
  const hasError = status === "error" && error !== null;
  const hasResult =
    status === "done" &&
    experiment !== null &&
    experiment.steps.length > 0 &&
    experiment.total_steps > 0;

  // Reset step index whenever a new experiment arrives
  useEffect(() => {
    if (hasResult) {
      setCurrentStep(0);
      setIsPlaying(false);
      setShowForm(false);
    }
  }, [hasResult, experiment?.experiment_id]);

  // Stop autoplay at last step
  useEffect(() => {
    if (!experiment) return;
    if (currentStep >= experiment.total_steps - 1) {
      setIsPlaying(false);
    }
  }, [currentStep, experiment]);

  const handleGenerate = useCallback(
    async (req: GenerateRequest) => {
      reset();
      setShowForm(true);
      await generate(req);
    },
    [generate, reset]
  );

  // Derived: text visible up to (and including) the current step
  const visibleCode = useMemo(() => {
    if (!experiment || experiment.steps.length === 0) return "";
    const step = experiment.steps[currentStep];
    if (!step) return "";
    // context = text generated BEFORE this step; selected_token = this step's token
    return step.context + step.selected_token;
  }, [experiment, currentStep]);

  // Derived: aggregate stats for the stats strip
  const stats = useMemo(() => {
    if (!experiment || experiment.steps.length === 0) return null;
    const entropies = experiment.steps
      .map((s) => s.entropy_before)
      .filter((e): e is number => e !== null);
    return {
      avgEntropy: entropies.length
        ? (entropies.reduce((a, b) => a + b, 0) / entropies.length).toFixed(3)
        : "—",
      maxEntropy: entropies.length ? Math.max(...entropies).toFixed(3) : "—",
      avgTopProb: experiment.steps.length
        ? formatPct(
            experiment.steps.reduce((s, st) => s + (st.top_tokens[0]?.probability ?? 0), 0) /
              experiment.steps.length,
            1
          )
        : "—",
    };
  }, [experiment]);

  // ── ERROR: show form with backend error (no stale visualization) ───────
  if (hasError && !isLoading) {
    return (
      <div className="mx-auto flex max-w-2xl flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold text-[#e6edf3]">Generate &amp; Visualize</h1>
          <p className="mt-1 text-sm text-[#8b949e]">
            Generation failed. The error below is from the backend — no placeholder
            output is shown.
          </p>
        </div>
        <Card>
          <PromptForm onSubmit={handleGenerate} isLoading={isLoading} error={error} />
        </Card>
      </div>
    );
  }

  // ── PRE-GENERATION: full-width prompt form ──────────────────────────────
  if (!hasResult && !isLoading) {
    return (
      <div className="mx-auto flex max-w-2xl flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold text-[#e6edf3]">Generate &amp; Visualize</h1>
          <p className="mt-1 text-sm leading-relaxed text-[#8b949e]">
            Qwen2.5-Coder generates code token-by-token with full decoding traces.
            Toggle Syncode C-grammar masking to compare raw vs constrained distributions,
            entropy shifts, and masked-token forensics at every step.
          </p>
        </div>
        <Card>
          <PromptForm onSubmit={handleGenerate} isLoading={isLoading} error={error} />
        </Card>
        <div className="grid grid-cols-2 gap-3 text-xs text-[#484f58] sm:grid-cols-4">
          {[
            ["Model", "Qwen2.5-Coder"],
            ["Runtime", "CPU · fp32"],
            ["Decoding", "Nucleus + trace"],
            ["Syncode", "C grammar mask"],
          ].map(([k, v]) => (
            <div key={k} className="rounded-md border border-surface-border bg-surface-raised p-2">
              <p className="text-[#484f58]">{k}</p>
              <p className="font-medium text-[#8b949e]">{v}</p>
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── LOADING ─────────────────────────────────────────────────────────────
  if (isLoading) {
    return (
      <div className="flex flex-col items-center gap-6 py-32">
        <Spinner size="lg" label="Generating tokens…" />
        <p className="max-w-sm text-center text-xs text-[#484f58]">
          Qwen2.5-Coder-1.5B-Instruct is generating on CPU.
          First run downloads weights (~3 GB); Syncode DFA build adds ~30 s once.
        </p>
        <Button variant="ghost" size="sm" onClick={reset}>
          Cancel
        </Button>
      </div>
    );
  }

  // ── POST-GENERATION: full visualization ─────────────────────────────────
  if (!experiment) return null;

  const activeStep = experiment.steps[currentStep];

  return (
    <div className="flex flex-col gap-3">

      {/* ── Compact re-generate strip ─────────────────────────────────── */}
      <div className="flex items-center gap-3 rounded-md border border-surface-border bg-surface-raised px-4 py-2">
        <span className="text-xs text-[#484f58]">
          <span className="text-accent-blue font-mono">
            {experiment.experiment_id.slice(0, 8)}…
          </span>
          {" · "}
          {experiment.model_name.split("/").pop()}
          {" · "}
          {experiment.total_steps} tokens
        </span>
        <Badge variant="neutral">{experiment.mode}</Badge>
        <button
          type="button"
          onClick={() => setShowForm((v) => !v)}
          className="ml-auto text-xs text-[#8b949e] hover:text-accent-blue"
        >
          {showForm ? "▾ Hide form" : "▸ New prompt"}
        </button>
        <Button variant="secondary" size="sm" onClick={() => { reset(); setShowForm(true); }}>
          Reset
        </Button>
      </div>

      {/* ── Collapsible prompt form ───────────────────────────────────── */}
      {showForm && (
        <Card>
          <PromptForm onSubmit={handleGenerate} isLoading={isLoading} error={error} />
        </Card>
      )}

      {/* ── Stats strip ───────────────────────────────────────────────── */}
      {stats && (
        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            { label: "Tokens", value: experiment.total_steps },
            { label: "Avg entropy", value: stats.avgEntropy, title: "Mean H = -Σp·log(p)" },
            { label: "Max entropy", value: stats.maxEntropy, title: "Most uncertain step" },
            { label: "Avg top-1 p", value: stats.avgTopProb, title: "Mean top-1 probability" },
          ].map(({ label, value, title }) => (
            <div
              key={label}
              title={title}
              className="rounded-md border border-surface-border bg-surface-raised px-2.5 py-1.5"
            >
              <p className="text-[10px] uppercase tracking-wider text-[#484f58]">{label}</p>
              <p className="mt-0.5 font-mono text-base font-semibold text-[#e6edf3]">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* ── Main split: code (narrow) + step detail / forensic (wide) ─ */}
      <div className="grid gap-3 lg:grid-cols-12">

        {/* Generated code — shows text built up to currentStep */}
        <section className="flex flex-col gap-1.5 lg:col-span-4">
          <div className="flex items-center gap-2">
            <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
              Generated Output
            </h2>
            <span className="ml-auto font-mono text-[10px] text-[#484f58]">
              up to step {currentStep + 1}
            </span>
          </div>
          {visibleCode ? (
            <CodeViewer
              code={visibleCode}
              className="min-h-32 max-h-[42vh]"
            />
          ) : (
            <div className="flex min-h-40 items-center justify-center rounded-md border border-accent-red/30 bg-red-900/10 px-4 text-sm text-accent-red">
              No token data for step {currentStep + 1} — trace may be corrupt.
            </div>
          )}
        </section>

        {/* Active step detail — wider forensic panel */}
        <section className="flex flex-col gap-1.5 lg:col-span-8">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
            Step Detail — Syncode Forensics
          </h2>
          {activeStep ? (
            <div className="overflow-y-auto max-h-[42vh] rounded-md border border-surface-border bg-surface-raised p-3">
              <StepViewer step={activeStep} />
            </div>
          ) : (
            <div className="flex h-40 items-center justify-center rounded-md border border-surface-border text-sm text-[#484f58]">
              No step selected
            </div>
          )}
        </section>
      </div>

      {/* ── Step player ───────────────────────────────────────────────── */}
      <StepPlayer
        totalSteps={experiment.total_steps}
        currentStep={currentStep}
        isPlaying={isPlaying}
        onStepChange={setCurrentStep}
        onPlayPause={() => setIsPlaying((v) => !v)}
        playIntervalMs={playIntervalMs}
        onIntervalChange={setPlayIntervalMs}
      />

      {/* ── Entropy chart ─────────────────────────────────────────────── */}
      <div className="rounded-md border border-surface-border bg-surface-raised px-3 py-2">
        <EntropyChart
          steps={experiment.steps}
          activeStep={currentStep}
          onStepClick={setCurrentStep}
        />
      </div>

      {/* ── Full decoding timeline ────────────────────────────────────── */}
      <section className="flex flex-col gap-1.5">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
          Decoding Timeline — {experiment.total_steps} step{experiment.total_steps !== 1 ? "s" : ""}
        </h2>
        <DecodingTimeline
          steps={experiment.steps}
          onStepSelect={setCurrentStep}
        />
      </section>

    </div>
  );
}
