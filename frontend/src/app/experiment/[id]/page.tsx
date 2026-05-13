"use client";

/**
 * Experiment viewer — Output Viewer + Token Visualization.
 * URL: /experiment/[id]
 *
 * Layout:
 *   ┌──────────────────────────────────────────────────────┐
 *   │ breadcrumb / metadata bar                            │
 *   ├──────────────────┬───────────────────────────────────┤
 *   │  stats strip     │                                   │
 *   ├──────────────────┤   Decoding Timeline               │
 *   │  Generated Code  │   (one TokenStep card per token)  │
 *   │  (CodeViewer)    │                                   │
 *   └──────────────────┴───────────────────────────────────┘
 */

import { useEffect, useMemo, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";

import { CodeViewer } from "@/components/output/CodeViewer";
import { DecodingTimeline } from "@/components/visualization/DecodingTimeline";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { getExperiment } from "@/lib/api";
import { formatDate, formatPct } from "@/lib/utils";
import type { ExperimentResult } from "@/types/decoding";

export default function ExperimentPage() {
  const { id } = useParams<{ id: string }>();
  const router = useRouter();

  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeLine, setActiveLine] = useState<number | undefined>(undefined);

  useEffect(() => {
    if (!id) return;
    getExperiment(id)
      .then(setExperiment)
      .catch((err: Error) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

  // Aggregate stats derived from the steps array
  const stats = useMemo(() => {
    if (!experiment || experiment.steps.length === 0) return null;
    const entropies = experiment.steps
      .map((s) => s.entropy_before)
      .filter((e): e is number => e !== null);
    const avgEntropy =
      entropies.length > 0
        ? entropies.reduce((a, b) => a + b, 0) / entropies.length
        : null;
    const maxEntropy = entropies.length > 0 ? Math.max(...entropies) : null;
    const minEntropy = entropies.length > 0 ? Math.min(...entropies) : null;
    // Average top-1 probability
    const avgTopProb =
      experiment.steps.length > 0
        ? experiment.steps.reduce(
            (sum, s) => sum + (s.top_tokens[0]?.probability ?? 0),
            0
          ) / experiment.steps.length
        : null;
    return { avgEntropy, maxEntropy, minEntropy, avgTopProb };
  }, [experiment]);

  if (loading) {
    return (
      <div className="flex justify-center py-32">
        <Spinner size="lg" label="Loading experiment…" />
      </div>
    );
  }

  if (error || !experiment) {
    return (
      <div className="flex flex-col items-center gap-4 py-32 text-center">
        <p className="text-accent-red">{error ?? "Experiment not found."}</p>
        <Button variant="secondary" onClick={() => router.push("/")}>
          ← Back to Generate
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-5">
      {/* ---------------------------------------------------------------- */}
      {/* Metadata bar                                                      */}
      {/* ---------------------------------------------------------------- */}
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/" className="text-sm text-[#8b949e] hover:text-accent-blue">
          ← Generate
        </Link>
        <span className="text-[#484f58]">/</span>
        <span className="font-mono text-xs text-[#484f58]">
          {experiment.experiment_id.slice(0, 8)}…
        </span>
        <Badge variant="neutral">{experiment.mode}</Badge>
        <Badge variant="info">{experiment.model_name.split("/").pop()}</Badge>
        <span className="ml-auto text-xs text-[#484f58]">
          {formatDate(experiment.created_at)}
        </span>
        <Button
          variant="secondary"
          size="sm"
          onClick={() => router.push(`/compare?a=${experiment.experiment_id}`)}
        >
          Open in Compare
        </Button>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Prompt                                                             */}
      {/* ---------------------------------------------------------------- */}
      <div className="rounded-md border border-surface-border bg-surface-raised px-4 py-3">
        <p className="text-[10px] uppercase tracking-wider text-[#484f58]">Prompt</p>
        <p className="mt-1 font-mono text-sm text-[#8b949e] line-clamp-2">
          {experiment.prompt}
        </p>
      </div>

      {/* ---------------------------------------------------------------- */}
      {/* Stats strip                                                        */}
      {/* ---------------------------------------------------------------- */}
      {stats && (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
          {[
            { label: "Steps", value: experiment.total_steps },
            {
              label: "Avg entropy",
              value: stats.avgEntropy !== null ? stats.avgEntropy.toFixed(3) : "—",
              title: "Mean Shannon entropy H = -Σ p·log(p) across all steps",
            },
            {
              label: "Max entropy",
              value: stats.maxEntropy !== null ? stats.maxEntropy.toFixed(3) : "—",
              title: "Most uncertain decoding step",
            },
            {
              label: "Avg top-1 prob",
              value: stats.avgTopProb !== null ? formatPct(stats.avgTopProb, 1) : "—",
              title: "Mean probability of the most likely token (greedy confidence)",
            },
          ].map(({ label, value, title }) => (
            <div
              key={label}
              title={title}
              className="rounded-md border border-surface-border bg-surface-raised px-3 py-2"
            >
              <p className="text-[10px] uppercase tracking-wider text-[#484f58]">{label}</p>
              <p className="mt-0.5 font-mono text-lg font-semibold text-[#e6edf3]">
                {value}
              </p>
            </div>
          ))}
        </div>
      )}

      {/* ---------------------------------------------------------------- */}
      {/* Main split: code viewer left, decoding timeline right             */}
      {/* ---------------------------------------------------------------- */}
      <div className="grid gap-5 lg:grid-cols-[1fr_1fr]">
        <section className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
            Generated Output
          </h2>
          <CodeViewer
            code={experiment.generated_code || "// (no output)"}
            activeLine={activeLine}
            className="min-h-48 max-h-[68vh]"
          />
        </section>

        <section className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
            Decoding Timeline — {experiment.total_steps} token
            {experiment.total_steps !== 1 ? "s" : ""}
          </h2>
          <div className="max-h-[68vh] overflow-y-auto pr-1">
            <DecodingTimeline
              steps={experiment.steps}
              onStepSelect={setActiveLine}
            />
          </div>
        </section>
      </div>
    </div>
  );
}
