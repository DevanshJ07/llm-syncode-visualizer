"use client";

/**
 * Experiment viewer page — Output Viewer + Token Visualization (Pages 2 & 3)
 *
 * URL: /experiment/[id]
 *
 * Layout:
 *   - top bar: experiment metadata
 *   - left panel: CodeViewer (generated code, clickable lines)
 *   - right panel: DecodingTimeline (step-by-step token data)
 *
 * Clicking a line in CodeViewer → scrolls to the corresponding step.
 * Clicking a step in DecodingTimeline → highlights the corresponding line.
 */

import { useEffect, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import Link from "next/link";

import { CodeViewer } from "@/components/output/CodeViewer";
import { DecodingTimeline } from "@/components/visualization/DecodingTimeline";
import { Badge } from "@/components/ui/Badge";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { getExperiment } from "@/lib/api";
import { formatDate } from "@/lib/utils";
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
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, [id]);

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
          Back to Generate
        </Button>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-6">
      {/* Breadcrumb / metadata */}
      <div className="flex flex-wrap items-center gap-3">
        <Link href="/" className="text-sm text-[#8b949e] hover:text-accent-blue">
          ← Generate
        </Link>
        <span className="text-[#484f58]">/</span>
        <span className="font-mono text-xs text-[#484f58]">{experiment.experiment_id}</span>
        <Badge variant={experiment.mode === "syncode" ? "valid" : "neutral"}>
          {experiment.mode}
        </Badge>
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

      {/* Prompt preview */}
      <div className="rounded-md border border-surface-border bg-surface-raised px-4 py-3">
        <p className="text-xs uppercase tracking-wider text-[#484f58]">Prompt</p>
        <p className="mt-1 font-mono text-sm text-[#8b949e] line-clamp-2">
          {experiment.prompt}
        </p>
      </div>

      {/* Main split layout */}
      <div className="grid gap-6 lg:grid-cols-2">
        <section className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
            Generated Code
          </h2>
          <CodeViewer
            code={experiment.generated_code || "/* No output yet */"}
            activeLine={activeLine}
            className="min-h-64 max-h-[70vh]"
          />
        </section>

        <section className="flex flex-col gap-2">
          <h2 className="text-xs font-semibold uppercase tracking-wider text-[#8b949e]">
            Decoding Steps ({experiment.total_steps})
          </h2>
          <div className="max-h-[70vh] overflow-y-auto pr-1">
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
