"use client";

/**
 * ComparePanel — side-by-side view of raw vs. Syncode-constrained generation.
 *
 * Each half renders:
 *   - generated code (CodeViewer)
 *   - aggregate stats (total steps, masked tokens, avg entropy)
 *
 * TODO Phase 2:
 *   - link step indices across both panels
 *   - diff-highlight tokens that differ between modes
 */

import { Card } from "@/components/ui/Card";
import { Badge } from "@/components/ui/Badge";
import { CodeViewer } from "@/components/output/CodeViewer";
import type { ExperimentResult } from "@/types/decoding";

interface PanelHalfProps {
  label: string;
  experiment: ExperimentResult | null;
}

function PanelHalf({ label, experiment }: PanelHalfProps) {
  if (!experiment) {
    return (
      <Card title={label} className="flex-1">
        <div className="flex h-64 items-center justify-center text-sm text-[#484f58]">
          No experiment loaded
        </div>
      </Card>
    );
  }

  const totalMasked = experiment.steps.reduce((s, st) => s + st.num_masked, 0);
  const avgEntropy =
    experiment.steps.length > 0
      ? experiment.steps.reduce((s, st) => s + (st.entropy_before ?? 0), 0) /
        experiment.steps.length
      : 0;

  return (
    <Card
      title={label}
      action={
        <Badge variant={experiment.mode === "syncode" ? "valid" : "neutral"}>
          {experiment.mode}
        </Badge>
      }
      className="flex-1"
    >
      <div className="flex flex-col gap-4">
        {/* Stats row */}
        <div className="flex gap-4 text-xs text-[#8b949e]">
          <span>{experiment.total_steps} steps</span>
          {experiment.mode === "syncode" && (
            <span className="text-token-masked">{totalMasked} total masked</span>
          )}
          <span>avg H = {avgEntropy.toFixed(3)}</span>
        </div>

        <CodeViewer code={experiment.generated_code} className="max-h-96" />
      </div>
    </Card>
  );
}

interface ComparePanelProps {
  raw: ExperimentResult | null;
  syncode: ExperimentResult | null;
}

export function ComparePanel({ raw, syncode }: ComparePanelProps) {
  return (
    <div className="flex flex-col gap-4 lg:flex-row">
      <PanelHalf label="Raw LLaMA" experiment={raw} />
      <PanelHalf label="Syncode Constrained" experiment={syncode} />
    </div>
  );
}
