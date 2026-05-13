"use client";

/**
 * DecodingTimeline — scrollable list of all decoding steps for an experiment.
 *
 * Renders one TokenStep card per generated token.
 * Tracks the active step so CodeViewer can highlight the corresponding line.
 *
 * TODO Phase 2: virtualise the list for experiments with 500+ steps.
 */

import { useState } from "react";

import { TokenStep } from "@/components/visualization/TokenStep";
import { Spinner } from "@/components/ui/Spinner";
import type { DecodingStep } from "@/types/decoding";

interface Props {
  steps: DecodingStep[];
  loading?: boolean;
  onStepSelect?: (stepIndex: number) => void;
}

export function DecodingTimeline({ steps, loading, onStepSelect }: Props) {
  const [activeStep, setActiveStep] = useState<number | null>(null);

  if (loading) {
    return (
      <div className="flex justify-center py-16">
        <Spinner label="Loading decoding steps…" />
      </div>
    );
  }

  if (steps.length === 0) {
    return (
      <div className="rounded-md border border-surface-border bg-surface p-8 text-center text-sm text-[#484f58]">
        No decoding steps to display.
        <br />
        Run a generation to see step-by-step token data.
      </div>
    );
  }

  const handleClick = (idx: number) => {
    setActiveStep(idx);
    onStepSelect?.(idx);
  };

  return (
    <div className="flex flex-col gap-2">
      <p className="text-xs text-[#484f58]">
        {steps.length} decoding step{steps.length !== 1 ? "s" : ""} — click to expand
      </p>
      {steps.map((step, i) => (
        <TokenStep
          key={step.step}
          step={step}
          isActive={activeStep === i}
          onClick={() => handleClick(i)}
        />
      ))}
    </div>
  );
}
