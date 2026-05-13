/**
 * useGeneration hook
 *
 * Manages the full lifecycle of a generation request:
 *   1. POST /generate → get experiment_id
 *   2. GET /experiment/{id} → get full result
 *
 * Used by the PromptForm component to trigger generation and
 * route the user to the experiment viewer on completion.
 */

"use client";

import { useState, useCallback } from "react";

import { postGenerate, getExperiment } from "@/lib/api";
import type { ExperimentResult, GenerateRequest } from "@/types/decoding";

type GenerationStatus = "idle" | "generating" | "fetching" | "done" | "error";

interface UseGenerationReturn {
  status: GenerationStatus;
  experiment: ExperimentResult | null;
  error: string | null;
  generate: (request: GenerateRequest) => Promise<string | null>;
  reset: () => void;
}

export function useGeneration(): UseGenerationReturn {
  const [status, setStatus] = useState<GenerationStatus>("idle");
  const [experiment, setExperiment] = useState<ExperimentResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const generate = useCallback(async (request: GenerateRequest) => {
    setStatus("generating");
    setError(null);
    setExperiment(null);

    try {
      const { experiment_id } = await postGenerate(request);

      setStatus("fetching");
      const result = await getExperiment(experiment_id);

      setExperiment(result);
      setStatus("done");
      return experiment_id;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      setError(message);
      setStatus("error");
      return null;
    }
  }, []);

  const reset = useCallback(() => {
    setStatus("idle");
    setExperiment(null);
    setError(null);
  }, []);

  return { status, experiment, error, generate, reset };
}
