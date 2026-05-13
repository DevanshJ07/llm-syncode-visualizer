"use client";

/**
 * useGeneration hook
 *
 * POST /generate now returns the full decoding trace inline, so there is
 * no longer a second GET /experiment/{id} round-trip.
 *
 * The hook converts the GenerateResponse into an ExperimentResult shape so
 * the rest of the UI (experiment page, timeline, charts) stays unchanged.
 */

import { useState, useCallback } from "react";

import { postGenerate } from "@/lib/api";
import type { ExperimentResult, GenerateRequest } from "@/types/decoding";

type GenerationStatus = "idle" | "generating" | "done" | "error";

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
      const response = await postGenerate(request);

      // Map the inline GenerateResponse onto the ExperimentResult shape
      // that the experiment viewer page and visualization components expect.
      const result: ExperimentResult = {
        experiment_id: response.experiment_id,
        prompt: response.prompt,
        mode: response.mode,
        // generated_text from response maps to generated_code on ExperimentResult
        generated_code: response.generated_text,
        steps: response.steps,
        total_steps: response.total_steps,
        model_name: response.model_name,
        created_at: new Date().toISOString(),
      };

      setExperiment(result);
      setStatus("done");
      return response.experiment_id;
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
