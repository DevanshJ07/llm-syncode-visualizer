"use client";

/**
 * useGeneration hook
 *
 * POST /generate returns the full decoding trace inline.
 * Empty or invalid traces are rejected — the hook surfaces HTTP 500 errors
 * from the backend and never enters "done" with zero steps.
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

      const result: ExperimentResult = {
        experiment_id: response.experiment_id,
        prompt: response.prompt,
        mode: response.mode,
        generated_code: response.generated_text,
        steps: response.steps,
        total_steps: response.total_steps,
        model_name: response.model_name,
        created_at: new Date().toISOString(),
      };

      // Final client-side guard — never show visualization with empty trace.
      if (result.steps.length === 0 || result.total_steps === 0) {
        throw new Error(
          "Backend returned HTTP 201 but decoding trace is empty. " +
            "This should not happen — check backend logs."
        );
      }

      setExperiment(result);
      setStatus("done");
      return response.experiment_id;
    } catch (err) {
      const message = err instanceof Error ? err.message : String(err);
      console.error("[useGeneration] failed:", message);
      setError(message);
      setExperiment(null);
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
