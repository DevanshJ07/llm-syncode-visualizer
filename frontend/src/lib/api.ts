/**
 * API client layer.
 *
 * All calls go through the Next.js rewrite proxy (/api → FastAPI).
 * Components and hooks should import from here, never fetch() directly.
 */

import type {
  ExperimentResult,
  GenerateRequest,
  GenerateResponse,
  StepResponse,
} from "@/types/decoding";

const BASE = "/api";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }

  return res.json() as Promise<T>;
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

export async function postGenerate(
  payload: GenerateRequest
): Promise<GenerateResponse> {
  return request<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
}

// ---------------------------------------------------------------------------
// Experiments
// ---------------------------------------------------------------------------

export async function getExperiment(id: string): Promise<ExperimentResult> {
  return request<ExperimentResult>(`/experiment/${id}`);
}

export async function getExperimentStep(
  id: string,
  step: number
): Promise<StepResponse> {
  return request<StepResponse>(`/experiment/${id}/steps/${step}`);
}

export async function listExperiments(): Promise<string[]> {
  return request<string[]>("/experiments");
}

// ---------------------------------------------------------------------------
// Health
// ---------------------------------------------------------------------------

export async function getHealth(): Promise<{ status: string }> {
  return request<{ status: string }>("/health");
}
