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
const DEBUG_API = process.env.NODE_ENV === "development";

/** Parse FastAPI error bodies into a human-readable string. */
export function formatApiError(status: number, body: string): string {
  try {
    const parsed = JSON.parse(body) as { detail?: unknown };
    const detail = parsed.detail;
    if (typeof detail === "string") {
      return `API ${status}: ${detail}`;
    }
    if (detail && typeof detail === "object") {
      const d = detail as Record<string, unknown>;
      const message =
        typeof d.message === "string"
          ? d.message
          : typeof d.error === "string"
            ? d.error
            : "Generation failed";
      const reasons = Array.isArray(d.reasons)
        ? (d.reasons as string[]).join("; ")
        : "";
      const genId =
        typeof d.generation_id === "string" ? ` [gen=${d.generation_id}]` : "";
      return `API ${status}: ${message}${reasons ? ` — ${reasons}` : ""}${genId}`;
    }
  } catch {
    // body is not JSON — use raw text
  }
  return `API ${status}: ${body.slice(0, 500)}`;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  if (DEBUG_API) {
    console.debug("[API request]", init?.method ?? "GET", path);
  }

  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });

  const bodyText = await res.text();

  if (DEBUG_API) {
    console.debug(
      "[API response]",
      path,
      "status=",
      res.status,
      "bytes=",
      bodyText.length
    );
  }

  if (!res.ok) {
    const message = formatApiError(res.status, bodyText);
    console.error("[API error]", path, message);
    throw new Error(message);
  }

  try {
    const data = JSON.parse(bodyText) as T;
    if (DEBUG_API && path === "/generate") {
      const preview = bodyText.slice(0, 1500);
      console.debug(
        "[API /generate payload preview]",
        preview + (bodyText.length > 1500 ? "…" : "")
      );
    }
    return data;
  } catch (parseErr) {
    console.error("[API JSON parse error]", path, parseErr);
    throw new Error(
      `API ${res.status}: response is not valid JSON (${String(parseErr)})`
    );
  }
}

/** Validate a successful /generate response — throws if trace is empty. */
export function assertValidGenerateResponse(response: GenerateResponse): void {
  const issues: string[] = [];
  if (!response.steps || response.steps.length === 0) {
    issues.push("response.steps is empty");
  }
  if (response.total_steps <= 0) {
    issues.push(`response.total_steps=${response.total_steps}`);
  }
  if (!response.generated_text || !response.generated_text.trim()) {
    issues.push("response.generated_text is empty");
  }
  if (response.status === "error") {
    issues.push(
      `response.status=error: ${response.message || "no message"}`
    );
  }
  if (issues.length > 0) {
    throw new Error(
      `Invalid generate response (${issues.join("; ")}). ` +
        `experiment_id=${response.experiment_id}`
    );
  }
}

// ---------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------

export async function postGenerate(
  payload: GenerateRequest
): Promise<GenerateResponse> {
  if (DEBUG_API) {
    console.debug("[API postGenerate] request", {
      prompt_len: payload.prompt.length,
      use_syncode: payload.use_syncode,
      max_new_tokens: payload.max_new_tokens,
    });
  }
  const response = await request<GenerateResponse>("/generate", {
    method: "POST",
    body: JSON.stringify(payload),
  });
  assertValidGenerateResponse(response);
  if (DEBUG_API) {
    console.debug("[API postGenerate] validated", {
      experiment_id: response.experiment_id,
      total_steps: response.total_steps,
      generated_text_len: response.generated_text.length,
    });
  }
  return response;
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
