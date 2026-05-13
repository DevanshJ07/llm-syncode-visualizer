/**
 * TypeScript mirrors of the Pydantic schemas in backend/app/models/schemas.py.
 * Keep these in sync whenever the backend schemas change.
 */

export interface TokenCandidate {
  token_id: number;
  token_str: string;
  probability: number;
  is_masked: boolean;
  is_selected: boolean;
}

export interface DecodingStep {
  step: number;
  context: string;
  top_tokens_before_syncode: TokenCandidate[];
  masked_tokens: number[];
  valid_tokens_after_syncode: TokenCandidate[];
  selected_token: string;
  entropy_before: number | null;
  entropy_after: number | null;
  num_masked: number;
}

export interface ExperimentResult {
  experiment_id: string;
  prompt: string;
  /** "raw" | "syncode" */
  mode: string;
  generated_code: string;
  steps: DecodingStep[];
  total_steps: number;
  model_name: string;
  created_at: string;
}

export interface GenerateRequest {
  prompt: string;
  use_syncode: boolean;
  top_k: number;
  max_new_tokens: number;
  temperature: number;
}

export interface GenerateResponse {
  experiment_id: string;
  status: string;
  message: string;
}

export interface StepResponse {
  step: DecodingStep;
  total_steps: number;
}

/** Convenience type for the compare view which holds two experiments side-by-side. */
export interface CompareState {
  raw: ExperimentResult | null;
  syncode: ExperimentResult | null;
}
