/**
 * TypeScript mirrors of the Pydantic schemas in backend/app/models/schemas.py.
 * Keep these in sync whenever the backend schemas change.
 */

// ---------------------------------------------------------------------------
// Core decoding data
// ---------------------------------------------------------------------------

/**
 * One candidate token at a decoding step.
 * Matches the JSON logging format from PROJECT_SPEC:
 *   { "token": "main", "probability": 0.42, "token_id": 1234 }
 */
export interface TopToken {
  token: string;       // decoded string (may contain spaces, newlines, special chars)
  probability: number; // softmax probability after temperature scaling [0, 1]
  token_id: number;    // vocabulary index
}

/** Legacy candidate model kept for future Syncode phase. */
export interface TokenCandidate {
  token_id: number;
  token_str: string;
  probability: number;
  is_masked: boolean;
  is_selected: boolean;
}

export interface DecodingStep {
  step: number;
  /** Decoded text generated before this step (context fed into the model). */
  context: string;

  // --- Real generation fields (Phase 2) ---
  /** Top-k candidates ranked by probability (after temperature scaling). */
  top_tokens: TopToken[];
  /** The token chosen by greedy decoding (argmax). */
  selected_token: string;
  /** Vocabulary index of the selected token. */
  selected_token_id: number;
  /** Shannon entropy H = -Σ p·log(p) over the full vocabulary distribution. */
  entropy_before: number | null;

  // --- Syncode fields (Phase 3, empty until Syncode is implemented) ---
  top_tokens_before_syncode: TokenCandidate[];
  masked_tokens: number[];
  valid_tokens_after_syncode: TokenCandidate[];
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

// ---------------------------------------------------------------------------
// API request / response shapes
// ---------------------------------------------------------------------------

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

/** Convenience type for the compare view. */
export interface CompareState {
  raw: ExperimentResult | null;
  syncode: ExperimentResult | null;
}
