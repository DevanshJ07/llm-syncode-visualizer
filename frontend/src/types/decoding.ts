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

/** Extended candidate model for Syncode before/after distributions. */
export interface TokenCandidate {
  token_id: number;
  token_str: string;
  probability: number;
  is_masked: boolean;
  is_selected: boolean;
}

/** A token rejected by Syncode grammar masking, carrying its raw probability. */
export interface MaskedTokenEntry {
  token: string;
  token_id: number;
  raw_prob: number;
}

export interface DecodingStep {
  step: number;
  /** Decoded text generated before this step (context fed into the model). */
  context: string;

  // --- Real generation fields ---
  /** Top-k candidates ranked by probability (after temperature scaling). */
  top_tokens: TopToken[];
  /** The token chosen by greedy decoding (argmax). */
  selected_token: string;
  /** Vocabulary index of the selected token. */
  selected_token_id: number;
  /** Shannon entropy H = -Σ p·log(p) over the full vocabulary distribution. */
  entropy_before: number | null;

  // --- Syncode fields ---
  top_tokens_before_syncode: TokenCandidate[];
  /** Rejected tokens with their raw probabilities (Syncode mode only). */
  masked_tokens: MaskedTokenEntry[];
  valid_tokens_after_syncode: TokenCandidate[];
  entropy_after: number | null;
  num_masked: number;

  // --- Syncode masking statistics per step ---
  vocab_size: number;
  valid_token_count: number;
  masked_token_count: number;
  masked_percentage: number;
  probability_mass_removed: number;

  // --- Parser recovery metadata ---
  /** True when the Syncode grammar parser threw at this step. */
  parser_error: boolean;
  /** Description of the parser exception (empty string when no error). */
  parser_error_message: string;
  /** True when raw logits were used because Syncode masking failed/was unavailable. */
  fallback_used: boolean;
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

/**
 * POST /generate response — full decoding trace returned inline.
 * The experiment is also persisted; use experiment_id with GET /experiment/{id}
 * if you need to retrieve it later.
 */
export interface GenerateResponse {
  // identity
  experiment_id: string;
  status: string;
  message: string;
  // generated output
  generated_text: string;
  model_name: string;
  mode: string;
  prompt: string;
  total_steps: number;
  // full decoding trace — one entry per generated token
  steps: DecodingStep[];
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
