SynViz — LLM Constrained Decoding Visualizer

SynViz is an interactive research-oriented platform for exploring how Large Language Models generate text at the token level under grammar-constrained decoding.

The platform visualizes the complete generation process step-by-step, allowing users to inspect how Syncode grammar constraints modify token probabilities, suppress invalid continuations, and reshape the decoding trajectory in real time.

Key Features
- Token-by-token generation tracing
- Raw vs Syncode-constrained decoding comparison
- Grammar-invalid token masking visualization
- Entropy tracking across decoding steps
- Token probability distribution inspection
- Interactive decoding timeline
- Constrained generation for C code synthesis
- FastAPI + React based research interface

Why This Project?
Most LLM interfaces only show the final generated output.

SynViz focuses on what happens between generated tokens.

By exposing probability distributions, masked tokens, entropy evolution, and parser-aware constraints, the platform helps researchers, students, and engineers better understand how constrained decoding influences model behavior during inference.

- Technology Stack
- Backend
- Python
- FastAPI
- Hugging Face Transformers
- Syncode
- Pydantic
- Frontend
- Next.js
- React
- TypeScript
- Tailwind CSS
- Recharts
- Research Concepts
- Large Language Models (LLMs)
- Constrained Decoding
- Formal Grammars
- Incremental Parsing
- Token Probability Analysis
- Entropy-Based Uncertainty Measurement
-AI Interpretability
---

## Project Structure

```
llm-syncode-visualizer/
├── backend/                    # FastAPI server
│   ├── main.py                 # App entry point + lifespan hooks
│   ├── requirements.txt
│   ├── .env.example
│   └── app/
│       ├── core/
│       │   └── config.py       # Settings (pydantic-settings)
│       ├── models/
│       │   └── schemas.py      # Pydantic request/response/data models
│       ├── api/
│       │   └── routes/
│       │       ├── generate.py     # POST /generate
│       │       └── experiments.py  # GET /experiment/{id}, GET /experiments
│       └── services/
│           ├── llm_service.py      # HuggingFace model wrapper
│           ├── syncode_service.py  # Syncode grammar-mask adapter
│           └── experiment_store.py # JSON-file persistence layer
│
├── frontend/                   # Next.js 14 app
│   ├── package.json
│   ├── next.config.ts          # API proxy rewrite → FastAPI
│   ├── tailwind.config.ts
│   └── src/
│       ├── app/
│       │   ├── layout.tsx
│       │   ├── page.tsx                    # Prompt Interface
│       │   ├── experiment/[id]/page.tsx    # Output + Token Viz
│       │   └── compare/page.tsx            # Compare View
│       ├── components/
│       │   ├── ui/             # Button, Card, Badge, Spinner
│       │   ├── layout/         # Navbar
│       │   ├── prompt/         # PromptForm, GenerationSettings
│       │   ├── visualization/  # DecodingTimeline, TokenStep, TokenProbabilityChart
│       │   ├── output/         # CodeViewer
│       │   └── compare/        # ComparePanel
│       ├── hooks/
│       │   └── useGeneration.ts
│       ├── lib/
│       │   ├── api.ts          # All fetch() wrappers
│       │   └── utils.ts
│       └── types/
│           └── decoding.ts     # TypeScript mirrors of Pydantic schemas
│
├── logs/                       # Experiment JSON files written here
├── docs/                       # Research notes, diagrams
└── PROJECT_SPEC.md
```

---

## Quick Start

### 1 — Backend

```bash
cd backend

# Create virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Syncode from source (requires git)
pip install git+https://github.com/uiuc-focal-lab/syncode.git

# Configure environment
cp .env.example .env
# edit .env: set MODEL_NAME, DEVICE, etc.

# Run
uvicorn main:app --reload --port 8000
```

API docs available at: http://localhost:8000/docs

### 2 — Frontend

```bash
cd frontend

# Install dependencies
npm install

# Configure environment
cp .env.local.example .env.local

# Run dev server
npm run dev
```

App available at: http://localhost:3000

---

## API Reference

| Method | Path | Description |
|---|---|---|
| `POST` | `/generate` | Submit a prompt; returns `experiment_id` |
| `GET` | `/experiment/{id}` | Full experiment with all decoding steps |
| `GET` | `/experiment/{id}/steps/{step}` | Single decoding step (1-indexed) |
| `GET` | `/experiments` | List all experiment IDs |
| `GET` | `/health` | Health check |

---

## JSON Log Format

Each decoding step is stored as:

```json
{
  "step": 1,
  "context": "void reverse(char *s",
  "top_tokens_before_syncode": [
    { "token_id": 12, "token_str": ")", "probability": 0.72, "is_masked": false, "is_selected": true }
  ],
  "masked_tokens": [342, 891],
  "valid_tokens_after_syncode": [...],
  "selected_token": ")",
  "entropy_before": 1.23,
  "entropy_after": 0.45,
  "num_masked": 2
}
```

---

## Implementation Priorities (Next Steps)

### Phase 2 — Core Inference (Backend)
1. Implement token-by-token generation loop in `llm_service.py` with logit capture
2. Plug `syncode_service.py` into the HuggingFace logits-processor hook
3. Capture `entropy_before` / `entropy_after` per step
4. Write populated `DecodingStep` objects to experiment store

### Phase 3 — Rich Visualization (Frontend)
1. Syntax highlighting in `CodeViewer` (shiki or prism-react-renderer)
2. Grouped before/after bars in `TokenProbabilityChart`
3. Entropy timeline chart across all steps (Recharts LineChart)
4. Step ↔ code line synchronisation
5. React Flow grammar tree visualisation

### Phase 4 — Polish
1. Streaming generation (Server-Sent Events)
2. Export experiment as JSON / CSV
3. Persistent experiment browser with search
4. Diff highlighting in compare view

---

## Notes for Researchers

- **Logging correctness > aesthetics** — the JSON experiment files are the primary
  research artifact; treat them as the ground truth.
- Syncode must be installed from source; it is not on PyPI yet.
- The model requires a CUDA GPU with ≥8 GB VRAM for Llama 3B in fp16.
  Set `DEVICE=cpu` for CPU-only (slow) testing.
- During UI development without a GPU, comment out `llm_service.load_model()`
  in `main.py` and mock experiments by placing hand-crafted JSON in `logs/experiments/`.
