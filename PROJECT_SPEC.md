# LLM Syncode Visualizer  
  
## Objective  
  
Build an interactive research platform for visualizing token-level generation of Llama 3B with and without Syncode constrained decoding for C code generation.  
  
The system should allow researchers to:  
- enter prompts,  
- generate C code,  
- inspect token probabilities at each decoding step,  
- compare raw LLM output vs Syncode-constrained output,  
- visualize masked/invalid tokens,  
- inspect grammar constraint effects.  
  
---  
  
# Core Workflow  
  
User Prompt  
→ Llama 3B Generation  
→ At every decoding step:  
 - top-k token candidates  
 - token probabilities  
 - entropy  
→ Syncode masking  
 - valid tokens  
 - invalid tokens  
 - masked tokens  
→ final selected token  
→ generated code  
→ store all intermediate decoding information as structured JSON  
→ frontend visualization layer  
  
---  
  
# Tech Stack  
  
Frontend:  
- Next.js  
- TailwindCSS  
- TypeScript  
- Recharts  
- React Flow  
  
Backend:  
- FastAPI  
- Python  
- HuggingFace Transformers  
  
Model:  
- Llama 3B  
  
Visualization Goals:  
- token probability charts  
- masked token highlighting  
- step-by-step decoding timeline  
- interactive code exploration  
- compare mode  
  
---  
  
# Required Pages  
  
## 1. Prompt Interface  
- prompt input  
- generation controls  
- syncode toggle  
- generation settings  
  
## 2. Output Viewer  
- syntax highlighted C code  
- clickable lines  
  
## 3. Token Visualization  
For each decoding step:  
- candidate tokens  
- probabilities  
- masked status  
- selected token  
- before/after syncode distributions  
  
## 4. Compare View  
- raw llama generation  
- syncode generation  
  
---  
  
# Required Backend APIs  
  
POST /generate  
GET /experiment/{id}  
GET /experiment/{id}/steps/{step}  
  
---  
  
# JSON Logging Format  
  
{  
 "step": 1,  
 "context": "",  
 "top_tokens_before_syncode": [],  
 "masked_tokens": [],  
 "valid_tokens_after_syncode": [],  
 "selected_token": ""  
}  
  
---  
  
# Folder Structure  
  
frontend/  
backend/  
logs/  
docs/  
  
---  
  
# Priority  
  
Backend logging correctness is more important than frontend beauty.  
  
The platform should prioritize research visualization over aesthetic design.