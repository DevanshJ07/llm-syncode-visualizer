/**
 * Home page — Prompt Interface (Page 1 from PROJECT_SPEC)
 *
 * Layout:
 *   - left: prompt form + settings
 *   - right: recent experiments list (placeholder)
 */

import { PromptForm } from "@/components/prompt/PromptForm";
import { Card } from "@/components/ui/Card";

export default function HomePage() {
  return (
    <div className="grid gap-8 lg:grid-cols-[2fr_1fr]">
      {/* Left: Prompt form */}
      <section className="flex flex-col gap-6">
        <div>
          <h1 className="text-2xl font-bold text-[#e6edf3]">
            Generate C Code
          </h1>
          <p className="mt-1 text-sm text-[#8b949e]">
            Enter a prompt to generate C code with Llama 3B. Toggle Syncode to
            apply grammar-constrained decoding and compare token distributions.
          </p>
        </div>

        <Card>
          <PromptForm />
        </Card>
      </section>

      {/* Right: Legend + recent experiments */}
      <aside className="flex flex-col gap-4">
        <Card title="Token Legend">
          <dl className="flex flex-col gap-2 text-sm">
            {[
              { color: "bg-token-selected", label: "Selected token" },
              { color: "bg-token-valid", label: "Valid (after Syncode)" },
              { color: "bg-token-masked", label: "Masked (grammar-invalid)" },
              { color: "bg-token-neutral", label: "Unselected candidate" },
            ].map(({ color, label }) => (
              <div key={label} className="flex items-center gap-2">
                <span className={`h-3 w-3 rounded-sm ${color}`} />
                <span className="text-[#8b949e]">{label}</span>
              </div>
            ))}
          </dl>
        </Card>

        <Card title="About">
          <p className="text-xs leading-relaxed text-[#8b949e]">
            SynViz visualises every autoregressive decoding step — including
            token probabilities, entropy, and Syncode grammar masking — so
            researchers can understand exactly how constrained decoding shapes
            C code generation.
          </p>
        </Card>
      </aside>
    </div>
  );
}
