"use client";

/**
 * Compare page — Page 4 from PROJECT_SPEC
 *
 * URL: /compare?a={experimentId}&b={experimentId}
 *
 * Allows researchers to load two experiments (raw vs. Syncode) side-by-side.
 * Pre-populates from URL query params when navigated from the experiment viewer.
 */

import { useEffect, useState } from "react";
import { useSearchParams } from "next/navigation";

import { ComparePanel } from "@/components/compare/ComparePanel";
import { Button } from "@/components/ui/Button";
import { Spinner } from "@/components/ui/Spinner";
import { getExperiment, listExperiments } from "@/lib/api";
import type { ExperimentResult } from "@/types/decoding";

export default function ComparePage() {
  const searchParams = useSearchParams();

  const [idA, setIdA] = useState(searchParams.get("a") ?? "");
  const [idB, setIdB] = useState(searchParams.get("b") ?? "");

  const [expA, setExpA] = useState<ExperimentResult | null>(null);
  const [expB, setExpB] = useState<ExperimentResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [recentIds, setRecentIds] = useState<string[]>([]);

  // Fetch recent experiment IDs for the dropdowns
  useEffect(() => {
    listExperiments()
      .then(setRecentIds)
      .catch(() => {});
  }, []);

  // Auto-load if query params provided
  useEffect(() => {
    if (idA) loadExperiment(idA, "a");
    if (idB) loadExperiment(idB, "b");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const loadExperiment = async (id: string, slot: "a" | "b") => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const exp = await getExperiment(id);
      if (slot === "a") setExpA(exp);
      else setExpB(exp);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div>
        <h1 className="text-2xl font-bold text-[#e6edf3]">Compare Experiments</h1>
        <p className="mt-1 text-sm text-[#8b949e]">
          Load two experiments side-by-side to compare raw vs. Syncode generation.
        </p>
      </div>

      {/* ID inputs */}
      <div className="grid gap-4 sm:grid-cols-2">
        {(["a", "b"] as const).map((slot) => {
          const id = slot === "a" ? idA : idB;
          const setId = slot === "a" ? setIdA : setIdB;
          const label = slot === "a" ? "Experiment A" : "Experiment B";
          return (
            <div key={slot} className="flex flex-col gap-1.5">
              <label className="text-xs font-medium uppercase tracking-wider text-[#8b949e]">
                {label}
              </label>
              <div className="flex gap-2">
                {recentIds.length > 0 ? (
                  <select
                    value={id}
                    onChange={(e) => setId(e.target.value)}
                    className="flex-1 rounded-md border border-surface-border bg-surface px-3 py-2 text-sm text-[#e6edf3] focus:border-accent-blue focus:outline-none"
                  >
                    <option value="">— select experiment —</option>
                    {recentIds.map((eid) => (
                      <option key={eid} value={eid}>
                        {eid.slice(0, 8)}…
                      </option>
                    ))}
                  </select>
                ) : (
                  <input
                    type="text"
                    value={id}
                    onChange={(e) => setId(e.target.value)}
                    placeholder="Paste experiment ID…"
                    className="flex-1 rounded-md border border-surface-border bg-surface px-3 py-2 font-mono text-xs text-[#e6edf3] focus:border-accent-blue focus:outline-none"
                  />
                )}
                <Button
                  variant="secondary"
                  size="sm"
                  onClick={() => loadExperiment(id, slot)}
                  disabled={!id || loading}
                >
                  Load
                </Button>
              </div>
            </div>
          );
        })}
      </div>

      {error && (
        <p className="rounded border border-accent-red/30 bg-red-900/20 px-3 py-2 text-sm text-accent-red">
          {error}
        </p>
      )}

      {loading ? (
        <div className="flex justify-center py-16">
          <Spinner label="Loading experiment…" />
        </div>
      ) : (
        <ComparePanel raw={expA} syncode={expB} />
      )}
    </div>
  );
}
