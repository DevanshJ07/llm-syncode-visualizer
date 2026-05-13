"use client";

/**
 * StepPlayer — step navigation slider + autoplay controls.
 *
 * Controls:
 *   |◀  go to first step
 *   ◀   previous step
 *   ▶/⏸ play/pause autoplay
 *   ▶   next step
 *   ▶|  go to last step
 *   speed selector (ms per step)
 *   range slider for direct scrubbing
 */

import { useEffect, useRef } from "react";
import { cn } from "@/lib/utils";

interface Props {
  totalSteps: number;
  currentStep: number;          // 0-indexed
  isPlaying: boolean;
  onStepChange: (idx: number) => void;
  onPlayPause: () => void;
  playIntervalMs?: number;
  onIntervalChange?: (ms: number) => void;
}

const SPEED_OPTIONS = [
  { label: "0.5×", ms: 2000 },
  { label: "1×",   ms: 1000 },
  { label: "2×",   ms: 500  },
  { label: "4×",   ms: 250  },
];

function Btn({
  onClick,
  disabled,
  title,
  children,
}: {
  onClick: () => void;
  disabled?: boolean;
  title?: string;
  children: React.ReactNode;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cn(
        "flex h-7 w-7 items-center justify-center rounded text-sm transition-colors",
        "bg-surface-raised border border-surface-border text-[#8b949e]",
        "hover:border-[#484f58] hover:text-[#e6edf3]",
        "disabled:opacity-30 disabled:pointer-events-none"
      )}
    >
      {children}
    </button>
  );
}

export function StepPlayer({
  totalSteps,
  currentStep,
  isPlaying,
  onStepChange,
  onPlayPause,
  playIntervalMs = 1000,
  onIntervalChange,
}: Props) {
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Autoplay: advance one step every playIntervalMs when playing.
  // Stops automatically at the last step.
  useEffect(() => {
    if (intervalRef.current) clearInterval(intervalRef.current);
    if (!isPlaying) return;

    intervalRef.current = setInterval(() => {
      onStepChange(Math.min(currentStep + 1, totalSteps - 1));
      // Parent stops playing when currentStep reaches totalSteps - 1.
    }, playIntervalMs);

    return () => {
      if (intervalRef.current) clearInterval(intervalRef.current);
    };
  }, [isPlaying, currentStep, totalSteps, playIntervalMs, onStepChange]);

  const atStart = currentStep === 0;
  const atEnd = currentStep === totalSteps - 1;

  return (
    <div className="flex flex-col gap-2 rounded-md border border-surface-border bg-surface-raised px-4 py-3">
      {/* Top row: buttons + step counter + speed */}
      <div className="flex items-center gap-2">
        {/* Transport buttons */}
        <Btn onClick={() => onStepChange(0)} disabled={atStart} title="First step">
          ⏮
        </Btn>
        <Btn onClick={() => onStepChange(currentStep - 1)} disabled={atStart} title="Previous step">
          ◀
        </Btn>
        <button
          type="button"
          onClick={onPlayPause}
          title={isPlaying ? "Pause" : "Play"}
          className={cn(
            "flex h-8 w-8 items-center justify-center rounded-full text-sm font-bold transition-colors",
            isPlaying
              ? "bg-accent-blue text-surface hover:bg-blue-400"
              : "bg-surface-raised border border-surface-border text-[#8b949e] hover:border-accent-blue hover:text-accent-blue"
          )}
        >
          {isPlaying ? "⏸" : "▶"}
        </button>
        <Btn onClick={() => onStepChange(currentStep + 1)} disabled={atEnd} title="Next step">
          ▶
        </Btn>
        <Btn onClick={() => onStepChange(totalSteps - 1)} disabled={atEnd} title="Last step">
          ⏭
        </Btn>

        {/* Step counter */}
        <span className="ml-2 font-mono text-xs text-[#8b949e]">
          step{" "}
          <span className="text-[#e6edf3]">{currentStep + 1}</span>
          {" / "}
          <span className="text-[#484f58]">{totalSteps}</span>
        </span>

        {/* Speed selector */}
        {onIntervalChange && (
          <div className="ml-auto flex items-center gap-1">
            <span className="text-[10px] text-[#484f58]">speed</span>
            {SPEED_OPTIONS.map((opt) => (
              <button
                key={opt.ms}
                type="button"
                onClick={() => onIntervalChange(opt.ms)}
                className={cn(
                  "rounded px-1.5 py-0.5 text-[10px] font-mono transition-colors",
                  playIntervalMs === opt.ms
                    ? "bg-accent-blue/20 text-accent-blue"
                    : "text-[#484f58] hover:text-[#8b949e]"
                )}
              >
                {opt.label}
              </button>
            ))}
          </div>
        )}
      </div>

      {/* Range slider */}
      <input
        type="range"
        min={0}
        max={totalSteps - 1}
        value={currentStep}
        onChange={(e) => onStepChange(Number(e.target.value))}
        className="w-full accent-accent-blue"
      />
    </div>
  );
}
