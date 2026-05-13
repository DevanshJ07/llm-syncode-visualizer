"use client";

/**
 * CodeViewer — displays syntax-highlighted C code with clickable line numbers.
 *
 * Clicking a line will call onLineClick(lineIndex) so parent components can
 * link lines to the corresponding decoding step in the visualization panel.
 *
 * Phase 2: integrate a proper syntax highlighter (e.g. shiki or prism-react-renderer).
 * For now we display plain monospace with basic token colouring.
 */

import { useState } from "react";
import { cn } from "@/lib/utils";

interface CodeViewerProps {
  code: string;
  /** Zero-indexed line number that is currently highlighted */
  activeLine?: number;
  onLineClick?: (lineIndex: number) => void;
  className?: string;
}

export function CodeViewer({
  code,
  activeLine,
  onLineClick,
  className,
}: CodeViewerProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  const lines = code.split("\n");

  return (
    <div
      className={cn(
        "code-block overflow-auto rounded-md border border-surface-border bg-surface",
        className
      )}
    >
      <table className="w-full border-collapse text-sm">
        <tbody>
          {lines.map((line, i) => (
            <tr
              key={i}
              onClick={() => onLineClick?.(i)}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
              className={cn(
                "group cursor-pointer transition-colors",
                activeLine === i && "bg-accent-blue/10",
                hovered === i && activeLine !== i && "bg-surface-raised"
              )}
            >
              {/* Line number gutter */}
              <td className="w-12 select-none border-r border-surface-border px-3 py-0.5 text-right text-[#484f58] group-hover:text-[#8b949e]">
                {i + 1}
              </td>
              {/* Code */}
              <td className="px-4 py-0.5 text-[#e6edf3] whitespace-pre">
                {line || " "}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
