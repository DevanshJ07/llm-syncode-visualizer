/**
 * Shared utility functions.
 */

import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

/** Merge Tailwind class names, resolving conflicts correctly. */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}

/** Format a probability (0–1) as a percentage string. */
export function formatPct(p: number, decimals = 1): string {
  return `${(p * 100).toFixed(decimals)}%`;
}

/** Truncate a string to maxLen, adding ellipsis if needed. */
export function truncate(s: string, maxLen: number): string {
  return s.length > maxLen ? s.slice(0, maxLen) + "…" : s;
}

/** ISO timestamp → human readable "May 13, 2026 11:04 AM" */
export function formatDate(iso: string): string {
  return new Date(iso).toLocaleString(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  });
}
