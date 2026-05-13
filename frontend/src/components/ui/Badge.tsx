import { cn } from "@/lib/utils";

type BadgeVariant = "masked" | "valid" | "selected" | "neutral" | "info";

interface BadgeProps {
  variant?: BadgeVariant;
  className?: string;
  children: React.ReactNode;
}

const variantClasses: Record<BadgeVariant, string> = {
  masked: "bg-red-900/40 text-token-masked border-token-masked/40",
  valid: "bg-green-900/40 text-token-valid border-token-valid/40",
  selected: "bg-blue-900/40 text-token-selected border-token-selected/40",
  neutral: "bg-[#21262d] text-token-neutral border-[#30363d]",
  info: "bg-purple-900/40 text-accent-purple border-accent-purple/40",
};

export function Badge({ variant = "neutral", className, children }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded border px-1.5 py-0.5 font-mono text-xs font-medium",
        variantClasses[variant],
        className
      )}
    >
      {children}
    </span>
  );
}
