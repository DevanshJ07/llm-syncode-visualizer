import { cn } from "@/lib/utils";
import { HTMLAttributes } from "react";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  title?: string;
  action?: React.ReactNode;
}

export function Card({ title, action, className, children, ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-lg border border-surface-border bg-surface-raised",
        className
      )}
      {...props}
    >
      {(title || action) && (
        <div className="flex items-center justify-between border-b border-surface-border px-4 py-3">
          {title && (
            <h3 className="text-sm font-semibold text-[#e6edf3]">{title}</h3>
          )}
          {action && <div>{action}</div>}
        </div>
      )}
      <div className="p-4">{children}</div>
    </div>
  );
}
