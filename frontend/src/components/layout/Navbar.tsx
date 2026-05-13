import Link from "next/link";

const NAV_LINKS = [
  { href: "/", label: "Generate" },
  { href: "/compare", label: "Compare" },
];

export function Navbar() {
  return (
    <nav className="border-b border-surface-border bg-surface-raised px-6 py-3">
      <div className="mx-auto flex max-w-7xl items-center justify-between">
        <Link href="/" className="flex items-center gap-2">
          <span className="text-lg font-bold tracking-tight text-accent-blue">
            SynViz
          </span>
          <span className="hidden text-sm text-[#8b949e] sm:inline">
            LLM Syncode Visualizer
          </span>
        </Link>

        <div className="flex items-center gap-6">
          {NAV_LINKS.map(({ href, label }) => (
            <Link
              key={href}
              href={href}
              className="text-sm text-[#8b949e] transition-colors hover:text-[#e6edf3]"
            >
              {label}
            </Link>
          ))}
        </div>
      </div>
    </nav>
  );
}
