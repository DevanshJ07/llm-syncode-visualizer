import Link from "next/link";

export default function NotFound() {
  return (
    <div className="flex flex-col items-center gap-4 py-32 text-center">
      <h2 className="text-4xl font-bold text-[#484f58]">404</h2>
      <p className="text-[#8b949e]">Page not found.</p>
      <Link
        href="/"
        className="text-sm text-accent-blue hover:underline"
      >
        Back to Generate
      </Link>
    </div>
  );
}
