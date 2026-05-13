import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  // Forward all /api/* calls to the FastAPI backend during development.
  // In production, handle this at the reverse-proxy layer (nginx/caddy).
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${process.env.BACKEND_URL ?? "http://localhost:8000"}/:path*`,
      },
    ];
  },
};

export default nextConfig;
