/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,

  /**
   * Proxy all /api/* requests to the FastAPI backend during development.
   * In production replace this with an nginx/caddy reverse-proxy rule.
   *
   * BACKEND_URL defaults to http://localhost:8000 so you can override it in
   * .env.local without touching this file.
   */
  async rewrites() {
    const backendUrl = process.env.BACKEND_URL ?? "http://localhost:8000";
    return [
      {
        source: "/api/:path*",
        destination: `${backendUrl}/:path*`,
      },
    ];
  },
};

export default nextConfig;
