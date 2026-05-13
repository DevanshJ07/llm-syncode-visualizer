import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        mono: ["JetBrains Mono", "Fira Code", "monospace"],
        sans: ["Inter", "system-ui", "sans-serif"],
      },
      colors: {
        // Research-tool palette: dark neutral base + accent colours
        surface: {
          DEFAULT: "#0f1117",
          raised: "#161b22",
          border: "#21262d",
        },
        accent: {
          blue: "#58a6ff",
          green: "#3fb950",
          red: "#f85149",
          yellow: "#d29922",
          purple: "#bc8cff",
        },
        token: {
          masked: "#f85149",    // invalid / masked tokens
          valid: "#3fb950",     // valid tokens after Syncode
          selected: "#58a6ff",  // the token that was chosen
          neutral: "#8b949e",   // unranked candidates
        },
      },
    },
  },
  plugins: [],
};

export default config;
