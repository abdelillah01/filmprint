/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./app/**/*.{js,ts,jsx,tsx}",
    "./components/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "var(--bg)",
        card: "var(--card)",
        raised: "var(--raised)",
        line: "var(--line)",
        "line-strong": "var(--line-strong)",
        ink: {
          DEFAULT: "var(--text)",
          2: "var(--text-2)",
          muted: "var(--muted)",
        },
        accent: {
          DEFAULT: "var(--accent)",
          ink: "var(--accent-ink)",
        },
        danger: "var(--danger)",
      },
      fontFamily: {
        sans: ["var(--font-geist-sans)", "system-ui", "sans-serif"],
        mono: ["var(--font-geist-mono)", "ui-monospace", "monospace"],
      },
      transitionTimingFunction: {
        smooth: "cubic-bezier(0.22, 1, 0.36, 1)",
      },
    },
  },
  plugins: [],
};
