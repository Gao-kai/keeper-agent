import tailwindcssAnimate from "tailwindcss-animate"

/** @type {import('tailwindcss').Config} */
export default {
  darkMode: ["class"],
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 深色科技感基础色板（后续功能开发时使用）
        background: "#0B0E14",
        surface: "#11151D",
        surfaceLight: "#1A1F2E",
        primary: {
          DEFAULT: "#6366F1",
          light: "#8B5CF6",
          dark: "#4F46E5",
        },
        foreground: "#F1F5F9",
        muted: "#94A3B8",
      },
      fontFamily: {
        sans: ['"PingFang SC"', "Inter", "system-ui", "sans-serif"],
      },
      borderRadius: {
        lg: "var(--radius)",
        md: "calc(var(--radius) - 2px)",
        sm: "calc(var(--radius) - 4px)",
      },
    },
  },
  plugins: [tailwindcssAnimate],
}
