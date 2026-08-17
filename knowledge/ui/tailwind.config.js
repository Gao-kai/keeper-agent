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
        // 明亮暖色调（参考 pi.dev 亮色主题：moonstone / parchment 色系）
        background: "#ebe7e4",        // moonstone 页面底色
        backgroundDeep: "#dacbc2",    // parchment 深一档暖色
        surface: "#f4f2f0",           // panel 卡片背景
        surfaceLight: "#eef1f3",      // panel-soft 浅色区块
        primary: {
          DEFAULT: "#4b607c",         // thread blue 主强调色
          light: "#6a9fcc",           // accent blue 高亮
          dark: "#394352",            // 按钮 hover 加深
        },
        foreground: "#252f3d",        // 深蓝灰主文字
        muted: "#5c5752",             // 暖灰次要文字
        success: "#5db87a",
        warning: "#e8993a",
        error: "#e8704f",
      },
      fontFamily: {
        sans: ['"PingFang SC"', "Inter", "system-ui", "sans-serif"],
        // 衬线字体用于页面主标题，增加质感
        serif: ['"Source Han Serif SC"', '"Noto Serif SC"', "Georgia", '"Times New Roman"', "serif"],
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
