import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// GitHub Pages serves this as a project site at /spiderman/ — every asset
// reference (including the weight fetches in src/inference/) must go through
// import.meta.env.BASE_URL rather than an absolute root path, or it 404s in
// production while still working in local dev (which serves from /).
export default defineConfig({
  base: "/spiderman/",
  plugins: [react()],
});
