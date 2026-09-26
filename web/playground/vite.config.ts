import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// Anticipates being served alongside the visualizer under the same GitHub
// Pages project site, at /spiderman/playground/ — not yet wired into the
// deploy workflow (that builds only @spiderman/visualizer today).
export default defineConfig({
  base: "/spiderman/playground/",
  plugins: [react()],
});
