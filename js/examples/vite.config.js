import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  server: {
    fs: {
      allow: ["../.."],
    },
  },
  build: {
    target: "esnext",
  },
  optimizeDeps: {
    exclude: ["egraph"],
  },
  plugins: [react()],
});
