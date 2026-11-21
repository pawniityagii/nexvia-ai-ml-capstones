import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base is read at runtime from VITE_API_BASE; it defaults to the same origin
// in production (the FastAPI backend serves the built viewer). In dev, the API
// paths below are proxied to the backend on :8000 so same-origin paths work.
const api = { target: "http://localhost:8000", changeOrigin: true };

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/analyze": api,
      "/cases": api,
      "/stats": api,
      "/health": api,
      "/disclaimer": api,
    },
  },
});
