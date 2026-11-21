import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// API base is read at runtime from VITE_API_BASE (defaults to localhost:8000).
export default defineConfig({
  plugins: [react()],
  server: { port: 5173 },
});
