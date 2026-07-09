import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev, proxy API + websocket to the FastAPI control plane so the app works
// from http://localhost:5173 with no CORS friction. The API base is configurable
// via VITE_API_BASE for production (default: same-origin, served behind nginx).
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": { target: "http://localhost:8000", changeOrigin: true },
      "/positions": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
      "/control": { target: "http://localhost:8000", changeOrigin: true },
      "/ws": { target: "ws://localhost:8000", ws: true },
    },
  },
  build: { outDir: "dist", sourcemap: false },
});
