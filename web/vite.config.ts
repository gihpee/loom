import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Фронт — отдельный образ за nginx. Он отдаётся по корню своего порта, а
// /admin и /v1 проксируются в оркестратор: так пути API не спорят с путями
// приложения и не нужен ни один скрипт-обёртка.
export default defineConfig({
  plugins: [react()],
  build: {
    // Одним файлом на каждый тип: панель маленькая, а разбиение на чанки
    // усложняет раздачу без всякой пользы.
    rollupOptions: { output: { manualChunks: undefined } },
  },
  server: {
    // `npm run dev` ходит в живой оркестратор — то же, что делает nginx в
    // образе, чтобы правка фронта не требовала пересборки.
    proxy: {
      "/admin": "http://127.0.0.1:8000",
      "/api": "http://127.0.0.1:8000",
      "/v1": "http://127.0.0.1:8000",
    },
  },
});
