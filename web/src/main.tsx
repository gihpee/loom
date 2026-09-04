import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { followSheen } from "./components/sheen";
import "./theme.css";

followSheen();

// Масштабирование выключено. Meta-тега недостаточно: Safari перестал слушать
// user-scalable, и щипок всё равно растягивал бы страницу.
for (const жест of ["gesturestart", "gesturechange", "gestureend"]) {
  addEventListener(жест, (e) => e.preventDefault(), { passive: false });
}

createRoot(document.getElementById("root")!).render(
  <StrictMode><App /></StrictMode>,
);
