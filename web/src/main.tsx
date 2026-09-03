import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { followSheen } from "./components/sheen";
import "./theme.css";

followSheen();

createRoot(document.getElementById("root")!).render(
  <StrictMode><App /></StrictMode>,
);
