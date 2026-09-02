import { useEffect, useState } from "react";
import { token } from "./lib/api";
import type { Node, Task } from "./lib/types";
import { Badge, Toasts, usePoll } from "./components";
import { Keys } from "./screens/Keys";
import { Models } from "./screens/Models";
import { Nodes } from "./screens/Nodes";
import { Overview } from "./screens/Overview";
import { Ray } from "./screens/Ray";
import { Release } from "./screens/Release";
import { Tasks } from "./screens/Tasks";

const SCREENS = ["Overview", "Nodes", "Models", "Ray", "Tasks", "Keys", "Release"] as const;
type Screen = (typeof SCREENS)[number];

const TITLE: Record<Screen, string> = {
  Overview: "Обзор", Nodes: "Узлы", Models: "Модели", Ray: "Ray",
  Tasks: "Задачи", Keys: "Ключи", Release: "Обновление",
};

function Shell() {
  const [screen, setScreen] = useState<Screen>(() => {
    const saved = localStorage.getItem("looma_screen") as Screen;
    return SCREENS.includes(saved) ? saved : "Overview";
  });
  const [secret, setSecret] = useState(token.get());
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents", 8000);
  const tasks = usePoll<{ tasks: Task[] }>("/admin/tasks", 8000);

  const go = (name: string) => {
    const target = name as Screen;
    if (!SCREENS.includes(target)) return;
    setScreen(target);
    localStorage.setItem("looma_screen", target);
  };

  // Цифры рядом с пунктами меню: видно, где что-то происходит, не заходя туда.
  const online = nodes.data?.nodes.length ?? 0;
  const active = (tasks.data?.tasks ?? []).filter(
    (t) => !["done", "failed", "cancelled", "gone"].includes(t.state)).length;
  const counts: Partial<Record<Screen, number>> = { Nodes: online, Tasks: active };

  useEffect(() => { document.title = `${TITLE[screen]} · Looma`; }, [screen]);

  const view = {
    Overview: <Overview go={go} />, Nodes: <Nodes />, Models: <Models />,
    Ray: <Ray />, Tasks: <Tasks />, Keys: <Keys />, Release: <Release />,
  }[screen];

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <b>looma</b>
          <span>{online ? `${online} online` : "offline"}</span>
        </div>
        <nav>
          {SCREENS.map((name) => (
            <button key={name} data-active={name === screen} onClick={() => go(name)}>
              {TITLE[name]}
              {counts[name] !== undefined && counts[name]! > 0 && (
                <span className="count">{counts[name]}</span>
              )}
            </button>
          ))}
        </nav>
        <div className="foot">
          <div className="field">
            <label>Админ-токен</label>
            <input type="password" value={secret} placeholder="не задан"
                   onChange={(e) => { setSecret(e.target.value); token.set(e.target.value); }} />
          </div>
          {nodes.error
            ? <Badge tone="bad">API недоступен</Badge>
            : <Badge tone="ok">API отвечает</Badge>}
        </div>
      </aside>
      <main className="main">{view}</main>
    </div>
  );
}

export const App = () => <Toasts><Shell /></Toasts>;
