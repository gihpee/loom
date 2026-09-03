import { useEffect, useState } from "react";
import { signOut, token, whoami, type Who } from "./lib/api";
import type { Node, Task } from "./lib/types";
import { Badge, Mark, Toasts, usePoll } from "./components";
import { Accounts } from "./screens/Accounts";
import { Cabinet } from "./screens/Cabinet";
import { Keys } from "./screens/Keys";
import { Landing } from "./screens/Landing";
import { Models } from "./screens/Models";
import { Nodes } from "./screens/Nodes";
import { Overview } from "./screens/Overview";
import { Ray } from "./screens/Ray";
import { SignIn } from "./screens/SignIn";
import { Release } from "./screens/Release";
import { Tasks } from "./screens/Tasks";

const SCREENS = ["Overview", "Nodes", "Models", "Ray", "Tasks", "Accounts", "Keys", "Release"] as const;
type Screen = (typeof SCREENS)[number];

const TITLE: Record<Screen, string> = {
  Overview: "Обзор", Nodes: "Узлы", Models: "Модели", Ray: "Ray",
  Tasks: "Задачи", Accounts: "Клиенты", Keys: "Ключи", Release: "Обновление",
};

function Shell({ who, onLeave }: { who: Who; onLeave: () => void }) {
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
    Ray: <Ray />, Tasks: <Tasks />, Accounts: <Accounts />,
    Keys: <Keys />, Release: <Release />,
  }[screen];

  return (
    <div className="shell">
      <aside className="side">
        <div className="brand">
          <Mark size={20} />
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
          <div className="who">
            <span>{who.email || "аварийный вход"}</span>
            <button className="btn ghost sm" onClick={onLeave}>выйти</button>
          </div>
        </div>
      </aside>
      <main className="main">{view}</main>
    </div>
  );
}

/** Три поверхности вместо одной, и правило доступа читается по адресу.
 *
 * Роутер свой на двадцать строк, а не библиотека: маршрутов четыре, и тащить
 * зависимость ради них значило бы обновлять её вместе с React каждый раз, когда
 * та решит поменять API.
 *
 * `/console` для панели, а НЕ `/admin`: этот префикс уже занят
 * проксированием API в nginx, и одинаковый путь означал бы, что страница и
 * запрос спорят за один адрес.
 */
type Surface = "landing" | "app" | "console" | "signin";

function surfaceOf(path: string): Surface {
  if (path.startsWith("/console")) return "console";
  if (path.startsWith("/app")) return "app";
  if (path.startsWith("/signin")) return "signin";
  return "landing";
}

function Router() {
  const [path, setPath] = useState(location.pathname);
  const [who, setWho] = useState<Who | null | undefined>(undefined);

  useEffect(() => {
    const back = () => setPath(location.pathname);
    addEventListener("popstate", back);
    return () => removeEventListener("popstate", back);
  }, []);

  const ask = () => whoami().then(setWho).catch(() => setWho(null));
  useEffect(() => { void ask(); }, [path]);

  const surface = surfaceOf(path);
  if (surface === "landing") return <Landing />;

  // Пока не спросили — ничего: мигнуть страницей входа человеку, который уже
  // вошёл, хуже, чем задержаться на долю секунды.
  if (who === undefined) return <div className="loading" />;

  if (!who) {
    return <SignIn onDone={() => { void ask(); }} />;
  }

  const leave = async () => {
    try { await signOut(); } finally { location.href = "/"; }
  };

  if (surface === "console") {
    if (who.role !== "admin") {
      return (
        <div className="denied">
          <h1>Панель — для администраторов</h1>
          <p>Вы вошли как {who.email}. Ваши ресурсы — в кабинете.</p>
          <a className="btn primary" href="/app">В кабинет</a>
        </div>
      );
    }
    return <Shell who={who} onLeave={leave} />;
  }
  return (
    <div className="app-shell">
      <header className="app-top">
        <a className="app-brand" href="/"><Mark size={22} /><span>Looma&nbsp;Float</span></a>
        <nav>
          {who.role === "admin" && <a href="/console">Панель</a>}
          <button className="btn ghost sm" onClick={leave}>Выйти</button>
        </nav>
      </header>
      <Cabinet who={who} />
    </div>
  );
}

export const App = () => <Toasts><Router /></Toasts>;
