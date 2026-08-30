import { useState } from "react";
import { token } from "./api";
import { Keys } from "./tabs/Keys";
import { Models } from "./tabs/Models";
import { Nodes } from "./tabs/Nodes";
import { Release } from "./tabs/Release";
import { Tasks } from "./tabs/Tasks";

const TABS = {
  Nodes: <Nodes />,
  Models: <Models />,
  Tasks: <Tasks />,
  Keys: <Keys />,
  Release: <Release />,
} as const;

type TabName = keyof typeof TABS;

export function App() {
  const [tab, setTab] = useState<TabName>(
    (localStorage.getItem("loom_tab") as TabName) in TABS
      ? (localStorage.getItem("loom_tab") as TabName)
      : "Nodes",
  );
  const [secret, setSecret] = useState(token.get());

  const pick = (name: TabName) => {
    setTab(name);
    localStorage.setItem("loom_tab", name);
  };

  return (
    <>
      <header>
        <h1>loom</h1>
        <nav>
          {(Object.keys(TABS) as TabName[]).map((name) => (
            <button key={name} className={name === tab ? "active" : ""}
                    onClick={() => pick(name)}>
              {name}
            </button>
          ))}
        </nav>
        <label className="token">
          <span>admin token</span>
          <input type="password" value={secret}
                 onChange={(e) => { setSecret(e.target.value); token.set(e.target.value); }} />
        </label>
      </header>
      {/* Каждая вкладка размонтируется при уходе: свои опросы и своё состояние
          формы, ничего не течёт между ними. */}
      <main>{TABS[tab]}</main>
    </>
  );
}
