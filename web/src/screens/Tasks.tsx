import { useMemo, useState } from "react";
import { get, grab, send } from "../lib/api";
import { duration, kb, short } from "../lib/format";
import type { Group, Task } from "../lib/types";
import {
  Badge, Button, Drawer, Empty, ErrorLine, Field, Modal,
  StateBadge, useAction, usePoll,
} from "../components";

const DONE = ["done", "failed", "cancelled", "gone"];

function RunTask({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const action = useAction(onDone);
  const [command, setCommand] = useState("");
  const [requirements, setRequirements] = useState("");
  const [gpus, setGpus] = useState("0");
  const [timeout, setTimeoutS] = useState("600");
  const [file, setFile] = useState<File | null>(null);

  const run = () => action.run(async () => {
    const body: Record<string, unknown> = {
      command,
      timeout_s: Number(timeout) || 3600,
      resources: { gpus: Number(gpus) || 0 },
    };
    const deps = requirements.split(/[,\s]+/).filter(Boolean);
    body.environment = deps.length ? { kind: "python", requirements: deps }
                                   : { kind: "none" };
    if (file) {
      const bytes = new Uint8Array(await file.arrayBuffer());
      let binary = "";
      for (const b of bytes) binary += String.fromCharCode(b);
      body.inputs = { [file.name]: btoa(binary) };
    }
    const created = await send<Task>("/admin/tasks", "POST", body);
    onClose();
    return created;
  }, "задача отправлена");

  return (
    <Modal title="Запустить задачу" onClose={onClose} footer={
      <div className="form-actions">
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="primary" onClick={run} disabled={action.busy || !command}>
          запустить
        </Button>
      </div>
    }>
      <div className="form-grid">
        <Field label="Команда" hint="кавычки понимаются: python -c &quot;print(1)&quot;">
          <input type="text" className="mono" autoFocus value={command}
                 placeholder='python -c "print(1)"'
                 onChange={(e) => setCommand(e.target.value)} />
        </Field>
      </div>
      <div className="form-grid three" style={{ marginTop: 12 }}>
        <Field label="Зависимости" hint="pip, через запятую">
          <input type="text" value={requirements} placeholder="numpy, pillow"
                 onChange={(e) => setRequirements(e.target.value)} />
        </Field>
        <Field label="GPU">
          <input type="number" min={0} value={gpus}
                 onChange={(e) => setGpus(e.target.value)} />
        </Field>
        <Field label="Таймаут, с">
          <input type="number" min={1} value={timeout}
                 onChange={(e) => setTimeoutS(e.target.value)} />
        </Field>
      </div>
      <div style={{ marginTop: 12 }}>
        <Field label="Входной файл" hint="ляжет в рабочий каталог задачи">
          <label className={`file${file ? " filled" : ""}`}>
            <input type="file" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
            <span>{file?.name ?? "выбрать файл"}</span>
          </label>
        </Field>
      </div>
      <p className="sub" style={{ marginTop: 16 }}>
        Результатом считается то, что задача запишет в каталог из
        <code> $LOOM_TASK_OUT</code>.
      </p>
    </Modal>
  );
}

export function Tasks() {
  const tasks = usePoll<{ tasks: Task[] }>("/admin/tasks");
  const groups = usePoll<{ groups: Group[] }>("/admin/groups");
  const action = useAction(tasks.refresh);
  const [running, setRunning] = useState(false);
  const [open, setOpen] = useState<Task | null>(null);
  const [logs, setLogs] = useState("");
  const [only, setOnly] = useState<"active" | "all" | "failed">("active");
  const [query, setQuery] = useState("");

  const list = useMemo(() => {
    let items = tasks.data?.tasks ?? [];
    if (only === "active") items = items.filter((t) => !DONE.includes(t.state));
    if (only === "failed") items = items.filter((t) => t.state === "failed");
    if (query) {
      const q = query.toLowerCase();
      items = items.filter((t) =>
        t.task_id.includes(q) || t.node_id.toLowerCase().includes(q) ||
        t.command.join(" ").toLowerCase().includes(q));
    }
    return items;
  }, [tasks.data, only, query]);

  const current = open
    ? (tasks.data?.tasks ?? []).find((t) => t.task_id === open.task_id) ?? open
    : null;

  const showLogs = (id: string) => action.run(async () => {
    const answer = await get<{ text: string }>(`/admin/tasks/${id}/logs?tail=400`);
    setLogs(answer.text || "(пусто)");
  });

  const groupsWithLabel = (groups.data?.groups ?? []);

  return (
    <div className="page">
      <header>
        <div>
          <h1>Задачи</h1>
          <p>{(tasks.data?.tasks ?? []).filter((t) => !DONE.includes(t.state)).length} выполняется</p>
        </div>
        <div className="actions">
          <Button kind="primary" onClick={() => setRunning(true)}>запустить</Button>
        </div>
      </header>

      <ErrorLine error={tasks.error} />

      <div className="tools">
        <input type="text" placeholder="поиск по id, узлу или команде"
               value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="seg">
          {(["active", "failed", "all"] as const).map((k) => (
            <button key={k} data-active={only === k} onClick={() => setOnly(k)}>
              {k === "active" ? "активные" : k === "failed" ? "упавшие" : "все"}
            </button>
          ))}
        </div>
      </div>

      <div className="card pad0">
        {list.length === 0 ? (
          <Empty title={tasks.loading ? "Загрузка…" : "Задач нет"}>
            {!tasks.loading && only === "active" &&
              "Активных задач нет. Переключитесь на «все», чтобы увидеть завершённые."}
          </Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>задача</th><th>команда</th><th>состояние</th>
                <th>время</th><th>результат</th><th />
              </tr>
            </thead>
            <tbody>
              {list.map((t) => (
                <tr key={t.task_id} className="clickable"
                    onClick={() => { setOpen(t); setLogs(""); }}>
                  <td>
                    <code>{short(t.task_id, 12)}</code>
                    <div className="sub">
                      {t.node_id}{t.group_id ? ` · rank ${t.rank}` : ""}
                    </div>
                  </td>
                  <td style={{ maxWidth: 320 }}>
                    <span className="mono" style={{ color: "var(--text-dim)" }}>
                      {t.command.join(" ").slice(0, 60)}
                    </span>
                  </td>
                  <td>
                    <StateBadge value={t.state}
                                pulse={!DONE.includes(t.state)} />
                    {t.error && <div className="sub"
                      style={{ color: "var(--bad)", maxWidth: 260 }}>
                      {t.error.slice(0, 70)}</div>}
                  </td>
                  <td className="num">{duration(t.seconds)}</td>
                  <td>
                    {t.results.length
                      ? <Badge tone="info">{t.results.length} файл(ов)</Badge>
                      : <span className="sub">—</span>}
                  </td>
                  <td style={{ width: 1 }}>
                    <Button kind="ghost" size="sm">детали</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {groupsWithLabel.length > 0 && (
        <section style={{ marginTop: 24 }}>
          <h2>Группы</h2>
          <div className="card pad0">
            <table>
              <thead><tr><th>группа</th><th>ранги</th><th /></tr></thead>
              <tbody>
                {groupsWithLabel.map((g) => (
                  <tr key={g.group_id}>
                    <td>
                      <code>{short(g.group_id, 14)}</code>
                      {g.label && <div className="sub"
                        style={{ color: "var(--accent)" }}>{g.label}</div>}
                    </td>
                    <td>
                      {g.ranks.map((r) => (
                        <span key={r.rank} style={{ marginRight: 12 }}>
                          <span className="sub" style={{ margin: 0 }}>{r.rank}:</span>{" "}
                          <code style={{ fontSize: 12 }}>{r.node_id}</code>
                        </span>
                      ))}
                    </td>
                    <td style={{ width: 1 }}>
                      <Button size="sm" kind="danger" onClick={() => action.run(
                        () => send(`/admin/groups/${g.group_id}/stop`, "POST"),
                        "группа снята")}>снять</Button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}

      {running && <RunTask onClose={() => setRunning(false)} onDone={tasks.refresh} />}

      {current && (
        <Drawer title={<code>{current.task_id}</code>} onClose={() => setOpen(null)}>
          <div style={{ display: "flex", gap: 8, marginBottom: 16 }}>
            <StateBadge value={current.state} />
            <span className="sub" style={{ margin: 0 }}>
              {current.node_id} · {duration(current.seconds)}
              {current.devices.length > 0 && ` · gpu ${current.devices.join(",")}`}
            </span>
            <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
              {DONE.includes(current.state) ? (
                <Button size="sm" kind="danger" onClick={() => action.run(
                  () => send(`/admin/tasks/${current.task_id}`, "DELETE"),
                  "убрано").then(() => setOpen(null))}>убрать</Button>
              ) : (
                <Button size="sm" kind="danger" onClick={() => action.run(
                  () => send(`/admin/tasks/${current.task_id}/stop`, "POST"),
                  "остановлено")}>остановить</Button>
              )}
            </span>
          </div>

          {current.error && (
            <div className="card" style={{ borderColor: "var(--bad-soft)",
                                           marginBottom: 16 }}>
              <b style={{ color: "var(--bad)", fontSize: 13 }}>Ошибка</b>
              <div className="sub" style={{ marginTop: 4 }}>{current.error}</div>
            </div>
          )}

          <section>
            <h2>Команда</h2>
            <pre className="block">{current.command.join(" ")}</pre>
          </section>

          {current.results.length > 0 && (
            <section>
              <h2>Результат</h2>
              <div className="card pad0">
                <table>
                  <tbody>
                    {current.results.map((f) => (
                      <tr key={f.name}>
                        <td><code>{f.name}</code></td>
                        <td className="num" style={{ width: 100 }}>{kb(f.size_bytes)} KB</td>
                        <td style={{ width: 1 }}>
                          <Button size="sm" onClick={() => action.run(() => grab(
                            `/admin/tasks/${current.task_id}/results/${encodeURIComponent(f.name)}`,
                            f.name.split("/").pop() ?? f.name))}>скачать</Button>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </section>
          )}

          <section>
            <h2>
              Логи
              <Button size="sm" kind="ghost"
                      onClick={() => showLogs(current.task_id)}>обновить</Button>
            </h2>
            {logs
              ? <pre className="block tall">{logs}</pre>
              : <div className="card"><Empty title="Логи не загружены">
                  Нажмите «обновить», чтобы забрать последние 400 строк с узла.
                </Empty></div>}
          </section>
        </Drawer>
      )}
    </div>
  );
}
