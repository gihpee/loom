import { useState } from "react";
import { base64, Field, Note, State, Table, useAction, useFile, usePoll } from "../ui";
import { download, get, send } from "../api";
import type { Group, Task } from "../types";

export function Tasks() {
  const tasks = usePoll<{ tasks: Task[] }>("/admin/tasks");
  const groups = usePoll<{ groups: Group[] }>("/admin/groups");
  const action = useAction(tasks.refresh);
  const file = useFile();

  const [command, setCommand] = useState("");
  const [requirements, setRequirements] = useState("");
  const [gpus, setGpus] = useState("0");
  const [timeout, setTimeoutS] = useState("600");
  const [logs, setLogs] = useState("");

  const run = () =>
    action.run(async () => {
      const bytes = await file.read();
      const body: Record<string, unknown> = {
        command,
        timeout_s: Number(timeout) || 3600,
        resources: { gpus: Number(gpus) || 0 },
      };
      const deps = requirements.split(/[,\s]+/).filter(Boolean);
      body.environment = deps.length
        ? { kind: "python", requirements: deps }
        : { kind: "none" };
      if (bytes) {
        const name = file.ref.current?.files?.[0]?.name ?? "input";
        body.inputs = { [name]: base64(bytes) };
      }
      const created = await send<Task>("/admin/tasks", "POST", body);
      return created;
    }, "отправлено");

  const showLogs = (id: string) =>
    action.run(async () => {
      const answer = await get<{ text: string }>(`/admin/tasks/${id}/logs?tail=300`);
      setLogs(answer.text || "(пусто)");
    });

  const groupRows = (groups.data?.groups ?? []).map((g) => [
    <><code>{g.group_id}</code>{g.label && <div className="acc">{g.label}</div>}</>,
    g.size,
    <>{g.ranks.map((r) => (
      <div key={r.rank}><span className="dim">{r.rank}:</span> {r.node_id}</div>
    ))}</>,
    <button onClick={() => action.run(
      () => send(`/admin/groups/${g.group_id}/stop`, "POST"), "снято")}>снять</button>,
  ]);

  const taskRows = (tasks.data?.tasks ?? []).map((t) => {
    const done = ["done", "failed", "cancelled", "gone"].includes(t.state);
    return [
      <><code>{t.task_id}</code>
        <div className="dim">{t.node_id}{t.group_id ? ` · rank ${t.rank}` : ""}</div></>,
      <span className="dim">{t.command.join(" ").slice(0, 70)}</span>,
      <><State value={t.state} />{t.error && <div className="dim">{t.error}</div>}</>,
      <>{t.seconds}s{t.devices.length > 0 &&
        <div className="dim">gpu {t.devices.join(",")}</div>}</>,
      t.results.length
        ? t.results.map((f) => (
            <div key={f.name}>
              <a href="#" onClick={(e) => {
                e.preventDefault();
                action.run(() => download(
                  `/admin/tasks/${t.task_id}/results/${encodeURIComponent(f.name)}`,
                  f.name.split("/").pop() ?? f.name));
              }}>{f.name}</a>{" "}
              <span className="dim">{(f.size_bytes / 1024).toFixed(1)} KB</span>
            </div>
          ))
        : <span className="dim">—</span>,
      <>
        <button onClick={() => showLogs(t.task_id)}>логи</button>{" "}
        {done
          ? <button onClick={() => action.run(
              () => send(`/admin/tasks/${t.task_id}`, "DELETE"), "убрано")}>убрать</button>
          : <button onClick={() => action.run(
              () => send(`/admin/tasks/${t.task_id}/stop`, "POST"), "остановлено")}>стоп</button>}
      </>,
    ];
  });

  return (
    <>
      <h2>Запустить задачу</h2>
      <div className="row">
        <Field label="команда" hint="кавычки понимаются">
          <input value={command} onChange={(e) => setCommand(e.target.value)}
                 placeholder='python -c "print(1)"' style={{ width: 340 }} />
        </Field>
        <Field label="зависимости" hint="pip, через запятую">
          <input value={requirements} onChange={(e) => setRequirements(e.target.value)}
                 placeholder="numpy, pillow" style={{ width: 200 }} />
        </Field>
        <Field label="GPU"><input value={gpus} onChange={(e) => setGpus(e.target.value)}
                                  style={{ width: 60 }} /></Field>
        <Field label="таймаут, с"><input value={timeout}
                                         onChange={(e) => setTimeoutS(e.target.value)}
                                         style={{ width: 90 }} /></Field>
        <Field label="входной файл" hint="ляжет в рабочий каталог">
          <input type="file" ref={file.ref} />
        </Field>
        <Field label="&nbsp;">
          <button className="action" onClick={run} disabled={action.busy || !command}>
            запустить
          </button>
        </Field>
      </div>
      <Note text={action.note} />

      <h2>Группы</h2>
      <Table head={["группа", "размер", "ранги", ""]} rows={groupRows}
             empty="групп нет" />

      <h2>Задачи</h2>
      <Table head={["задача", "команда", "состояние", "время", "результат", ""]}
             rows={taskRows} empty="задач нет" />

      {logs && (<><h2>Логи</h2><pre className="logs">{logs}</pre></>)}
    </>
  );
}
