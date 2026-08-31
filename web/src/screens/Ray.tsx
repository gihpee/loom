import { useEffect, useState } from "react";
import { get, send } from "../lib/api";
import { ago, short } from "../lib/format";
import type { Group, GroupHealth, Node, StageHealth, Task } from "../lib/types";
import {
  Badge, Button, Confirm, Empty, ErrorLine, Field, FilePick,
  StateBadge, useAction, usePoll, useToast,
} from "../components";

/** Как узнать Ray-группу среди прочих.
 *
 *  По команде, а не по метке: метку задаёт человек, а команда — это то, что
 *  на узле действительно работает, и соврать она не может. */
const RAY = "loom_ray.server";

const TEMPLATE = `import os
import ray

# ДО импорта ray. Плазма-сокет ложится внутрь временного каталога Ray, а путь
# unix-сокета не может быть длиннее 103 байт — каталог задачи в лимит не
# влезает. LOOM_TASK_TMP агент даёт как раз для этого.
os.environ.setdefault("RAY_TMPDIR", os.environ["LOOM_TASK_TMP"])

ray.init()          # подключится к кластеру, который уже поднял ранг 0


@ray.remote
def work(n):        # имя латиницей: Ray кодирует его в ASCII
    return n * n


answers = ray.get([work.remote(i) for i in range(100)])

# Результат — только то, что легло сюда. Всё остальное считается черновиком.
with open(os.path.join(os.environ["LOOM_TASK_OUT"], "answer.txt"), "w") as f:
    f.write(str(sum(answers)))
`;

/** Как этот узел даётся соседям.
 *
 *  Данных о конкретной ПАРЕ у нас нет: узлы докладывают о себе, а не друг о
 *  друге. Поэтому здесь то же топологическое правило, что и в оркестраторе. */
function reach(n?: Node): [string, "ok" | "warn" | "bad"] {
  if (!n) return ["узел отключился", "bad"];
  if (n.reachable) return ["принимает входящие", "ok"];
  if (n.symmetric_nat) return ["симметричный NAT — только через реле", "bad"];
  return ["за NAT, пробивается", "warn"];
}

/** Что ранг делает прямо сейчас. «running» — про процесс, а не про
 *  готовность: пока в кластере не все ранги, работать нельзя. */
function phase(s: StageHealth): string {
  if (s.state === "pending") return "в очереди";
  if (s.state === "provisioning") return "ставит ray";
  if (s.state === "failed") return "упал";
  if (s.state === "cancelled") return "снят";
  if (s.state !== "running") return s.state;
  if (!s.stage) return "стартует";
  if (s.ready) return "в кластере";
  const seen = s.stage.nodes ?? 0, want = s.stage.size ?? 0;
  return want > 1 ? `ждёт ранги (${seen}/${want})` : s.stage.status;
}

function Cluster({ group, nodes, onStop }: {
  group: Group; nodes: Node[]; onStop: () => void;
}) {
  const [health, setHealth] = useState<GroupHealth | null>(null);
  useEffect(() => {
    // Готовность спрашивается у самих рангов, а не выводится из состояния
    // задачи: процесс запускается за минуты до того, как кластер соберётся.
    let alive = true;
    const pull = () => get<GroupHealth>(`/admin/groups/${group.group_id}/health`)
      .then((b) => alive && setHealth(b)).catch(() => undefined);
    pull();
    const timer = setInterval(pull, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, [group.group_id]);

  const ranks = health?.stages ?? group.ranks.map((r) => ({
    ...r, state: "?", error: "", seconds: 0, ready: false, stage: null,
  } as StageHealth));
  const alive = ranks.filter((r) => r.ready).length;
  const byId = new Map(nodes.map((n) => [n.node_id, n]));
  // Пара пойдёт через реле, если ни один из двоих не принимает входящие и
  // хоть у кого-то симметричный NAT. Для Ray это заметно дороже, чем для
  // конвейера: через тот же путь идёт весь его обмен, а не 8 КБ на токен.
  const relayed = ranks.length > 1 && ranks.every((r) => {
    const n = byId.get(r.node_id);
    return n && !n.reachable;
  }) && ranks.some((r) => byId.get(r.node_id)?.symmetric_nat);

  return (
    <div className="card" style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap" }}>
        <b>{group.label || "без метки"}</b>
        <code style={{ color: "var(--text-dim)", fontSize: 12 }}>
          {short(group.group_id, 14)}
        </code>
        <span className="sub" style={{ margin: 0 }}>{ago(group.submitted_at)}</span>
        <span style={{ marginLeft: "auto", display: "flex", gap: 8, alignItems: "center" }}>
          <Badge tone={alive === ranks.length ? "ok" : "warn"}
                 pulse={alive !== ranks.length}>
            {alive} из {ranks.length} в кластере
          </Badge>
          <Button size="sm" kind="danger" onClick={onStop}>снять</Button>
        </span>
      </div>

      {relayed && (
        <div className="sub" style={{ margin: 0, color: "var(--warn)" }}>
          часть пар идёт через реле: для Ray это заметно медленнее, чем для
          конвейера — через тот же путь проходит весь его обмен
        </div>
      )}

      <div style={{ display: "grid", gap: 6 }}>
        {ranks.map((r) => (
          <div key={r.rank} style={{
            display: "flex", gap: 10, alignItems: "baseline", flexWrap: "wrap",
            padding: "7px 10px", borderRadius: 8,
            background: "var(--raised)", border: "1px solid var(--line-soft)",
          }}>
            <b style={{ fontSize: 12.5 }}>
              rank {r.rank}{r.rank === 0 && <span className="sub"> · голова</span>}
            </b>
            <code style={{ color: "var(--text-dim)", fontSize: 12 }}>{r.node_id}</code>
            <Badge tone={reach(byId.get(r.node_id))[1]}>
              {reach(byId.get(r.node_id))[0]}
            </Badge>
            <span style={{ marginLeft: "auto", display: "flex", gap: 8,
                           alignItems: "center" }}>
              <span className="sub" style={{ margin: 0 }}>{phase(r)}</span>
              <StateBadge value={r.ready ? "ready" : r.state}
                          pulse={!r.ready && r.state === "running"} />
            </span>
            {r.error && (
              <div className="sub" style={{ width: "100%", color: "var(--bad)" }}>
                {r.error}
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function Ray() {
  const groups = usePoll<{ groups: Group[] }>("/admin/groups", 6000);
  const tasks = usePoll<{ tasks: Task[] }>("/admin/tasks", 6000);
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents", 8000);
  const action = useAction(groups.refresh);
  const toast = useToast();

  const [node, setNode] = useState("");
  const [size, setSize] = useState("1");
  const [label, setLabel] = useState("");
  const [version, setVersion] = useState("");
  const [script, setScript] = useState("");
  const [scriptName, setScriptName] = useState("");
  const [stopping, setStopping] = useState<Group | null>(null);

  const free = (nodes.data?.nodes ?? []).filter((n) => n.accepts_tasks);
  // Группа считается Ray-группой, если её нулевой ранг это и запускает.
  const rayTasks = new Set((tasks.data?.tasks ?? [])
    .filter((t) => t.command.some((c) => c.includes(RAY)))
    .map((t) => t.group_id));
  const live = (groups.data?.groups ?? []).filter((g) => rayTasks.has(g.group_id));

  const pick = (file: File) => {
    const reader = new FileReader();
    reader.onload = () => {
      // FileReader отдаёт data:...;base64,XXXX — API нужен только хвост.
      setScript(String(reader.result).split(",")[1] ?? "");
      setScriptName(file.name);
    };
    reader.readAsDataURL(file);
  };

  const launch = () => action.run(async () => {
    await send("/admin/ray", "POST", {
      node_ids: node ? [node] : [],
      size: node ? undefined : Number(size) || 1,
      script: script || undefined,
      label: label || undefined,
      ray_version: version || undefined,
    });
    setScript(""); setScriptName(""); setLabel("");
  }, script ? "кластер поднимается, скрипт запустится сам" : "кластер поднимается");

  // Узлы, которые соседям даются тяжело. Видеть это надо ДО запуска: после
  // него «медленно» и «сломалось» выглядят одинаково.
  const awkward = free.filter((n) => !n.reachable && n.symmetric_nat);

  return (
    <div className="page">
      <header>
        <div>
          <h1>Ray</h1>
          <p>кластер под задачу: живёт, пока живёт задача, и умирает вместе с ней</p>
        </div>
      </header>

      <ErrorLine error={groups.error || nodes.error} />

      <section>
        <h2>Поднять кластер</h2>
        <div className="card" style={{ display: "grid", gap: 14 }}>
          <div style={{ display: "grid", gap: 12,
                        gridTemplateColumns: "repeat(auto-fit, minmax(190px, 1fr))" }}>
            <Field label="узел" hint={free.length ? "" : "нет узлов, берущих работу"}>
              <select value={node} onChange={(e) => setNode(e.target.value)}>
                <option value="">выбрать самые свободные</option>
                {free.map((n) => (
                  <option key={n.node_id} value={n.node_id}>
                    {n.node_id} · {n.gpus_free}/{n.gpus_total} GPU
                  </option>
                ))}
              </select>
            </Field>
            <Field label="узлов"
                   hint={node ? "узел выбран вручную" : `доступно ${free.length}`}>
              <input type="number" min={1} max={Math.max(1, free.length)}
                     value={node ? 1 : size} disabled={!!node}
                     onChange={(e) => setSize(e.target.value)} />
            </Field>
            <Field label="метка" hint="чтобы найти его потом">
              <input value={label} onChange={(e) => setLabel(e.target.value)}
                     placeholder="перебор-гиперпараметров" />
            </Field>
            <Field label="версия ray" hint="пусто — последняя">
              <input className="mono" value={version}
                     onChange={(e) => setVersion(e.target.value)} placeholder="2.58.0" />
            </Field>
            <Field label="точка входа"
                   hint={scriptName || "без неё кластер просто стоит и ждёт"}>
              <FilePick label="выбрать .py" accept=".py" onPick={pick} />
            </Field>
          </div>

          <div style={{ display: "flex", gap: 8, alignItems: "center",
                        flexWrap: "wrap" }}>
            <Button kind="primary" disabled={action.busy || free.length === 0}
                    onClick={launch}>
              поднять кластер
            </Button>
            <span className="sub" style={{ margin: 0 }}>
              {awkward.length > 0
                ? `${awkward.length} из ${free.length} узлов за симметричным NAT: `
                  + "с ними кластер пойдёт через реле"
                : "ранги находят друг друга через агента: порты соседей он "
                  + "держит у себя на локалхосте, и Ray про NAT не узнаёт"}
            </span>
          </div>
        </div>
      </section>

      <section>
        <h2>Работают</h2>
        {live.length === 0 ? (
          <div className="card pad0">
            <Empty title="Кластеров нет">
              Ray здесь — обычная задача: агент не отличает её от любой другой
              и про Ray ничего не знает.
            </Empty>
          </div>
        ) : (
          <div style={{ display: "grid", gap: 12 }}>
            {live.map((g) => (
              <Cluster key={g.group_id} group={g} nodes={nodes.data?.nodes ?? []}
                       onStop={() => setStopping(g)} />
            ))}
          </div>
        )}
      </section>

      <section>
        <h2>Как писать задачу</h2>
        <div className="card">
          <pre className="block pickable">{TEMPLATE}</pre>
          <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
            <Button size="sm" onClick={() => {
              navigator.clipboard.writeText(TEMPLATE);
              toast("ok", "скопировано");
            }}>скопировать шаблон</Button>
          </div>
        </div>
      </section>

      <section>
        <h2>Что здесь поедет, а что нет</h2>
        <div className="card pad0">
          <table>
            <thead>
              <tr><th>форма задачи</th><th>обмен между узлами</th><th>на этом железе</th></tr>
            </thead>
            <tbody>
              <tr>
                <td>независимые куски</td><td className="sub">ничего</td>
                <td><Badge tone="ok">да</Badge></td>
              </tr>
              <tr>
                <td>конвейер по слоям</td>
                <td className="sub">активации, ~8 КБ на токен</td>
                <td><Badge tone="ok">да</Badge></td>
              </tr>
              <tr>
                <td>тензорный параллелизм</td>
                <td className="sub">allreduce внутри каждого слоя</td>
                <td><Badge tone="bad">нет</Badge></td>
              </tr>
              <tr>
                <td>обучение DDP / FSDP</td>
                <td className="sub">градиенты каждый шаг = размер модели</td>
                <td><Badge tone="bad">нет</Badge></td>
              </tr>
            </tbody>
          </table>
        </div>
        <p className="sub">
          Ray не объединяет VRAM: одна аллокация CUDA не может лежать на двух
          машинах. Он планировщик и транспорт — разрезать задачу должна
          стратегия, и две нижние на домашних каналах не работают. Обучение
          модели на 7B требует ~14 ГБ обмена на шаг: на канале 100 Мбит это
          двадцать минут за шаг. Ray запустит это и не предупредит.
        </p>
      </section>

      {stopping && (
        <Confirm
          title="Снять кластер?"
          body={`«${stopping.label || stopping.group_id}» и всё, что в нём считается.`}
          action="снять"
          onClose={() => setStopping(null)}
          onConfirm={() => action.run(
            () => send(`/admin/groups/${stopping.group_id}/stop`, "POST"),
            "снят").finally(() => setStopping(null))}
        />
      )}
    </div>
  );
}
