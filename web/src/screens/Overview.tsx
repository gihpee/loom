import { gb } from "../lib/format";
import type { Connect, Group, Node, Task, VersionMap } from "../lib/types";
import { Badge, Empty, ErrorLine, Stat, StateBadge, usePoll } from "../components";
import { Topology, strandOf, type NodeState } from "./Topology";

/** Экран, которого не было: что вообще происходит, одним взглядом.
 *  Оператор открывает панель, чтобы понять «всё ли в порядке», а не чтобы
 *  читать пять таблиц подряд. */
export function Overview({ go }: { go: (screen: string) => void }) {
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents");
  const tasks = usePoll<{ tasks: Task[] }>("/admin/tasks");
  const groups = usePoll<{ groups: Group[] }>("/admin/groups");
  const serving = usePoll<{ data: { id: string }[] }>("/v1/models");
  const release = usePoll<VersionMap>("/admin/release", 10000);
  const connect = usePoll<Connect>("/admin/connect", 30000);

  const list = nodes.data?.nodes ?? [];
  const gpusFree = list.reduce((n, x) => n + x.gpus_free, 0);
  const gpusAll = list.reduce((n, x) => n + x.gpus_total, 0);
  const vram = list.reduce((n, x) => n + x.vram_free_bytes, 0);
  const refusing = list.filter((n) => !n.accepts_tasks);
  const active = (tasks.data?.tasks ?? []).filter(
    (t) => !["done", "failed", "cancelled", "gone"].includes(t.state));
  const failed = (tasks.data?.tasks ?? []).filter((t) => t.state === "failed");
  const up = new Set((serving.data?.data ?? []).map((m) => m.id));
  const models = (groups.data?.groups ?? []).filter((g) => g.label);
  const versions = Object.keys(release.data?.versions ?? {});

  // Чьё держит каждый узел. Из групп: имя у группы есть только у модели, у
  // арендованного кластера его нет — по этому и различаем, не заводя третьего
  // источника правды.
  const held = new Map<string, NodeState>();
  for (const g of groups.data?.groups ?? []) {
    const state: NodeState = g.label ? "inference" : "rented";
    for (const rank of g.ranks ?? []) {
      if (rank.node_id && held.get(rank.node_id) !== "rented") {
        held.set(rank.node_id, state);
      }
    }
  }
  const strands = list.map((n) => strandOf(n, held));
  const lost = strands.filter((s) => s.state === "lost").length;

  const problems: { text: string; where: string }[] = [];
  if (connect.data?.severity === "warn")
    problems.push({ text: connect.data.warning ?? "адрес недостижим снаружи — узлы не подключатся",
                    where: "Nodes" });
  for (const n of refusing)
    problems.push({ text: `${n.node_id}: ${n.refusal}`, where: "Nodes" });
  for (const n of list.filter((x) => x.update_error))
    problems.push({ text: `${n.node_id}: ${n.update_error}`, where: "Release" });
  for (const g of models.filter((g) => !up.has(g.label)))
    problems.push({ text: `${g.label} ещё не отвечает`, where: "Models" });
  for (const t of failed.slice(0, 3))
    problems.push({ text: `${t.task_id}: ${t.error || "упала"}`, where: "Tasks" });

  return (
    <div className="page">
      <header>
        <div>
          <h1>Обзор</h1>
          <p>{list.length ? `${list.length} узлов на связи` : "узлы не подключены"}</p>
        </div>
      </header>

      <ErrorLine error={nodes.error} />

      {list.length > 0 && (
        <section className="topo">
          <Topology strands={strands} />
          <div className="topo-legend">
            <span><i data-state="idle" />простаивает</span>
            <span><i data-state="inference" />инференс</span>
            <span><i data-state="rented" />у клиента</span>
            <span><i data-state="updating" />обновляется</span>
            {lost > 0 && <span><i data-state="lost" />замолчал ({lost})</span>}
          </div>
        </section>
      )}

      <section>
        <div className="grid stats">
          <Stat label="Узлы" value={list.length}
                sub={refusing.length ? `${refusing.length} не берут работу` : "все принимают"} />
          <Stat label="GPU свободно" value={gpusFree} unit={`из ${gpusAll}`}
                sub={`${gb(vram)} GB VRAM`} />
          <Stat label="Модели" value={[...up].length}
                sub={models.length > up.size
                  ? `${models.length - up.size} поднимается`
                  : "все отвечают"} />
          <Stat label="Задачи" value={active.length}
                sub={failed.length ? `${failed.length} упало` : "без отказов"} />
          <Stat label="Версии агента" value={versions.length || "—"}
                sub={release.data?.release
                  ? `выкатка ${release.data.release.version} · ${release.data.release.wave_percent}%`
                  : "релиз не опубликован"} />
        </div>
      </section>

      <section>
        <h2>Требует внимания</h2>
        {problems.length === 0 ? (
          <div className="card"><Empty title="Всё в порядке">
            Узлы на связи, модели отвечают, задачи не падают.
          </Empty></div>
        ) : (
          <div className="card pad0">
            <table>
              <tbody>
                {problems.slice(0, 8).map((p, i) => (
                  <tr className="clickable" key={i} onClick={() => go(p.where)}>
                    <td style={{ width: 1 }}><Badge tone="warn">{p.where}</Badge></td>
                    <td>{p.text}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {models.length > 0 && (
        <section>
          <h2>Модели</h2>
          <div className="card pad0">
            <table>
              <thead><tr><th>модель</th><th>стадий</th><th>состояние</th></tr></thead>
              <tbody>
                {models.map((g) => (
                  <tr className="clickable" key={g.group_id} onClick={() => go("Models")}>
                    <td><b>{g.label}</b></td>
                    <td className="num">{g.size}</td>
                    <td><StateBadge value={up.has(g.label) ? "ready" : "loading"}
                                    pulse={!up.has(g.label)} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      )}
    </div>
  );
}
