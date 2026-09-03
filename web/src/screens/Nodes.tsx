import { useMemo, useState } from "react";
import { gb } from "../lib/format";
import type { Connect, Node } from "../lib/types";
import {
  Badge, Bar, Button, Drawer, Empty, ErrorLine, usePoll,
} from "../components";

function linkBadge(n: Node) {
  if (!n.peer_id) return <Badge tone="dim">нет p2p</Badge>;
  if (n.symmetric_nat) return <Badge tone="warn">symmetric NAT</Badge>;
  // Только про входящие соединения — и ни слова про реле, которого может и не
  // быть. Прежняя надпись «relay» читалась как «идёт через реле» и спорила с
  // логом узла, где в это же время стоит «0 relay».
  return n.reachable
    ? <Badge tone="ok">принимает</Badge>
    : <Badge tone="warn">за NAT</Badge>;
}

export function Nodes() {
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents");
  const connect = usePoll<Connect>("/admin/connect", 30000);
  const [query, setQuery] = useState("");
  const [only, setOnly] = useState<"all" | "idle" | "problem">("all");
  const [open, setOpen] = useState<Node | null>(null);

  const list = useMemo(() => {
    let items = nodes.data?.nodes ?? [];
    if (query) {
      const q = query.toLowerCase();
      items = items.filter((n) =>
        n.node_id.toLowerCase().includes(q) || n.gpu_name.toLowerCase().includes(q));
    }
    if (only === "idle") items = items.filter((n) => n.tasks_running === 0);
    if (only === "problem") items = items.filter((n) => !n.accepts_tasks || n.update_error);
    return [...items].sort((a, b) => a.node_id.localeCompare(b.node_id));
  }, [nodes.data, query, only]);

  const current = open ? (nodes.data?.nodes ?? []).find(
    (n) => n.node_id === open.node_id) ?? open : null;

  return (
    <div className="page">
      <header>
        <div>
          <h1>Узлы</h1>
          <p>
            звонят на <code>{connect.data?.dial_address ?? "…"}</code>
            {connect.data?.severity === "warn" && (
              <> · <span style={{ color: "var(--bad)" }}>снаружи недостижим</span></>
            )}
          </p>
        </div>
      </header>

      <ErrorLine error={nodes.error} />

      <div className="tools">
        <input type="text" placeholder="поиск по имени или карте"
               value={query} onChange={(e) => setQuery(e.target.value)} />
        <div className="seg">
          {(["all", "idle", "problem"] as const).map((k) => (
            <button key={k} data-active={only === k} onClick={() => setOnly(k)}>
              {k === "all" ? "все" : k === "idle" ? "простаивают" : "с проблемой"}
            </button>
          ))}
        </div>
        <span className="sub" style={{ marginLeft: "auto" }}>
          {list.length} из {nodes.data?.nodes.length ?? 0}
        </span>
      </div>

      <div className="card pad0">
        {list.length === 0 ? (
          <Empty title={nodes.loading ? "Загрузка…" : "Ничего не найдено"}>
            {!nodes.loading && "Ключ для подключения выдаётся на вкладке Keys."}
          </Empty>
        ) : (
          <table>
            <thead>
              <tr>
                <th>узел</th><th>железо</th><th>GPU</th><th>VRAM</th>
                <th>состояние</th><th>снаружи</th><th>агент</th><th />
              </tr>
            </thead>
            <tbody>
              {list.map((n) => {
                const busy = n.gpus_total - n.gpus_free;
                return (
                  <tr key={n.node_id} className="clickable" onClick={() => setOpen(n)}>
                    <td>
                      <b>{n.node_id}</b>
                      <div className="sub">{n.seconds_since_seen}s назад</div>
                    </td>
                    <td>
                      {n.gpu_name || n.device || "—"}
                      {n.cuda_version && <div className="sub">CUDA {n.cuda_version}</div>}
                    </td>
                    <td style={{ minWidth: 92 }}>
                      <span className="num">{n.gpus_free}/{n.gpus_total}</span>
                      <Bar percent={n.gpus_total ? (busy / n.gpus_total) * 100 : 0}
                           tone={n.gpus_free === 0 ? "bad" : undefined} />
                    </td>
                    <td className="num">{gb(n.vram_free_bytes)} GB</td>
                    <td>
                      {n.accepts_tasks
                        ? <Badge tone="ok">{n.tasks_running || 0} задач</Badge>
                        : <Badge tone="bad">не берёт</Badge>}
                    </td>
                    <td>{linkBadge(n)}</td>
                    <td>
                      <code>{n.agent_version}</code>
                      {n.update_error && <div className="sub"
                        style={{ color: "var(--bad)" }}>обновление не идёт</div>}
                    </td>
                    <td style={{ width: 1 }}>
                      <Button kind="ghost" size="sm">детали</Button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      {current && (
        <Drawer title={<><code>{current.node_id}</code></>} onClose={() => setOpen(null)}>
          <div className="grid stats" style={{ marginBottom: 24 }}>
            <div className="card stat">
              <div className="label">GPU</div>
              <div className="value">{current.gpus_free}<small> из {current.gpus_total}</small></div>
              <div className="sub">{gb(current.vram_free_bytes)} GB свободно</div>
            </div>
            <div className="card stat">
              <div className="label">Кэш окружений</div>
              <div className="value">{gb(current.env_cache_bytes)}<small> GB</small></div>
              <div className="sub">{current.environment_kinds.join(", ") || "—"}</div>
            </div>
            <div className="card stat">
              <div className="label">Кэш моделей</div>
              <div className="value">{gb(current.model_cache_bytes)}<small> GB</small></div>
              <div className="sub">веса, скачанные один раз на узел</div>
            </div>
            <div className="card stat">
              <div className="label">Диск</div>
              {/* Ноль означает «агент ещё не умеет о нём рассказывать», а не
                  «места нет»: старая версия это поле просто не заполняет, и
                  показывать её как заполненный под завязку узел — врать. */}
              <div className="value">
                {current.disk_total_bytes
                  ? <>{gb(current.disk_free_bytes)}<small> из {gb(current.disk_total_bytes)} GB</small></>
                  : <small>агент не сообщает</small>}
              </div>
              <div className="sub">том с кэшами и задачами</div>
            </div>
          </div>

          <section>
            <h2>Железо</h2>
            <div className="card">
              <Row k="карта" v={current.gpu_name || "—"} />
              <Row k="устройство" v={current.device} />
              <Row k="CUDA драйвера" v={current.cuda_version || "не определена"} />
              <Row k="RAM хоста" v={`${current.host_ram_gb.toFixed(0)} GB`} />
              <Row k="регион" v={current.region || "—"} />
            </div>
          </section>

          <section>
            <h2>Канал к соседям</h2>
            <div className="card">
              <Row k="peer id" v={current.peer_id
                ? <code>{current.peer_id.slice(0, 20)}…</code> : "p2p выключен"} />
              <Row k="дозвонимость" v={
                current.symmetric_nat ? "symmetric NAT — только через реле"
                  : current.reachable ? "принимает входящие" : "только через реле"} />
              <Row k="прямых / через оркестратор"
                   v={`${current.direct} / ${current.relayed}`} />
              {current.link_rtt_ms > 0 && <Row k="RTT" v={`${current.link_rtt_ms} ms`} />}
            </div>
          </section>

          <section>
            <h2>Агент</h2>
            <div className="card">
              <Row k="версия" v={<code>{current.agent_version}</code>} />
              <Row k="обновление" v={current.update_state || "молчит"} />
              {current.update_version &&
                <Row k="предложено" v={<code>{current.update_version}</code>} />}
              {current.update_error && <Row k="не выходит"
                v={<span style={{ color: "var(--bad)" }}>{current.update_error}</span>} />}
              {!current.accepts_tasks &&
                <Row k="отказ" v={<span style={{ color: "var(--bad)" }}>{current.refusal}</span>} />}
            </div>
          </section>
        </Drawer>
      )}
    </div>
  );
}

function Row({ k, v }: { k: string; v: React.ReactNode }) {
  return (
    <div style={{
      display: "flex", gap: 16, padding: "7px 0",
      borderBottom: "1px solid var(--line-soft)",
    }}>
      <span style={{ color: "var(--text-mute)", minWidth: 180, fontSize: 12.5 }}>{k}</span>
      <span style={{ flex: 1 }}>{v}</span>
    </div>
  );
}
