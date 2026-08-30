import { gb } from "../api";
import type { Node } from "../types";
import { Table, usePoll } from "../ui";

interface Connect {
  dial_address: string; source: string; severity: string;
  self_check: boolean | null; warning: string | null; agent_image: string;
}

function link(node: Node) {
  if (!node.peer_id) return <span className="dim">нет p2p</span>;
  const total = node.direct + node.relayed;
  const how = node.symmetric_nat
    ? <span className="warn">symmetric NAT</span>
    : node.reachable
      ? <span className="ok">direct</span>
      : <span className="warn">relay</span>;
  return (
    <>
      {how}
      {total > 0 && (
        <div className="dim">
          {Math.round(node.direct_share * 100)}% direct · {total} сообщ.
          {node.link_rtt_ms > 0 && ` · ${node.link_rtt_ms} ms`}
        </div>
      )}
    </>
  );
}

export function Nodes() {
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents");
  const connect = usePoll<Connect>("/admin/connect", 30000);

  const rows = (nodes.data?.nodes ?? []).map((n) => [
    <><b>{n.node_id}</b><div className="dim">agent {n.agent_version}</div></>,
    <>{n.gpu_name || n.device || "—"}
      {n.cuda_version && <div className="dim">CUDA {n.cuda_version}</div>}</>,
    `${n.gpus_free} / ${n.gpus_total}`,
    `${gb(n.vram_free_bytes)} GB`,
    n.accepts_tasks
      ? <><span className="ok">принимает</span>
          <div className="dim">{n.environment_kinds.join(", ")}</div></>
      : <><span className="bad">не принимает</span>
          <div className="dim">{n.refusal}</div></>,
    <>{n.tasks_running} задач<div className="dim">кэш {gb(n.env_cache_bytes)} GB</div></>,
    link(n),
    <span className="dim">{n.seconds_since_seen}s</span>,
  ]);

  return (
    <>
      <h2>Узлы</h2>
      <Table
        head={["узел", "железо", "GPU", "VRAM", "состояние", "нагрузка", "канал", "виден"]}
        rows={rows}
        empty="ни один узел не подключён"
      />
      {connect.data && (
        <>
          <h2>Подключение</h2>
          <p>
            Узлы звонят на <b>{connect.data.dial_address}</b>{" "}
            <span className="dim">({connect.data.source})</span>
          </p>
          {connect.data.severity === "warn" && (
            <p className="bad">
              {connect.data.warning ?? "снаружи этот адрес недостижим — узлы не подключатся"}
            </p>
          )}
          <p className="dim">Ключ выдаётся во вкладке Keys, адрес уже внутри него.</p>
        </>
      )}
    </>
  );
}
