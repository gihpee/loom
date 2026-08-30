import { useEffect, useRef, useState } from "react";
import { get, message, send } from "../lib/api";
import { gb } from "../lib/format";
import type { Group, GroupHealth, Node, StageHealth } from "../lib/types";
import {
  Badge, Bar, Button, Confirm, Empty, ErrorLine, Field, Modal,
  StateBadge, useAction, usePoll, useToast,
} from "../components";

/** "running" — про процесс, а не про готовность отвечать: веса грузятся
 *  минутами, и всё это время состояние задачи одинаково. */
function phase(s: StageHealth): [string, number] {
  if (s.state === "pending") return ["в очереди", 6];
  if (s.state === "provisioning") return ["окружение и веса", 30];
  if (s.state === "failed") return ["упала", 100];
  if (s.state === "cancelled") return ["снята", 100];
  if (s.state !== "running") return [s.state, 0];
  if (!s.stage) return ["стартует", 60];
  return s.ready ? ["готова", 100] : ["грузит веса", 82];
}

function Pipeline({ group }: { group: Group }) {
  const [health, setHealth] = useState<GroupHealth | null>(null);
  useEffect(() => {
    let alive = true;
    const pull = () => get<GroupHealth>(`/admin/groups/${group.group_id}/health`)
      .then((b) => alive && setHealth(b)).catch(() => undefined);
    pull();
    const timer = setInterval(pull, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, [group.group_id]);

  const stages = health?.stages ?? group.ranks.map((r) => ({
    ...r, state: "?", error: "", seconds: 0, ready: false, stage: null,
  } as StageHealth));

  return (
    <div style={{ display: "grid", gap: 8 }}>
      {stages.map((s) => {
        const [what, percent] = phase(s);
        return (
          <div key={s.rank} style={{
            padding: "8px 10px", borderRadius: 8,
            background: "var(--raised)", border: "1px solid var(--line-soft)",
          }}>
            <div style={{ display: "flex", gap: 10, alignItems: "baseline",
                          flexWrap: "wrap", marginBottom: 6 }}>
              <b style={{ fontSize: 12.5 }}>rank {s.rank}</b>
              <code style={{ color: "var(--text-dim)", fontSize: 12 }}>{s.node_id}</code>
              {s.stage?.layers && (
                <span className="sub" style={{ margin: 0 }}>
                  слои {s.stage.layers[0]}–{s.stage.layers[1]}
                </span>
              )}
              <span style={{ marginLeft: "auto", display: "flex", gap: 8,
                             alignItems: "center" }}>
                <span className="sub" style={{ margin: 0 }}>{what}</span>
                <StateBadge value={s.ready ? "ready" : s.state}
                            pulse={!s.ready && s.state === "running"} />
              </span>
            </div>
            {s.error && (
              <div className="sub" style={{ color: "var(--bad)", marginBottom: 6 }}>
                {s.error}
              </div>
            )}
            <Bar percent={percent} tone={s.state === "failed" ? "bad"
                                        : s.ready ? "ok" : undefined} />
          </div>
        );
      })}
    </div>
  );
}

function Deploy({ nodes, onClose, onDone }: {
  nodes: Node[]; onClose: () => void; onDone: () => void;
}) {
  const action = useAction(onDone);
  const toast = useToast();
  const [repo, setRepo] = useState("");
  const [label, setLabel] = useState("");
  const [dtype, setDtype] = useState("bfloat16");
  const [device, setDevice] = useState("cuda");
  const [stages, setStages] = useState("2");
  const [byVram, setByVram] = useState(true);
  const [picked, setPicked] = useState<string[]>([]);
  const [layers, setLayers] = useState<number | null>(null);

  const describe = () => action.run(async () => {
    const info = await send<{ num_layers: number; architecture: string }>(
      "/admin/models/describe", "POST", { repo });
    setLayers(info.num_layers);
    toast("ok", `${repo}: ${info.num_layers} слоёв, ${info.architecture || "?"}`);
  });

  const deploy = () => action.run(async () => {
    const body: Record<string, unknown> = { repo, dtype, device, by_vram: byVram };
    if (label) body.label = label;
    if (picked.length) body.node_ids = picked;
    else body.stages = Number(stages) || 1;
    const created = await send<{
      label: string; model: { num_layers: number };
      split: { rank: number; node_id: string; start_layer: number; end_layer: number }[];
    }>("/admin/deploy", "POST", body);
    toast("ok", `${created.label}: ${created.model.num_layers} слоёв\n` +
      created.split.map((s) =>
        `rank ${s.rank}  ${s.start_layer}–${s.end_layer}  ${s.node_id}`).join("\n"));
    onClose();
  });

  const takers = nodes.filter((n) => n.accepts_tasks);
  const count = picked.length || Number(stages) || 1;

  return (
    <Modal title="Развернуть модель" onClose={onClose} footer={
      <div className="form-actions">
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="primary" onClick={deploy} disabled={action.busy || !repo}>
          {action.busy ? "…" : "развернуть"}
        </Button>
      </div>
    }>
      <div className="form-grid two">
        <Field label="Модель на HuggingFace" hint="владелец/название">
          <input type="text" className="mono" value={repo} autoFocus
                 placeholder="Qwen/Qwen3-8B"
                 onChange={(e) => { setRepo(e.target.value); setLayers(null); }} />
        </Field>
        <Field label="Имя для клиентов" hint="по умолчанию — хвост repo">
          <input type="text" value={label} onChange={(e) => setLabel(e.target.value)} />
        </Field>
        <Field label="Точность">
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option>bfloat16</option><option>float16</option><option>float32</option>
          </select>
        </Field>
        <Field label="Устройство">
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option>cuda</option><option>cpu</option>
          </select>
        </Field>
      </div>

      <div style={{ margin: "16px 0", display: "flex", gap: 8, alignItems: "center" }}>
        <Button size="sm" onClick={describe} disabled={!repo || action.busy}>
          узнать число слоёв
        </Button>
        {layers !== null && (
          <span className="sub" style={{ margin: 0 }}>
            {layers} слоёв → примерно по {Math.floor(layers / count)} на стадию
          </span>
        )}
      </div>

      <div className="form-grid two">
        <Field label="Узлы" hint="ничего не выбрано — возьмёт самые свободные">
          <select multiple size={4} value={picked}
                  onChange={(e) => setPicked(
                    Array.from(e.target.selectedOptions, (o) => o.value))}>
            {takers.map((n) => (
              <option key={n.node_id} value={n.node_id}>
                {n.node_id} — {gb(n.vram_free_bytes)} GB, {n.gpus_free} GPU
              </option>
            ))}
          </select>
        </Field>
        <div style={{ display: "grid", gap: 12, alignContent: "start" }}>
          <Field label="Стадий" hint="если узлы не выбраны">
            <input type="number" min={1} value={stages} disabled={picked.length > 0}
                   onChange={(e) => setStages(e.target.value)} />
          </Field>
          <label style={{ display: "flex", gap: 8, alignItems: "center",
                          color: "var(--text-dim)", fontSize: 13 }}>
            <input type="checkbox" checked={byVram} style={{ width: "auto" }}
                   onChange={(e) => setByVram(e.target.checked)} />
            резать слои пропорционально свободной VRAM
          </label>
        </div>
      </div>

      <p className="sub" style={{ marginTop: 16 }}>
        Веса качает сама стадия, только свой диапазон. Первый запуск на узле —
        минуты: ставится окружение и качаются веса. Дальше из кэша.
      </p>
    </Modal>
  );
}

function Chat({ model }: { model: string }) {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState("128");
  const [text, setText] = useState("");
  const [stats, setStats] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLPreElement>(null);

  const ask = async () => {
    if (!model || !prompt || busy) return;
    setBusy(true); setText(""); setStats("");
    const began = Date.now();
    let first: number | null = null, pieces = 0, acc = "";
    try {
      const r = await fetch("/v1/chat/completions", {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model, stream: true, messages: [{ role: "user", content: prompt }],
          max_tokens: Number(maxTokens) || 128,
        }),
      });
      if (!r.ok || !r.body) {
        const b = await r.json().catch(() => ({}));
        throw new Error(b?.error?.message ?? `HTTP ${r.status}`);
      }
      const reader = r.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        // SSE: события разделены пустой строкой, последнее может быть неполным.
        const events = buffer.split("\n\n");
        buffer = events.pop() ?? "";
        for (const event of events) {
          const line = event.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;
          let parsed: any;
          try { parsed = JSON.parse(raw); } catch { continue; }
          if (parsed.error) { acc += `\n[${parsed.error.message}]`; setText(acc); continue; }
          const piece = parsed.choices?.[0]?.delta?.content ?? "";
          if (piece) {
            if (first === null) first = (Date.now() - began) / 1000;
            pieces += 1; acc += piece; setText(acc);
            box.current?.scrollTo({ top: box.current.scrollHeight });
          }
        }
      }
      const total = (Date.now() - began) / 1000;
      setStats(`${total.toFixed(1)}s` +
        (first !== null ? ` · первый токен ${first.toFixed(1)}s` : "") +
        (pieces ? ` · ${(pieces / total).toFixed(1)} кусков/с` : ""));
    } catch (e) {
      setText((prev) => prev + (prev ? "\n\n" : "") + message(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="card">
      <div style={{ display: "flex", gap: 8, alignItems: "flex-end", marginBottom: 12 }}>
        <div style={{ flex: 1 }}>
          <Field label={`Запрос к ${model}`}>
            <input type="text" value={prompt} placeholder="привет"
                   onChange={(e) => setPrompt(e.target.value)}
                   onKeyDown={(e) => e.key === "Enter" && ask()} />
          </Field>
        </div>
        <div style={{ width: 110 }}>
          <Field label="max_tokens">
            <input type="number" value={maxTokens}
                   onChange={(e) => setMaxTokens(e.target.value)} />
          </Field>
        </div>
        <Button kind="primary" onClick={ask} disabled={busy || !prompt}>
          {busy ? "генерация…" : "спросить"}
        </Button>
      </div>
      <pre className="block tall" ref={box}>{text || "ответ появится здесь, потоком"}</pre>
      {stats && <div className="sub">{stats}</div>}
    </div>
  );
}

export function Models() {
  const groups = usePoll<{ groups: Group[] }>("/admin/groups", 5000);
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents", 5000);
  const serving = usePoll<{ data: { id: string }[] }>("/v1/models", 5000);
  const action = useAction(groups.refresh);
  const [deploying, setDeploying] = useState(false);
  const [asking, setAsking] = useState("");
  const [dropping, setDropping] = useState<Group | null>(null);

  const up = new Set((serving.data?.data ?? []).map((m) => m.id));
  const models = (groups.data?.groups ?? []).filter((g) => g.label);

  return (
    <div className="page">
      <header>
        <div>
          <h1>Модели</h1>
          <p>{models.length
            ? `${up.size} отвечает, ${models.length - up.size} поднимается`
            : "ничего не развёрнуто"}</p>
        </div>
        <div className="actions">
          <Button kind="primary" onClick={() => setDeploying(true)}>развернуть</Button>
        </div>
      </header>

      <ErrorLine error={groups.error} />

      {models.length === 0 ? (
        <div className="card">
          <Empty title="Моделей нет">
            Модель режется по слоям между узлами и разворачивается группой стадий.
          </Empty>
        </div>
      ) : (
        <div className="grid cards">
          {models.map((g) => (
            <div className="card" key={g.group_id}>
              <div style={{ display: "flex", alignItems: "center", gap: 10,
                            marginBottom: 12 }}>
                <b style={{ fontSize: 15 }}>{g.label}</b>
                {up.has(g.label)
                  ? <Badge tone="ok">отвечает</Badge>
                  : <Badge tone="warn" pulse>поднимается</Badge>}
                <span style={{ marginLeft: "auto", display: "flex", gap: 6 }}>
                  {up.has(g.label) && (
                    <Button size="sm" onClick={() => setAsking(g.label)}>спросить</Button>
                  )}
                  <Button size="sm" kind="danger"
                          onClick={() => setDropping(g)}>снять</Button>
                </span>
              </div>
              <Pipeline group={g} />
            </div>
          ))}
        </div>
      )}

      {asking && (
        <section style={{ marginTop: 24 }}>
          <h2>Проверка</h2>
          <Chat model={asking} />
        </section>
      )}

      {deploying && (
        <Deploy nodes={nodes.data?.nodes ?? []} onClose={() => setDeploying(false)}
                onDone={groups.refresh} />
      )}

      {dropping && (
        <Confirm
          title={`Снять ${dropping.label}?`}
          action="снять"
          onClose={() => setDropping(null)}
          onConfirm={() => action.run(
            () => send(`/admin/groups/${dropping.group_id}/stop`, "POST"),
            `${dropping.label} снята`)}
          body={<>
            <p>Все {dropping.size} стадии будут остановлены, модель перестанет отвечать.</p>
            <p className="sub" style={{ marginTop: 8 }}>
              Веса и окружение останутся в кэше узлов — повторный запуск будет быстрым.
            </p>
          </>}
        />
      )}
    </div>
  );
}
