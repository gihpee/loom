import { ReactNode, useEffect, useRef, useState } from "react";
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
  const [engine, setEngine] = useState("torch");
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
    const body: Record<string, unknown> = { repo, dtype, device, engine, by_vram: byVram };
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
  // vLLM без карты не поднимается вовсе. Сказать это здесь, а не дать
  // оркестратору отказать после нажатия: сочетание видно на экране целиком,
  // и объяснять его лучше рядом с тем, что его создало.
  const clash = engine === "vllm" && device !== "cuda"
    ? "vLLM работает только на cuda" : "";

  return (
    <Modal title="Развернуть модель" onClose={onClose} footer={
      <div className="form-actions">
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="primary" onClick={deploy}
                disabled={action.busy || !repo || !!clash}>
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
        <Field label="Движок"
               hint={clash || (engine === "vllm"
                 ? "несколько запросов в одном шаге — параллельные клиенты складывают пропускную способность, а не делят её"
                 : "работает везде, в том числе на cpu; параллельные запросы делят одну карту по очереди")}>
          <select value={engine} onChange={(e) => setEngine(e.target.value)}>
            <option value="torch">transformers — переносимый</option>
            <option value="vllm">vLLM — батчинг, только cuda</option>
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
        {engine === "vllm" && " Для vLLM первый запуск дольше: он ставится в " +
          "окружение узла отдельно и весит немало."}
      </p>
    </Modal>
  );
}

interface Usage { prompt_tokens: number; completion_tokens: number }
interface Timings {
  ttft_ms: number; total_ms: number; decode_ms: number;
  decode_tokens_per_s: number; tokens_per_s: number;
  inter_token_ms_p50: number; inter_token_ms_p95: number; inter_token_ms_max: number;
  stages: number;
}

function Metric({ label, value, unit }: { label: string; value: ReactNode; unit?: string }) {
  return (
    <div>
      <div style={{ color: "var(--text-mute)", fontSize: 11, textTransform: "uppercase",
                    letterSpacing: ".05em" }}>{label}</div>
      <div className="num" style={{ fontSize: 17, fontWeight: 600, marginTop: 2 }}>
        {value}{unit && <small style={{ fontSize: 12, color: "var(--text-mute)",
                                        fontWeight: 400 }}> {unit}</small>}
      </div>
    </div>
  );
}

function Chat({ model }: { model: string }) {
  const toast = useToast();
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState("128");
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [usage, setUsage] = useState<Usage | null>(null);
  const [timings, setTimings] = useState<Timings | null>(null);
  const [live, setLive] = useState<{ pieces: number; first: number | null; elapsed: number }>(
    { pieces: 0, first: null, elapsed: 0 });
  const [events, setEvents] = useState<unknown[]>([]);
  const box = useRef<HTMLPreElement>(null);

  const ask = async () => {
    if (!model || !prompt || busy) return;
    setBusy(true); setText(""); setUsage(null); setTimings(null); setEvents([]);
    setLive({ pieces: 0, first: null, elapsed: 0 });
    const began = Date.now();
    const collected: unknown[] = [];
    let first: number | null = null, pieces = 0, acc = "";
    const ticker = setInterval(
      () => setLive((s) => ({ ...s, elapsed: (Date.now() - began) / 1000 })), 100);
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
        const chunks = buffer.split("\n\n");
        buffer = chunks.pop() ?? "";
        for (const chunk of chunks) {
          const line = chunk.split("\n").find((l) => l.startsWith("data: "));
          if (!line) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") continue;
          let parsed: any;
          try { parsed = JSON.parse(raw); } catch { continue; }
          collected.push(parsed);
          if (parsed.error) {
            // Стадия шлёт ошибку строкой; клиентские — объектом. Раньше
            // читалось только второе, и в окне появлялось «[undefined]».
            const what = typeof parsed.error === "string"
              ? parsed.error : parsed.error.message ?? JSON.stringify(parsed.error);
            acc += (acc ? "\n\n" : "") + `[${what}]`;
            setText(acc);
            continue;
          }
          if (parsed.usage) setUsage(parsed.usage);
          if (parsed.timings) setTimings(parsed.timings);
          const piece = parsed.choices?.[0]?.delta?.content ?? "";
          if (piece) {
            if (first === null) { first = (Date.now() - began) / 1000; }
            pieces += 1; acc += piece;
            setText(acc);
            setLive({ pieces, first, elapsed: (Date.now() - began) / 1000 });
            box.current?.scrollTo({ top: box.current.scrollHeight });
          }
        }
      }
      setEvents(collected);
    } catch (e) {
      setText((prev) => prev + (prev ? "\n\n" : "") + message(e));
    } finally {
      clearInterval(ticker);
      setLive((s) => ({ ...s, elapsed: (Date.now() - began) / 1000 }));
      setBusy(false);
    }
  };

  const save = () => {
    const body = JSON.stringify(
      { model, prompt, text, usage, timings, events }, null, 2);
    const url = URL.createObjectURL(new Blob([body], { type: "application/json" }));
    const a = document.createElement("a");
    a.href = url;
    a.download = `${model}-${new Date().toISOString().slice(0, 19)}.json`;
    document.body.appendChild(a); a.click(); a.remove();
    URL.revokeObjectURL(url);
  };

  const perSecond = timings?.decode_tokens_per_s
    ?? (live.pieces && live.elapsed ? live.pieces / live.elapsed : 0);

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

      <pre className="block tall" ref={box}>{text || "ответ появится здесь, стримом"}</pre>

      {(busy || live.pieces > 0 || timings) && (
        <div style={{
          display: "grid", gap: 16, marginTop: 14, padding: "14px 0 0",
          borderTop: "1px solid var(--line-soft)",
          gridTemplateColumns: "repeat(auto-fit, minmax(112px, 1fr))",
        }}>
          <Metric label="время" value={(timings ? timings.total_ms / 1000 : live.elapsed)
            .toFixed(1)} unit="s" />
          <Metric label="первый токен"
                  value={timings ? (timings.ttft_ms / 1000).toFixed(2)
                                 : live.first?.toFixed(2) ?? "—"} unit="s" />
          <Metric label="скорость" value={perSecond ? perSecond.toFixed(1) : "—"}
                  unit="tok/s" />
          <Metric label="токенов"
                  value={usage?.completion_tokens ?? live.pieces} />
          {timings && <Metric label="между токенами"
                              value={timings.inter_token_ms_p50.toFixed(0)} unit="ms p50" />}
          {timings && <Metric label="худший разрыв"
                              value={timings.inter_token_ms_max.toFixed(0)} unit="ms" />}
          {timings && <Metric label="стадий" value={timings.stages} />}
          {usage && <Metric label="промпт" value={usage.prompt_tokens} unit="tok" />}
        </div>
      )}

      {(text || events.length > 0) && !busy && (
        <div style={{ display: "flex", gap: 8, marginTop: 14 }}>
          <Button size="sm" onClick={() => {
            navigator.clipboard.writeText(text); toast("ok", "ответ скопирован");
          }}>копировать ответ</Button>
          <Button size="sm" onClick={save}>скачать JSON</Button>
          <span className="sub" style={{ margin: "auto 0 auto 0" }}>
            {events.length} событий в потоке
          </span>
        </div>
      )}
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
