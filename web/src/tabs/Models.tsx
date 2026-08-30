import { useCallback, useEffect, useRef, useState } from "react";
import { gb, get, send } from "../api";
import type { Group, GroupHealth, Node, StageHealth } from "../types";
import { Field, Note, State, useAction, usePoll } from "../ui";

/** Что стадия делает сейчас. "running" — про процесс, а не про готовность:
 *  веса грузятся минутами, и всё это время состояние задачи одинаково. */
function phase(stage: StageHealth): [string, number] {
  if (stage.state === "pending") return ["в очереди", 5];
  if (stage.state === "provisioning") return ["окружение и веса", 25];
  if (stage.state === "failed") return ["упала", 100];
  if (stage.state === "cancelled") return ["снята", 100];
  if (stage.state !== "running") return [stage.state, 0];
  if (!stage.stage) return ["стартует", 55];
  return stage.ready ? ["готова", 100] : ["грузит веса", 80];
}

function Stages({ group }: { group: Group }) {
  const [health, setHealth] = useState<GroupHealth | null>(null);
  useEffect(() => {
    let alive = true;
    const pull = () =>
      get<GroupHealth>(`/admin/groups/${group.group_id}/health`)
        .then((body) => alive && setHealth(body))
        .catch(() => undefined);
    pull();
    const timer = setInterval(pull, 4000);
    return () => { alive = false; clearInterval(timer); };
  }, [group.group_id]);

  const stages = health?.stages ?? group.ranks.map((r) => ({
    ...r, state: "?", error: "", seconds: 0, ready: false, stage: null,
  } as StageHealth));

  return (
    <>
      {stages.map((s) => {
        const [what, percent] = phase(s);
        return (
          <div className="stage" key={s.rank}>
            <div className="head">
              <b>rank {s.rank}</b>
              <span className="dim">{s.node_id}</span>
              {s.stage?.layers && (
                <span className="dim">
                  слои [{s.stage.layers[0]}, {s.stage.layers[1]})
                </span>
              )}
              <State value={s.ready ? "ready" : s.state} />
              <span className="dim">{what}</span>
              {s.seconds > 0 && <span className="dim">{s.seconds}s</span>}
            </div>
            {s.error && <div className="bad small">{s.error}</div>}
            <div className="bar"><i style={{ width: `${percent}%` }} /></div>
          </div>
        );
      })}
    </>
  );
}

function Chat({ model }: { model: string }) {
  const [prompt, setPrompt] = useState("");
  const [maxTokens, setMaxTokens] = useState("128");
  const [out, setOut] = useState("");
  const [busy, setBusy] = useState(false);
  const box = useRef<HTMLPreElement>(null);

  const ask = useCallback(async () => {
    if (!model || !prompt) return;
    setBusy(true);
    setOut("");
    const started = Date.now();
    let first: number | null = null;
    let pieces = 0;
    try {
      const response = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          model, stream: true,
          messages: [{ role: "user", content: prompt }],
          max_tokens: Number(maxTokens) || 128,
        }),
      });
      if (!response.ok || !response.body) {
        const body = await response.json().catch(() => ({}));
        throw new Error(body?.error?.message ?? response.status);
      }
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      let text = "";
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
          const payload = line.slice(6).trim();
          if (payload === "[DONE]") continue;
          let parsed: any;
          try { parsed = JSON.parse(payload); } catch { continue; }
          if (parsed.error) { text += `\n[${parsed.error.message}]`; continue; }
          const piece = parsed.choices?.[0]?.delta?.content ?? "";
          if (piece) {
            if (first === null) first = (Date.now() - started) / 1000;
            pieces += 1;
            text += piece;
            setOut(text);
            box.current?.scrollTo({ top: box.current.scrollHeight });
          }
        }
      }
      const seconds = (Date.now() - started) / 1000;
      setOut(text + `\n\n[${seconds.toFixed(1)}s` +
        (first !== null ? `, первый через ${first.toFixed(1)}s` : "") +
        (pieces ? `, ${(pieces / seconds).toFixed(1)} кусков/с]` : "]"));
    } catch (exc) {
      setOut((prev) => prev + (prev ? "\n\n" : "") + String(exc));
    } finally {
      setBusy(false);
    }
  }, [model, prompt, maxTokens]);

  return (
    <>
      <div className="row">
        <Field label="запрос">
          <input value={prompt} onChange={(e) => setPrompt(e.target.value)}
                 onKeyDown={(e) => e.key === "Enter" && ask()}
                 placeholder="привет" style={{ width: 420 }} />
        </Field>
        <Field label="max_tokens">
          <input value={maxTokens} onChange={(e) => setMaxTokens(e.target.value)}
                 style={{ width: 90 }} />
        </Field>
        <Field label="&nbsp;">
          <button className="action" onClick={ask} disabled={busy || !model}>
            {busy ? "…" : "спросить"}
          </button>
        </Field>
      </div>
      <pre className="logs" ref={box}>{out || "ответ появится здесь, потоком"}</pre>
    </>
  );
}

export function Models() {
  const groups = usePoll<{ groups: Group[] }>("/admin/groups", 5000);
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents", 5000);
  const serving = usePoll<{ data: { id: string }[] }>("/v1/models", 5000);
  const action = useAction(groups.refresh);

  const [repo, setRepo] = useState("");
  const [label, setLabel] = useState("");
  const [dtype, setDtype] = useState("bfloat16");
  const [device, setDevice] = useState("cuda");
  const [stages, setStages] = useState("2");
  const [byVram, setByVram] = useState(true);
  const [picked, setPicked] = useState<string[]>([]);
  const [asking, setAsking] = useState("");

  const describe = () =>
    action.run(async () => {
      const info = await send<{ repo: string; num_layers: number; architecture: string }>(
        "/admin/models/describe", "POST", { repo });
      action.setNote(`${info.repo}: ${info.num_layers} слоёв, ${info.architecture || "?"}`);
    });

  const deploy = () =>
    action.run(async () => {
      const body: Record<string, unknown> = { repo, dtype, device, by_vram: byVram };
      if (label) body.label = label;
      if (picked.length) body.node_ids = picked;
      else body.stages = Number(stages) || 1;
      const created = await send<{
        label: string;
        model: { num_layers: number };
        split: { rank: number; node_id: string; start_layer: number; end_layer: number }[];
      }>("/admin/deploy", "POST", body);
      action.setNote(
        `${created.label}: ${created.model.num_layers} слоёв\n` +
        created.split.map((s) =>
          `  rank ${s.rank}  [${s.start_layer}, ${s.end_layer})  ${s.node_id}`).join("\n"));
    });

  const up = new Set((serving.data?.data ?? []).map((m) => m.id));
  const deployed = (groups.data?.groups ?? []).filter((g) => g.label);
  const takers = (nodes.data?.nodes ?? []).filter((n) => n.accepts_tasks);

  return (
    <>
      <h2>Развернуть</h2>
      <div className="row">
        <Field label="модель" hint="имя на huggingface">
          <input value={repo} onChange={(e) => setRepo(e.target.value)}
                 placeholder="Qwen/Qwen3-8B" style={{ width: 240 }} />
        </Field>
        <Field label="имя для клиентов" hint="по умолчанию — хвост repo">
          <input value={label} onChange={(e) => setLabel(e.target.value)}
                 style={{ width: 190 }} />
        </Field>
        <Field label="dtype">
          <select value={dtype} onChange={(e) => setDtype(e.target.value)}>
            <option>bfloat16</option><option>float16</option><option>float32</option>
          </select>
        </Field>
        <Field label="device">
          <select value={device} onChange={(e) => setDevice(e.target.value)}>
            <option>cuda</option><option>cpu</option>
          </select>
        </Field>
        <Field label="&nbsp;">
          <button onClick={describe} disabled={action.busy || !repo}>слоёв?</button>
        </Field>
      </div>
      <div className="row">
        <Field label="узлы" hint="пусто — возьмёт самые свободные">
          <select multiple size={3} value={picked} style={{ minWidth: 300 }}
                  onChange={(e) => setPicked(
                    Array.from(e.target.selectedOptions, (o) => o.value))}>
            {takers.map((n) => (
              <option key={n.node_id} value={n.node_id}>
                {n.node_id} · {gb(n.vram_free_bytes)} GB
              </option>
            ))}
          </select>
        </Field>
        <Field label="стадий" hint="если узлы не выбраны">
          <input value={stages} onChange={(e) => setStages(e.target.value)}
                 style={{ width: 60 }} />
        </Field>
        <Field label="разрез слоёв">
          <label className="check">
            <input type="checkbox" checked={byVram}
                   onChange={(e) => setByVram(e.target.checked)} /> по VRAM
          </label>
        </Field>
        <Field label="&nbsp;">
          <button className="action" onClick={deploy} disabled={action.busy || !repo}>
            развернуть
          </button>
        </Field>
      </div>
      <p className="dim">
        Веса качает сама стадия, только свой диапазон. Первый запуск на узле —
        минуты: окружение и веса. Дальше из кэша.
      </p>
      <Note text={action.note} />

      <h2>Развёрнуто</h2>
      {deployed.length === 0 && <p className="dim">ничего не развёрнуто</p>}
      {deployed.map((g) => (
        <div className="group" key={g.group_id}>
          <div className="head">
            <b className="acc">{g.label}</b>
            <State value={up.has(g.label) ? "ready" : "loading"} />
            <span className="dim">{g.size} стадии</span>
            <button onClick={() => action.run(
              () => send(`/admin/groups/${g.group_id}/stop`, "POST"), "снято")}>
              снять
            </button>
            {up.has(g.label) && (
              <button onClick={() => setAsking(g.label)}>спросить</button>
            )}
          </div>
          <Stages group={g} />
        </div>
      ))}

      <h2>Проверить{asking && <span className="acc"> · {asking}</span>}</h2>
      <Chat model={asking} />
    </>
  );
}
