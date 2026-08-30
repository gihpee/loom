import { useState } from "react";
import { send } from "../api";
import type { JoinKey } from "../types";
import { Field, Note, Table, useAction, usePoll } from "../ui";

export function Keys() {
  const keys = usePoll<{ keys: JoinKey[] }>("/admin/keys", 10000);
  const action = useAction(keys.refresh);
  const [label, setLabel] = useState("");
  const [maxNodes, setMaxNodes] = useState("0");
  const [command, setCommand] = useState("");

  const issue = () =>
    action.run(async () => {
      const key = await send<JoinKey>("/admin/keys", "POST", {
        label,
        max_nodes: Number(maxNodes) || 0,
      });
      setCommand(
        `docker run -d --gpus all --restart unless-stopped --network host \\\n` +
        `  -v loom-data:/var/lib/loom ${key.agent_image} --key ${key.key}`,
      );
      setLabel("");
    });

  const rows = (keys.data?.keys ?? []).map((k) => [
    <code>{k.key_id}</code>,
    k.label || <span className="dim">—</span>,
    `${k.nodes?.length ?? 0}${k.max_nodes ? ` / ${k.max_nodes}` : ""}`,
    <button onClick={() => action.run(() => send(`/admin/keys/${k.key_id}`, "DELETE"))}>
      отозвать
    </button>,
  ]);

  return (
    <>
      <h2>Выдать ключ</h2>
      <div className="row">
        <Field label="метка" hint="чья машина — видно только вам">
          <input value={label} onChange={(e) => setLabel(e.target.value)}
                 placeholder="машина Миши" style={{ width: 220 }} />
        </Field>
        <Field label="лимит узлов" hint="0 — без лимита">
          <input value={maxNodes} onChange={(e) => setMaxNodes(e.target.value)}
                 style={{ width: 90 }} />
        </Field>
        <Field label="&nbsp;">
          <button className="action" onClick={issue} disabled={action.busy}>выдать</button>
        </Field>
      </div>
      <Note text={action.note} />
      {command && (
        <>
          <h2>Команда для владельца машины</h2>
          <pre className="pick">{command}</pre>
          <p className="dim">
            Адрес оркестратора внутри ключа. <code>--network host</code> нужен для
            p2p, том — чтобы кэш и обновления пережили перезапуск.
          </p>
        </>
      )}
      <h2>Ключи</h2>
      <Table head={["id", "метка", "узлов", ""]} rows={rows} empty="ключей нет" />
    </>
  );
}
