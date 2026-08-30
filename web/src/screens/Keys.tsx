import { useState } from "react";
import { send } from "../lib/api";
import type { Connect, JoinKey } from "../lib/types";
import {
  Button, Confirm, Empty, ErrorLine, Field, Modal,
  useAction, usePoll, useToast,
} from "../components";

export function Keys() {
  const keys = usePoll<{ keys: JoinKey[] }>("/admin/keys", 10000);
  const connect = usePoll<Connect>("/admin/connect", 30000);
  const action = useAction(keys.refresh);
  const toast = useToast();
  const [issuing, setIssuing] = useState(false);
  const [label, setLabel] = useState("");
  const [maxNodes, setMaxNodes] = useState("0");
  const [command, setCommand] = useState("");
  const [revoking, setRevoking] = useState<JoinKey | null>(null);

  const issue = () => action.run(async () => {
    const key = await send<JoinKey>("/admin/keys", "POST",
      { label, max_nodes: Number(maxNodes) || 0 });
    setCommand(
      `docker run -d --gpus all --restart unless-stopped --network host \\\n` +
      `  -v loom-data:/var/lib/loom ${key.agent_image} --key ${key.key}`);
    setLabel(""); setIssuing(false);
  }, "ключ выдан");

  const list = keys.data?.keys ?? [];

  return (
    <div className="page">
      <header>
        <div>
          <h1>Ключи подключения</h1>
          <p>адрес оркестратора внутри ключа — владелец машины больше ничего не вводит</p>
        </div>
        <div className="actions">
          <Button kind="primary" onClick={() => setIssuing(true)}>выдать ключ</Button>
        </div>
      </header>

      <ErrorLine error={keys.error} />

      {command && (
        <section>
          <h2>Команда для владельца машины</h2>
          <div className="card">
            <pre className="block pickable">{command}</pre>
            <div style={{ display: "flex", gap: 8, marginTop: 12 }}>
              <Button size="sm" onClick={() => {
                navigator.clipboard.writeText(command);
                toast("ok", "скопировано");
              }}>скопировать</Button>
              <Button size="sm" kind="ghost" onClick={() => setCommand("")}>скрыть</Button>
            </div>
            <p className="sub" style={{ marginTop: 12 }}>
              <code>--network host</code> — иначе прямой канал между узлами невозможен.{" "}
              <code>-v loom-data</code> — иначе кэш и обновления не переживут перезапуск.
            </p>
          </div>
        </section>
      )}

      <div className="card pad0">
        {list.length === 0 ? (
          <Empty title="Ключей нет">
            Каждый ключ — приглашение для одной или нескольких машин.
          </Empty>
        ) : (
          <table>
            <thead>
              <tr><th>id</th><th>метка</th><th>узлы</th><th>лимит</th><th /></tr>
            </thead>
            <tbody>
              {list.map((k) => (
                <tr key={k.key_id}>
                  <td><code>{k.key_id}</code></td>
                  <td>{k.label || <span className="sub">—</span>}</td>
                  <td>
                    {k.nodes?.length
                      ? k.nodes.map((n) => <div key={n}><code style={{ fontSize: 12 }}>{n}</code></div>)
                      : <span className="sub">никто не подключился</span>}
                  </td>
                  <td className="num">{k.max_nodes || "∞"}</td>
                  <td style={{ width: 1 }}>
                    <Button size="sm" kind="danger"
                            onClick={() => setRevoking(k)}>отозвать</Button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {issuing && (
        <Modal title="Выдать ключ" onClose={() => setIssuing(false)} footer={
          <div className="form-actions">
            <Button kind="ghost" onClick={() => setIssuing(false)}>отмена</Button>
            <Button kind="primary" onClick={issue} disabled={action.busy}>выдать</Button>
          </div>
        }>
          <div className="form-grid two">
            <Field label="Метка" hint="видно только вам">
              <input type="text" autoFocus value={label} placeholder="машина 1"
                     onChange={(e) => setLabel(e.target.value)} />
            </Field>
            <Field label="Лимит узлов" hint="0 — без лимита">
              <input type="number" min={0} value={maxNodes}
                     onChange={(e) => setMaxNodes(e.target.value)} />
            </Field>
          </div>
          <p className="sub" style={{ marginTop: 16 }}>
            Узлы будут звонить на <code>{connect.data?.dial_address ?? "…"}</code>.
            {connect.data?.severity === "warn" && (
              <span style={{ color: "var(--bad)" }}>
                {" "}Этот адрес недостижим снаружи — подключиться не получится.
              </span>
            )}
          </p>
        </Modal>
      )}

      {revoking && (
        <Confirm
          title={`Отозвать ключ ${revoking.key_id}?`}
          action="отозвать"
          onClose={() => setRevoking(null)}
          onConfirm={() => action.run(
            () => send(`/admin/keys/${revoking.key_id}`, "DELETE"), "ключ отозван")}
          body={<p>
            По нему больше нельзя будет подключиться.
            {revoking.nodes?.length
              ? ` Уже подключённые ${revoking.nodes.length} узлов продолжат работать.`
              : ""}
          </p>}
        />
      )}
    </div>
  );
}
