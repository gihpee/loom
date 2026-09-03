import { useState } from "react";
import { send } from "../lib/api";
import { mb } from "../lib/format";
import type { Node, VersionMap } from "../lib/types";
import {
  Badge, Bar, Button, Confirm, Empty, ErrorLine, Field, Modal,
  StateBadge, useAction, usePoll, useToast,
} from "../components";

export function Release() {
  const map = usePoll<VersionMap>("/admin/release", 5000);
  const nodes = usePoll<{ nodes: Node[] }>("/admin/agents", 4000);
  const action = useAction(map.refresh);
  const toast = useToast();
  const [publishing, setPublishing] = useState(false);
  const [version, setVersion] = useState("");
  const [signature, setSignature] = useState("");
  const [archive, setArchive] = useState<File | null>(null);
  const [stopping, setStopping] = useState(false);

  const readManifest = async (file: File) => {
    try {
      const parsed = JSON.parse(await file.text());
      setVersion(parsed.version ?? "");
      setSignature(parsed.signature ?? "");
      toast("ok", `манифест ${parsed.version}`);
    } catch (e) {
      toast("bad", `манифест не читается: ${e}`);
    }
  };

  const publish = () => action.run(async () => {
    if (!archive) throw new Error("нужен архив .tar.gz");
    const bytes = new Uint8Array(await archive.arrayBuffer());
    let binary = "";
    for (const b of bytes) binary += String.fromCharCode(b);
    await send("/admin/release", "POST",
      { version, signature, archive: btoa(binary) });
    setPublishing(false);
  }, "опубликовано, выкатка на 0%");

  const wave = (percent: number) => action.run(async () => {
    const answer = await send<{ wave_percent: number; nodes_told: number }>(
      "/admin/release/wave", "POST", { percent });
    toast("ok", `выкатка ${answer.wave_percent}% · уведомлено узлов: ${answer.nodes_told}`);
  });

  const current = map.data?.release;
  const list = nodes.data?.nodes ?? [];
  const onTarget = current
    ? list.filter((n) => n.agent_version === current.version).length : 0;

  return (
    <div className="page">
      <header>
        <div>
          <h1>Обновление агента</h1>
          <p>{current
            ? `${current.version} · выкатка ${current.wave_percent}%`
            : "релиз не опубликован"}</p>
        </div>
        <div className="actions">
          <Button kind="primary" onClick={() => setPublishing(true)}>опубликовать</Button>
        </div>
      </header>

      <ErrorLine error={map.error} />

      {current ? (
        <section>
          <div className="card">
            <div style={{ display: "flex", alignItems: "center", gap: 12,
                          marginBottom: 14 }}>
              <b style={{ fontSize: 16 }}>{current.version}</b>
              <span className="sub" style={{ margin: 0 }}>{mb(current.size_bytes)} MB</span>
              <Badge tone={current.wave_percent === 0 ? "dim"
                          : current.wave_percent === 100 ? "info" : "warn"}>
                выкатка {current.wave_percent}%
              </Badge>
              <span style={{ marginLeft: "auto" }}>
                {onTarget} из {list.length} узлов обновились
              </span>
            </div>
            <Bar percent={list.length ? (onTarget / list.length) * 100 : 0}
                 tone={onTarget === list.length && list.length > 0 ? "ok" : undefined} />
            <div style={{ display: "flex", gap: 8, marginTop: 16, flexWrap: "wrap" }}>
              {[10, 25, 50, 100].map((p) => (
                <Button key={p} onClick={() => wave(p)} disabled={action.busy}
                        kind={p === 100 ? "primary" : "default"}>
                  {p === 100 ? "весь парк" : `${p}%`}
                </Button>
              ))}
              <Button kind="danger" onClick={() => setStopping(true)}
                      style={{ marginLeft: "auto" }}>остановить выкатку</Button>
            </div>
            <p className="sub" style={{ marginTop: 14 }}>
              Волна выбирается по имени узла — узел остаётся в ней при переподключении.
              Ступень переводится вручную: регистрация не означает, что версия работает.
            </p>
          </div>
        </section>
      ) : (
        <div className="card">
          <Empty title="Релиз не опубликован">
            Архив и манифест делает <code>scripts/sign_release.py sign</code>.
            Подпись ставится ключом, которого у оркестратора нет.
          </Empty>
        </div>
      )}

      <section>
        <h2>Узлы</h2>
        <div className="card pad0">
          {list.length === 0 ? <Empty title="Узлов нет" /> : (
            <table>
              <thead>
                <tr><th>узел</th><th>версия</th><th>обновление</th><th>что мешает</th></tr>
              </thead>
              <tbody>
                {list.map((n) => (
                  <tr key={n.node_id}
                      // Проход утка́ по списку: строка светится, пока узел
                      // тянет и ставит релиз, и гаснет, когда он это сделал.
                      // Только на время настоящей раскатки — постоянная
                      // анимация в приборной доске мешает читать состояние.
                      data-weft={["fetching", "downloaded"].includes(n.update_state)
                                 ? "on" : undefined}>
                    <td><code>{n.node_id}</code></td>
                    <td>
                      <code>{n.agent_version}</code>
                      {current && n.agent_version === current.version &&
                        <Badge tone="ok">целевая</Badge>}
                    </td>
                    <td>
                      {n.update_state
                        ? <>
                            <StateBadge value={n.update_state}
                                        pulse={n.update_state === "fetching"} />
                            {n.update_version && (
                              <span className="sub" style={{ marginLeft: 6 }}>
                                → {n.update_version}
                              </span>
                            )}
                          </>
                        : <span className="sub">молчит</span>}
                    </td>
                    <td style={{ maxWidth: 380 }}>
                      {n.update_error
                        ? <span style={{ color: "var(--bad)", fontSize: 12.5 }}>
                            {n.update_error}
                          </span>
                        : <span className="sub">—</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      </section>

      {publishing && (
        <Modal title="Опубликовать релиз" onClose={() => setPublishing(false)} footer={
          <div className="form-actions">
            <Button kind="ghost" onClick={() => setPublishing(false)}>отмена</Button>
            <Button kind="primary" onClick={publish}
                    disabled={action.busy || !version || !signature || !archive}>
              опубликовать
            </Button>
          </div>
        }>
          <div className="form-grid two">
            <Field label="Манифест .json" hint="заполнит версию и подпись">
              <label className={`file${version ? " filled" : ""}`}>
                <input type="file" accept=".json" onChange={(e) => {
                  const f = e.target.files?.[0];
                  if (f) readManifest(f);
                }} />
                <span>{version ? `манифест ${version}` : "выбрать манифест"}</span>
              </label>
            </Field>
            <Field label="Архив .tar.gz">
              <label className={`file${archive ? " filled" : ""}`}>
                <input type="file" accept=".gz,.tgz"
                       onChange={(e) => setArchive(e.target.files?.[0] ?? null)} />
                <span>{archive?.name ?? "выбрать архив"}</span>
              </label>
            </Field>
          </div>
          <div className="form-grid" style={{ marginTop: 12 }}>
            <Field label="Версия">
              <input type="text" className="mono" value={version} placeholder="0.2.0"
                     onChange={(e) => setVersion(e.target.value)} />
            </Field>
            <Field label="Подпись" hint="ed25519, 128 hex">
              <input type="text" className="mono" value={signature}
                     onChange={(e) => setSignature(e.target.value)} />
            </Field>
          </div>
          <p className="sub" style={{ marginTop: 16 }}>
            Оба файла — из одного прогона <code>sign_release.py sign</code>, иначе
            подпись не подойдёт к архиву. Публикация не начинает выкатку.
          </p>
        </Modal>
      )}

      {stopping && (
        <Confirm
          title="Остановить выкатку?"
          action="остановить"
          onClose={() => setStopping(false)}
          onConfirm={() => action.run(
            () => send("/admin/release/withdraw", "POST"), "выкатка остановлена")}
          body={<>
            <p>Новым узлам релиз предлагаться не будет.</p>
            <p className="sub" style={{ marginTop: 8 }}>
              Уже обновившиеся останутся на новой версии: вернуть их можно только
              выпустив следующую — узлы отвергают откат на более старую.
            </p>
          </>}
        />
      )}
    </div>
  );
}
