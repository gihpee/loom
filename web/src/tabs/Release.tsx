import { useState } from "react";
import { mb, send } from "../api";
import type { VersionMap } from "../types";
import { base64, Field, Note, Table, useAction, useFile, usePoll } from "../ui";

export function Release() {
  const map = usePoll<VersionMap>("/admin/release", 5000);
  const action = useAction(map.refresh);
  const manifest = useFile();
  const archive = useFile();
  const [version, setVersion] = useState("");
  const [signature, setSignature] = useState("");

  // Манифест только ЗАПОЛНЯЕТ поля. Выбор архива их не трогает — это разные
  // input'ы и разное состояние.
  const readManifest = async () => {
    const body = await manifest.text();
    if (!body) return;
    try {
      const parsed = JSON.parse(body);
      setVersion(parsed.version ?? "");
      setSignature(parsed.signature ?? "");
      action.setNote(`манифест: ${parsed.version}, sha256 ${(parsed.sha256 ?? "").slice(0, 12)}…`);
    } catch (exc) {
      action.setNote(`манифест не читается: ${exc}`);
    }
  };

  const publish = () =>
    action.run(async () => {
      const bytes = await archive.read();
      if (!bytes) throw new Error("нужен архив .tar.gz");
      const published = await send<{ version: string }>("/admin/release", "POST", {
        version, signature, archive: base64(bytes),
      });
      return published;
    }, "опубликовано, выкатка на 0%");

  const wave = (percent: number) =>
    action.run(async () => {
      const answer = await send<{ wave_percent: number; nodes_told: number }>(
        "/admin/release/wave", "POST", { percent });
      action.setNote(
        `выкатка ${answer.wave_percent}% · уведомлено узлов: ${answer.nodes_told}`);
    });

  const versions = Object.entries(map.data?.versions ?? {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, count]) => [
      <code>{name}</code>, count,
      `${Math.round((count / (map.data?.nodes_total || 1)) * 100)}%`,
    ]);
  const current = map.data?.release;

  return (
    <>
      <h2>Выкатка</h2>
      {current ? (
        <>
          <p>
            <b>{current.version}</b> · {mb(current.size_bytes)} MB ·{" "}
            <b>{current.wave_percent}%</b>{" "}
            <span className="dim">
              в волне {map.data?.nodes_in_wave}, уже на ней{" "}
              {map.data?.nodes_on_target} из {map.data?.nodes_total}
            </span>
          </p>
          <div className="row">
            {[10, 25, 50, 100].map((p) => (
              <button key={p} onClick={() => wave(p)} disabled={action.busy}>
                {p === 100 ? "весь парк" : `${p}%`}
              </button>
            ))}
            <button className="danger" onClick={() => action.run(
              () => send("/admin/release/withdraw", "POST"), "выкатка остановлена")}>
              остановить
            </button>
          </div>
          <p className="dim">
            Волна выбирается по имени узла, поэтому узел остаётся в ней при
            переподключении. Ступень переводится вручную: регистрация не означает,
            что версия работает.
          </p>
        </>
      ) : (
        <p className="dim">ничего не опубликовано</p>
      )}
      <Note text={action.note} />

      <h2>Версии в парке</h2>
      <Table head={["версия", "узлов", "доля"]} rows={versions} empty="узлов нет" />

      <h2>Опубликовать</h2>
      <div className="row">
        <Field label="манифест .json" hint="заполнит версию и подпись">
          <input type="file" accept=".json" ref={manifest.ref} onChange={readManifest} />
        </Field>
        <Field label="архив .tar.gz">
          <input type="file" accept=".gz,.tgz" ref={archive.ref} />
        </Field>
        <Field label="&nbsp;">
          <button className="action" onClick={publish}
                  disabled={action.busy || !version || !signature}>
            опубликовать
          </button>
        </Field>
      </div>
      <div className="row">
        <Field label="версия">
          <input value={version} onChange={(e) => setVersion(e.target.value)}
                 style={{ width: 110 }} />
        </Field>
        <Field label="подпись" hint="ed25519, 128 hex">
          <input value={signature} onChange={(e) => setSignature(e.target.value)}
                 style={{ width: 460 }} />
        </Field>
      </div>
      <p className="dim">
        Оба файла делает <code>scripts/sign_release.py sign</code> — из одного
        прогона, иначе подпись не подойдёт к архиву. Публикация не начинает выкатку.
      </p>
    </>
  );
}
