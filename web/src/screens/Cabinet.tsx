/** Кабинет клиента: что он может взять, что уже потратил и чем подписывается.
 *
 * Инференс клиент не размещает — на это права только у администратора. Он им
 * пользуется. А кластер арендует сам, вытесняя при нехватке узлов базовую
 * загрузку и оплачивая часы, пока держит ресурс.
 */
import { useState } from "react";
import {
  Badge, Button, Empty, ErrorLine, Field, Modal, Stat, useAction, usePoll, useToast,
} from "../components";
import { send, type Who } from "../lib/api";

interface KeyRow {
  id: number; hint: string; name: string;
  created_at: string; last_used_at: string | null; revoked_at: string | null;
}
interface LeaseLine {
  resource: string; gpu_hours: number; cost: number; currency: string;
  running: number; leases: number;
}
interface TokenLine { model: string; prompt: number; completion: number }
interface Usage { leases: LeaseLine[]; tokens: TokenLine[] }
interface Cluster {
  id: number; group_id: string; label: string; nodes: number; gpus: number;
  opened_at: string; alive: boolean;
}

const money = (kopecks: number, currency: string) =>
  `${(kopecks / 100).toLocaleString("ru-RU", { minimumFractionDigits: 2 })} ${
    currency === "RUB" ? "₽" : currency}`;

export function Cabinet({ who }: { who: Who }) {
  const usage = usePoll<Usage>("/api/usage", 15000);
  const clusters = usePoll<{ clusters: Cluster[] }>("/api/compute", 10000);
  const keys = usePoll<{ keys: KeyRow[] }>("/api/keys", 20000);
  const [renting, setRenting] = useState(false);
  const [fresh, setFresh] = useState("");

  const totals = usage.data?.leases ?? [];
  const tokens = usage.data?.tokens ?? [];
  const spent = totals.reduce((sum, l) => sum + l.cost, 0);
  const currency = totals[0]?.currency ?? "RUB";
  const running = totals.reduce((sum, l) => sum + l.running, 0);

  return (
    <div className="page">
      <ErrorLine error={usage.error || keys.error} />
      <header>
        <div>
          <h1>Мои ресурсы</h1>
          <p>{who.email}</p>
        </div>
        <div className="actions">
          <Button kind="primary" onClick={() => setRenting(true)}>
            Арендовать кластер
          </Button>
        </div>
      </header>

      <div className="grid stats">
        <Stat label="Потрачено" value={money(spent, currency)} />
        <Stat label="Сейчас работает" value={running} sub="аренд" />
        <Stat label="Токенов выдано"
              value={tokens.reduce((s, t) => s + t.completion, 0).toLocaleString("ru-RU")} />
      </div>

      <Clusters clusters={clusters} onChanged={() => { clusters.refresh(); usage.refresh(); }} />

      <section>
        <h2>Потребление</h2>
        {totals.length === 0 ? (
          <Empty title="Пока ничего не израсходовано">
            Аренда кластера считается по GPU-часам, инференс — по токенам.
          </Empty>
        ) : (
          <div className="card pad0"><table>
            <thead>
              <tr><th>Ресурс</th><th>GPU-часов</th><th>Аренд</th><th>Стоимость</th></tr>
            </thead>
            <tbody>
              {totals.map((l) => (
                <tr key={l.resource}>
                  <td className="mono">
                    {l.resource}
                    {l.running > 0 && <Badge tone="ok" pulse>идёт</Badge>}
                  </td>
                  <td className="num">{l.gpu_hours.toFixed(2)}</td>
                  <td className="num">{l.leases}</td>
                  <td className="num">{money(l.cost, l.currency)}</td>
                </tr>
              ))}
            </tbody>
          </table></div>
        )}
      </section>

      {tokens.length > 0 && (
        <section>
          <h2>Инференс</h2>
          <table>
            <thead><tr><th>Модель</th><th>Промпт</th><th>Ответ</th></tr></thead>
            <tbody>
              {tokens.map((t) => (
                <tr key={t.model}>
                  <td className="mono">{t.model}</td>
                  <td className="num">{t.prompt.toLocaleString("ru-RU")}</td>
                  <td className="num">{t.completion.toLocaleString("ru-RU")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </section>
      )}

      <Keys keys={keys} fresh={fresh} setFresh={setFresh} />

      {renting && (
        <RentCluster onClose={() => setRenting(false)}
                     onDone={() => { usage.refresh(); clusters.refresh(); }} />
      )}
    </div>
  );
}

function Clusters({ clusters, onChanged }: {
  clusters: ReturnType<typeof usePoll<{ clusters: Cluster[] }>>;
  onChanged: () => void;
}) {
  const action = useAction(onChanged);
  const rows = clusters.data?.clusters ?? [];
  if (rows.length === 0) return null;
  return (
    <section>
      <h2>Мои кластеры</h2>
      <div className="card pad0"><table>
        <thead>
          <tr><th>Кластер</th><th>Узлов</th><th>С</th><th>Состояние</th><th /></tr>
        </thead>
        <tbody>
          {rows.map((c) => (
            <tr key={c.group_id}>
              <td className="mono">{c.group_id}</td>
              <td className="num">{c.nodes}</td>
              <td>{new Date(c.opened_at).toLocaleString("ru-RU")}</td>
              <td>{c.alive
                ? <Badge tone="ok" pulse>работает</Badge>
                : <Badge tone="warn">не отвечает</Badge>}</td>
              <td>
                <Button size="sm" kind="ghost" disabled={action.busy}
                        onClick={() => action.run(async () => {
                          await send(`/api/compute/${c.group_id}`, "DELETE");
                        }, "кластер снят")}>снять</Button>
              </td>
            </tr>
          ))}
        </tbody>
      </table></div>
      <p className="sub">
        Счёт идёт, пока кластер держит ресурс. Снятие возвращает платформе
        модели, которые были подвинуты ради этой аренды.
      </p>
    </section>
  );
}

function Keys({ keys, fresh, setFresh }: {
  keys: ReturnType<typeof usePoll<{ keys: KeyRow[] }>>;
  fresh: string; setFresh: (v: string) => void;
}) {
  const toast = useToast();
  const action = useAction(() => keys.refresh());
  const rows = (keys.data?.keys ?? []).filter((k) => !k.revoked_at);

  const create = () => action.run(async () => {
    const made = await send<{ key: string }>("/api/keys", "POST", { name: "" });
    // Показывается один раз: в базе только хэш, и восстановить его нельзя.
    setFresh(made.key);
  });

  return (
    <section>
      <h2>Ключи API</h2>
      <div className="tools">
        <Button size="sm" onClick={create} disabled={action.busy}>Создать ключ</Button>
      </div>
      {fresh && (
        <div className="notice">
          <p><b>Сохраните ключ сейчас — он больше не будет показан.</b></p>
          <code className="mono">{fresh}</code>
          <Button size="sm" onClick={() => { navigator.clipboard?.writeText(fresh); toast("ok", "скопирован"); }}>
            скопировать
          </Button>
        </div>
      )}
      {rows.length === 0 ? (
        <Empty title="Ключей нет">Ключ подписывает запросы к инференсу.</Empty>
      ) : (
        <div className="card pad0"><table>
          <thead><tr><th>Ключ</th><th>Создан</th><th>Использован</th><th /></tr></thead>
          <tbody>
            {rows.map((k) => (
              <tr key={k.id}>
                <td className="mono">{k.hint}…</td>
                <td>{new Date(k.created_at).toLocaleDateString("ru-RU")}</td>
                <td>{k.last_used_at
                  ? new Date(k.last_used_at).toLocaleDateString("ru-RU")
                  : <span className="sub">ни разу</span>}</td>
                <td>
                  <Button size="sm" kind="ghost"
                          onClick={() => action.run(async () => {
                            await send(`/api/keys/${k.id}`, "DELETE");
                          })}>
                    отозвать
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table></div>
      )}
    </section>
  );
}

function RentCluster({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const toast = useToast();
  const action = useAction(() => { onDone(); onClose(); });
  const [size, setSize] = useState("2");
  const [hours, setHours] = useState("6");

  const rent = () => action.run(async () => {
    // /api/compute, а не /admin/ray: админский префикс клиенту закрыт, и
    // раньше аренда из кабинета упиралась в 403.
    const made = await send<{ group_id: string }>("/api/compute", "POST", {
      size: Number(size) || 1,
      hours: Math.max(1, Number(hours) || 1),
    });
    toast("ok", `кластер ${made.group_id} поднимается`);
  });

  return (
    <Modal title="Арендовать Ray-кластер" onClose={onClose} footer={
      <div className="form-actions">
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="primary" onClick={rent} disabled={action.busy}>
          {action.busy ? "…" : "арендовать"}
        </Button>
      </div>
    }>
      <div className="form-grid two">
        <Field label="Узлов" hint="по одной карте с узла">
          <input type="number" min={1} value={size}
                 onChange={(e) => setSize(e.target.value)} />
        </Field>
        <Field label="Часов" hint="не больше 24 — потолок на одну аренду">
          <input type="number" min={1} max={24} value={hours}
                 onChange={(e) => setHours(e.target.value)} />
        </Field>
      </div>
      <p className="sub" style={{ marginTop: 14 }}>
        Свободных узлов может не хватить — тогда платформа снимет часть своих
        моделей и вернёт их, когда аренда закончится. Счёт идёт за всё время,
        пока кластер держит ресурс.
      </p>
    </Modal>
  );
}
