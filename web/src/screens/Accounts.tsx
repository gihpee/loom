/** Учётные записи и ставки. Клиента заводит администратор — регистрация по
 *  приглашению, а не самообслуживанием: сеть в закрытой бете, и за каждую
 *  машину в ней мы отвечаем. */
import { useState } from "react";
import {
  Badge, Button, Empty, ErrorLine, Field, Modal, useAction, usePoll,
} from "../components";
import { send } from "../lib/api";

interface Account {
  id: number; email: string; role: string; display_name: string; disabled: boolean;
}
interface Rate { resource: string; per_hour: number; currency: string }

export function Accounts() {
  const list = usePoll<{ accounts: Account[] }>("/admin/accounts", 15000);
  const rates = usePoll<{ rates: Rate[] }>("/admin/rates", 30000);
  const action = useAction(() => { list.refresh(); rates.refresh(); });
  const [adding, setAdding] = useState(false);

  return (
    <div className="screen">
      <ErrorLine error={list.error || rates.error} />
      <header className="screen-head">
        <h1>Клиенты</h1>
        <Button kind="primary" onClick={() => setAdding(true)}>Завести запись</Button>
      </header>

      {(list.data?.accounts ?? []).length === 0 ? (
        <Empty title="Записей нет">
          Первую заводят из командной строки: scripts/create_admin.py
        </Empty>
      ) : (
        <table className="table">
          <thead><tr><th>Почта</th><th>Роль</th><th>Состояние</th><th /></tr></thead>
          <tbody>
            {(list.data?.accounts ?? []).map((a) => (
              <tr key={a.id}>
                <td>{a.email}</td>
                <td className="mono">{a.role}</td>
                <td>{a.disabled
                  ? <Badge tone="bad">отключён</Badge>
                  : <Badge tone="ok">работает</Badge>}</td>
                <td>
                  <Button size="sm" kind="ghost" disabled={action.busy}
                          onClick={() => action.run(async () => {
                            await send(`/admin/accounts/${a.id}/disabled`, "POST",
                                       { disabled: !a.disabled });
                          }, a.disabled ? "включён" : "отключён")}>
                    {a.disabled ? "включить" : "отключить"}
                  </Button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}

      <section>
        <h2>Ставки</h2>
        <p className="sub">
          В копейках за GPU-час. Ставка запоминается в аренде: поднятая завтра
          цена не перепишет вчерашние счета.
        </p>
        <div className="cards">
          {(rates.data?.rates ?? []).map((r) => (
            <RateCard key={r.resource} rate={r} onSaved={() => rates.refresh()} />
          ))}
        </div>
      </section>

      {adding && <NewAccount onClose={() => setAdding(false)}
                             onDone={() => list.refresh()} />}
    </div>
  );
}

function RateCard({ rate, onSaved }: { rate: Rate; onSaved: () => void }) {
  const action = useAction(onSaved);
  const [value, setValue] = useState(String(rate.per_hour));
  return (
    <div className="card stat">
      <div className="label mono">{rate.resource}</div>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <input type="number" min={0} value={value}
               onChange={(e) => setValue(e.target.value)} />
        <Button size="sm" disabled={action.busy}
                onClick={() => action.run(async () => {
                  await send("/admin/rates", "POST", {
                    resource: rate.resource, per_hour: Number(value) || 0,
                  });
                }, "ставка сохранена")}>ok</Button>
      </div>
      <div className="sub">{(Number(value) / 100).toFixed(2)} ₽ за GPU-час</div>
    </div>
  );
}

function NewAccount({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const action = useAction(() => { onDone(); onClose(); });
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState("client");

  return (
    <Modal title="Новая учётная запись" onClose={onClose} footer={
      <div className="form-actions">
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="primary" disabled={action.busy || !email || password.length < 10}
                onClick={() => action.run(async () => {
                  await send("/admin/accounts", "POST", { email, password, role });
                }, "запись заведена")}>завести</Button>
      </div>
    }>
      <div className="form-grid two">
        <Field label="Почта"><input type="email" value={email}
               onChange={(e) => setEmail(e.target.value)} /></Field>
        <Field label="Роль">
          <select value={role} onChange={(e) => setRole(e.target.value)}>
            <option value="client">клиент</option>
            <option value="admin">администратор</option>
          </select>
        </Field>
      </div>
      <Field label="Пароль" hint="не короче десяти символов — единственная защита
                                  утёкшей базы это цена перебора">
        <input type="password" value={password}
               onChange={(e) => setPassword(e.target.value)} />
      </Field>
    </Modal>
  );
}
