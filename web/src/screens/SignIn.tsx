/** Вход. Единственная страница, доступная без представления. */
import { useState } from "react";
import { Mark } from "../components";
import { message, signIn } from "../lib/api";

export function SignIn({ onDone }: { onDone: () => void }) {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setBusy(true);
    setError("");
    try {
      await signIn(email, password);
      onDone();
    } catch (err) {
      setError(message(err));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="signin">
      <form className="signin-card" onSubmit={submit}>
        <a className="signin-brand" href="/"><Mark size={34} /><span>Looma Float</span></a>
        <h1>Вход в кабинет</h1>
        <label>
          Почта
          <input type="email" value={email} autoFocus autoComplete="username"
                 onChange={(e) => setEmail(e.target.value)} />
        </label>
        <label>
          Пароль
          <input type="password" value={password} autoComplete="current-password"
                 onChange={(e) => setPassword(e.target.value)} />
        </label>
        {/* Причина одна на оба случая — так и отвечает сервер: «нет такого
            адреса» и «неверный пароль» вместе рассказывают, кто у нас
            зарегистрирован. */}
        {error && <p className="signin-error">{error}</p>}
        <button type="submit" disabled={busy || !email || !password}>
          {busy ? "…" : "Войти"}
        </button>
        <p className="signin-fine">
          Регистрация по приглашению: учётные записи заводит администратор.
        </p>
      </form>
    </div>
  );
}
