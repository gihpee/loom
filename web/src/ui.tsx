import { ReactNode, useCallback, useEffect, useRef, useState } from "react";
import { ApiError, get } from "./api";

/** Опрос сервера, который НЕ трогает то, что набрано в формах.
 *
 *  Прошлая админка перерисовывала страницу целиком каждые три секунды, и
 *  наполовину заполненная форма исчезала под руками. Здесь состояние сервера и
 *  состояние формы — разные вещи, и обновляется только первое.
 */
export function usePoll<T>(path: string, everyMs = 3000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string>("");
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    let alive = true;
    const controller = new AbortController();
    const pull = async () => {
      try {
        const body = await get<T>(path, controller.signal);
        if (alive) {
          setData(body);
          setError("");
        }
      } catch (exc) {
        if (alive && !(exc instanceof DOMException)) {
          setError(exc instanceof ApiError ? exc.message : String(exc));
        }
      }
    };
    pull();
    const timer = setInterval(pull, everyMs);
    return () => {
      alive = false;
      controller.abort();
      clearInterval(timer);
    };
  }, [path, everyMs, tick]);

  return { data, error, refresh };
}

/** Действие с кнопки: занятость, ошибка и обновление после успеха. */
export function useAction(after?: () => void) {
  const [busy, setBusy] = useState(false);
  const [note, setNote] = useState("");
  const run = useCallback(
    async (what: () => Promise<unknown>, success?: string) => {
      setBusy(true);
      setNote("");
      try {
        await what();
        if (success) setNote(success);
        after?.();
      } catch (exc) {
        setNote(exc instanceof ApiError ? exc.message : String(exc));
      } finally {
        setBusy(false);
      }
    },
    [after],
  );
  return { busy, note, setNote, run };
}

export function Field({ label, hint, children }: {
  label: string; hint?: string; children: ReactNode;
}) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
      {hint ? <em>{hint}</em> : null}
    </label>
  );
}

export function Table({ head, rows, empty = "пусто" }: {
  head: string[]; rows: ReactNode[][]; empty?: string;
}) {
  if (!rows.length) return <p className="dim">{empty}</p>;
  return (
    <table>
      <thead>
        <tr>{head.map((h) => <th key={h}>{h}</th>)}</tr>
      </thead>
      <tbody>
        {rows.map((row, i) => (
          <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>
        ))}
      </tbody>
    </table>
  );
}

const STATE_CLASS: Record<string, string> = {
  running: "ok", done: "ok", ready: "ok",
  pending: "warn", provisioning: "warn", loading: "warn",
  failed: "bad", cancelled: "dim", gone: "dim",
};

export function State({ value }: { value: string }) {
  return <span className={STATE_CLASS[value] ?? "dim"}>{value}</span>;
}

export function Note({ text }: { text: string }) {
  return text ? <pre className="note">{text}</pre> : null;
}

/** Прочитать выбранный файл, не теряя остальные поля формы. */
export function useFile() {
  const ref = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  const read = async (): Promise<Uint8Array | null> => {
    const file = ref.current?.files?.[0];
    if (!file) return null;
    return new Uint8Array(await file.arrayBuffer());
  };
  const text = async (): Promise<string | null> => {
    const file = ref.current?.files?.[0];
    return file ? file.text() : null;
  };
  return { ref, name, setName, read, text };
}

export function base64(bytes: Uint8Array): string {
  let binary = "";
  for (const byte of bytes) binary += String.fromCharCode(byte);
  return btoa(binary);
}
