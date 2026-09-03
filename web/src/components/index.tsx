import {
  createContext, ReactNode, useCallback, useContext, useEffect,
  useRef, useState,
} from "react";
import { get, message } from "../lib/api";

/* --------------------------------------------------------------- данные */

/** Опрос сервера, который НЕ трогает то, что набрано в формах.
 *
 *  Состояние сервера и состояние формы — разные вещи; обновляется только
 *  первое. В прошлой панели этого разделения не было, и наполовину заполненная
 *  форма исчезала под руками каждые три секунды.
 */
export function usePoll<T>(path: string | null, everyMs = 4000) {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [tick, setTick] = useState(0);
  const refresh = useCallback(() => setTick((n) => n + 1), []);

  useEffect(() => {
    if (!path) return;
    let alive = true;
    const stop = new AbortController();
    const pull = async () => {
      try {
        const body = await get<T>(path, stop.signal);
        if (!alive) return;
        setData(body); setError(""); setLoading(false);
      } catch (e) {
        if (!alive || e instanceof DOMException) return;
        setError(message(e)); setLoading(false);
      }
    };
    pull();
    const timer = setInterval(pull, everyMs);
    return () => { alive = false; stop.abort(); clearInterval(timer); };
  }, [path, everyMs, tick]);

  return { data, error, loading, refresh };
}

/* --------------------------------------------------------------- тосты */

type Toast = { id: number; kind: "ok" | "bad"; text: string };
const ToastCtx = createContext<(kind: "ok" | "bad", text: string) => void>(() => {});
export const useToast = () => useContext(ToastCtx);

export function Toasts({ children }: { children: ReactNode }) {
  const [items, setItems] = useState<Toast[]>([]);
  const push = useCallback((kind: "ok" | "bad", text: string) => {
    const id = Date.now() + Math.random();
    setItems((prev) => [...prev, { id, kind, text }]);
    // Ошибки держатся дольше: их читают, а не замечают краем глаза.
    setTimeout(() => setItems((prev) => prev.filter((t) => t.id !== id)),
               kind === "bad" ? 9000 : 4000);
  }, []);
  return (
    <ToastCtx.Provider value={push}>
      {children}
      <div className="toasts">
        {items.map((t) => (
          <div className={`toast ${t.kind}`} key={t.id}>
            <div className="body">{t.text}</div>
            <button className="btn ghost sm"
                    onClick={() => setItems((p) => p.filter((x) => x.id !== t.id))}>✕</button>
          </div>
        ))}
      </div>
    </ToastCtx.Provider>
  );
}

/** Действие с кнопки: занятость, тост об исходе, обновление данных. */
export function useAction(after?: () => void) {
  const toast = useToast();
  const [busy, setBusy] = useState(false);
  const run = useCallback(async (what: () => Promise<unknown>, ok?: string) => {
    setBusy(true);
    try {
      await what();
      if (ok) toast("ok", ok);
      after?.();
    } catch (e) {
      toast("bad", message(e));
      throw e;
    } finally {
      setBusy(false);
    }
  }, [after, toast]);
  return { busy, run };
}

/* ----------------------------------------------------------- примитивы */

export function Button({ kind = "default", size, children, ...rest }: {
  kind?: "default" | "primary" | "ghost" | "danger"; size?: "sm";
} & React.ButtonHTMLAttributes<HTMLButtonElement>) {
  const cls = ["btn", kind === "default" ? "" : kind, size ?? ""].filter(Boolean);
  return <button className={cls.join(" ")} {...rest}>{children}</button>;
}

export function Badge({ tone = "dim", pulse, children }: {
  tone?: "ok" | "warn" | "bad" | "dim" | "info"; pulse?: boolean; children: ReactNode;
}) {
  return (
    <span className={`badge ${tone}${pulse ? " pulse" : ""}`}>
      <i className="dot" />{children}
    </span>
  );
}

const TONE: Record<string, "ok" | "warn" | "bad" | "dim"> = {
  running: "ok", done: "ok", ready: "ok",
  pending: "warn", provisioning: "warn", starting: "warn", loading: "warn",
  fetching: "warn", downloaded: "warn",
  failed: "bad", refused: "bad",
  cancelled: "dim", gone: "dim", idle: "dim",
};
export const StateBadge = ({ value, pulse }: { value: string; pulse?: boolean }) =>
  <Badge tone={TONE[value] ?? "dim"} pulse={pulse}>{value}</Badge>;

export function Field({ label, hint, children }: {
  label: string; hint?: string; children: ReactNode;
}) {
  return (
    <div className="field">
      <label>{label}</label>
      {children}
      {hint && <span className="hint">{hint}</span>}
    </div>
  );
}

/** Файловый вход, который не выглядит как файловый вход браузера. */
export function FilePick({ label, accept, onPick }: {
  label: string; accept?: string; onPick?: (file: File) => void;
}) {
  const ref = useRef<HTMLInputElement>(null);
  const [name, setName] = useState("");
  return (
    <label className={`file${name ? " filled" : ""}`}>
      <input ref={ref} type="file" accept={accept} onChange={(e) => {
        const file = e.target.files?.[0];
        setName(file?.name ?? "");
        if (file && onPick) onPick(file);
      }} />
      <span>{name || label}</span>
      {name && <span style={{ marginLeft: "auto", color: "var(--text-mute)" }}>✕</span>}
    </label>
  );
}

export const useFileRef = () => {
  const ref = useRef<HTMLInputElement>(null);
  return ref;
};

export function Stat({ label, value, unit, sub }: {
  label: string; value: ReactNode; unit?: string; sub?: ReactNode;
}) {
  return (
    <div className="card stat">
      <div className="label">{label}</div>
      <div className="value">{value}{unit && <small> {unit}</small>}</div>
      {sub && <div className="sub">{sub}</div>}
    </div>
  );
}

export function Empty({ title, children }: { title: string; children?: ReactNode }) {
  return <div className="empty"><h3>{title}</h3>{children && <p>{children}</p>}</div>;
}

export function Bar({ percent, tone }: { percent: number; tone?: "ok" | "bad" }) {
  return (
    <div className={`bar${tone ? ` ${tone}` : ""}`}>
      <i style={{ width: `${Math.max(0, Math.min(100, percent))}%` }} />
    </div>
  );
}

/* ------------------------------------------------------------ слои */

function useEscape(onClose: () => void) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
}

export function Modal({ title, onClose, children, footer }: {
  title: string; onClose: () => void; children: ReactNode; footer?: ReactNode;
}) {
  useEscape(onClose);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="modal" role="dialog" aria-modal>
        <div className="sheet-head">
          <h3>{title}</h3>
          <Button kind="ghost" size="sm" className="close" onClick={onClose}>✕</Button>
        </div>
        <div className="sheet-body">{children}</div>
        {footer && <div className="sheet-body" style={{ paddingTop: 0 }}>{footer}</div>}
      </div>
    </>
  );
}

export function Drawer({ title, onClose, children }: {
  title: ReactNode; onClose: () => void; children: ReactNode;
}) {
  useEscape(onClose);
  return (
    <>
      <div className="scrim" onClick={onClose} />
      <div className="drawer" role="dialog" aria-modal>
        <div className="sheet-head">
          <h3>{title}</h3>
          <Button kind="ghost" size="sm" className="close" onClick={onClose}>✕</Button>
        </div>
        <div className="sheet-body">{children}</div>
      </div>
    </>
  );
}

/** Подтверждение для того, что нельзя отменить. */
export function Confirm({ title, body, action, onClose, onConfirm }: {
  title: string; body: ReactNode; action: string;
  onClose: () => void; onConfirm: () => void;
}) {
  return (
    <Modal title={title} onClose={onClose} footer={
      <div className="form-actions" style={{ borderTop: 0, paddingTop: 0 }}>
        <Button kind="ghost" onClick={onClose}>отмена</Button>
        <Button kind="danger" onClick={() => { onConfirm(); onClose(); }}>{action}</Button>
      </div>
    }>
      {body}
    </Modal>
  );
}

export function ErrorLine({ error }: { error: string }) {
  if (!error) return null;
  const gate = error.includes("403") || error.toLowerCase().includes("token");
  return (
    <div className="card" style={{ borderColor: "var(--bad-soft)", marginBottom: 16 }}>
      <b style={{ color: "var(--bad)" }}>{gate ? "Нужен админ-токен" : "Ошибка"}</b>
      <div className="sub" style={{ marginTop: 4 }}>{error}</div>
    </div>
  );
}

/** Знак Looma: «L», собранная из переплетения. Две нити основы идут вниз, две
 *  нити утка́ уходят вправо, и в каждом пересечении одна проходит поверх другой —
 *  отсюда просветы. Нижняя кромка скруглена: там уто́к разворачивается обратно,
 *  как на настоящей кромке полотна.
 *
 *  Цвет берётся из currentColor: знак живёт на трёх поверхностях, и заливать
 *  его акцентом намертво значило бы чинить его отдельно при каждой смене темы. */
export function Mark({ size = 32 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 100 100" fill="currentColor"
         aria-hidden="true">
      {/* основа: левая нить ныряет под нижний уто́к и выходит скруглённой кромкой */}
      <rect x="8" y="6" width="20" height="45" />
      <path d="M8 77 H28 V94 H25 A17 17 0 0 1 8 77 Z" />
      {/* основа: правая нить ныряет под верхний уто́к */}
      <rect x="34" y="6" width="20" height="19" />
      <rect x="34" y="51" width="20" height="43" />
      {/* уто́к: верхний идёт поверх правой нити, нижний — поверх левой */}
      <rect x="31" y="28" width="61" height="20" />
      <rect x="8" y="54" width="23" height="20" />
      <rect x="57" y="54" width="35" height="20" />
    </svg>
  );
}
