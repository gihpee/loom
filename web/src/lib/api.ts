/** Единственный способ ходить в API: кто ты, разбор ошибки, отмена.
 *
 * Представиться можно двумя способами. Сессия ездит в cookie и ставится входом
 * по паролю — так ходит человек. Админ-токен остаётся запасным входом на случай,
 * когда база недоступна или пароль потерян; он же был единственным до появления
 * учётных записей.
 */

const TOKEN = "looma_token";

export const token = {
  get: () => localStorage.getItem(TOKEN) ?? "",
  set: (v: string) => localStorage.setItem(TOKEN, v),
};

function head(json = true): Record<string, string> {
  const h: Record<string, string> = {};
  if (json) h["Content-Type"] = "application/json";
  const t = token.get();
  if (t) h["X-Looma-Admin-Token"] = t;
  return h;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) { super(message); }
}

async function boom(r: Response): Promise<never> {
  let detail = `HTTP ${r.status}`;
  try { detail = (await r.json())?.error?.message ?? detail; } catch { /* не JSON */ }
  throw new ApiError(detail, r.status);
}

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const r = await fetch(path, { headers: head(false), signal,
                                credentials: "same-origin" });
  if (!r.ok) await boom(r);
  return r.json();
}

export async function send<T>(path: string, method: "POST" | "DELETE",
                              body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method, headers: head(), credentials: "same-origin",
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!r.ok) await boom(r);
  return r.json();
}

/** Файл отдаётся байтами, а токен едет в заголовке — ссылкой это не сделать. */
export async function grab(path: string, filename: string): Promise<void> {
  const r = await fetch(path, { headers: head(false) });
  if (!r.ok) await boom(r);
  const url = URL.createObjectURL(await r.blob());
  const a = document.createElement("a");
  a.href = url; a.download = filename;
  document.body.appendChild(a); a.click(); a.remove();
  URL.revokeObjectURL(url);
}

export const message = (e: unknown) =>
  e instanceof ApiError ? e.message : e instanceof Error ? e.message : String(e);

/** Кто вошёл. `null` — никто: страница входа решает, что с этим делать. */
export interface Who {
  id: number | null;
  email: string;
  role: string;
  display_name?: string;
  how: string;
}

export async function whoami(): Promise<Who | null> {
  try {
    return await get<Who>("/api/me");
  } catch (e) {
    // 401 — это не поломка, а ответ «никто». Отличать его от настоящей
    // ошибки важно: иначе страница входа показывает «сервер недоступен».
    if (e instanceof ApiError && e.status === 401) return null;
    throw e;
  }
}

export const signIn = (email: string, password: string) =>
  send<Who>("/api/session", "POST", { email, password });

export const signOut = () => send<{ signed_out: boolean }>("/api/session", "DELETE");
