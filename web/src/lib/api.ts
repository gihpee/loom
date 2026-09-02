/** Единственный способ ходить в API: токен, разбор ошибки, отмена. */

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
  const r = await fetch(path, { headers: head(false), signal });
  if (!r.ok) await boom(r);
  return r.json();
}

export async function send<T>(path: string, method: "POST" | "DELETE",
                              body?: unknown): Promise<T> {
  const r = await fetch(path, {
    method, headers: head(),
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
