/** Один способ ходить в API: токен, разбор ошибки, отмена. */

const TOKEN_KEY = "loom_token";

export const token = {
  get: () => localStorage.getItem(TOKEN_KEY) ?? "",
  set: (value: string) => localStorage.setItem(TOKEN_KEY, value),
};

function headers(json = true): Record<string, string> {
  const head: Record<string, string> = {};
  if (json) head["Content-Type"] = "application/json";
  const value = token.get();
  if (value) head["X-Loom-Admin-Token"] = value;
  return head;
}

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
  }
}

async function fail(response: Response): Promise<never> {
  let detail = `${response.status}`;
  try {
    const body = await response.json();
    detail = body?.error?.message ?? detail;
  } catch {
    /* тело не JSON — остаётся код */
  }
  throw new ApiError(detail, response.status);
}

export async function get<T>(path: string, signal?: AbortSignal): Promise<T> {
  const response = await fetch(path, { headers: headers(false), signal });
  if (!response.ok) await fail(response);
  return response.json();
}

export async function send<T>(
  path: string,
  method: "POST" | "DELETE",
  body?: unknown,
): Promise<T> {
  const response = await fetch(path, {
    method,
    headers: headers(),
    body: body === undefined ? undefined : JSON.stringify(body),
  });
  if (!response.ok) await fail(response);
  return response.json();
}

/** Скачать файл, который отдаётся байтами: токен едет в заголовке, поэтому
 *  обычной ссылкой это не сделать. */
export async function download(path: string, filename: string): Promise<void> {
  const response = await fetch(path, { headers: headers(false) });
  if (!response.ok) await fail(response);
  const url = URL.createObjectURL(await response.blob());
  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = filename;
  document.body.appendChild(anchor);
  anchor.click();
  anchor.remove();
  URL.revokeObjectURL(url);
}

export const gb = (bytes = 0) => (bytes / 1073741824).toFixed(1);
export const mb = (bytes = 0) => (bytes / 1048576).toFixed(1);
