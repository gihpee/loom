export const gb = (b = 0) => (b / 1073741824).toFixed(1);
export const mb = (b = 0) => (b / 1048576).toFixed(1);
export const kb = (b = 0) => (b / 1024).toFixed(1);

export function duration(seconds: number): string {
  if (seconds < 60) return `${seconds.toFixed(seconds < 10 ? 1 : 0)}s`;
  const m = Math.floor(seconds / 60);
  if (m < 60) return `${m}m ${Math.round(seconds % 60)}s`;
  return `${Math.floor(m / 60)}h ${m % 60}m`;
}

export function ago(unixSeconds: number): string {
  const delta = Date.now() / 1000 - unixSeconds;
  if (delta < 60) return "только что";
  if (delta < 3600) return `${Math.floor(delta / 60)} мин назад`;
  if (delta < 86400) return `${Math.floor(delta / 3600)} ч назад`;
  return `${Math.floor(delta / 86400)} дн назад`;
}

/** Длинные идентификаторы: середина не нужна, а концы различают. */
export const short = (id: string, keep = 8) =>
  id.length <= keep * 2 + 1 ? id : `${id.slice(0, keep)}…${id.slice(-4)}`;
