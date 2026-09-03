/** Блик на стекле идёт за курсором.
 *
 *  Именно это делает стекло живым: неподвижное отражение читается как рисунок
 *  на плашке, а не как свет на изогнутой поверхности. Подписка одна на весь
 *  документ — по одной на карточку означало бы полсотни слушателей на лендинге.
 *  Замер положения отложен до кадра: иначе каждое движение мыши заставляло бы
 *  браузер считать раскладку заново. */
import { calmMotion } from "./Weave";

const CARDS = ".glass, .card, .lp-surface, .lp-fact, .lp-q, .key-woven, .signin-card";

export function followSheen() {
  if (calmMotion()) return;
  let queued = false;
  let last: PointerEvent | null = null;

  addEventListener("pointermove", (e) => {
    last = e;
    if (queued) return;
    queued = true;
    requestAnimationFrame(() => {
      queued = false;
      const event = last;
      if (!event) return;
      const card = (event.target as Element | null)?.closest?.(CARDS) as HTMLElement | null;
      if (!card) return;
      const box = card.getBoundingClientRect();
      card.style.setProperty("--mx", `${((event.clientX - box.left) / box.width) * 100}%`);
      card.style.setProperty("--my", `${((event.clientY - box.top) / box.height) * 100}%`);
    });
  }, { passive: true });
}
