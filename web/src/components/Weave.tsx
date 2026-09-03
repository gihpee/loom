/** Ткацкая грамматика движения — один примитив на все три поверхности.
 *
 * Четыре стадии, и это не декоративная лексика, а имена состояний:
 *
 *   основа   нити-направляющие, видны всегда слабым тоном;
 *   зев      нити расходятся, освобождая проход — перед раскрытием;
 *   уток     светящийся челнок идёт слева направо, протаскивая контент;
 *   прибой   нити садятся на место с лёгким пружинным перехлёстом.
 *
 * Зачем один примитив вместо fade-in на каждой секции: разрозненные появления
 * читаются как «тут кто-то добавил анимацию», а повторяющаяся форма — как
 * язык. К тому же она совпадает с предметом: полотно действительно ткут слева
 * направо, и раскрытие ключа API «выткано» ровно потому, что показывается
 * один раз.
 *
 * Срабатывает единожды. Повторное проигрывание при обратном скролле — самый
 * заметный признак анимации ради анимации.
 */
import { useEffect, useRef, useState } from "react";

export const calmMotion = () =>
  typeof matchMedia === "function" &&
  matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Ссылка на узел, который выткётся, когда попадёт в поле зрения. */
export function useWeaveReveal<T extends HTMLElement>(threshold = 0.3) {
  const ref = useRef<T>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    if (calmMotion()) {
      // Никакого раскрытия: контент просто на месте. Отключать анимацию,
      // ломая при этом появление, — худший из вариантов.
      node.style.setProperty("--progress", "1");
      return;
    }
    const eye = new IntersectionObserver(([entry]) => {
      if (!entry.isIntersecting) return;
      node.style.setProperty("--progress", "1");
      eye.disconnect();
    }, { threshold });
    eye.observe(node);
    return () => eye.disconnect();
  }, [threshold]);

  return ref;
}

/** Обёртка для случаев, когда своя разметка не нужна. */
export function WeaveReveal({ children, delay = 0, className = "" }: {
  children: React.ReactNode; delay?: number; className?: string;
}) {
  const ref = useWeaveReveal<HTMLDivElement>();
  return (
    <div ref={ref} className={`weave-reveal ${className}`.trim()}
         style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

/** Ткать по команде, а не по скроллу: ключ API, смена состояния узла.
 *
 * Возвращает состояние и две команды. `unweave` — не «спрятать», а именно
 * заткать обратно: у показанного один раз ключа исчезновение должно выглядеть
 * необратимым, потому что оно и есть необратимое.
 */
export function useWoven(initial = false) {
  const [woven, setWoven] = useState(initial);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    node.style.setProperty("--progress", woven || calmMotion() ? "1" : "0");
  }, [woven]);

  return { ref, woven, weave: () => setWoven(true), unweave: () => setWoven(false) };
}
