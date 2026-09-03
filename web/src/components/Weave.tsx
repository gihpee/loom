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
import { useEffect, useLayoutEffect, useRef, useState } from "react";

export const calmMotion = () =>
  typeof matchMedia === "function" &&
  matchMedia("(prefers-reduced-motion: reduce)").matches;

/** Ссылка на узел, который выткётся, когда попадёт в поле зрения.
 *
 * Содержимое видно ПО УМОЛЧАНИЮ, а прячет его скрипт — и только затем, чтобы
 * тут же показать. Обратный порядок (спрятано в CSS, показывает наблюдатель)
 * выглядит так же ровно до первой причины, по которой наблюдатель не
 * сработал: вкладка была скрыта, элемент так и не пересёк порог, скрипт не
 * выполнился. Тогда посетитель видит пустой раздел с одним заголовком, и
 * виновата в этом анимация, которой там не должно быть вовсе.
 *
 * Правило простое: украшение не имеет права прятать содержимое.
 */
export function useWeaveReveal<T extends HTMLElement>(threshold = 0.3) {
  const ref = useRef<T>(null);
  const hidden = useRef(false);

  // Прячем только то, что ещё ниже экрана. Уже видимое не трогаем совсем:
  // анимировать появление того, на что человек уже смотрит, незачем, а вот
  // спрятать первый экран в ожидании события, которое может не прийти, —
  // ровно та ошибка, из-за которой разделы оказывались пустыми.
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node || calmMotion()) return;
    if (node.getBoundingClientRect().top < window.innerHeight) return;
    node.style.setProperty("--progress", "0");
    hidden.current = true;
  }, []);

  useEffect(() => {
    const node = ref.current;
    if (!node || !hidden.current) return;

    const show = (animate: boolean) => {
      if (!hidden.current) return;
      hidden.current = false;
      // В скрытой вкладке переходы не идут: они замирают на начальном кадре,
      // то есть на спрятанном. Тогда показываем разом, без анимации.
      if (!animate) node.style.transition = "none";
      node.style.setProperty("--progress", "1");
    };

    const eye = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) { show(!document.hidden); eye.disconnect(); }
    }, { threshold });
    eye.observe(node);

    // Страховка: если наблюдатель почему-либо не позвал — показываем сами.
    // Опоздать с появлением несравнимо лучше, чем не появиться никогда.
    const rescue = setTimeout(() => { show(!document.hidden); eye.disconnect(); }, 2000);
    return () => { clearTimeout(rescue); eye.disconnect(); };
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

  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    node.style.setProperty("--progress", woven || calmMotion() ? "1" : "0");
  }, [woven]);

  return { ref, woven, weave: () => setWoven(true), unweave: () => setWoven(false) };
}
