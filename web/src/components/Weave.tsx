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
/** Событие «к этой карточке идёт уто́к». Слушает фоновый станок: он выводит
 *  нить от полотна к её кромке ровно в том же такте, в каком открывается клип. */
export const WEFT = "looma:weft";

export interface WeftPass {
  left: number; right: number; top: number; bottom: number;
  lead: number; sweep: number;
}

/** Поставить величину, не проиграв переход к ней. Нужно там, где --progress
 *  всё-таки переходный (показ ключа): первая установка — это состояние, а не
 *  движение, и ключ не должен затыкаться обратно на глазах у того, кто его ещё
 *  не открывал. */
function setAtOnce(node: HTMLElement, value: string) {
  const kept = node.style.transition;
  node.style.transition = "none";
  node.style.setProperty("--progress", value);
  void node.offsetWidth;
  node.style.transition = kept;
}

export function useWeaveReveal<T extends HTMLElement>(threshold = 0.3) {
  const ref = useRef<T>(null);
  const waiting = useRef(false);

  // Прячем только то, что ещё ниже экрана. Уже видимое не трогаем совсем:
  // анимировать появление того, на что человек уже смотрит, незачем, а вот
  // спрятать первый экран в ожидании события, которое может не прийти, — ровно
  // та ошибка, из-за которой разделы однажды оказались пустыми.
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node || calmMotion()) return;
    // clientHeight, а не innerHeight: последний на момент этого замера может
    // быть ещё нулевым. Не знаем размера окна — не прячем ничего: тогда просто
    // не будет анимации, а не пустого первого экрана в ожидании страховки.
    const viewport = document.documentElement.clientHeight || window.innerHeight;
    if (!viewport) return;
    if (node.getBoundingClientRect().top < viewport * 0.9) return;
    node.classList.add("weave-wait");
    waiting.current = true;
  }, []);

  useEffect(() => {
    const node = ref.current;
    if (!node || !waiting.current) return;

    const show = (animate: boolean) => {
      if (!waiting.current) return;
      waiting.current = false;
      node.classList.remove("weave-wait");
      if (!animate) return;              // вкладка скрыта — показываем разом

      node.classList.add("weaving");
      node.addEventListener("animationend",
        () => node.classList.remove("weaving"), { once: true });

      // Станок ведёт нить к этой карточке. Тайминги берутся с самого элемента,
      // а не задаются здесь второй раз: разъехавшись, нить и клип превратились
      // бы в два разных движения вместо одного.
      const css = getComputedStyle(node);
      const ms = (name: string, fallback: number) => {
        const raw = css.getPropertyValue(name).trim();
        const v = parseFloat(raw);
        if (!Number.isFinite(v)) return fallback;
        return raw.endsWith("ms") ? v : v * 1000;
      };
      const box = node.getBoundingClientRect();
      window.dispatchEvent(new CustomEvent(WEFT, { detail: {
        left: box.left, right: box.right, top: box.top, bottom: box.bottom,
        lead: ms("--lead", 240), sweep: ms("--sweep", 900),
      } }));
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

/** Слой с нитями: голая основа перед челноком и сам челнок на кромке.
 *  Отдельным элементом, потому что ::before и ::after карточки заняты бликом и
 *  плёнкой стекла. */
export function Weft() {
  return <i className="weft" aria-hidden="true" />;
}

export function WeaveReveal({ children, delay = 0, className = "" }: {
  children: React.ReactNode; delay?: number; className?: string;
}) {
  const ref = useWeaveReveal<HTMLDivElement>();
  return (
    <div ref={ref} className={`weave-reveal ${className}`.trim()}
         style={{ "--lead": `${240 + delay}ms` } as React.CSSProperties}>
      {children}
      <Weft />
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

  const first = useRef(true);
  useLayoutEffect(() => {
    const node = ref.current;
    if (!node) return;
    const value = woven || calmMotion() ? "1" : "0";
    // Первая установка — состояние, а не движение: ключ не должен затыкаться
    // обратно на глазах у того, кто его ещё не открывал.
    if (first.current) { first.current = false; setAtOnce(node, value); return; }
    node.style.setProperty("--progress", value);
  }, [woven]);

  return { ref, woven, weave: () => setWoven(true), unweave: () => setWoven(false) };
}
