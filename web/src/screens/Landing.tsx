/** Публичная главная.
 *
 * Что здесь НЕ делается и почему. Не пересказывается устройство системы —
 * человеку, который решает, брать ли мощность, безразлично, как она
 * оркестрируется. Не обещается «ноль простоя»: провайдеров ещё нет, и говорить
 * о их выгоде рано. Не перечисляются виды владельцев карт — это внутренняя
 * сегментация, а не то, что продаётся.
 *
 * Остаётся одно утверждение и две возможности под ним. Всё остальное —
 * подробности, которые раскрываются по наведению, а не вываливаются сразу.
 */
import { useEffect, useRef, useState } from "react";
import { Mark } from "../components";

/* ------------------------------------------------------------------ ткань
   Знак Looma — полосы, ткацкий стан. Отсюда и графика: нити основы, по которым
   идут импульсы. Это не украшение ради движения: сеть из отдельных машин,
   собирающаяся в одно полотно, — ровно то, что продаётся, и показать это
   короче, чем описать. */
function Weave() {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    if (!c) return;
    const ctx = c.getContext("2d");
    if (!ctx) return;

    const slow = matchMedia("(prefers-reduced-motion: reduce)").matches;
    let raf = 0;
    let w = 0, h = 0;
    const dpr = Math.min(devicePixelRatio || 1, 2);

    const size = () => {
      w = c.clientWidth; h = c.clientHeight;
      c.width = w * dpr; c.height = h * dpr;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    size();
    addEventListener("resize", size);

    const LINES = 11;
    // Импульс на нити: где он, как быстро идёт и насколько ярок.
    const pulses = Array.from({ length: 16 }, () => ({
      line: Math.floor(Math.random() * LINES),
      at: Math.random(),
      speed: 0.0016 + Math.random() * 0.0032,
      life: 0.4 + Math.random() * 0.6,
    }));

    const draw = (t: number) => {
      ctx.clearRect(0, 0, w, h);
      const step = h / (LINES - 1);

      for (let i = 0; i < LINES; i++) {
        const y = i * step;
        // Лёгкая волна: полотно живое, но не мельтешит.
        const sway = slow ? 0 : Math.sin(t / 2600 + i * 0.7) * 7;
        ctx.beginPath();
        ctx.moveTo(0, y + sway);
        ctx.bezierCurveTo(w * 0.3, y - sway, w * 0.7, y + sway * 2, w, y - sway);
        // Нити должны читаться сами по себе: при прежней прозрачности
        // оставались одни импульсы, и полотно выглядело случайными пятнами.
        ctx.strokeStyle = `rgba(69, 200, 192, ${0.14 + (i % 3) * 0.06})`;
        ctx.lineWidth = i % 4 === 0 ? 1.4 : 1;
        ctx.stroke();
      }

      for (const p of pulses) {
        if (!slow) p.at += p.speed;
        if (p.at > 1) { p.at = 0; p.line = Math.floor(Math.random() * LINES); }
        const y = p.line * step;
        const sway = slow ? 0 : Math.sin(t / 2600 + p.line * 0.7) * 7;
        const x = p.at * w;
        const yy = y + sway * Math.sin(p.at * Math.PI);
        // Короткий след вдоль нити, а не круглое пятно: импульс должен
        // выглядеть идущим ПО нити, иначе связь с ней теряется.
        const tail = ctx.createLinearGradient(x - 90, 0, x + 10, 0);
        tail.addColorStop(0, "rgba(120, 240, 230, 0)");
        tail.addColorStop(1, `rgba(120, 240, 230, ${0.75 * p.life})`);
        ctx.strokeStyle = tail;
        ctx.lineWidth = 1.8;
        ctx.beginPath();
        ctx.moveTo(Math.max(0, x - 90), yy);
        ctx.lineTo(x, yy);
        ctx.stroke();

        const head = ctx.createRadialGradient(x, yy, 0, x, yy, 9);
        head.addColorStop(0, `rgba(160, 250, 240, ${0.9 * p.life})`);
        head.addColorStop(1, "rgba(160, 250, 240, 0)");
        ctx.fillStyle = head;
        ctx.fillRect(x - 9, yy - 9, 18, 18);
      }
      raf = requestAnimationFrame(draw);
    };
    raf = requestAnimationFrame(draw);

    return () => { cancelAnimationFrame(raf); removeEventListener("resize", size); };
  }, []);

  return <canvas ref={canvas} className="weave" aria-hidden="true" />;
}

/* ------------------------------------------------------- появление по скроллу */
function useReveal<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    const node = ref.current;
    if (!node) return;
    const eye = new IntersectionObserver(([e]) => {
      if (e.isIntersecting) { setSeen(true); eye.disconnect(); }
    }, { rootMargin: "-60px" });
    eye.observe(node);
    return () => eye.disconnect();
  }, []);
  return { ref, seen };
}

function Reveal({ children, delay = 0 }: { children: React.ReactNode; delay?: number }) {
  const { ref, seen } = useReveal<HTMLDivElement>();
  return (
    <div ref={ref} className="reveal" data-seen={seen}
         style={{ transitionDelay: `${delay}ms` }}>
      {children}
    </div>
  );
}

const PRODUCTS = [
  {
    id: "looma-inference",
    title: "Инференс",
    line: "Готовый API вместо своей карты.",
    detail: "Совместим с OpenAI: меняется адрес и ключ, остальной код остаётся. " +
      "Большие модели работают там, где не помещаются целиком.",
    facts: ["оплата за токены", "потоковая выдача", "модели до 70B"],
  },
  {
    id: "looma-compute",
    title: "Кластер",
    line: "Свой код на арендованных картах.",
    detail: "Обычный Ray: подключаетесь с ноутбука одной строкой и считаете " +
      "что угодно — обучение, симуляции, рендер.",
    facts: ["оплата за GPU-часы", "почасовая аренда", "локальный порт"],
  },
];

const WHY = [
  { head: "Дешевле облака", text: "Мощность берётся у тех, у кого она уже есть, — без дата-центров в цене." },
  { head: "Без обязательств", text: "Ни минимального срока, ни резерва. Счёт идёт по факту, посекундно." },
  { head: "Ничего не настраивать", text: "Ни VPC, ни квот, ни заявок на доступ. Ключ — и работаете." },
];

export function Landing() {
  return (
    <div className="landing">
      <header className="lp-top">
        <div className="lp-wrap lp-nav">
          <a className="lp-brand" href="/"><Mark size={28} /><span>Looma&nbsp;Float</span></a>
          <nav>
            <a href="#products">Возможности</a>
            <a className="lp-enter" href="/app">Войти</a>
          </nav>
        </div>
      </header>

      <section className="lp-hero">
        <Weave />
        <div className="lp-wrap lp-hero-body">
          <p className="lp-kicker">Маркетплейс вычислений</p>
          <h1>Мощность,<br />когда она нужна</h1>
          <p className="lp-lead">
            Инференс больших моделей и аренда GPU-кластеров. Платите за
            использованное — без дата-центров в цене.
          </p>
          <div className="lp-cta">
            <a className="lp-primary" href="/app">Начать</a>
            <a className="lp-secondary" href="#products">Возможности</a>
          </div>
        </div>
      </section>

      <section className="lp-section" id="products">
        <div className="lp-wrap">
          <Reveal><h2 className="lp-h2">Возможности</h2></Reveal>
          <div className="lp-products">
            {PRODUCTS.map((p, i) => (
              <Reveal key={p.id} delay={i * 90}>
                <article className="lp-product">
                  <h3>{p.title}</h3>
                  <p className="lp-line">{p.line}</p>
                  <p className="lp-detail">{p.detail}</p>
                  <ul>{p.facts.map((f) => <li key={f}>{f}</li>)}</ul>
                  <code className="lp-id">{p.id}</code>
                </article>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="lp-section lp-why">
        <div className="lp-wrap">
          <div className="lp-why-grid">
            {WHY.map((w, i) => (
              <Reveal key={w.head} delay={i * 80}>
                <div className="lp-why-item">
                  <h3>{w.head}</h3>
                  <p>{w.text}</p>
                </div>
              </Reveal>
            ))}
          </div>
        </div>
      </section>

      <section className="lp-section lp-final">
        <div className="lp-wrap">
          <Reveal>
            <h2 className="lp-h2">Попробовать</h2>
            <p className="lp-lead">
              Доступ по приглашению: сеть в закрытой бете, и мы отвечаем за
              каждую машину в ней.
            </p>
            <a className="lp-primary" href="/app">Войти в кабинет</a>
          </Reveal>
        </div>
      </section>

      <footer className="lp-foot">
        <div className="lp-wrap">
          <Mark size={24} />
          <p>Looma Float</p>
        </div>
      </footer>
    </div>
  );
}
