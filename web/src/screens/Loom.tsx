/** Станок: фон всего лендинга.
 *
 * Это не орнамент. Здесь показан ровно один факт архитектуры, который иначе
 * приходится читать словами: **у узла нет входящих портов**. Он сам тянет одну
 * нить к оркестратору, и весь трафик идёт по ней в обе стороны.
 *
 * Отсюда всё остальное:
 *
 *   — от каждого узла ровно ОДНА нить, а не связь туда-обратно;
 *   — импульсы по ней идут в обе стороны: канал один, трафика в нём много;
 *   — нить узла НЕ упирается в центр, а продолжается утко́м полотна. Сеть
 *     отдельных машин, собирающаяся в одно полотно, — это не метафора в
 *     подписи, это то, что видно.
 *
 * Полотно ткётся по мере прокрутки: наверху нити разрознены, к последнему
 * экрану сплетены. Скролл и есть рассказ.
 */
import { useEffect, useRef } from "react";
import { calmMotion } from "../components/Weave";

export type Highlight = "" | "inference" | "compute";

const ROWS = 15;           // нитей утка = подключённых узлов
const WARPS = 22;          // нитей основы: структура самой платформы
const PULSES = 22;         // ограничено намеренно: просадка кадров хуже, чем меньше движения

interface Pulse { row: number; at: number; speed: number; back: boolean }

export function Loom({ progress, highlight }: {
  progress: React.MutableRefObject<number>;
  highlight: React.MutableRefObject<Highlight>;
}) {
  const canvas = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;

    const calm = calmMotion();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    let w = 0, h = 0, raf = 0;

    const size = () => {
      w = c.clientWidth; h = c.clientHeight;
      if (!w || !h) return;          // раскладка ещё не случилась — ждём наблюдателя
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    size();
    // ResizeObserver, а не событие окна: замер на монтировании может застать
    // нулевую высоту, и тогда холст навсегда остаётся буфером 0 пикселей —
    // без единой ошибки в консоли, просто пустое место.
    const watch = new ResizeObserver(size);
    watch.observe(c);

    const pulses: Pulse[] = Array.from({ length: PULSES }, () => ({
      row: Math.floor(Math.random() * ROWS),
      at: Math.random(),
      speed: 0.0022 + Math.random() * 0.004,
      back: Math.random() < 0.45,
    }));

    // Пучок, который выходит из полотна при наведении на «кластер»: не любые
    // нити, а соседние — вытесняется группа узлов, а не случайные.
    const BUNDLE = [5, 6, 7, 8];
    let lift = 0;

    /** Геометрия кадра. Считается один раз за кадр, а не на каждую нить. */
    const frame = () => {
      const p = Math.min(1, Math.max(0, progress.current));
      // Полотно смещено вправо: текст читается слева, и они не спорят за
      // одно место. По центру заголовок ложился прямо на переплетение.
      const cx = w > 900 ? w * 0.68 : w * 0.5;
      const cy = h * 0.5;
      const fw = Math.min(w * 0.52, 620) * (0.42 + 0.58 * p);
      const fh = Math.min(h * 0.46, 400);
      return { p, cx, cy, left: cx - fw / 2, right: cx + fw / 2,
               top: cy - fh / 2, rowGap: fh / (ROWS - 1) };
    };

    const rowY = (f: ReturnType<typeof frame>, i: number) => f.top + i * f.rowGap;

    /** Точка узла на периферии. Нечётные слева, чётные справа — нить входит в
     *  полотно со своей стороны и идёт утко́м насквозь. */
    const nodeAt = (f: ReturnType<typeof frame>, i: number) => {
      const leftSide = i % 2 === 0;
      const spread = (i / (ROWS - 1) - 0.5) * 1.35;
      return {
        x: leftSide ? f.left - w * 0.16 : w * 0.97,
        y: f.cy + spread * h * 0.46,
        leftSide,
      };
    };

    /** Смещение нити пучка наружу: 0 — в полотне, 1 — вышла. */
    const bundleLift = (i: number) =>
      BUNDLE.includes(i) ? lift * (28 + 10 * Math.abs(i - 6.5)) : 0;

    /** Путь нити: от узла к своей кромке и утко́м через полотно. */
    const thread = (f: ReturnType<typeof frame>, i: number) => {
      const node = nodeAt(f, i);
      const y = rowY(f, i) - bundleLift(i);
      const entry = node.leftSide ? f.left : f.right;
      const exit = node.leftSide ? f.right : f.left;
      return { node, y, entry, exit };
    };

    const drawThread = (f: ReturnType<typeof frame>, i: number, t: number) => {
      const { node, y, entry, exit } = thread(f, i);
      ctx.beginPath();
      ctx.moveTo(node.x, node.y);
      // Провис к полотну: прямая линия читалась бы как схема, а не как нить.
      ctx.bezierCurveTo((node.x + entry) / 2, node.y, (node.x + entry) / 2, y, entry, y);
      ctx.lineTo(exit, y);
      const lifted = bundleLift(i) > 0.5;
      ctx.strokeStyle = lifted
        ? `rgba(79, 232, 206, ${.2 + .25 * lift})`
        : `rgba(31, 191, 168, ${.10 + .05 * Math.sin(t / 1800 + i)})`;
      ctx.lineWidth = 1;
      ctx.stroke();

      // Узел: маленький, но всегда виден — он источник нити.
      ctx.beginPath();
      ctx.arc(node.x, node.y, 2.2, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(79, 232, 206, .5)";
      ctx.fill();
    };

    /** Основа: вертикальные нити платформы. Видны всегда слабым тоном. */
    const drawWarp = (f: ReturnType<typeof frame>) => {
      const bottom = f.top + (ROWS - 1) * f.rowGap;
      for (let k = 0; k < WARPS; k++) {
        const x = f.left + (k / (WARPS - 1)) * (f.right - f.left);
        ctx.beginPath();
        ctx.moveTo(x, f.top);
        ctx.lineTo(x, bottom);
        ctx.strokeStyle = "rgba(31, 191, 168, .09)";
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    };

    /** Переплетение: уто́к идёт поверх основы через клетку, а не сплошь.
     *  Именно шахматный порядок и читается как ткань, а не как сетка. */
    const drawWeave = (f: ReturnType<typeof frame>) => {
      const step = (f.right - f.left) / (WARPS - 1);
      for (let i = 0; i < ROWS; i++) {
        const lifted = bundleLift(i);
        if (lifted > 0.5) continue;          // вышедшая нить не переплетена
        const y = rowY(f, i);
        for (let k = 0; k < WARPS - 1; k++) {
          if ((i + k) % 2) continue;
          const x0 = f.left + k * step;
          ctx.beginPath();
          ctx.moveTo(x0, y);
          ctx.lineTo(x0 + step, y);
          ctx.strokeStyle = `rgba(79, 232, 206, ${.10 + .22 * f.p})`;
          ctx.lineWidth = 1.3;
          ctx.stroke();
        }
      }
    };

    /** Свечение — слоями и в режиме сложения, как в разборе техники: источник
     *  света не абстрактный, а сами пересечения нитей. */
    const drawGlow = (f: ReturnType<typeof frame>) => {
      ctx.globalCompositeOperation = "lighter";
      const bottom = f.top + (ROWS - 1) * f.rowGap;
      const cx = (f.left + f.right) / 2, cy = (f.top + bottom) / 2;
      for (const [r, a] of [[Math.max(f.right - f.left, 1) * .75, .16],
                            [Math.max(f.right - f.left, 1) * .38, .12]] as const) {
        const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);
        g.addColorStop(0, `rgba(31, 191, 168, ${a * (0.35 + 0.65 * f.p)})`);
        g.addColorStop(1, "rgba(31, 191, 168, 0)");
        ctx.fillStyle = g;
        ctx.fillRect(cx - r, cy - r, r * 2, r * 2);
      }
      ctx.globalCompositeOperation = "source-over";
    };

    const drawPulses = (f: ReturnType<typeof frame>, t: number) => {
      ctx.globalCompositeOperation = "lighter";
      for (const pulse of pulses) {
        if (!calm) pulse.at += pulse.speed;
        if (pulse.at > 1) {
          pulse.at = 0;
          pulse.row = Math.floor(Math.random() * ROWS);
          pulse.back = Math.random() < 0.45;
        }
        const { node, y, entry, exit } = thread(f, pulse.row);
        // Доля пути: сначала подвод от узла, потом уто́к через полотно.
        const k = pulse.back ? 1 - pulse.at : pulse.at;
        let x: number, yy: number;
        if (k < 0.42) {
          const u = k / 0.42;
          x = node.x + (entry - node.x) * u;
          yy = node.y + (y - node.y) * u;
        } else {
          const u = (k - 0.42) / 0.58;
          x = entry + (exit - entry) * u;
          yy = y;
        }
        const g = ctx.createRadialGradient(x, yy, 0, x, yy, 7);
        g.addColorStop(0, "rgba(120, 245, 224, .85)");
        g.addColorStop(1, "rgba(120, 245, 224, 0)");
        ctx.fillStyle = g;
        ctx.fillRect(x - 7, yy - 7, 14, 14);
      }
      ctx.globalCompositeOperation = "source-over";
    };

    const draw = (t: number) => {
      if (!w || !h) { raf = requestAnimationFrame(draw); return; }
      const f = frame();
      // Пучок выходит из полотна ровно на «кластер»: аренда вытесняет базовую
      // загрузку. На «инференс» нити остаются в полотне — он и есть полотно.
      const want = highlight.current === "compute" ? 1 : 0;
      lift += (want - lift) * (calm ? 1 : 0.08);

      ctx.clearRect(0, 0, w, h);
      drawGlow(f);
      drawWarp(f);
      for (let i = 0; i < ROWS; i++) drawThread(f, i, t);
      drawWeave(f);
      drawPulses(f, t);

      if (!calm) raf = requestAnimationFrame(draw);
    };

    if (calm) {
      // Замирает на уже сплетённом полотне: неподвижная картинка должна быть
      // законченной, а не остановленной на полпути.
      progress.current = 1;
      draw(0);
    } else {
      raf = requestAnimationFrame(draw);
    }

    return () => { cancelAnimationFrame(raf); watch.disconnect(); };
  }, [progress, highlight]);

  return <canvas ref={canvas} className="loom" aria-hidden="true" />;
}
