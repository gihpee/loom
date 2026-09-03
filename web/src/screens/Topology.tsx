/** Мини-топология: оркестратор и нити к узлам.
 *
 * Здесь мотив станка работает **прибором**, а не рассказом. На лендинге нить
 * объясняет идею; тут она отвечает на вопрос «что сейчас с сетью» и обязана
 * читаться за секунду:
 *
 *   цвет нити  — чьё держит узел: простаивает, инференс, клиент, обновляется;
 *   пульс      — живой обмен по gRPC. Нить тускнеет, когда узел замолчал.
 *
 * Поэтому здесь нет стадии «прибой»: пружинный доводчик уместен там, где
 * что-то раскрывается, и неуместен там, где показывают состояние. Пульс ровный
 * и одинаковый на всех нитях — глаз должен ловить различия в ЦВЕТЕ и в том,
 * какая нить погасла, а не в том, какая красивее движется.
 */
import { useEffect, useRef } from "react";
import { calmMotion } from "../components/Weave";

export type NodeState = "idle" | "inference" | "rented" | "updating" | "lost";

export interface Strand {
  id: string;
  state: NodeState;
  /** Секунд с последнего слова узла: по нему нить и тускнеет. */
  silent: number;
}

const TONE: Record<NodeState, string> = {
  idle:      "31, 191, 168",
  inference: "79, 232, 206",
  rented:    "234, 245, 242",
  updating:  "229, 185, 92",
  lost:      "242, 109, 120",
};

/** Молчание, после которого нить считается погасшей. Втрое больше периода
 *  телеметрии: одна пропущенная посылка — это не потеря связи. */
const SILENT_LIMIT = 45;

export function Topology({ strands }: { strands: Strand[] }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  const data = useRef(strands);
  data.current = strands;

  useEffect(() => {
    const c = canvas.current;
    const ctx = c?.getContext("2d");
    if (!c || !ctx) return;

    const calm = calmMotion();
    const dpr = Math.min(devicePixelRatio || 1, 2);
    let w = 0, h = 0, raf = 0;

    const size = () => {
      w = c.clientWidth; h = c.clientHeight;
      if (!w || !h) return;
      c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    size();
    const watch = new ResizeObserver(size);
    watch.observe(c);

    const draw = (t: number) => {
      if (!w || !h) { raf = requestAnimationFrame(draw); return; }
      const rows = data.current;
      ctx.clearRect(0, 0, w, h);

      const cx = w / 2, cy = h / 2;
      const radius = Math.min(w, h) * 0.40;

      // Оркестратор: единственная точка, до которой можно дозвониться.
      ctx.beginPath();
      ctx.arc(cx, cy, 7, 0, Math.PI * 2);
      ctx.fillStyle = "rgba(79, 232, 206, .9)";
      ctx.fill();
      ctx.beginPath();
      ctx.arc(cx, cy, 13, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(31, 191, 168, .35)";
      ctx.lineWidth = 1;
      ctx.stroke();

      rows.forEach((strand, i) => {
        const angle = (i / Math.max(1, rows.length)) * Math.PI * 2 - Math.PI / 2;
        const nx = cx + Math.cos(angle) * radius;
        const ny = cy + Math.sin(angle) * radius;
        const quiet = Math.min(1, strand.silent / SILENT_LIMIT);
        const tone = TONE[strand.state];
        // Тускнеет пропорционально молчанию: не «связь есть/нет», а насколько
        // давно узел говорил. Резкий порог скрывал бы, что всё уже плохо.
        const alpha = 0.5 * (1 - quiet * 0.8);

        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(nx, ny);
        ctx.strokeStyle = `rgba(${tone}, ${alpha})`;
        ctx.lineWidth = strand.state === "rented" ? 1.6 : 1.2;
        ctx.stroke();

        ctx.beginPath();
        ctx.arc(nx, ny, 3.4, 0, Math.PI * 2);
        ctx.fillStyle = `rgba(${tone}, ${0.35 + 0.55 * (1 - quiet)})`;
        ctx.fill();

        // Пульс — ровный, одинаковый на всех нитях. Он про то, что обмен идёт,
        // а не про то, какая нить интереснее. Замолчавшая его не показывает.
        if (!calm && quiet < 0.9) {
          const phase = ((t / 1500 + i * 0.17) % 1);
          const px = cx + (nx - cx) * phase;
          const py = cy + (ny - cy) * phase;
          const glow = ctx.createRadialGradient(px, py, 0, px, py, 6);
          glow.addColorStop(0, `rgba(${tone}, ${0.75 * (1 - quiet)})`);
          glow.addColorStop(1, `rgba(${tone}, 0)`);
          ctx.fillStyle = glow;
          ctx.fillRect(px - 6, py - 6, 12, 12);
        }
      });

      if (!calm) raf = requestAnimationFrame(draw);
    };

    if (calm) draw(0); else raf = requestAnimationFrame(draw);
    return () => { cancelAnimationFrame(raf); watch.disconnect(); };
  }, []);

  return <canvas ref={canvas} className="topology" aria-hidden="true" />;
}

/** Состояние узла по тому, что о нём известно. Отдельной функцией: правило
 *  должно читаться целиком, а не собираться из веток внутри отрисовки. */
export function strandOf(node: {
  node_id: string; seconds_since_seen: number; update_state: string;
}, held: Map<string, NodeState>): Strand {
  const silent = Math.max(0, node.seconds_since_seen);
  const state: NodeState =
    silent >= SILENT_LIMIT ? "lost"
    : node.update_state && node.update_state !== "idle" ? "updating"
    : held.get(node.node_id) ?? "idle";
  return { id: node.node_id, state, silent };
}
