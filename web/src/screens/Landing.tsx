/** Публичная главная.
 *
 * Что здесь НЕ делается и почему. Не пересказывается устройство системы —
 * человеку, решающему, брать ли мощность, безразлично, чем она оркестрируется.
 * Не обещается «ноль простоя»: провайдеров ещё нет, говорить об их выгоде
 * рано. Нет отзывов и тарифов — их не существует, а придуманные видны сразу.
 *
 * Концепт держится на форме, а не на подписи: полотно на фоне ткётся по мере
 * прокрутки и отзывается на наведение. Текст говорит, что можно взять; фон
 * показывает, как это устроено.
 */
import { useEffect, useRef, useState } from "react";
import { Mark } from "../components";
import { WeaveReveal } from "../components/Weave";
import { Loom, type Highlight } from "./Loom";

/* Три возможности одной строкой. Текст короткий намеренно: подпись под иконкой
   читают взглядом, а не абзацем, и длинная фраза здесь просто не читается. */
const OPTIONS = [
  {
    id: "inference",
    title: "Инференс",
    text: "Готовый API. Как у OpenAI.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
           strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M4 5h16v10H9.5L4 19V5Z" />
        <path d="m12 8 1.1 2.3 2.3 1.1-2.3 1.1L12 15l-1.1-2.5-2.3-1.1 2.3-1.1L12 8Z" />
      </svg>
    ),
  },
  {
    id: "compute",
    title: "Ray-кластер",
    text: "Свой код на чужих картах.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
           strokeLinecap="round" aria-hidden="true">
        <circle cx="6" cy="7" r="1.9" /><circle cx="18" cy="6" r="1.9" />
        <circle cx="12" cy="13" r="2.1" />
        <circle cx="6" cy="18" r="1.9" /><circle cx="18" cy="17" r="1.9" />
        <path d="M7.6 8.3 10.4 11.5M16.4 7.4 13.7 11.3M10.4 14.5 7.6 16.5M13.7 14.4 16.4 15.7" />
      </svg>
    ),
  },
  {
    id: "weights",
    title: "Свои модели",
    text: "Ваши веса — по запросу.",
    icon: (
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"
           strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M12 3v9M9 6l3-3 3 3" />
        <path d="M4 13v5a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-5" />
      </svg>
    ),
  },
];

const FACTS = [
  {
    id: "outbound",
    weight: "core",
    head: "Одно исходящее соединение",
    text: "У машины-поставщика нет и не будет входящих портов: это домашний " +
      "компьютер за роутером, который никто не настраивает. Узел сам открывает " +
      "канал наружу, и всё идёт обратно по нему же — команды, активации модели, " +
      "порт до кластера.",
  },
  {
    id: "anchor",
    weight: "mid",
    head: "Платформа — первый клиент своей сети",
    text: "Пока прямого арендатора нет, карты занимает инференс. Приходит " +
      "клиент за кластером — модели уступают ему узлы и возвращаются, когда " +
      "аренда кончилась.",
  },
  {
    id: "isolation",
    weight: "mid",
    head: "Чужой код в песочнице",
    text: "Задача идёт под отдельным пользователем, в своём каталоге, с " +
      "ограничениями по памяти и процессам. Владелец машины сдаёт мощность, а " +
      "не доступ к себе.",
  },
];

export function Landing() {
  const progress = useRef(0);
  const highlight = useRef<Highlight>("");
  const [lit, setLit] = useState<Highlight>("");

  // Полотно ткётся прокруткой. Значение пишем в ref, а не в состояние: иначе
  // каждый кадр скролла перерисовывал бы всё дерево.
  useEffect(() => {
    const onScroll = () => {
      const total = document.body.scrollHeight - innerHeight;
      progress.current = total > 0 ? Math.min(1, scrollY / (total * 0.85)) : 1;
    };
    onScroll();
    addEventListener("scroll", onScroll, { passive: true });
    return () => removeEventListener("scroll", onScroll);
  }, []);

  const point = (what: Highlight) => ({
    onMouseEnter: () => { highlight.current = what; setLit(what); },
    onMouseLeave: () => { highlight.current = ""; setLit(""); },
    onFocus: () => { highlight.current = what; setLit(what); },
    onBlur: () => { highlight.current = ""; setLit(""); },
  });

  return (
    <div className="landing">
      <Loom progress={progress} highlight={highlight} />

      <header className="lp-top">
        <div className="lp-wrap lp-nav">
          <a className="lp-brand" href="/"><Mark size={26} /><span>Looma&nbsp;Float</span></a>
          <nav>
            <a href="#deploy">Возможности</a>
            <a href="#demo">Демо</a>
            <a href="#how">Как устроено</a>
            <a className="lp-enter" href="/app">Войти</a>
          </nav>
        </div>
      </header>

      <section className="lp-hero">
        <div className="lp-wrap lp-hero-body">
          <h1>Мощность,<br />когда она нужна</h1>
          <p className="lp-lead">
            Считаем на распределённой сети GPU. Берите{" "}
            <a href="#deploy" className="lp-point" data-lit={lit === "inference"}
               {...point("inference")}>инференс</a>{" "}
            или{" "}
            <a href="#deploy" className="lp-point" data-lit={lit === "compute"}
               {...point("compute")}>кластер</a>{" "}
            — платите за использованное.
          </p>
          <div className="lp-cta">
            <a className="lp-primary" href="/app">Начать</a>
          </div>
        </div>
      </section>

      <section className="lp-section" id="deploy">
        <div className="lp-wrap">
          <h2 className="lp-h2">Разворачивайте что хотите</h2>
          <p className="lp-sub">И когда хотите.</p>
          <div className="lp-deploy">
            <Arcs />
            {OPTIONS.map((o, i) => (
              <WeaveReveal key={o.id} className="lp-opt" delay={i * 120}>
                <span className="lp-chip">{o.icon}</span>
                <h3>{o.title}</h3>
                <p>{o.text}</p>
              </WeaveReveal>
            ))}
          </div>
        </div>
      </section>

      <Demo />

      <section className="lp-section" id="how">
        <div className="lp-wrap">
          <h2 className="lp-h2">Как устроено</h2>
          <div className="lp-facts">
            {FACTS.map((f, i) => (
              <WeaveReveal key={f.id} delay={i * 110}
                           className={`lp-fact lp-fact--${f.weight}`}>
                <h3>{f.head}</h3>
                <p>{f.text}</p>
              </WeaveReveal>
            ))}
          </div>
        </div>
      </section>

      <section className="lp-section lp-final">
        <div className="lp-wrap">
          <WeaveReveal>
            <h2>Полотно уже соткано</h2>
            <p className="lp-lead">
              Сеть работает, счёт считается, модели отвечают. Осталось выдать
              вам ключ.
            </p>
            <a className="lp-primary" href="/app">Войти в кабинет</a>
          </WeaveReveal>
        </div>
      </section>

      <footer className="lp-foot">
        <div className="lp-wrap">
          <Mark size={22} />
          <p>Looma Float</p>
        </div>
      </footer>
    </div>
  );
}



/** Нити между возможностями. Не украшение: они те же, что на фоновом полотне, —
 *  три способа взять мощность идут по одной сети, а не стоят порознь. */
function Arcs() {
  return (
    <svg className="lp-arcs" viewBox="0 0 1000 120" preserveAspectRatio="none"
         aria-hidden="true">
      {[0, 1, 2, 3].map((i) => (
        <path key={i} fill="none" stroke="currentColor" strokeWidth="1"
              d={`M0 ${64 + i * 7} C 250 ${18 + i * 7}, 750 ${18 + i * 7}, 1000 ${64 + i * 7}`} />
      ))}
    </svg>
  );
}

/** Живое демо: та самая модель, что развёрнута на парке, отвечает прямо здесь.
 *
 *  Смысл блока — не «попробуйте чат», а «посмотрите на скорость». Поэтому
 *  показатели считаются по-настоящему: время до первого токена меряется
 *  отдельно от скорости выдачи, иначе ожидание очереди размазалось бы по всей
 *  генерации и цифра польстила бы нам вдвое.
 *
 *  Блока нет вовсе, когда сеть пуста: поле ввода, которое ничего не отвечает,
 *  хуже отсутствующего раздела.
 */
function Demo() {
  const [ready, setReady] = useState<{ model: string; nodes: number } | null>(null);
  const [prompt, setPrompt] = useState("");
  const [answer, setAnswer] = useState("");
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);
  const [stat, setStat] = useState({ tokens: 0, first: 0, rate: 0 });

  useEffect(() => {
    let alive = true;
    fetch("/api/demo")
      .then((r) => (r.ok ? r.json() : null))
      .then((d) => { if (alive && d?.model) setReady(d); })
      .catch(() => {});
    return () => { alive = false; };
  }, []);

  if (!ready) return null;

  const ask = async () => {
    if (busy || !prompt.trim()) return;
    setBusy(true); setAnswer(""); setError(""); setStat({ tokens: 0, first: 0, rate: 0 });
    const started = performance.now();
    let first = 0, tokens = 0, text = "";
    try {
      const res = await fetch("/api/demo", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt }),
      });
      if (!res.ok || !res.body) {
        const why = await res.json().catch(() => null);
        throw new Error(why?.error?.message ?? "сейчас не отвечает");
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let tail = "";
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        tail += decoder.decode(value, { stream: true });
        const lines = tail.split("\n");
        tail = lines.pop() ?? "";          // последняя строка может быть неполной
        for (const line of lines) {
          if (!line.startsWith("data:")) continue;
          const payload = line.slice(5).trim();
          if (!payload || payload === "[DONE]") continue;
          let piece: any;
          try { piece = JSON.parse(payload); } catch { continue; }
          if (piece.error) throw new Error(piece.error.message);
          const delta: string = piece.choices?.[0]?.delta?.content ?? "";
          if (!delta) continue;
          if (!first) first = performance.now() - started;
          tokens += 1;
          text += delta;
          const spent = (performance.now() - started - first) / 1000;
          setAnswer(text);
          setStat({ tokens, first, rate: spent > 0 ? tokens / spent : 0 });
        }
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <section className="lp-section" id="demo">
      <div className="lp-wrap">
        <h2 className="lp-h2">Убедитесь сами</h2>
        <p className="lp-sub">
          {ready.model} разрезана по домашним машинам — {ready.nodes}{" "}
          {ready.nodes === 1 ? "узел" : ready.nodes < 5 ? "узла" : "узлов"}. Спросите её.
        </p>

        <div className="lp-demo glass">
          <textarea
            value={prompt} rows={2} maxLength={400}
            placeholder="Спросите что-нибудь"
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); ask(); }
            }}
          />
          <button className="lp-ask" onClick={ask} disabled={busy || !prompt.trim()}
                  aria-label="Спросить">
            {busy ? <i className="lp-spin" /> : "→"}
          </button>
        </div>

        {(answer || error || busy) && (
          <div className="lp-out">
            {error
              ? <p className="lp-fail">{error}</p>
              : <p className="lp-answer">{answer}<span className="lp-caret" data-on={busy} /></p>}
            {stat.tokens > 0 && (
              <dl className="lp-stats">
                <div><dt>скорость</dt><dd>{stat.rate.toFixed(1)} <b>ток/с</b></dd></div>
                <div><dt>первый токен</dt><dd>{(stat.first / 1000).toFixed(2)} <b>с</b></dd></div>
                <div><dt>выдано</dt><dd>{stat.tokens} <b>ток</b></dd></div>
              </dl>
            )}
          </div>
        )}
      </div>
    </section>
  );
}
