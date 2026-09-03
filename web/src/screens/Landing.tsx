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
import { WeaveReveal, useWeaveReveal } from "../components/Weave";
import { Loom, type Highlight } from "./Loom";

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

const FAQ = [
  {
    q: "Можно ли уже сдавать свою карту?",
    a: "Нет. Провайдерской стороны пока не существует: сеть работает на нашем " +
      "железе. Сдача мощности — следующий этап, и мы не хотим показывать её " +
      "раньше, чем сможем отвечать за выплаты.",
  },
  {
    q: "Как происходит оплата?",
    a: "Пока никак. Потребление считается посекундно и по токенам, счёт виден " +
      "в кабинете, но списаний нет. Мы сначала научились считать честно, а " +
      "приём платежей добавим, когда будет что предъявить.",
  },
  {
    q: "Всё ли потребление попадает в счёт?",
    a: "Почти. Токены потокового инференса пока не учитываются: счётчики " +
      "приходят в последнем куске потока, минуя оркестратор. Обычные ответы " +
      "считаются полностью. Это записано у нас в документации, а не спрятано.",
  },
  {
    q: "Как получить доступ?",
    a: "По приглашению. Учётные записи заводит администратор — сеть в закрытой " +
      "бете, и мы отвечаем за каждую машину в ней.",
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
            <a href="#surfaces">Возможности</a>
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
            <a href="#surfaces" className="lp-point" data-lit={lit === "inference"}
               {...point("inference")}>инференс</a>{" "}
            или{" "}
            <a href="#surfaces" className="lp-point" data-lit={lit === "compute"}
               {...point("compute")}>кластер</a>{" "}
            — платите за использованное.
          </p>
          <div className="lp-cta">
            <a className="lp-primary" href="/app">Начать</a>
          </div>
        </div>
      </section>

      <section className="lp-section" id="surfaces">
        <div className="lp-wrap">
          <h2 className="lp-h2">Две поверхности</h2>
          <div className="lp-surfaces">
            <Surface
              featured
              id="looma-inference"
              title="Инференс"
              line="Готовый API вместо своей карты."
              detail="Совместим с OpenAI: меняется адрес и ключ, остальной код
                      остаётся. Модель разрезана по слоям между картами, поэтому
                      работает то, что не помещается ни в одну из них."
              facts={["оплата за токены", "потоковая выдача", "батчинг запросов"]}
              {...point("inference")}
            />
            <Surface
              id="looma-compute"
              title="Кластер"
              line="Свой код на арендованных картах."
              detail="Обычный Ray: подключаетесь с ноутбука одной строкой.
                      Кластер собирается на узлах, у которых нет ни одного
                      открытого порта."
              facts={["оплата за GPU-часы", "почасовая аренда", "локальный порт"]}
              {...point("compute")}
            />
          </div>
        </div>
      </section>

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

      <section className="lp-section" id="faq">
        <div className="lp-wrap lp-faq-wrap">
          <h2 className="lp-h2">Чего пока нет</h2>
          <p className="lp-sub">
            Продукт молодой, и честнее сказать это самим, чем дать выяснить в
            процессе.
          </p>
          <div className="lp-faq">
            {FAQ.map((item) => <Question key={item.q} {...item} />)}
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

function Surface({ id, title, line, detail, facts, featured, ...rest }: {
  id: string; title: string; line: string; detail: string; facts: string[];
  featured?: boolean;
} & React.HTMLAttributes<HTMLElement>) {
  const ref = useWeaveReveal<HTMLElement>(0.25);
  return (
    <article ref={ref} {...rest}
             className={`lp-surface weave-reveal${featured ? " card--featured" : ""}`}>
      <h3>{title}</h3>
      <p className="lp-line">{line}</p>
      <p className="lp-detail">{detail}</p>
      <ul>{facts.map((f) => <li key={f}>{f}</li>)}</ul>
      <code className="lp-id">{id}</code>
    </article>
  );
}

function Question({ q, a }: { q: string; a: string }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="lp-q" data-open={open}>
      <button onClick={() => setOpen(!open)} aria-expanded={open}>
        <span>{q}</span>
        <i aria-hidden="true" />
      </button>
      <div className="lp-a"><p>{a}</p></div>
    </div>
  );
}
