-- Журнал потребления: что и сколько израсходовал каждый.
--
-- Только на добавление. Закрытая запись правится ровно один раз — в момент
-- закрытия, — и больше не меняется никогда: счёт, который можно переписать
-- задним числом, не счёт.

-- Ставки. Отдельно от журнала, потому что меняются: сегодняшняя цена не должна
-- переписывать вчерашние записи.
CREATE TABLE IF NOT EXISTS rates (
    resource   TEXT        PRIMARY KEY,
    -- Копейки за GPU-час, а не рубли: деньги в плавающей точке — способ
    -- получить 0.1 + 0.2 = 0.30000000000000004 в счёте.
    per_hour   BIGINT      NOT NULL CHECK (per_hour >= 0),
    currency   TEXT        NOT NULL DEFAULT 'RUB',
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Аренда: ресурс занят с такого-то момента по такой-то.
CREATE TABLE IF NOT EXISTS leases (
    id          BIGSERIAL   PRIMARY KEY,
    account_id  BIGINT      NOT NULL REFERENCES accounts (id) ON DELETE RESTRICT,
    resource    TEXT        NOT NULL,
    -- Группа задач, которой это соответствует. По ней аренда и закрывается,
    -- когда группа исчезла сама.
    group_id    TEXT        NOT NULL,
    label       TEXT        NOT NULL DEFAULT '',
    nodes       INTEGER     NOT NULL DEFAULT 0,
    gpus        INTEGER     NOT NULL DEFAULT 0,
    -- Ставка ЗАПИСЫВАЕТСЯ в аренду, а не берётся из rates при подсчёте:
    -- иначе изменение цены переписало бы историю.
    per_hour    BIGINT      NOT NULL DEFAULT 0,
    currency    TEXT        NOT NULL DEFAULT 'RUB',
    opened_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    closed_at   TIMESTAMPTZ,
    -- Как закрылась: released — сняли намеренно; vanished — группы не стало, и
    -- мы закрыли по последнему наблюдению. Разница важна при разборе: во
    -- втором случае конец времени приблизителен.
    closed_why  TEXT,
    CONSTRAINT leases_closed_after_opened CHECK (closed_at IS NULL OR closed_at >= opened_at)
);

CREATE INDEX IF NOT EXISTS leases_account_idx ON leases (account_id, opened_at);
-- Открытые: их ищут при каждой сверке, и их всегда мало на фоне закрытых.
CREATE INDEX IF NOT EXISTS leases_open_idx ON leases (group_id) WHERE closed_at IS NULL;

-- Токены инференса: другая единица, поэтому другая таблица. Складывать часы и
-- токены в одну означало бы столбец, смысл которого зависит от соседнего.
CREATE TABLE IF NOT EXISTS token_usage (
    id                BIGSERIAL   PRIMARY KEY,
    account_id        BIGINT      NOT NULL REFERENCES accounts (id) ON DELETE RESTRICT,
    model             TEXT        NOT NULL,
    prompt_tokens     INTEGER     NOT NULL DEFAULT 0,
    completion_tokens INTEGER     NOT NULL DEFAULT 0,
    at                TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS token_usage_account_idx ON token_usage (account_id, at);
