-- Учётные записи, сессии и ключи API.
--
-- Первая миграция проекта: до неё всё персистентное лежало в JSON-файлах,
-- переписываемых целиком при каждой записи. Для десятка ключей подключения это
-- было правильно; для журнала потребления, который растёт без конца, — нет.

CREATE TABLE IF NOT EXISTS accounts (
    id           BIGSERIAL PRIMARY KEY,
    email        TEXT        NOT NULL,
    password     TEXT        NOT NULL,
    role         TEXT        NOT NULL CHECK (role IN ('client', 'admin')),
    display_name TEXT        NOT NULL DEFAULT '',
    -- Отметка, а не флаг: «когда отключили» отвечает и на вопрос «отключён ли»,
    -- и на вопрос «с какого момента», а второй понадобится при разборе счетов.
    disabled_at  TIMESTAMPTZ,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- По lower(email), а не по email: иначе Ivan@ и ivan@ — два разных человека,
-- и второй зарегистрируется поверх первого, ничего не заметив.
CREATE UNIQUE INDEX IF NOT EXISTS accounts_email_key ON accounts (lower(email));

CREATE TABLE IF NOT EXISTS sessions (
    -- Первичный ключ — хэш, а не сам токен: утёкшая база не должна давать
    -- работающих сессий.
    token_hash   TEXT        PRIMARY KEY,
    account_id   BIGINT      NOT NULL REFERENCES accounts (id) ON DELETE CASCADE,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    expires_at   TIMESTAMPTZ NOT NULL
);

CREATE INDEX IF NOT EXISTS sessions_account_idx ON sessions (account_id);
-- По сроку: истёкшие подчищаются пачкой, и без индекса это полный проход по
-- таблице, которая растёт с каждым входом.
CREATE INDEX IF NOT EXISTS sessions_expires_idx ON sessions (expires_at);

CREATE TABLE IF NOT EXISTS api_keys (
    id           BIGSERIAL   PRIMARY KEY,
    account_id   BIGINT      NOT NULL REFERENCES accounts (id) ON DELETE CASCADE,
    token_hash   TEXT        NOT NULL UNIQUE,
    -- Начало ключа: владелец узнаёт свой среди нескольких, а секрета в нём нет.
    hint         TEXT        NOT NULL,
    name         TEXT        NOT NULL DEFAULT '',
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_used_at TIMESTAMPTZ,
    revoked_at   TIMESTAMPTZ
);

CREATE INDEX IF NOT EXISTS api_keys_account_idx ON api_keys (account_id);
