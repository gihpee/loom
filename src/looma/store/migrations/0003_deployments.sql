-- Развёртывания моделей: чем их поднимали и что с ними стало.
--
-- Нужно ровно для одного — вернуть на место снятое. Без этой таблицы каждая
-- аренда убивала бы инференс НАВСЕГДА: группа снята, а чем её поднимали, знал
-- только тот HTTP-запрос, который давно закончился.

CREATE TABLE IF NOT EXISTS deployments (
    group_id    TEXT        PRIMARY KEY,
    label       TEXT        NOT NULL DEFAULT '',
    -- Тело запроса, которым это подняли. Им же и поднимем обратно: узлы
    -- подберутся заново, потому что прежние к тому времени могут быть заняты.
    request     JSONB       NOT NULL,
    -- Кто просил. Аренда закроется — вернём от его же имени.
    account_id  BIGINT      REFERENCES accounts (id) ON DELETE SET NULL,
    -- running | evicted | gone
    state       TEXT        NOT NULL DEFAULT 'running',
    -- Защищённое не вытесняется никогда, что бы ни просил арендатор.
    protected   BOOLEAN     NOT NULL DEFAULT FALSE,
    -- Какой арендой снято. По ней и вернём, когда та закроется.
    evicted_by  BIGINT      REFERENCES leases (id) ON DELETE SET NULL,
    evicted_at  TIMESTAMPTZ,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Снятые ищут при каждом закрытии аренды, и их всегда мало на фоне остальных.
CREATE INDEX IF NOT EXISTS deployments_evicted_idx
    ON deployments (evicted_by) WHERE state = 'evicted';
