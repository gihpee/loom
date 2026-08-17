-- Loom persistent catalog / billing schema (Postgres).
-- Live perf data (latencies, RTT) lives in Redis (see store.py key layout);
-- this DDL covers the durable side: model catalog, node registry, allocations,
-- telemetry export for billing.

CREATE TABLE IF NOT EXISTS models (
    model_id        TEXT PRIMARY KEY,          -- e.g. "qwen3-0.6b"
    hf_name         TEXT NOT NULL,             -- upstream weights reference
    num_layers      INTEGER NOT NULL,
    per_layer_bytes BIGINT  NOT NULL,          -- ModelInfo.decoder_layer_io_bytes(roofline=false)
    embedding_bytes BIGINT  NOT NULL,
    tie_embedding   BOOLEAN NOT NULL DEFAULT FALSE,
    model_info_json JSONB   NOT NULL,          -- full serialized ModelInfo
    demand_qps      DOUBLE PRECISION NOT NULL DEFAULT 0,
    priority        DOUBLE PRECISION NOT NULL DEFAULT 1,
    price_willing   DOUBLE PRECISION NOT NULL DEFAULT 1,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    deleted_at      TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS nodes (
    node_id         TEXT PRIMARY KEY,
    pubkey          TEXT NOT NULL,             -- onboarding keypair (Phase 4)
    region          TEXT NOT NULL DEFAULT 'default',
    device          TEXT NOT NULL,             -- "cuda" | "mlx" | ...
    gpu_name        TEXT,
    num_gpus        INTEGER NOT NULL DEFAULT 1,
    vram_total_bytes BIGINT NOT NULL,
    tflops_fp16     DOUBLE PRECISION NOT NULL,
    mem_bandwidth_gbps DOUBLE PRECISION NOT NULL,
    price_per_gpu_hour DOUBLE PRECISION NOT NULL DEFAULT 0,
    declared_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    calibrated      BOOLEAN NOT NULL DEFAULT FALSE,  -- benchmark-calibrated vs declared
    banned          BOOLEAN NOT NULL DEFAULT FALSE,
    last_seen_at    TIMESTAMPTZ
);

-- Resource Broker output: which slice of which node serves which model.
CREATE TABLE IF NOT EXISTS allocations (
    allocation_id   BIGSERIAL PRIMARY KEY,
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    node_id         TEXT NOT NULL REFERENCES nodes(node_id),
    vram_quota_bytes BIGINT NOT NULL,
    start_layer     INTEGER,
    end_layer       INTEGER,
    backend_type    TEXT NOT NULL,             -- "vllm" | "sglang" | "mlx"
    state           TEXT NOT NULL DEFAULT 'planned',  -- planned|loading|serving|draining|released
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    released_at     TIMESTAMPTZ,
    UNIQUE (model_id, node_id, created_at)
);
CREATE INDEX IF NOT EXISTS allocations_by_model ON allocations (model_id) WHERE released_at IS NULL;
CREATE INDEX IF NOT EXISTS allocations_by_node  ON allocations (node_id)  WHERE released_at IS NULL;

-- Usage export from ReportTelemetry, billing-ready aggregation source.
CREATE TABLE IF NOT EXISTS usage_records (
    record_id       BIGSERIAL PRIMARY KEY,
    node_id         TEXT NOT NULL REFERENCES nodes(node_id),
    model_id        TEXT NOT NULL REFERENCES models(model_id),
    window_start    TIMESTAMPTZ NOT NULL,
    window_end      TIMESTAMPTZ NOT NULL,
    tokens_in       BIGINT NOT NULL DEFAULT 0,
    tokens_out      BIGINT NOT NULL DEFAULT 0,
    gpu_seconds     DOUBLE PRECISION NOT NULL DEFAULT 0,
    vram_byte_seconds NUMERIC NOT NULL DEFAULT 0,
    reported_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS usage_by_node_window  ON usage_records (node_id, window_start);
CREATE INDEX IF NOT EXISTS usage_by_model_window ON usage_records (model_id, window_start);

-- Reputation / anomaly log (Phase 4 MVP: declared vs measured deviations).
CREATE TABLE IF NOT EXISTS node_anomalies (
    anomaly_id      BIGSERIAL PRIMARY KEY,
    node_id         TEXT NOT NULL REFERENCES nodes(node_id),
    kind            TEXT NOT NULL,             -- 'perf_deviation' | 'quota_violation' | ...
    details_json    JSONB NOT NULL,
    observed_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);
