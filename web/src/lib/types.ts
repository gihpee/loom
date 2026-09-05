export interface Node {
  node_id: string; region: string; agent_version: string;
  device: string; gpu_name: string; cuda_version: string;
  gpus_total: number; gpus_free: number; vram_free_bytes: number; host_ram_gb: number;
  accepts_tasks: boolean; refusal: string; environment_kinds: string[];
  tasks_running: number; env_cache_bytes: number; model_cache_bytes: number;
  disk_free_bytes: number; disk_total_bytes: number;
  connected_at: number; seconds_since_seen: number;
  peer_id: string; symmetric_nat: boolean; reachable: boolean;
  in_network: boolean; visible_addrs: string[];
  direct: number; relayed: number; direct_share: number; link_rtt_ms: number;
  update_state: string; update_version: string; update_error: string;
}

export interface ResultFile { name: string; size_bytes: number; digest: string }

export interface Task {
  task_id: string; node_id: string; command: string[];
  state: string; error: string; exit_code: number;
  devices: number[]; seconds: number; submitted_at: number;
  results: ResultFile[]; group_id: string; rank: number;
  // Задача была на узле, когда оркестратор о ней забыл: её приняли
  // обратно по докладу узла, и команды у неё нет — узел её не хранит.
  adopted?: boolean;
}

export interface Group {
  group_id: string; label: string; size: number;
  ranks: { rank: number; task_id: string; node_id: string }[];
  submitted_at: number;
  // Все ранги закончились. Считает оркестратор — иначе каждый экран сводил бы
  // список задач с составом группы сам.
  finished?: boolean;
}

export interface StageHealth {
  rank: number; task_id: string; node_id: string;
  state: string; error: string; seconds: number; ready: boolean;
  // Ответ самой нагрузки, как она его отдала. У стадии инференса свои поля,
  // у ранга Ray свои — оркестратор его не переписывает и о нём не судит.
  stage: {
    status: string;
    layers?: [number, number] | null;   // стадия инференса
    nodes?: number; size?: number; error?: string;   // ранг Ray
    client_port?: number; python?: string; ray?: string;
  } | null;
}

export interface GroupHealth {
  group_id: string; label: string; ready: boolean; stages: StageHealth[];
}

export interface Release {
  version: string; sha256: string; wave_percent: number;
  published_at: number; size_bytes: number;
}

export interface VersionMap {
  release: Release | null; versions: Record<string, number>;
  nodes_total: number; nodes_on_target: number; nodes_in_wave: number;
}

export interface JoinKey {
  key_id: string; label: string; max_nodes: number; nodes: string[];
  revoked: boolean; key?: string; address?: string; agent_image?: string;
}

export interface Connect {
  dial_address: string; source: string; severity: string;
  self_check: boolean | null; warning: string | null; agent_image: string;
}
