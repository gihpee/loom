export interface Node {
  node_id: string;
  region: string;
  agent_version: string;
  device: string;
  gpu_name: string;
  cuda_version: string;
  gpus_total: number;
  gpus_free: number;
  vram_free_bytes: number;
  host_ram_gb: number;
  accepts_tasks: boolean;
  refusal: string;
  environment_kinds: string[];
  tasks_running: number;
  env_cache_bytes: number;
  seconds_since_seen: number;
  peer_id: string;
  symmetric_nat: boolean;
  reachable: boolean;
  direct: number;
  relayed: number;
  direct_share: number;
  link_rtt_ms: number;
}

export interface ResultFile {
  name: string;
  size_bytes: number;
  digest: string;
}

export interface Task {
  task_id: string;
  node_id: string;
  command: string[];
  state: string;
  error: string;
  exit_code: number;
  devices: number[];
  seconds: number;
  submitted_at: number;
  results: ResultFile[];
  group_id: string;
  rank: number;
}

export interface Group {
  group_id: string;
  label: string;
  size: number;
  ranks: { rank: number; task_id: string; node_id: string }[];
  submitted_at: number;
}

export interface StageHealth {
  rank: number;
  task_id: string;
  node_id: string;
  state: string;
  error: string;
  seconds: number;
  ready: boolean;
  stage: { status: string; layers: [number, number] | null } | null;
}

export interface GroupHealth {
  group_id: string;
  label: string;
  ready: boolean;
  stages: StageHealth[];
}

export interface Release {
  version: string;
  sha256: string;
  wave_percent: number;
  published_at: number;
  size_bytes: number;
}

export interface VersionMap {
  release: Release | null;
  versions: Record<string, number>;
  nodes_total: number;
  nodes_on_target: number;
  nodes_in_wave: number;
}

export interface JoinKey {
  key_id: string;
  label: string;
  max_nodes: number;
  nodes: string[];
  revoked: boolean;
  key?: string;
  address?: string;
  agent_image?: string;
}
