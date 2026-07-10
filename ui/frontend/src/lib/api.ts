export type StageState = "pending" | "running" | "blocked" | "failed" | "done" | "skipped"

export interface JobStage {
  state: StageState
  message?: string
  progress?: number
  started_at?: number | null
  finished_at?: number | null
  meta?: Record<string, unknown>
}

export interface PipelineJob {
  id: string
  name: string
  created_at: number
  input?: { type?: string; source?: string }
  current_stage?: string | null
  stages: Record<string, JobStage>
  preview_url?: string
  vet?: VetReport
  log_tail?: string[]
}

export interface VetReport {
  pass?: boolean
  frames?: number
  seconds?: number
  hard?: Record<string, { pass?: boolean; [key: string]: unknown }>
  advisory?: Record<string, { ok?: boolean; [key: string]: unknown }>
}

export interface ExamMetrics {
  nominal?: { pass?: boolean; success_rate?: number; mpkpe_m?: number; ee_pos_error_m?: number; held_out_seed?: number }
  push?: { pass?: boolean; success_rate?: number; mpkpe_m?: number; force_n?: number; held_out_seed?: number }
  repeatability?: number
  [key: string]: unknown
}

export interface SimExam {
  verdict?: string
  at?: number
  exam_id?: string | null
  metrics?: ExamMetrics
  policy_sha256?: string
}

export interface Dance {
  id: string
  name: string
  created_at: number
  updated_at: number
  status: "draft" | "sim-verified" | "show-ready" | string
  duration_s?: number | null
  motion_csv?: string | null
  policy_path?: string | null
  preview?: string | null
  vet?: VetReport | null
  sim_exam?: SimExam | null
  source_job?: string | null
  notes?: string
  policy_sha256?: string | null
  incident?: { at?: number; detail?: string; second?: number; [key: string]: unknown } | null
  audio?: { track?: string; source?: string; align?: { audio_delay_s?: number; performance_s?: number }; attached_at?: number; [key: string]: unknown } | null
  repeatability?: { consecutive_clean?: number; total_runs?: number; last_run_at?: number | null; history?: Array<{ passed?: boolean; at?: number; metrics?: ExamMetrics; policy_sha256?: string }> }
  repeatability_target?: number
}

export interface Show {
  id: string
  dance_id: string
  dance_name: string
  operator: string
  created_at: number
  steps?: Record<string, { at?: number; confirmed?: boolean; value?: number }>
  deploy?: { requested_at?: number; note?: string } | null
  outcome?: { result: "clean" | "aborted" | "incident"; notes?: string; at?: number } | null
  closed: boolean
  mode: "rehearsal" | "live"
  setlist_id?: string | null
  next_step?: string | null
  checklist_complete?: boolean
  checklist_spec?: Array<{ key: string; kind: "confirm" | "number"; title: string; detail: string }>
}

export interface RunStatus {
  running: boolean
  show_id?: string | null
  dance_id?: string | null
  mode?: string | null
  phase: string
  fall_detected?: boolean
  last_lines?: string[]
  started_at?: number | null
}

export interface SystemStatus {
  checked_at?: number
  reachable?: boolean
  stale?: boolean
  gpu?: { utilization_pct?: number; memory_used_mb?: number; memory_total_mb?: number; temperature_c?: number; name?: string; [key: string]: unknown } | null
  jobs?: Array<{ name: string; iteration?: number; max_iteration?: number; mean_reward?: number; mean_episode_length?: number; wandb_url?: string }>
  cost?: { hours?: number; rate_vnd_per_hour?: number; accrued_vnd?: number; accrued_usd?: number; cap_vnd?: number; cap_fraction?: number; over_cap?: boolean }
  detail?: string
}

export interface Venue {
  id: string
  name: string
  radius_m: number
  margin_m: number
  max_excursion_m: number
  notes?: string
}

export interface SetListItem {
  dance_id: string
  dance_name?: string
  gap_after_s?: number
  note?: string
  status?: string
  show_ready?: boolean
  blockers?: string[]
}

export interface SetList {
  id: string
  name: string
  notes?: string
  created_at?: number
  updated_at?: number
  items: SetListItem[]
  show_ready?: boolean
  duration_s?: number
  blockers?: string[]
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, options)
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`
    try {
      const data = await response.json()
      message = data.detail || data.message || message
    } catch { /* response is not JSON */ }
    throw new ApiError(response.status, message)
  }
  return response.json() as Promise<T>
}

export const api = {
  get: <T>(path: string) => request<T>(path),
  send: <T>(path: string, method: "POST" | "DELETE" | "PATCH" | "PUT", body?: unknown) => request<T>(path, {
    method,
    headers: body === undefined ? undefined : { "content-type": "application/json" },
    body: body === undefined ? undefined : JSON.stringify(body),
  }),
  upload: <T>(path: string, form: FormData) => request<T>(path, { method: "POST", body: form }),
}
