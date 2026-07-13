import { useState, type ReactNode } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { ArchiveRestore, Box, Cloud, CloudOff, Cpu, Database, Download, Gauge, HardDrive, Network, RefreshCw, Settings2, Upload, Wifi } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { EmptyState, InlineAlert, Metric, PageHeader, StatusBadge } from "@/components/console-ui"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api } from "@/lib/api"
import { fmtDate, fmtHMS, fmtMoney } from "@/lib/utils"

interface CloudInfo { config?: Record<string, string | number | null>; last_test?: { ok?: boolean; detail?: string; [key: string]: unknown } | null }
interface BodyModels { ready?: boolean; smpl?: boolean; smplx?: boolean; detail?: string; [key: string]: unknown }

function Stat({ label, value, accent }: { label: string; value: ReactNode; accent?: boolean }) {
  return <div className={`rounded-md border px-2.5 py-1.5 ${accent ? "border-blue-500/30 bg-blue-500/[.07]" : "border-border bg-background/40"}`}><div className="text-[9px] font-semibold uppercase tracking-wide text-muted-foreground">{label}</div><div className={`mt-0.5 font-mono text-xs font-bold ${accent ? "text-blue-300" : "text-foreground"}`}>{value}</div></div>
}

function SystemOverview({ data }: { data: ConsoleData }) {
  const gpu = data.system?.gpu
  const cost = data.system?.cost
  const jobs = data.system?.jobs ?? []
  const cap = (cost?.cap_fraction ?? 0) * 100
  return <div className="space-y-4">
    {!data.system?.reachable && <InlineAlert tone="warning" title="Cloud GPU is unreachable" body={data.system?.detail ?? "Training may still be running; this panel is showing the last known state."} />}
    <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-4"><Card><CardContent className="pt-5"><Metric label="GPU utilization" value={gpu?.utilization_pct != null ? `${Math.round(gpu.utilization_pct)}%` : "Offline"} detail={gpu?.name ?? "No live device"} accent={gpu ? "blue" : undefined} /></CardContent></Card><Card><CardContent className="pt-5"><Metric label="VRAM" value={gpu?.memory_used_mb != null ? `${(gpu.memory_used_mb / 1024).toFixed(1)} GB` : "—"} detail={gpu?.memory_total_mb ? `of ${(gpu.memory_total_mb / 1024).toFixed(0)} GB` : "not reported"} /></CardContent></Card><Card><CardContent className="pt-5"><Metric label="Cloud spend" value={fmtMoney(cost?.accrued_vnd)} detail={`${cost?.hours?.toFixed(1) ?? "—"} instance hours`} accent={cost?.over_cap ? "red" : undefined} /></CardContent></Card><Card><CardContent className="pt-5"><Metric label="Training jobs" value={jobs.length} detail={jobs.length ? "active on box" : "no active jobs"} accent={jobs.length ? "green" : undefined} /></CardContent></Card></div>
    <div className="grid gap-4 xl:grid-cols-[1fr_360px]"><Card><CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Cpu /> Training monitor</div><CardTitle className="mt-2">Active workloads</CardTitle></div><Badge variant={jobs.length ? "success" : "secondary"}>{jobs.length ? "live" : "idle"}</Badge></CardHeader><CardContent>{jobs.length ? <div className="space-y-3">{jobs.map((job) => { const pct = job.iteration && job.max_iteration ? job.iteration / job.max_iteration * 100 : 0; const rate = job.iteration_time_s ? 1 / job.iteration_time_s : null; return <div key={job.name} className="rounded-lg border border-border bg-background/25 p-4"><div className="flex items-center justify-between gap-2"><div className="text-xs font-semibold">{job.name}</div><StatusBadge status="running" /></div><Progress className="mt-3" value={pct} /><div className="mt-1.5 flex justify-between font-mono text-[10px] text-muted-foreground"><span>{job.iteration?.toLocaleString() ?? "—"} / {job.max_iteration?.toLocaleString() ?? "—"} iters</span><span>{pct.toFixed(1)}%</span></div><div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-3"><Stat label="ETA (stage)" value={fmtHMS(job.eta_s)} accent /><Stat label="Elapsed (stage)" value={fmtHMS(job.elapsed_s)} /><Stat label="Speed" value={rate ? `${rate.toFixed(2)} it/s` : "—"} /><Stat label="Reward" value={job.mean_reward?.toFixed(2) ?? "—"} /><Stat label="Ep length" value={job.mean_episode_length?.toFixed(0) ?? "—"} />{job.wandb_url ? <a className="flex flex-col justify-center rounded-md border border-blue-500/25 bg-blue-500/[.06] px-2.5 py-1.5 text-[10px] font-semibold text-blue-300 hover:bg-blue-500/[.12]" href={job.wandb_url} target="_blank" rel="noreferrer">W&B ↗</a> : <Stat label="W&B" value="—" />}</div></div>})}</div> : <EmptyState icon={Cpu} title="No active training" body="The current box snapshot reports no running trainer." />}</CardContent></Card><Card><CardHeader><div className="panel-kicker"><Gauge /> Budget</div><CardTitle className="mt-2">Spend guardrail</CardTitle></CardHeader><CardContent><div className="flex items-end justify-between"><span className="text-3xl font-semibold">{fmtMoney(cost?.accrued_vnd)}</span><span className="text-xs text-muted-foreground">cap {fmtMoney(cost?.cap_vnd)}</span></div><Progress className="mt-4 h-3" value={Math.min(100, cap)} indicatorClassName={cost?.over_cap ? "bg-red-500" : "bg-blue-500"} /><div className="mt-2 flex justify-between text-[10px] text-muted-foreground"><span>{cap.toFixed(1)}% consumed</span><span>{cost?.over_cap ? "OVER CAP" : "within cap"}</span></div><div className="mt-5 rounded-lg border border-amber-500/20 bg-amber-500/[.06] p-3 text-[10px] leading-5 text-amber-100/70">GreenNode billing ends only when the instance is deleted—not when it is stopped.</div></CardContent></Card></div>
  </div>
}

function SettingsPanel({ data }: { data: ConsoleData }) {
  const queryClient = useQueryClient()
  const cloud = useQuery({ queryKey: ["cloud"], queryFn: () => api.get<CloudInfo>("/api/cloud") })
  const body = useQuery({ queryKey: ["bodymodels"], queryFn: () => api.get<BodyModels>("/api/bodymodels") })
  const [config, setConfig] = useState<Record<string, string>>({})
  const [archivePath, setArchivePath] = useState("")
  const cloudMutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body?: unknown }) => api.send(path, "POST", body),
    onSuccess: () => { toast.success("Cloud configuration updated"); queryClient.invalidateQueries({ queryKey: ["cloud"] }); queryClient.invalidateQueries({ queryKey: ["system"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  const modelInstall = useMutation({ mutationFn: () => api.send("/api/bodymodels/install", "POST"), onSuccess: () => { toast.success("Body model install complete"); queryClient.invalidateQueries({ queryKey: ["bodymodels"] }) }, onError: (error: Error) => toast.error(error.message) })
  const libraryImport = useMutation({ mutationFn: () => api.send("/api/library/import", "POST", { archive_path: archivePath }), onSuccess: () => { toast.success("Library imported"); queryClient.invalidateQueries({ queryKey: ["dances"] }) }, onError: (error: Error) => toast.error(error.message) })
  const existing = cloud.data?.config ?? {}
  return <div className="grid gap-4 xl:grid-cols-2"><Card><CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Network /> Cloud transport</div><CardTitle className="mt-2">GreenNode connection</CardTitle></div><StatusBadge status={cloud.data?.last_test?.ok ? "ready" : data.system?.reachable ? "ready" : "offline"} /></CardHeader><CardContent className="space-y-3"><div className="grid gap-3 sm:grid-cols-2">{["transport", "host", "port", "user", "key_path", "jupyter_url", "jupyter_token"].map((key) => <div key={key} className={key === "jupyter_url" || key === "jupyter_token" ? "sm:col-span-2" : ""}><label className="metric-label">{key.replaceAll("_", " ")}</label><Input className="mt-2" value={config[key] ?? ""} onChange={(event) => setConfig((current) => ({ ...current, [key]: event.target.value }))} placeholder={String(existing[key] ?? "not configured")} /></div>)}</div><div className="flex gap-2"><Button onClick={() => cloudMutation.mutate({ path: "/api/cloud/config", body: { ...config, ...(config.port ? { port: Number(config.port) } : {}) } })}><Settings2 /> Save</Button><Button variant="outline" onClick={() => cloudMutation.mutate({ path: "/api/cloud/test" })}><RefreshCw /> Test connection</Button></div>{cloud.data?.last_test?.detail && <div className="rounded-lg border border-border p-3 text-xs text-muted-foreground">{cloud.data.last_test.detail}</div>}</CardContent></Card>
    <div className="space-y-4"><Card><CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Database /> Body models</div><CardTitle className="mt-2">SMPL assets</CardTitle></div><StatusBadge status={body.data?.ready ? "ready" : "blocked"} /></CardHeader><CardContent><div className="grid grid-cols-2 gap-3"><div className="rounded-lg border border-border p-3"><div className="metric-label">SMPL</div><div className="mt-2 text-sm font-semibold">{body.data?.smpl ? "Installed" : "Missing"}</div></div><div className="rounded-lg border border-border p-3"><div className="metric-label">SMPL-X</div><div className="mt-2 text-sm font-semibold">{body.data?.smplx ? "Installed" : "Missing"}</div></div></div><Button className="mt-3" variant="outline" onClick={() => modelInstall.mutate()}><HardDrive /> Detect & install</Button></CardContent></Card>
    <Card><CardHeader><div className="panel-kicker"><ArchiveRestore /> Library portability</div><CardTitle className="mt-2">Backup & restore</CardTitle></CardHeader><CardContent className="space-y-3"><Button variant="outline" className="w-full" asChild><a href="/api/library/export"><Download /> Export dance library</a></Button><div className="flex gap-2"><Input value={archivePath} onChange={(event) => setArchivePath(event.target.value)} placeholder="Path to .tar.gz archive" /><Button variant="outline" disabled={!archivePath} onClick={() => libraryImport.mutate()}><Upload /> Import</Button></div></CardContent></Card></div>
  </div>
}

export function SystemScreen({ data }: { data: ConsoleData }) {
  return <div><PageHeader eyebrow="Infrastructure" title="System" description="Read-only compute telemetry plus configuration for the local engine, cloud box, and portable dance library." actions={<Badge variant={data.system?.reachable ? "success" : "secondary"}>{data.system?.reachable ? <Cloud className="mr-1.5 h-3.5 w-3.5" /> : <CloudOff className="mr-1.5 h-3.5 w-3.5" />}{data.system?.reachable ? "connected" : "offline"}</Badge>} /><Tabs defaultValue="monitor"><TabsList><TabsTrigger value="monitor">Monitor</TabsTrigger><TabsTrigger value="settings">Configuration</TabsTrigger></TabsList><TabsContent value="monitor"><SystemOverview data={data} /></TabsContent><TabsContent value="settings"><SettingsPanel data={data} /></TabsContent></Tabs></div>
}
