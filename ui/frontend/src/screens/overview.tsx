import { useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { Activity, AlertOctagon, Bot, Check, Cloud, Cpu, Gauge, MapPin, Radio, ShieldCheck, Square, TimerReset, Zap } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { InlineAlert, Metric, PageHeader, StatusBadge } from "@/components/console-ui"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type Show } from "@/lib/api"
import { fmtDate, fmtDuration, fmtMoney, fmtPercent, shortHash } from "@/lib/utils"

const STATES = [
  { key: "preflight", label: "Preflight" },
  { key: "deploying", label: "Deploy" },
  { key: "dancing", label: "Dance" },
  { key: "stand", label: "Handback" },
  { key: "done", label: "Done" },
]

function normalizedPhase(phase: string, running: boolean) {
  if (["fall", "stopped"].includes(phase)) return "incident"
  if (["launching"].includes(phase)) return "preflight"
  if (["arming"].includes(phase)) return "deploying"
  if (["performing"].includes(phase)) return "dancing"
  if (["ramp-to-damping"].includes(phase)) return "stand"
  if (["ended"].includes(phase) || (!running && phase !== "idle")) return "done"
  return phase === "idle" ? "idle" : phase
}

function OutcomeButtons({ show }: { show: Show }) {
  const queryClient = useQueryClient()
  const outcome = useMutation({
    mutationFn: (result: "clean" | "aborted" | "incident") => api.send(`/api/shows/${show.id}/outcome`, "POST", { result }),
    onSuccess: (_data, result) => {
      toast.success(`Outcome recorded: ${result}`)
      queryClient.invalidateQueries({ queryKey: ["shows"] })
      queryClient.invalidateQueries({ queryKey: ["dances"] })
    },
    onError: (error: Error) => toast.error(error.message),
  })
  return (
    <div className="grid grid-cols-3 gap-2">
      <Button variant="outline" className="border-emerald-500/30 text-emerald-300 hover:bg-emerald-500/10" onClick={() => outcome.mutate("clean")}>Clean</Button>
      <Button variant="outline" className="border-amber-500/30 text-amber-300 hover:bg-amber-500/10" onClick={() => outcome.mutate("aborted")}>Aborted</Button>
      <Button variant="destructive" onClick={() => outcome.mutate("incident")}><AlertOctagon /> Incident</Button>
    </div>
  )
}

function LiveRunCard({ data, onPerform }: { data: ConsoleData; onPerform: () => void }) {
  const queryClient = useQueryClient()
  const [now, setNow] = useState(Date.now() / 1000)
  useEffect(() => {
    if (!data.run.running) return
    const timer = window.setInterval(() => setNow(Date.now() / 1000), 1_000)
    return () => window.clearInterval(timer)
  }, [data.run.running])

  const activeDance = data.dances.find((dance) => dance.id === data.run.dance_id)
  const activeShow = data.shows.find((show) => show.id === data.run.show_id) ?? data.shows.find((show) => !show.closed)
  const phase = normalizedPhase(data.run.phase, data.run.running)
  const phaseIndex = STATES.findIndex((item) => item.key === phase)
  const elapsed = data.run.started_at ? Math.max(0, now - data.run.started_at) : 0
  const duration = (activeDance?.duration_s ?? 0) + 8
  const progress = data.run.running && duration ? Math.min(98, elapsed / duration * 100) : phase === "done" ? 100 : 0
  const incident = phase === "incident" || data.run.fall_detected

  const stop = useMutation({
    mutationFn: () => api.send<{ stopped: boolean; detail: string }>("/api/shows/runs/current/stop", "POST"),
    onSuccess: (result) => {
      result.stopped ? toast.warning(result.detail) : toast.info(result.detail)
      queryClient.invalidateQueries({ queryKey: ["current-run"] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  return (
    <Card className={`relative overflow-hidden ${incident ? "border-red-500/35 shadow-danger" : data.run.running ? "border-blue-500/35 shadow-glow" : ""}`} data-testid="live-run-card">
      <div className={`absolute inset-x-0 top-0 h-px ${incident ? "bg-red-500" : data.run.running ? "bg-blue-500" : "bg-border"}`} />
      <CardHeader className="flex-row items-start justify-between gap-4 space-y-0 border-b border-border/70 pb-4">
        <div>
          <div className="panel-kicker"><Radio className={data.run.running ? "text-blue-400" : "text-muted-foreground"} /> Live run control</div>
          <CardTitle className="mt-3 text-xl">{incident ? "Incident response" : data.run.running ? activeDance?.name ?? "Performance in progress" : "Robot is standing by"}</CardTitle>
          <p className="mt-1 text-xs text-muted-foreground">{data.run.running ? `${data.run.mode ?? "live"} • ${fmtDuration(elapsed)} elapsed` : "No policy currently owns the robot."}</p>
        </div>
        <StatusBadge status={incident ? "incident" : data.run.running ? phase : "idle"} />
      </CardHeader>
      <CardContent className="space-y-5 pt-5">
        <div className="grid gap-3 sm:grid-cols-3">
          <div className="rounded-lg border border-border bg-background/35 p-3"><div className="metric-label">Show</div><div className="mt-1.5 truncate text-sm font-semibold">{activeDance?.name ?? "No active show"}</div></div>
          <div className="rounded-lg border border-border bg-background/35 p-3"><div className="metric-label">Policy</div><div className="mt-1.5 truncate font-mono text-xs text-blue-300">{shortHash(activeDance?.policy_sha256)}</div></div>
          <div className="rounded-lg border border-border bg-background/35 p-3"><div className="metric-label">Venue</div><div className="mt-1.5 flex items-center gap-1.5 truncate text-sm font-semibold"><MapPin className="h-3.5 w-3.5 text-blue-400" />{data.venues?.active.name ?? "Not selected"}</div></div>
        </div>

        <div>
          <div className="mb-2 flex items-center justify-between text-[11px] text-muted-foreground"><span>Performance timeline</span><span className="font-mono">{fmtDuration(elapsed)} / {activeDance ? fmtDuration(duration) : "—"}</span></div>
          <Progress value={progress} indicatorClassName={incident ? "bg-red-500" : "bg-blue-500"} className="h-2.5" />
          <div className="mt-3 grid grid-cols-5 gap-1">
            {STATES.map((state, index) => {
              const active = state.key === phase
              const complete = phaseIndex > index || phase === "done"
              return <div key={state.key} className="text-center"><div className={`mx-auto mb-1.5 h-2 w-2 rounded-full ${active ? incident ? "bg-red-400 ring-4 ring-red-500/15" : "bg-blue-400 ring-4 ring-blue-500/15" : complete ? "bg-emerald-400" : "bg-slate-700"}`} /><span className={`text-[9px] uppercase tracking-wide ${active ? "text-foreground" : "text-muted-foreground"}`}>{state.label}</span></div>
            })}
          </div>
        </div>

        {data.run.running ? (
          <div className="rounded-xl border border-red-500/30 bg-red-500/[.07] p-4">
            <div className="mb-3 flex items-center justify-between gap-3"><div><div className="text-sm font-bold text-red-200">REMOTE B-DAMP IS THE PRIMARY STOP</div><div className="mt-0.5 text-[11px] text-red-200/60">Software STOP sends SIGTERM, then the runtime damps the robot soft.</div></div><ShieldCheck className="h-6 w-6 text-red-400" /></div>
            <Button variant="destructive" size="lg" className="h-14 w-full text-lg font-black tracking-wide" onClick={() => stop.mutate()} disabled={stop.isPending} data-testid="stop-show"><Square className="fill-current" /> STOP SHOW — DAMP ROBOT</Button>
          </div>
        ) : activeShow && !activeShow.closed ? (
          <div className="space-y-3 rounded-xl border border-amber-500/25 bg-amber-500/[.06] p-4"><div><div className="text-sm font-semibold text-amber-200">Outcome required</div><div className="mt-1 text-xs text-amber-100/60">Close this run before another show. Incident demotes the dance immediately.</div></div><OutcomeButtons show={activeShow} /></div>
        ) : (
          <div className="flex flex-col items-start justify-between gap-3 rounded-lg border border-border bg-background/30 p-4 sm:flex-row sm:items-center"><div><div className="text-sm font-semibold">Ready for the next cue</div><div className="mt-1 text-xs text-muted-foreground">Open Perform to select a show-ready dance and run the safety checks.</div></div><Button onClick={onPerform}><Zap /> Open Perform</Button></div>
        )}

        {!!data.run.last_lines?.length && <ScrollArea className="h-24 rounded-lg border border-border bg-black/30 p-3"><pre className="whitespace-pre-wrap font-mono text-[10px] leading-5 text-slate-400">{data.run.last_lines.join("\n")}</pre></ScrollArea>}
      </CardContent>
    </Card>
  )
}

export function OverviewScreen({ data, onPerform }: { data: ConsoleData; onPerform: () => void }) {
  const ready = data.dances.filter((dance) => dance.status === "show-ready").length
  const cleanRuns = data.shows.filter((show) => show.outcome?.result === "clean").length
  const latestShows = useMemo(() => data.shows.slice(0, 5), [data.shows])
  const gpu = data.system?.gpu
  const cost = data.system?.cost
  const activeTraining = data.system?.jobs?.[0]

  return (
    <div>
      <PageHeader eyebrow="Mission control" title="Operator overview" description="Live performance ownership, safety state, and the few numbers that matter before the next cue." actions={<Badge variant={data.system?.reachable ? "success" : "secondary"} className="h-8 px-3"><Cloud className="mr-1.5 h-3.5 w-3.5" />GPU {data.system?.reachable ? "online" : "offline"}</Badge>} />
      {data.system?.cost?.over_cap && <InlineAlert className="mb-5" tone="warning" title="Cloud spend is over the recorded cap" body={`${fmtMoney(cost?.accrued_vnd)} accrued against ${fmtMoney(cost?.cap_vnd)}. Confirm the GPU instance was deleted if training is complete.`} />}

      <div className="grid gap-4 xl:grid-cols-[minmax(0,1.45fr)_minmax(330px,.75fr)]">
        <LiveRunCard data={data} onPerform={onPerform} />
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-1">
          <Card>
            <CardHeader className="pb-4"><div className="panel-kicker"><Cpu /> System</div><CardTitle className="mt-2">Compute & training</CardTitle></CardHeader>
            <CardContent className="space-y-5">
              <div className="grid grid-cols-2 gap-5"><Metric label="GPU utilization" value={gpu?.utilization_pct != null ? `${Math.round(gpu.utilization_pct)}%` : "Offline"} accent={gpu ? "blue" : undefined} detail={gpu?.name ?? data.system?.detail} /><Metric label="Cloud spend" value={fmtMoney(cost?.accrued_vnd)} accent={cost?.over_cap ? "red" : undefined} detail={`${cost?.hours?.toFixed(1) ?? "—"} box hours`} /></div>
              <div className="rounded-lg border border-border bg-background/35 p-3">
                <div className="mb-2 flex items-center justify-between"><span className="text-xs font-semibold">{activeTraining?.name ?? "No active training"}</span><StatusBadge status={activeTraining ? "running" : "idle"} /></div>
                <Progress value={activeTraining?.iteration && activeTraining.max_iteration ? activeTraining.iteration / activeTraining.max_iteration * 100 : 0} />
                <div className="mt-2 flex justify-between font-mono text-[10px] text-muted-foreground"><span>{activeTraining?.iteration?.toLocaleString() ?? 0} / {activeTraining?.max_iteration?.toLocaleString() ?? "—"}</span><span>reward {activeTraining?.mean_reward?.toFixed(2) ?? "—"}</span></div>
              </div>
            </CardContent>
          </Card>
          <Card>
            <CardHeader className="pb-4"><div className="panel-kicker"><Gauge /> Readiness</div><CardTitle className="mt-2">Show fleet</CardTitle></CardHeader>
            <CardContent><div className="grid grid-cols-2 gap-5"><Metric label="Show-ready" value={ready} accent="green" detail={`of ${data.dances.length} dances`} /><Metric label="Clean runs" value={cleanRuns} detail={`${data.shows.length} recorded`} /></div><div className="mt-5 space-y-2">{data.dances.slice(0, 3).map((dance) => <div key={dance.id} className="flex items-center justify-between rounded-md border border-border/70 px-3 py-2"><span className="truncate text-xs font-medium">{dance.name}</span><StatusBadge status={dance.status} /></div>)}</div></CardContent>
          </Card>
        </div>
      </div>

      <div className="mt-4 grid gap-4 lg:grid-cols-2">
        <Card>
          <CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Activity /> Recent operations</div><CardTitle className="mt-2">Run history</CardTitle></div><Badge variant="secondary">{data.shows.length} events</Badge></CardHeader>
          <CardContent className="space-y-2">
            {latestShows.map((show) => <div key={show.id} className="flex items-center gap-3 rounded-lg border border-border/70 bg-background/25 p-3"><div className={`rounded-md p-2 ${show.outcome?.result === "clean" ? "bg-emerald-500/10 text-emerald-300" : show.outcome?.result === "incident" ? "bg-red-500/10 text-red-300" : "bg-muted text-muted-foreground"}`}>{show.outcome?.result === "clean" ? <Check className="h-4 w-4" /> : <TimerReset className="h-4 w-4" />}</div><div className="min-w-0 flex-1"><div className="truncate text-xs font-semibold">{show.dance_name}</div><div className="mt-0.5 text-[10px] text-muted-foreground">{fmtDate(show.outcome?.at ?? show.created_at)} • {show.operator}</div></div><StatusBadge status={show.outcome?.result ?? "open"} /></div>)}
          </CardContent>
        </Card>
        <Card className="code-grid">
          <CardHeader><div className="panel-kicker"><Bot /> Safety posture</div><CardTitle className="mt-2">Authority chain</CardTitle></CardHeader>
          <CardContent>
            <div className="grid grid-cols-5 gap-1">{data.phases.map((phase, index) => <div key={phase.phase} className="relative rounded-lg border border-border bg-card/90 p-3 text-center"><div className="mx-auto mb-2 flex h-7 w-7 items-center justify-center rounded-full bg-blue-500/10 font-mono text-[10px] text-blue-300">0{index + 1}</div><div className="text-[10px] font-bold uppercase tracking-wide">{phase.phase.replace("_", " ")}</div><div className="mt-1 hidden text-[9px] text-muted-foreground sm:block">{phase.owner}</div></div>)}</div>
            <div className="mt-4 flex items-center gap-3 rounded-lg border border-blue-500/20 bg-blue-500/[.06] p-3"><ShieldCheck className="h-5 w-5 shrink-0 text-blue-400" /><div className="text-xs leading-5 text-blue-100/70">Policy ownership is explicit. The damping remote remains in the operator’s hand for every robot-moving phase.</div></div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
