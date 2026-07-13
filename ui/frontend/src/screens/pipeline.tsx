import { useMemo, useRef, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, Check, ChevronRight, CircleDashed, Clock3, FileVideo2, Gauge, LoaderCircle, Play, RotateCcw, ScrollText, UploadCloud, X } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { ScrollArea } from "@/components/ui/scroll-area"
import { EmptyState, InlineAlert, PageHeader, StatusBadge } from "@/components/console-ui"
import { RobotPreview } from "@/components/robot-preview"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type PipelineJob, type StageState, type VideoQuality } from "@/lib/api"
import { cn, fmtDate } from "@/lib/utils"

const DIM_LABELS: Record<string, string> = {
  framerate: "Framerate", resolution: "Resolution", lighting: "Lighting",
  sharpness_snappy: "Sharpness / snappy", movement_feasibility: "Movement feasibility",
}
const scoreColor = (s?: number) => s == null ? "text-muted-foreground" : s >= 7 ? "text-emerald-400" : s >= 5 ? "text-amber-400" : "text-red-400"

function QualityGate({ q }: { q: VideoQuality }) {
  if (q.verdict === "unreadable" || q.verdict === "error") {
    return <Card className="border-red-500/40"><CardContent className="flex items-center gap-2 pt-5 text-sm font-semibold text-red-300"><AlertTriangle className="h-4 w-4" /> Video quality check: {q.recommendation}</CardContent></Card>
  }
  const tone = q.verdict === "good" ? "success" : q.verdict === "acceptable" ? "warning" : "destructive"
  return <Card>
    <CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Gauge /> Video quality check</div><CardTitle className="mt-2">Upload rubric</CardTitle></div><div className="flex items-center gap-3"><Badge variant={tone as "success" | "warning" | "destructive"}>{q.verdict}</Badge><div className="text-right"><div className={cn("font-mono text-xl font-bold", scoreColor(q.overall_score))}>{q.overall_score}/10</div><div className="text-[9px] uppercase tracking-wide text-muted-foreground">overall</div></div></div></CardHeader>
    <CardContent>
      <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
        {Object.entries(q.dimensions ?? {}).map(([k, d]) => <div key={k} className="rounded-lg border border-border bg-background/25 p-3"><div className="flex items-center justify-between"><span className="text-[11px] font-semibold">{DIM_LABELS[k] ?? k}</span><span className={cn("font-mono text-sm font-bold", scoreColor(d.score))}>{d.score}/10</span></div><Progress value={(d.score ?? 0) * 10} className="mt-1.5 h-1" /><div className="mt-1.5 font-mono text-[9px] text-muted-foreground">{d.value}</div><div className="mt-1 text-[10px] leading-4 text-muted-foreground">{d.note}{d.flag ? <span className="text-amber-400"> · {d.flag}</span> : null}</div></div>)}
        <div className="rounded-lg border border-blue-500/25 bg-blue-500/[.06] p-3"><div className="flex items-center justify-between"><span className="text-[11px] font-semibold">Dance difficulty</span><span className="font-mono text-sm font-bold text-blue-300">{q.difficulty?.score}/10</span></div><Progress value={(q.difficulty?.score ?? 0) * 10} className="mt-1.5 h-1" indicatorClassName="bg-blue-500" /><div className="mt-1.5 text-[10px] font-semibold uppercase tracking-wide text-blue-300">{q.difficulty?.value}</div><div className="mt-1 text-[10px] leading-4 text-muted-foreground">{q.difficulty?.note}</div></div>
      </div>
      {!!q.blockers?.length && <InlineAlert className="mt-3" tone="danger" title="Blockers — a better clip will train much better" body={q.blockers.join(" · ")} />}
      {!!q.flags?.length && <div className="mt-2 text-[10px] text-amber-400">⚠ {q.flags.join(" · ")}</div>}
      <div className="mt-3 rounded-lg border border-border bg-background/25 p-3 text-xs text-muted-foreground">{q.recommendation}</div>
    </CardContent>
  </Card>
}

const STAGES = [
  { key: "extract", label: "Extract", note: "Video → body pose" },
  { key: "retarget", label: "Retarget", note: "Pose → G1 motion" },
  { key: "train", label: "Train", note: "Robust policy" },
  { key: "verify", label: "Verify", note: "Latency + held-out" },
  { key: "export", label: "Export", note: "Show bundle" },
]

function StageIcon({ state }: { state?: StageState }) {
  if (state === "done" || state === "skipped") return <Check className="h-4 w-4" />
  if (state === "running") return <LoaderCircle className="h-4 w-4 animate-spin" />
  if (state === "failed") return <X className="h-4 w-4" />
  if (state === "blocked") return <Clock3 className="h-4 w-4" />
  return <CircleDashed className="h-4 w-4" />
}

function JobProgress({ job, compact = false }: { job: PipelineJob; compact?: boolean }) {
  const values = Object.values(job.stages)
  const complete = values.filter((stage) => ["done", "skipped"].includes(stage.state)).length
  const currentProgress = values.find((stage) => stage.state === "running")?.progress ?? 0
  const progress = (complete + currentProgress) / Math.max(1, values.length) * 100
  return <div className={compact ? "mt-2" : ""}><Progress value={progress} className="h-1.5" /><div className="mt-1.5 flex justify-between text-[9px] uppercase tracking-wide text-muted-foreground"><span>{job.current_stage ?? "Complete"}</span><span>{Math.round(progress)}%</span></div></div>
}

export function PipelineScreen({ data }: { data: ConsoleData }) {
  const queryClient = useQueryClient()
  const picker = useRef<HTMLInputElement>(null)
  const [dragging, setDragging] = useState(false)
  const [selectedId, setSelectedId] = useState<string | null>(data.jobs[0]?.id ?? null)
  const selectedSummary = data.jobs.find((job) => job.id === selectedId)
  const detail = useQuery({
    queryKey: ["job", selectedId],
    queryFn: () => api.get<PipelineJob>(`/api/jobs/${selectedId}`),
    enabled: !!selectedId,
    refetchInterval: selectedSummary?.current_stage ? 3_000 : false,
  })
  const selected = detail.data ?? selectedSummary

  const upload = useMutation({
    mutationFn: async (file: File) => {
      const form = new FormData()
      form.append("video", file)
      return api.upload<PipelineJob>("/api/jobs/upload", form)
    },
    onSuccess: (job) => {
      toast.success("Pipeline job created")
      setSelectedId(job.id)
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  const action = useMutation({
    mutationFn: ({ jobId, action }: { jobId: string; action: "retry" | "approve-train" }) => api.send<PipelineJob>(`/api/jobs/${jobId}/${action}`, "POST"),
    onSuccess: (_job, variables) => {
      toast.success(variables.action === "retry" ? "Stage re-queued" : "Training approved")
      queryClient.invalidateQueries({ queryKey: ["jobs"] })
      queryClient.invalidateQueries({ queryKey: ["job", variables.jobId] })
    },
    onError: (error: Error) => toast.error(error.message),
  })

  const trainNeedsApproval = selected?.stages.train && selected.stages.retarget?.state === "done" && selected.stages.train.state !== "done" && !(selected.stages.train.meta?.approved)
  const failedStage = selected && Object.entries(selected.stages).find(([, stage]) => stage.state === "failed")
  const retryable = selected?.current_stage && ["failed", "blocked"].includes(selected.stages[selected.current_stage]?.state)

  const sortedJobs = useMemo(() => [...data.jobs].sort((a, b) => b.created_at - a.created_at), [data.jobs])

  const acceptFile = (file?: File) => {
    if (!file) return
    if (!(file.type.startsWith("video/") || file.name.toLowerCase().endsWith(".csv"))) return toast.error("Choose a video or motion CSV")
    upload.mutate(file)
  }

  return (
    <div>
      <PageHeader eyebrow="Create" title="Pipeline studio" description="Turn a reference video into a verified G1 policy. Every expensive or safety-sensitive transition remains explicit." actions={<Button onClick={() => picker.current?.click()} disabled={upload.isPending}><UploadCloud /> Upload motion</Button>} />
      <input ref={picker} type="file" accept="video/*,.csv" className="hidden" onChange={(event) => acceptFile(event.target.files?.[0])} />

      <Card className={cn("mb-4 border-dashed transition-colors", dragging && "border-blue-500/60 bg-blue-500/[.06]")} onDragOver={(event) => { event.preventDefault(); setDragging(true) }} onDragLeave={() => setDragging(false)} onDrop={(event) => { event.preventDefault(); setDragging(false); acceptFile(event.dataTransfer.files[0]) }} data-testid="upload-dropzone">
        <CardContent className="flex flex-col items-center justify-center py-8 text-center sm:flex-row sm:text-left">
          <div className="mb-3 rounded-xl border border-blue-500/20 bg-blue-500/10 p-3 text-blue-300 sm:mb-0 sm:mr-4"><FileVideo2 className="h-6 w-6" /></div>
          <div><div className="text-sm font-semibold">Drop a dance video or motion CSV</div><div className="mt-1 text-xs text-muted-foreground">15 seconds–4 minutes. Upload streams directly into the job store with disk guardrails.</div></div>
          <Button className="mt-4 sm:ml-auto sm:mt-0" variant="outline" onClick={() => picker.current?.click()} disabled={upload.isPending}>{upload.isPending ? <LoaderCircle className="animate-spin" /> : <UploadCloud />} {upload.isPending ? "Uploading…" : "Choose file"}</Button>
        </CardContent>
      </Card>

      <div className="grid gap-4 xl:grid-cols-[310px_minmax(0,1fr)]">
        <Card>
          <CardHeader className="border-b border-border/70 pb-4"><div className="flex items-center justify-between"><CardTitle>Jobs</CardTitle><Badge variant="secondary">{sortedJobs.length}</Badge></div></CardHeader>
          <ScrollArea className="h-[640px]">
            <div className="space-y-1 p-2">
              {sortedJobs.map((job) => <button key={job.id} onClick={() => setSelectedId(job.id)} className={cn("w-full rounded-lg border border-transparent p-3 text-left transition-colors hover:bg-accent/60", selectedId === job.id && "border-blue-500/25 bg-blue-500/[.08]")}><div className="flex items-start gap-2.5"><div className="mt-0.5 rounded-md bg-muted p-1.5"><FileVideo2 className="h-3.5 w-3.5" /></div><div className="min-w-0 flex-1"><div className="truncate text-xs font-semibold">{job.name}</div><div className="mt-1 text-[10px] text-muted-foreground">{fmtDate(job.created_at)} • {job.input?.type ?? "input"}</div><JobProgress job={job} compact /></div><ChevronRight className="mt-1 h-3.5 w-3.5 text-muted-foreground" /></div></button>)}
              {!sortedJobs.length && <div className="p-3"><EmptyState icon={FileVideo2} title="No pipeline jobs" body="Upload a reference video to start the five-stage workflow." /></div>}
            </div>
          </ScrollArea>
        </Card>

        {selected ? <div className="space-y-4">
          <Card>
            <CardHeader className="flex-row items-start justify-between gap-4 space-y-0 border-b border-border/70"><div><div className="panel-kicker"><Gauge /> Pipeline progress</div><CardTitle className="mt-2 text-lg">{selected.name}</CardTitle><p className="mt-1 font-mono text-[10px] text-muted-foreground">{selected.id}</p></div><StatusBadge status={selected.current_stage ? selected.stages[selected.current_stage]?.state : "done"} /></CardHeader>
            <CardContent className="pt-5">
              <RobotPreview className="mb-5" url={selected.preview_url} title={`${selected.name} robot preview`} duration={selected.vet?.seconds} />
              <div className="grid gap-2 md:grid-cols-5">
                {STAGES.map((stage, index) => {
                  const record = selected.stages[stage.key]
                  const state = record?.state ?? "pending"
                  return <div key={stage.key} className={cn("relative rounded-lg border p-3", state === "running" && "border-blue-500/40 bg-blue-500/[.07]", state === "failed" && "border-red-500/35 bg-red-500/[.06]", state === "done" && "border-emerald-500/20 bg-emerald-500/[.04]")}><div className={cn("mb-3 flex h-8 w-8 items-center justify-center rounded-full border", state === "running" ? "border-blue-500/40 bg-blue-500/15 text-blue-300" : state === "failed" ? "border-red-500/40 bg-red-500/15 text-red-300" : state === "done" ? "border-emerald-500/30 bg-emerald-500/10 text-emerald-300" : "border-border bg-muted text-muted-foreground")}><StageIcon state={state} /></div><div className="text-xs font-semibold">{index + 1}. {stage.label}</div><div className="mt-1 text-[10px] text-muted-foreground">{stage.note}</div><div className="mt-3"><StatusBadge status={state} /></div>{state === "running" && <Progress value={(record.progress ?? 0) * 100} className="mt-3 h-1" />}</div>
                })}
              </div>
              {failedStage && <InlineAlert className="mt-4" tone="danger" title={`${failedStage[0]} failed`} body={failedStage[1].message || "Open the stage log for the failure trace."} />}
              {selected.current_stage && selected.stages[selected.current_stage]?.message && !failedStage && <InlineAlert className="mt-4" tone={selected.stages[selected.current_stage].state === "blocked" ? "warning" : "info"} title={`${selected.current_stage}: ${selected.stages[selected.current_stage].state}`} body={selected.stages[selected.current_stage].message} />}
              <div className="mt-4 flex flex-wrap gap-2">
                {trainNeedsApproval && <Button onClick={() => action.mutate({ jobId: selected.id, action: "approve-train" })}><Play /> Approve training</Button>}
                {retryable && <Button variant="outline" onClick={() => action.mutate({ jobId: selected.id, action: "retry" })}><RotateCcw /> Retry stage</Button>}
                {selected.preview_url && <Button variant="outline" asChild><a href={selected.preview_url} target="_blank" rel="noreferrer"><FileVideo2 /> Open preview file</a></Button>}
              </div>
            </CardContent>
          </Card>

          {selected.quality && <QualityGate q={selected.quality} />}

          <div className="grid gap-4 lg:grid-cols-2">
            <Card>
              <CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><ScrollText /> Stage log</div><CardTitle className="mt-2">Latest output</CardTitle></div><Badge variant="secondary">tail 40</Badge></CardHeader>
              <CardContent><ScrollArea className="h-64 rounded-lg border border-border bg-black/35"><pre className="whitespace-pre-wrap p-4 font-mono text-[10px] leading-5 text-slate-400">{selected.log_tail?.join("\n") || "No log output yet."}</pre></ScrollArea></CardContent>
            </Card>
            <Card>
              <CardHeader><div className="panel-kicker"><AlertTriangle /> Motion gate</div><CardTitle className="mt-2">Vetting report</CardTitle></CardHeader>
              <CardContent>
                {selected.vet ? <div className="space-y-2">{Object.entries(selected.vet.hard ?? {}).map(([name, value]) => <div key={name} className="flex items-center justify-between rounded-lg border border-border px-3 py-2.5"><div><div className="text-xs font-semibold capitalize">{name.replaceAll("_", " ")}</div><div className="mt-0.5 text-[10px] text-muted-foreground">hard safety check</div></div><StatusBadge status={value.pass === false ? "failed" : "pass"} /></div>)}<div className="pt-2 text-[10px] text-muted-foreground">{selected.vet.frames?.toLocaleString() ?? "—"} frames • {selected.vet.seconds?.toFixed(1) ?? "—"} seconds</div></div> : <EmptyState title="Awaiting retarget" body="The excursion, joint-limit, grounding, and floorwork gate appears after retargeting." />}
              </CardContent>
            </Card>
          </div>
        </div> : <Card><CardContent className="p-6"><EmptyState title="Select a job" body="Choose a pipeline job to inspect its stages, logs, preview, and failure reason." /></CardContent></Card>}
      </div>
    </div>
  )
}
