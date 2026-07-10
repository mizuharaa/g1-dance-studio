import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertTriangle, ArchiveRestore, AudioLines, BarChart3, CheckCircle2, FileKey2, Gauge, History, Library, Music2, Plus, ShieldCheck, Trash2 } from "lucide-react"
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Progress } from "@/components/ui/progress"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { EmptyState, InlineAlert, Metric, PageHeader, StatusBadge } from "@/components/console-ui"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type Dance } from "@/lib/api"
import { cn, fmtDate, fmtDuration, fmtPercent, shortHash } from "@/lib/utils"

interface PolicyVersion { version_id?: string; id?: string; created_at?: number; policy_sha256?: string; notes?: string; [key: string]: unknown }

function ManageDanceDialog({ dance, open, onOpenChange }: { dance: Dance; open: boolean; onOpenChange: (open: boolean) => void }) {
  const queryClient = useQueryClient()
  const [policyPath, setPolicyPath] = useState(dance.policy_path ?? "")
  const [audioPath, setAudioPath] = useState("")
  const [bpm, setBpm] = useState("118")
  useEffect(() => setPolicyPath(dance.policy_path ?? ""), [dance.policy_path])
  const mutate = useMutation({
    mutationFn: ({ path, method = "POST", body }: { path: string; method?: "POST" | "DELETE"; body?: unknown }) => api.send(path, method, body),
    onSuccess: () => {
      toast.success("Dance updated")
      queryClient.invalidateQueries({ queryKey: ["dances"] })
      onOpenChange(false)
    },
    onError: (error: Error) => toast.error(error.message),
  })
  return <Dialog open={open} onOpenChange={onOpenChange}><DialogContent><DialogHeader><DialogTitle>Manage {dance.name}</DialogTitle><DialogDescription>Artifact changes reset or preserve safety status exactly as enforced by the backend.</DialogDescription></DialogHeader><div className="space-y-5">
    <div><label className="metric-label">Policy path</label><div className="mt-2 flex gap-2"><Input value={policyPath} onChange={(event) => setPolicyPath(event.target.value)} placeholder="data/policies/.../policy.onnx" /><Button variant="outline" onClick={() => mutate.mutate({ path: `/api/dances/${dance.id}/policy`, body: { policy_path: policyPath } })}><FileKey2 /> Attach</Button></div><p className="mt-2 text-[10px] leading-4 text-muted-foreground">Attaching a policy resets verification to draft. Re-run the signed sim exam before promotion.</p></div>
    <Separator />
    <div><label className="metric-label">Music source path</label><div className="mt-2 flex gap-2"><Input value={audioPath} onChange={(event) => setAudioPath(event.target.value)} placeholder="C:\\music\\track.mp3" /><Button variant="outline" onClick={() => mutate.mutate({ path: `/api/dances/${dance.id}/audio`, body: { source_path: audioPath } })}><Music2 /> Attach</Button></div><div className="mt-2 flex items-center gap-2"><Input className="w-28" type="number" value={bpm} onChange={(event) => setBpm(event.target.value)} /><Button variant="ghost" onClick={() => mutate.mutate({ path: `/api/dances/${dance.id}/audio`, body: { bpm: Number(bpm) } })}>Generate click track</Button>{dance.audio && <Button variant="ghost" className="ml-auto text-red-300" onClick={() => mutate.mutate({ path: `/api/dances/${dance.id}/audio`, method: "DELETE" })}><Trash2 /> Remove</Button>}</div></div>
  </div><DialogFooter><Button variant="outline" onClick={() => onOpenChange(false)}>Close</Button></DialogFooter></DialogContent></Dialog>
}

function VersionsPanel({ dance }: { dance: Dance }) {
  const queryClient = useQueryClient()
  const versions = useQuery({ queryKey: ["versions", dance.id], queryFn: () => api.get<{ versions: PolicyVersion[] }>(`/api/dances/${dance.id}/versions`) })
  const rollback = useMutation({
    mutationFn: (versionId: string) => api.send(`/api/dances/${dance.id}/rollback`, "POST", { version_id: versionId }),
    onSuccess: () => { toast.success("Files restored; status reset to draft"); queryClient.invalidateQueries({ queryKey: ["dances"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  if (!versions.data?.versions.length) return <EmptyState icon={ArchiveRestore} title="No stored policy versions" body="A version snapshot is created when a show-ready policy is promoted." />
  return <div className="space-y-2">{versions.data.versions.map((version, index) => { const id = version.version_id ?? version.id ?? String(index); return <div key={id} className="flex flex-col gap-3 rounded-lg border border-border bg-background/25 p-3 sm:flex-row sm:items-center"><div className="min-w-0 flex-1"><div className="font-mono text-xs font-semibold text-blue-300">{id.slice(0, 14)}</div><div className="mt-1 text-[10px] text-muted-foreground">{fmtDate(version.created_at)} • {shortHash(version.policy_sha256)}</div></div><Button variant="outline" size="sm" onClick={() => rollback.mutate(id)} disabled={rollback.isPending}><ArchiveRestore /> Restore</Button></div>})}</div>
}

function DanceDetail({ dance, data }: { dance: Dance; data: ConsoleData }) {
  const queryClient = useQueryClient()
  const [manage, setManage] = useState(false)
  const runs = data.shows.filter((show) => show.dance_id === dance.id)
  const nominal = dance.sim_exam?.metrics?.nominal
  const push = dance.sim_exam?.metrics?.push
  const chart = [
    { name: "Nominal", survival: (nominal?.success_rate ?? 0) * 100, mpkpe: (nominal?.mpkpe_m ?? 0) * 100 },
    { name: "Push", survival: (push?.success_rate ?? 0) * 100, mpkpe: (push?.mpkpe_m ?? 0) * 100 },
  ]
  const promote = useMutation({
    mutationFn: () => api.send(`/api/dances/${dance.id}/promote`, "POST", { status: "show-ready" }),
    onSuccess: () => { toast.success("Dance promoted to show-ready"); queryClient.invalidateQueries({ queryKey: ["dances"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  return <>
    <Card className="overflow-hidden">
      <div className="h-1 bg-gradient-to-r from-blue-600 via-blue-400 to-transparent" />
      <CardHeader className="flex-row items-start justify-between gap-3 space-y-0 border-b border-border/70"><div><div className="panel-kicker"><Library /> Dance record</div><CardTitle className="mt-2 text-xl">{dance.name}</CardTitle><div className="mt-2 flex flex-wrap items-center gap-2"><StatusBadge status={dance.status} /><Badge variant="secondary">{fmtDuration(dance.duration_s)}</Badge><span className="font-mono text-[10px] text-muted-foreground">{dance.id}</span></div></div><div className="flex flex-wrap justify-end gap-2"><Button variant="outline" onClick={() => setManage(true)}>Manage</Button>{dance.status === "sim-verified" && <Button onClick={() => promote.mutate()}><ShieldCheck /> Promote</Button>}</div></CardHeader>
      <CardContent className="pt-5">
        {dance.incident && <InlineAlert className="mb-4" tone="danger" title="Dance demoted after incident" body={String(dance.incident.detail ?? "Review the incident record before any new promotion.")} />}
        <Tabs defaultValue="stats">
          <TabsList><TabsTrigger value="stats">Stats</TabsTrigger><TabsTrigger value="runs">Run history</TabsTrigger><TabsTrigger value="versions">Versions</TabsTrigger><TabsTrigger value="contract">Contract</TabsTrigger></TabsList>
          <TabsContent value="stats">
            <div className="grid gap-4 lg:grid-cols-[1fr_1.1fr]">
              <div className="grid grid-cols-2 gap-3">
                <div className="rounded-lg border border-border bg-background/30 p-4"><Metric label="Held-out survival" value={fmtPercent(nominal?.success_rate, 1)} accent={(nominal?.success_rate ?? 0) >= .99 ? "green" : "amber"} detail="nominal" /></div>
                <div className="rounded-lg border border-border bg-background/30 p-4"><Metric label="Push survival" value={fmtPercent(push?.success_rate, 1)} accent={(push?.success_rate ?? 0) >= .99 ? "green" : "amber"} detail={`${push?.force_n ? Math.round(push.force_n) + " N equivalent" : "held-out push"}`} /></div>
                <div className="rounded-lg border border-border bg-background/30 p-4"><Metric label="Tracking error" value={nominal?.mpkpe_m != null ? `${(nominal.mpkpe_m * 100).toFixed(1)} cm` : "—"} detail="MPKPE nominal" /></div>
                <div className="rounded-lg border border-border bg-background/30 p-4"><Metric label="Repeatability" value={`${dance.repeatability?.consecutive_clean ?? 0}/${dance.repeatability_target ?? 3}`} accent={(dance.repeatability?.consecutive_clean ?? 0) >= (dance.repeatability_target ?? 3) ? "green" : "amber"} detail={`${dance.repeatability?.total_runs ?? 0} signed exams`} /></div>
              </div>
              <div className="h-60 rounded-lg border border-border bg-background/25 p-3"><ResponsiveContainer width="100%" height="100%"><BarChart data={chart} margin={{ top: 12, right: 8, left: -18, bottom: 0 }}><CartesianGrid stroke="#1e293b" vertical={false} /><XAxis dataKey="name" tick={{ fill: "#94a3b8", fontSize: 10 }} axisLine={false} tickLine={false} /><YAxis domain={[0, 100]} tick={{ fill: "#64748b", fontSize: 9 }} axisLine={false} tickLine={false} /><Tooltip contentStyle={{ background: "#0b101a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }} /><Bar dataKey="survival" name="Survival %" fill="#3b82f6" radius={[5, 5, 0, 0]} /></BarChart></ResponsiveContainer></div>
            </div>
            <div className="mt-4 grid gap-3 md:grid-cols-3">
              <div className="rounded-lg border border-dashed border-border p-3"><div className="metric-label">40 / 60 / 80 ms latency gate</div><div className="mt-2 text-sm font-semibold text-muted-foreground">Not exposed by API</div><div className="mt-1 text-[10px] leading-4 text-muted-foreground">Frontend will render these results once the dance API includes gap_check conditions.</div></div>
              <div className="rounded-lg border border-dashed border-border p-3"><div className="metric-label">Training iterations</div><div className="mt-2 text-sm font-semibold text-muted-foreground">Not exposed by API</div><div className="mt-1 text-[10px] leading-4 text-muted-foreground">The global training monitor is available under System.</div></div>
              <div className="rounded-lg border border-dashed border-border p-3"><div className="metric-label">Training cost</div><div className="mt-2 text-sm font-semibold text-muted-foreground">Not exposed by API</div><div className="mt-1 text-[10px] leading-4 text-muted-foreground">No per-dance cost attribution is currently returned.</div></div>
            </div>
          </TabsContent>
          <TabsContent value="runs">{runs.length ? <div className="space-y-2">{runs.map((show) => <div key={show.id} className="flex items-center gap-3 rounded-lg border border-border p-3"><div className={cn("rounded-md p-2", show.outcome?.result === "clean" ? "bg-emerald-500/10 text-emerald-300" : show.outcome?.result === "incident" ? "bg-red-500/10 text-red-300" : "bg-muted text-muted-foreground")}><History className="h-4 w-4" /></div><div className="min-w-0 flex-1"><div className="text-xs font-semibold">{show.mode} performance</div><div className="mt-1 truncate text-[10px] text-muted-foreground">{fmtDate(show.outcome?.at ?? show.created_at)} • {show.operator} {show.outcome?.notes ? `• ${show.outcome.notes}` : ""}</div></div><StatusBadge status={show.outcome?.result ?? "open"} /></div>)}</div> : <EmptyState icon={History} title="No hardware runs" body="Recorded show outcomes for this dance will appear here." />}</TabsContent>
          <TabsContent value="versions"><VersionsPanel dance={dance} /></TabsContent>
          <TabsContent value="contract"><div className="space-y-2 text-xs">{[["Policy", dance.policy_path], ["Policy SHA", dance.policy_sha256], ["Motion", dance.motion_csv], ["Audio", dance.audio?.track], ["Source job", dance.source_job]].map(([label, value]) => <div key={label as string} className="grid gap-1 rounded-lg border border-border p-3 sm:grid-cols-[130px_1fr]"><span className="font-semibold text-muted-foreground">{label}</span><span className="break-all font-mono text-[10px]">{value || "Not attached"}</span></div>)}{dance.notes && <div className="rounded-lg border border-border p-3 text-muted-foreground">{dance.notes}</div>}</div></TabsContent>
        </Tabs>
      </CardContent>
    </Card>
    <ManageDanceDialog dance={dance} open={manage} onOpenChange={setManage} />
  </>
}

export function DancesScreen({ data }: { data: ConsoleData }) {
  const [selectedId, setSelectedId] = useState<string | null>(data.dances[0]?.id ?? null)
  const [filter, setFilter] = useState("all")
  const selected = data.dances.find((dance) => dance.id === selectedId) ?? data.dances[0]
  const filtered = useMemo(() => data.dances.filter((dance) => filter === "all" || dance.status === filter), [data.dances, filter])
  return <div>
    <PageHeader eyebrow="Library" title="Dances & stats" description="Safety provenance, held-out performance, hardware outcomes, and policy versions in one operator-readable record." actions={<Button variant="outline" asChild><a href="/api/library/export"><ArchiveRestore /> Export library</a></Button>} />
    <div className="mb-4 flex flex-wrap gap-2">{["all", "show-ready", "sim-verified", "draft"].map((item) => <Button key={item} variant={filter === item ? "secondary" : "ghost"} size="sm" onClick={() => setFilter(item)}>{item === "all" ? "All dances" : item.replace("-", " ")}</Button>)}</div>
    <div className="grid gap-4 xl:grid-cols-[320px_minmax(0,1fr)]">
      <Card>
        <CardHeader className="border-b border-border/70 pb-4"><div className="flex items-center justify-between"><CardTitle>Dance library</CardTitle><Badge variant="secondary">{filtered.length}</Badge></div></CardHeader>
        <CardContent className="space-y-2 p-2">{filtered.map((dance) => <button key={dance.id} onClick={() => setSelectedId(dance.id)} className={cn("w-full rounded-lg border p-3 text-left transition-colors hover:bg-accent/50", selected?.id === dance.id ? "border-blue-500/30 bg-blue-500/[.07]" : "border-transparent")}><div className="flex items-start justify-between gap-2"><div className="min-w-0"><div className="truncate text-xs font-semibold">{dance.name}</div><div className="mt-1 text-[10px] text-muted-foreground">{fmtDuration(dance.duration_s)} • updated {fmtDate(dance.updated_at, false)}</div></div><StatusBadge status={dance.status} /></div><Progress className="mt-3 h-1" value={(dance.repeatability?.consecutive_clean ?? 0) / (dance.repeatability_target ?? 3) * 100} indicatorClassName={dance.status === "show-ready" ? "bg-emerald-500" : "bg-blue-500"} /></button>)}{!filtered.length && <EmptyState icon={Library} title="No dances in this view" body="Change the status filter or complete a pipeline job." />}</CardContent>
      </Card>
      {selected ? <DanceDetail dance={selected} data={data} /> : <Card><CardContent className="p-6"><EmptyState icon={Library} title="No dance records" body="The first verified pipeline export will appear here." /></CardContent></Card>}
    </div>
  </div>
}
