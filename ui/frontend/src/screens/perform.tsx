import { useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertOctagon, ArrowDown, ArrowUp, Check, ChevronRight, CircleAlert, GripVertical, ListMusic, MapPin, Plus, Radio, ShieldAlert, ShieldCheck, Sparkles, Trash2, Users, Volume2, WifiOff, X } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { EmptyState, InlineAlert, PageHeader, StatusBadge } from "@/components/console-ui"
import { RobotPreview } from "@/components/robot-preview"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type Dance, type SetList, type SetListItem, type Show } from "@/lib/api"
import { cn, fmtDate, fmtDuration } from "@/lib/utils"
import { dancePreviewUrl } from "@/lib/preview"

const RUN_PHRASE = "I AM PRESENT WITH THE DAMPING REMOTE"

interface ChecklistItem { key?: string; title?: string; label?: string; detail?: string; ready?: boolean; pass?: boolean; blocker?: boolean; status?: string; [key: string]: unknown }
interface ChecklistReport { ready?: boolean; items?: ChecklistItem[]; blockers?: string[]; confirm_keys?: string[]; [key: string]: unknown }

function OutcomeCapture({ show }: { show: Show }) {
  const queryClient = useQueryClient()
  const [notes, setNotes] = useState("")
  const mutation = useMutation({
    mutationFn: (result: string) => api.send(`/api/shows/${show.id}/outcome`, "POST", { result, notes }),
    onSuccess: (_data, result) => { toast.success(`Outcome recorded: ${result}`); queryClient.invalidateQueries({ queryKey: ["shows"] }); queryClient.invalidateQueries({ queryKey: ["dances"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  return <Card className="border-amber-500/30 bg-amber-500/[.05]"><CardContent className="pt-5"><div className="flex flex-col gap-4 lg:flex-row lg:items-end"><div className="min-w-0 flex-1"><div className="panel-kicker text-amber-300"><CircleAlert /> Outcome required</div><div className="mt-2 text-sm font-semibold">{show.dance_name} • {show.mode}</div><div className="mt-1 text-[11px] text-amber-100/60">An open show blocks the next run. Incident immediately demotes the dance.</div><Input className="mt-3" value={notes} onChange={(event) => setNotes(event.target.value)} placeholder="Optional operator notes — include the second of any anomaly" /></div><div className="grid shrink-0 grid-cols-3 gap-2"><Button variant="outline" className="border-emerald-500/30 text-emerald-300" onClick={() => mutation.mutate("clean")}>Clean</Button><Button variant="outline" className="border-amber-500/30 text-amber-300" onClick={() => mutation.mutate("aborted")}>Aborted</Button><Button variant="destructive" onClick={() => mutation.mutate("incident")}><AlertOctagon /> Incident</Button></div></div></CardContent></Card>
}

function ChecklistPanel({ dance }: { dance: Dance }) {
  const [acks, setAcks] = useState<string[]>([])
  const report = useQuery({ queryKey: ["checklist", dance.id, acks], queryFn: () => api.send<ChecklistReport>(`/api/dances/${dance.id}/checklist`, "POST", { acks }) })
  const items = report.data?.items ?? []
  const confirmKeys = report.data?.confirm_keys ?? []
  return <div className="space-y-3"><div className="flex items-center justify-between"><div><div className="text-sm font-semibold">Pre-show evaluator</div><div className="mt-1 text-[10px] text-muted-foreground">Robot reachability and venue are live; operator acknowledgements are explicit.</div></div><StatusBadge status={report.data?.ready ? "ready" : "blocked"} /></div>
    {confirmKeys.length > 0 && <div className="grid gap-2 sm:grid-cols-2">{confirmKeys.map((key) => <button key={key} className={cn("flex items-center gap-2 rounded-lg border p-3 text-left text-xs", acks.includes(key) ? "border-emerald-500/30 bg-emerald-500/[.06]" : "border-border")} onClick={() => setAcks((current) => current.includes(key) ? current.filter((item) => item !== key) : [...current, key])}><span className={cn("flex h-4 w-4 items-center justify-center rounded border", acks.includes(key) && "border-emerald-500 bg-emerald-500 text-black")}>{acks.includes(key) && <Check className="h-3 w-3" />}</span>{key.replaceAll("_", " ")}</button>)}</div>}
    {items.length > 0 ? <div className="space-y-2">{items.map((item, index) => { const pass = item.ready ?? item.pass ?? item.status === "pass"; return <div key={item.key ?? index} className="flex items-start gap-3 rounded-lg border border-border p-3"><div className={cn("mt-0.5 rounded-full p-1", pass ? "bg-emerald-500/10 text-emerald-300" : "bg-amber-500/10 text-amber-300")}>{pass ? <Check className="h-3 w-3" /> : <CircleAlert className="h-3 w-3" />}</div><div className="min-w-0 flex-1"><div className="text-xs font-semibold">{item.title ?? item.label ?? item.key}</div>{item.detail && <div className="mt-1 text-[10px] leading-4 text-muted-foreground">{item.detail}</div>}</div></div> })}</div> : <div className="rounded-lg border border-dashed border-border p-3 text-xs text-muted-foreground">Checklist response contains no item details on this backend build. The run endpoint still enforces show-ready, audio, robot reachability, single-run lock, and typed confirmation.</div>}
  </div>
}

function InlineRunGate({ dance, blocked }: { dance: Dance; blocked: boolean }) {
  const queryClient = useQueryClient()
  const [operator, setOperator] = useState("")
  const [phrase, setPhrase] = useState("")
  const [mode, setMode] = useState<"rehearsal" | "live">("rehearsal")
  const [exitStand, setExitStand] = useState(true)
  const [free, setFree] = useState(false)
  const confirmed = phrase === RUN_PHRASE && operator.trim().length > 0 && !blocked
  const run = useMutation({
    mutationFn: () => api.send(`/api/shows/${dance.id}/run`, "POST", { confirmation: phrase, operator, mode, exit_stand: exitStand, free, audio_mode: "laptop" }),
    onSuccess: () => { toast.success("Show started — keep the damping remote in hand"); setPhrase(""); queryClient.invalidateQueries({ queryKey: ["current-run"] }); queryClient.invalidateQueries({ queryKey: ["shows"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  return <div className="rounded-xl border-2 border-red-300 bg-red-50 p-4 shadow-sm" data-testid="show-warning-gate">
    <div className="flex items-start gap-3"><div className="rounded-lg bg-red-600 p-2 text-white"><ShieldAlert className="h-5 w-5" /></div><div className="min-w-0"><div className="text-sm font-black text-red-900">Physical damping remote required</div><p className="mt-1 text-[11px] leading-5 text-red-800/75">This robot has no torque-cut hardware e-stop. Hold B-damp for the full performance. Software STOP makes the robot go soft.</p></div></div>
    <div className="mt-4 grid min-w-0 gap-3 sm:grid-cols-2"><div className="min-w-0"><label className="metric-label text-red-800">Operator name</label><Input className="mt-2 border-red-200 bg-white" value={operator} onChange={(event) => setOperator(event.target.value)} placeholder="Operator name" /></div><div className="min-w-0"><label className="metric-label text-red-800">Run mode</label><div className="mt-2 grid grid-cols-2 gap-2"><Button variant={mode === "rehearsal" ? "secondary" : "outline"} onClick={() => setMode("rehearsal")}>Rehearsal</Button><Button variant={mode === "live" ? "secondary" : "outline"} onClick={() => setMode("live")}>Live show</Button></div></div></div>
    <div className="mt-3 grid gap-2 sm:grid-cols-2"><button className={cn("rounded-lg border bg-white p-3 text-left text-xs font-semibold", exitStand ? "border-blue-400 text-blue-800" : "border-slate-200 text-slate-600")} onClick={() => setExitStand(!exitStand)}>{exitStand ? "✓ " : ""}Stand and hand back at finish</button><button className={cn("rounded-lg border bg-white p-3 text-left text-xs font-semibold", free ? "border-amber-400 text-amber-800" : "border-slate-200 text-slate-600")} onClick={() => setFree(!free)}>{free ? "✓ " : ""}Untethered/free configuration</button></div>
    <div className="mt-4 rounded-lg border border-red-200 bg-white p-3"><label className="text-[10px] font-black uppercase tracking-[.13em] text-red-800">Type this exact warning phrase to unlock</label><div className="mt-2 select-all rounded-md bg-red-950 px-3 py-2 text-center font-mono text-[10px] font-bold text-white sm:text-xs">{RUN_PHRASE}</div><Input data-testid="run-confirmation" className={cn("mt-2 border-red-200 bg-white font-mono", phrase && phrase !== RUN_PHRASE && "border-red-500 ring-2 ring-red-100")} value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder="Type phrase exactly" autoComplete="off" /></div>
    <Button data-testid="start-show" variant="destructive" size="lg" className="mt-3 h-12 w-full text-base font-black" disabled={!confirmed || run.isPending} onClick={() => run.mutate()}><Radio /> {run.isPending ? "Starting…" : blocked ? "Resolve blockers before show" : "RUN SHOW"}</Button>
  </div>
}

function PerformTab({ data }: { data: ConsoleData }) {
  const ready = data.dances.filter((dance) => dance.status === "show-ready")
  const [selectedId, setSelectedId] = useState<string | null>(ready[0]?.id ?? null)
  const selected = data.dances.find((dance) => dance.id === selectedId) ?? ready[0]
  const openShow = data.shows.find((show) => !show.closed)
  return <div className="space-y-4">
    {openShow && <OutcomeCapture show={openShow} />}
    <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,.85fr)]">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Sparkles /> Select act</div><CardTitle className="mt-2">Show-ready dances</CardTitle></div><Badge variant="success">{ready.length} ready</Badge></CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          {ready.map((dance) => <button key={dance.id} onClick={() => setSelectedId(dance.id)} className={cn("hover-lift rounded-xl border p-4 text-left", selected?.id === dance.id ? "border-blue-400 bg-blue-50 shadow-sm" : "border-slate-200 bg-white hover:bg-blue-50/40")}><div className="flex items-start justify-between gap-2"><div className="rounded-lg bg-blue-100 p-2 text-blue-700"><Volume2 className="h-4 w-4" /></div><StatusBadge status={dance.status} /></div><div className="mt-4 text-sm font-bold text-slate-900">{dance.name}</div><div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500"><span>{fmtDuration(dance.duration_s)}</span><span>•</span><span>{dance.audio ? "music attached" : "no music"}</span></div><div className="mt-3 text-[10px] font-semibold"><span className={dance.audio ? "text-emerald-700" : "text-red-600"}>{dance.audio ? "Audio ready" : "Blocked: attach audio"}</span></div></button>)}
          {!ready.length && <div className="sm:col-span-2"><EmptyState icon={ShieldCheck} title="No show-ready dances" body="A dance needs three clean signed exams, a pinned policy, and an explicit promotion." /></div>}
        </CardContent>
      </Card>
      <Card className="relative overflow-hidden"><CardHeader><div className="panel-kicker"><ShieldCheck /> Show controls</div><CardTitle className="mt-2">{selected?.name ?? "Select a dance"}</CardTitle></CardHeader><CardContent>{selected ? <div className="space-y-4"><RobotPreview url={dancePreviewUrl(selected)} title={`${selected.name} performance preview`} duration={selected.duration_s} /><div className="grid grid-cols-2 gap-3"><div className="hover-lift rounded-lg border border-slate-200 bg-white p-3"><div className="metric-label">Duration</div><div className="mt-2 text-lg font-bold">{fmtDuration(selected.duration_s)}</div></div><div className="hover-lift rounded-lg border border-slate-200 bg-white p-3"><div className="metric-label">Clean exams</div><div className="mt-2 text-lg font-bold">{selected.repeatability?.consecutive_clean ?? 0}/{selected.repeatability_target ?? 3}</div></div></div>{!selected.audio && <InlineAlert tone="danger" title="Audio missing" body="The live run endpoint refuses a silent dance." />}{openShow && <InlineAlert title="Resolve the open show first" body="Record Clean, Aborted, or Incident above." />}<InlineRunGate dance={selected} blocked={!selected.audio || !!openShow || data.run.running} /><ChecklistPanel dance={selected} /></div> : <EmptyState title="Select an act" body="Choose a show-ready dance to evaluate its preflight." />}</CardContent></Card>
    </div>
    <Card><CardHeader><div className="panel-kicker"><Users /> Control ownership</div><CardTitle className="mt-2">Walk-on → policy → walk-off</CardTitle></CardHeader><CardContent><div className="grid gap-2 md:grid-cols-5">{data.phases.map((phase, index) => <div key={phase.phase} className="rounded-lg border border-border bg-background/25 p-3"><div className="flex items-center gap-2"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500/10 font-mono text-[10px] text-blue-300">{index + 1}</span><span className="text-[10px] font-bold uppercase tracking-wide">{phase.phase.replaceAll("_", " ")}</span></div><div className="mt-2 text-[10px] font-semibold text-blue-300">{phase.owner}</div><p className="mt-1 line-clamp-3 text-[9px] leading-4 text-muted-foreground">{phase.note}</p></div>)}</div></CardContent></Card>
  </div>
}

function SetlistsTab({ data }: { data: ConsoleData }) {
  const queryClient = useQueryClient()
  const [selectedId, setSelectedId] = useState<string | null>(data.setlists[0]?.id ?? null)
  const selected = data.setlists.find((setlist) => setlist.id === selectedId)
  const [newName, setNewName] = useState("")
  const mutation = useMutation({
    mutationFn: ({ path, method = "POST", body }: { path: string; method?: "POST" | "DELETE"; body?: unknown }) => api.send<SetList>(path, method, body),
    onSuccess: (result) => { if (result?.id) setSelectedId(result.id); queryClient.invalidateQueries({ queryKey: ["setlists"] }); toast.success("Setlist updated") },
    onError: (error: Error) => toast.error(error.message),
  })
  const saveItems = (items: SetListItem[]) => selected && mutation.mutate({ path: `/api/setlists/${selected.id}`, body: { items: items.map(({ dance_id, gap_after_s, note }) => ({ dance_id, gap_after_s, note })) } })
  const move = (index: number, direction: number) => { if (!selected) return; const items = [...selected.items]; const target = index + direction; if (target < 0 || target >= items.length) return; [items[index], items[target]] = [items[target], items[index]]; saveItems(items) }
  return <div className="grid gap-4 xl:grid-cols-[300px_minmax(0,1fr)]"><Card><CardHeader><div className="panel-kicker"><ListMusic /> Setlists</div><CardTitle className="mt-2">Show programs</CardTitle></CardHeader><CardContent className="space-y-2"><div className="flex gap-2"><Input value={newName} onChange={(event) => setNewName(event.target.value)} placeholder="New setlist" /><Button size="icon" disabled={!newName.trim()} onClick={() => { mutation.mutate({ path: "/api/setlists", body: { name: newName } }); setNewName("") }}><Plus /></Button></div>{data.setlists.map((setlist) => <button key={setlist.id} onClick={() => setSelectedId(setlist.id)} className={cn("w-full rounded-lg border p-3 text-left", selectedId === setlist.id ? "border-blue-500/35 bg-blue-500/[.07]" : "border-border")}><div className="flex items-center justify-between gap-2"><span className="truncate text-xs font-semibold">{setlist.name}</span><StatusBadge status={setlist.show_ready ? "ready" : "blocked"} /></div><div className="mt-1 text-[10px] text-muted-foreground">{setlist.items.length} acts • {fmtDuration(setlist.duration_s)}</div></button>)}</CardContent></Card>
    <Card><CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><GripVertical /> Builder</div><CardTitle className="mt-2">{selected?.name ?? "Select a setlist"}</CardTitle></div>{selected && <Button variant="ghost" className="text-red-300" onClick={() => mutation.mutate({ path: `/api/setlists/${selected.id}`, method: "DELETE" })}><Trash2 /> Delete</Button>}</CardHeader><CardContent>{selected ? <div className="space-y-3">
      {(selected.blockers?.length || !selected.show_ready) && <InlineAlert title="Program has show-ready blockers" body={selected.blockers?.join(" • ") || "Every act must be show-ready before the program can run."} />}
      {selected.items.map((item, index) => { const dance = data.dances.find((entry) => entry.id === item.dance_id); return <div key={`${item.dance_id}-${index}`} className="grid gap-3 rounded-lg border border-border bg-background/25 p-3 sm:grid-cols-[auto_1fr_auto]"><div className="flex h-8 w-8 items-center justify-center rounded-md bg-muted font-mono text-xs">{index + 1}</div><div className="min-w-0"><div className="flex items-center gap-2"><span className="truncate text-xs font-semibold">{dance?.name ?? item.dance_name ?? item.dance_id}</span><StatusBadge status={dance?.status ?? item.status} /></div><div className="mt-2 flex items-center gap-2 text-[10px] text-muted-foreground"><span>Gap after</span><Input className="h-7 w-20" type="number" value={item.gap_after_s ?? 8} onChange={(event) => { const items = [...selected.items]; items[index] = { ...items[index], gap_after_s: Number(event.target.value) }; saveItems(items) }} /><span>seconds</span></div></div><div className="flex gap-1"><Button variant="ghost" size="icon" onClick={() => move(index, -1)}><ArrowUp /></Button><Button variant="ghost" size="icon" onClick={() => move(index, 1)}><ArrowDown /></Button><Button variant="ghost" size="icon" className="text-red-300" onClick={() => saveItems(selected.items.filter((_, itemIndex) => itemIndex !== index))}><X /></Button></div></div> })}
      {!selected.items.length && <EmptyState icon={ListMusic} title="Empty program" body="Add a dance below to begin the show timeline." />}
      <Separator />
      <div className="flex flex-wrap gap-2">{data.dances.filter((dance) => !selected.items.some((item) => item.dance_id === dance.id)).map((dance) => <Button key={dance.id} size="sm" variant="outline" onClick={() => saveItems([...selected.items, { dance_id: dance.id, gap_after_s: 8 }])}><Plus /> {dance.name}</Button>)}</div>
    </div> : <EmptyState icon={ListMusic} title="No setlist selected" body="Create a show program or select one from the left." />}</CardContent></Card></div>
}

function VenuesTab({ data }: { data: ConsoleData }) {
  const queryClient = useQueryClient()
  const [name, setName] = useState("")
  const [radius, setRadius] = useState("2")
  const mutation = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => api.send(path, "POST", body),
    onSuccess: () => { toast.success("Venue updated"); queryClient.invalidateQueries({ queryKey: ["venues"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  return <div className="grid gap-4 lg:grid-cols-2"><Card><CardHeader><div className="panel-kicker"><MapPin /> Active venue</div><CardTitle className="mt-2">{data.venues?.active.name ?? "No venue"}</CardTitle></CardHeader><CardContent className="space-y-3">{data.venues?.venues.map((venue) => <button key={venue.id} className={cn("flex w-full items-center gap-3 rounded-lg border p-3 text-left", venue.id === data.venues?.active.id ? "border-blue-500/35 bg-blue-500/[.07]" : "border-border")} onClick={() => mutation.mutate({ path: "/api/venues/active", body: { key: venue.id } })}><div className="rounded-md bg-blue-500/10 p-2 text-blue-300"><MapPin className="h-4 w-4" /></div><div className="flex-1"><div className="text-xs font-semibold">{venue.name}</div><div className="mt-1 text-[10px] text-muted-foreground">{venue.radius_m} m radius • {venue.max_excursion_m} m motion limit</div></div>{venue.id === data.venues?.active.id && <Badge variant="success">active</Badge>}</button>)}</CardContent></Card><Card><CardHeader><div className="panel-kicker"><Plus /> Venue registry</div><CardTitle className="mt-2">Add venue</CardTitle></CardHeader><CardContent className="space-y-3"><div><label className="metric-label">Venue name</label><Input className="mt-2" value={name} onChange={(event) => setName(event.target.value)} placeholder="Stage A" /></div><div><label className="metric-label">Clear radius (m)</label><Input className="mt-2" type="number" min="0.6" step="0.1" value={radius} onChange={(event) => setRadius(event.target.value)} /></div><InlineAlert tone="info" title="The active venue drives motion vetting" body="The hard excursion gate uses radius minus the 0.5 m safety margin." /><Button className="w-full" disabled={!name.trim()} onClick={() => mutation.mutate({ path: "/api/venues", body: { name, radius_m: Number(radius), make_active: true } })}><MapPin /> Save and activate</Button></CardContent></Card></div>
}

export function PerformScreen({ data }: { data: ConsoleData }) {
  return <div><PageHeader eyebrow="Operate" title="Show mode" description="Select the act, watch the robot preview, complete every safety check, then type the physical-remote confirmation to run." actions={data.run.running ? <Badge variant="destructive" className="animate-pulse">SHOW RUNNING</Badge> : <Badge variant="secondary">Robot standing by</Badge>} /><Tabs defaultValue="perform"><TabsList><TabsTrigger value="perform">Perform</TabsTrigger><TabsTrigger value="setlists">Setlists</TabsTrigger><TabsTrigger value="venues">Venues</TabsTrigger></TabsList><TabsContent value="perform"><PerformTab data={data} /></TabsContent><TabsContent value="setlists"><SetlistsTab data={data} /></TabsContent><TabsContent value="venues"><VenuesTab data={data} /></TabsContent></Tabs></div>
}
