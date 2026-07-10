import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { AlertOctagon, ArrowDown, ArrowUp, Check, ChevronRight, CircleAlert, GripVertical, ListMusic, MapPin, Plus, Radio, ShieldAlert, ShieldCheck, Sparkles, Trash2, Users, Volume2, WifiOff, X } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { Input } from "@/components/ui/input"
import { Separator } from "@/components/ui/separator"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { EmptyState, InlineAlert, PageHeader, StatusBadge } from "@/components/console-ui"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type Dance, type SetList, type SetListItem, type Show } from "@/lib/api"
import { cn, fmtDate, fmtDuration } from "@/lib/utils"

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

function RunShowDialog({ dance, open, onOpenChange }: { dance: Dance | null; open: boolean; onOpenChange: (value: boolean) => void }) {
  const queryClient = useQueryClient()
  const [operator, setOperator] = useState("")
  const [phrase, setPhrase] = useState("")
  const [mode, setMode] = useState<"rehearsal" | "live">("rehearsal")
  const [exitStand, setExitStand] = useState(true)
  const [free, setFree] = useState(false)
  useEffect(() => { if (!open) { setPhrase(""); setFree(false) } }, [open])
  const run = useMutation({
    mutationFn: () => api.send(`/api/shows/${dance!.id}/run`, "POST", { confirmation: phrase, operator, mode, exit_stand: exitStand, free, audio_mode: "laptop" }),
    onSuccess: () => { toast.success("Show started — keep the damping remote in hand"); onOpenChange(false); queryClient.invalidateQueries({ queryKey: ["current-run"] }); queryClient.invalidateQueries({ queryKey: ["shows"] }) },
    onError: (error: Error) => toast.error(error.message),
  })
  if (!dance) return null
  const confirmed = phrase === RUN_PHRASE && operator.trim().length > 0
  return <Dialog open={open} onOpenChange={onOpenChange}><DialogContent className="max-w-2xl border-red-500/30"><DialogHeader><DialogTitle className="flex items-center gap-2 text-red-200"><ShieldAlert className="h-5 w-5" /> Arm live robot show</DialogTitle><DialogDescription>{dance.name} • {fmtDuration(dance.duration_s)} • policy and audio contract enforced server-side.</DialogDescription></DialogHeader>
    <div className="space-y-4">
      <div className="rounded-xl border border-red-500/35 bg-red-500/10 p-4 text-center"><div className="text-lg font-black tracking-wide text-red-200">REMOTE B-DAMP = PRIMARY STOP</div><div className="mt-1 text-xs text-red-200/65">This G1 has no torque-cut hardware e-stop. Hold the damping remote for the entire run.</div></div>
      <div className="grid gap-3 sm:grid-cols-2"><div><label className="metric-label">Operator</label><Input className="mt-2" value={operator} onChange={(event) => setOperator(event.target.value)} placeholder="Operator name" /></div><div><label className="metric-label">Run mode</label><div className="mt-2 grid grid-cols-2 gap-2"><Button variant={mode === "rehearsal" ? "secondary" : "outline"} onClick={() => setMode("rehearsal")}>Rehearsal</Button><Button variant={mode === "live" ? "secondary" : "outline"} onClick={() => setMode("live")}>Live</Button></div></div></div>
      <div className="grid gap-2 sm:grid-cols-2"><button className={cn("rounded-lg border p-3 text-left", exitStand ? "border-blue-500/40 bg-blue-500/[.07]" : "border-border")} onClick={() => setExitStand(!exitStand)}><div className="flex items-center gap-2 text-xs font-semibold"><span className={cn("flex h-4 w-4 items-center justify-center rounded border", exitStand && "border-blue-500 bg-blue-500 text-white")}>{exitStand && <Check className="h-3 w-3" />}</span> Stand at finish</div><p className="mt-1 text-[10px] text-muted-foreground">Policy holds final pose, then overlaps onboard handback.</p></button><button className={cn("rounded-lg border p-3 text-left", free ? "border-amber-500/40 bg-amber-500/[.07]" : "border-border")} onClick={() => setFree(!free)}><div className="flex items-center gap-2 text-xs font-semibold"><span className={cn("flex h-4 w-4 items-center justify-center rounded border", free && "border-amber-500 bg-amber-500 text-black")}>{free && <Check className="h-3 w-3" />}</span> Untethered/free config</div><p className="mt-1 text-[10px] text-muted-foreground">Only use the hardware-validated free policy and a clear venue.</p></button></div>
      {free && <InlineAlert tone="warning" title="Highest-risk configuration" body="No tether catches the robot. This remains a conscious operator choice even after sim and hardware validation." />}
      <div><label className="metric-label">Type the exact confirmation phrase</label><div className="mt-2 rounded-md bg-black/35 p-2.5 text-center font-mono text-[11px] text-red-300">{RUN_PHRASE}</div><Input data-testid="run-confirmation" className={cn("mt-2 font-mono", phrase && phrase !== RUN_PHRASE && "border-red-500/60")} value={phrase} onChange={(event) => setPhrase(event.target.value)} placeholder="Type phrase exactly" autoComplete="off" /></div>
    </div>
    <DialogFooter><Button variant="outline" onClick={() => onOpenChange(false)}>Cancel</Button><Button data-testid="start-show" variant="destructive" size="lg" disabled={!confirmed || run.isPending} onClick={() => run.mutate()}><Radio /> {run.isPending ? "Starting…" : "RUN SHOW"}</Button></DialogFooter>
  </DialogContent></Dialog>
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

function PerformTab({ data }: { data: ConsoleData }) {
  const ready = data.dances.filter((dance) => dance.status === "show-ready")
  const [selectedId, setSelectedId] = useState<string | null>(ready[0]?.id ?? null)
  const [runOpen, setRunOpen] = useState(false)
  const selected = data.dances.find((dance) => dance.id === selectedId) ?? ready[0]
  const openShow = data.shows.find((show) => !show.closed)
  return <div className="space-y-4">
    {openShow && <OutcomeCapture show={openShow} />}
    <div className="grid gap-4 xl:grid-cols-[minmax(0,1.15fr)_minmax(360px,.85fr)]">
      <Card>
        <CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><Sparkles /> Select act</div><CardTitle className="mt-2">Show-ready dances</CardTitle></div><Badge variant="success">{ready.length} ready</Badge></CardHeader>
        <CardContent className="grid gap-3 sm:grid-cols-2">
          {ready.map((dance) => <button key={dance.id} onClick={() => setSelectedId(dance.id)} className={cn("rounded-xl border p-4 text-left transition-colors", selected?.id === dance.id ? "border-blue-500/45 bg-blue-500/[.08] shadow-glow" : "border-border bg-background/25 hover:bg-accent/50")}><div className="flex items-start justify-between gap-2"><div className="rounded-lg bg-blue-500/10 p-2 text-blue-300"><Volume2 className="h-4 w-4" /></div><StatusBadge status={dance.status} /></div><div className="mt-4 text-sm font-semibold">{dance.name}</div><div className="mt-1 flex items-center gap-2 text-[10px] text-muted-foreground"><span>{fmtDuration(dance.duration_s)}</span><span>•</span><span>{dance.audio ? "music attached" : "no music"}</span></div><div className="mt-3 flex items-center gap-2 text-[10px]"><span className={cn("status-dot", dance.audio ? "bg-emerald-400" : "bg-red-400")} /><span className={dance.audio ? "text-emerald-300" : "text-red-300"}>{dance.audio ? "Audio ready" : "Blocked: attach audio"}</span></div></button>)}
          {!ready.length && <div className="sm:col-span-2"><EmptyState icon={ShieldCheck} title="No show-ready dances" body="A dance needs three clean signed exams, a pinned policy, and an explicit promotion." /></div>}
        </CardContent>
      </Card>
      <Card className="relative overflow-hidden"><div className="absolute right-0 top-0 h-48 w-48 rounded-full bg-blue-500/5 blur-3xl" /><CardHeader><div className="panel-kicker"><ShieldCheck /> Perform mode</div><CardTitle className="mt-2">{selected?.name ?? "Select a dance"}</CardTitle></CardHeader><CardContent>{selected ? <div className="space-y-4"><div className="grid grid-cols-2 gap-3"><div className="rounded-lg border border-border p-3"><div className="metric-label">Duration</div><div className="mt-2 text-lg font-semibold">{fmtDuration(selected.duration_s)}</div></div><div className="rounded-lg border border-border p-3"><div className="metric-label">Clean exams</div><div className="mt-2 text-lg font-semibold">{selected.repeatability?.consecutive_clean ?? 0}/{selected.repeatability_target ?? 3}</div></div></div><ChecklistPanel dance={selected} /><Button size="lg" className="w-full" disabled={!selected.audio || !!openShow || data.run.running} onClick={() => setRunOpen(true)}><Radio /> Arm run show</Button>{!selected.audio && <InlineAlert tone="danger" title="Audio missing" body="The live run endpoint refuses a silent dance." />}{openShow && <InlineAlert title="Resolve the open show first" body="Record Clean, Aborted, or Incident above." />}</div> : <EmptyState title="Select an act" body="Choose a show-ready dance to evaluate its preflight." />}</CardContent></Card>
    </div>
    <Card><CardHeader><div className="panel-kicker"><Users /> Control ownership</div><CardTitle className="mt-2">Walk-on → policy → walk-off</CardTitle></CardHeader><CardContent><div className="grid gap-2 md:grid-cols-5">{data.phases.map((phase, index) => <div key={phase.phase} className="rounded-lg border border-border bg-background/25 p-3"><div className="flex items-center gap-2"><span className="flex h-6 w-6 items-center justify-center rounded-full bg-blue-500/10 font-mono text-[10px] text-blue-300">{index + 1}</span><span className="text-[10px] font-bold uppercase tracking-wide">{phase.phase.replaceAll("_", " ")}</span></div><div className="mt-2 text-[10px] font-semibold text-blue-300">{phase.owner}</div><p className="mt-1 line-clamp-3 text-[9px] leading-4 text-muted-foreground">{phase.note}</p></div>)}</div></CardContent></Card>
    <RunShowDialog dance={selected ?? null} open={runOpen} onOpenChange={setRunOpen} />
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
  return <div><PageHeader eyebrow="Operate" title="Shows & setlists" description="Performance mode makes blockers explicit, keeps the stop path visible, and never hides who owns the robot." actions={data.run.running ? <Badge variant="destructive" className="animate-pulse">SHOW RUNNING</Badge> : <Badge variant="secondary">standby</Badge>} /><Tabs defaultValue="perform"><TabsList><TabsTrigger value="perform">Perform</TabsTrigger><TabsTrigger value="setlists">Setlists</TabsTrigger><TabsTrigger value="venues">Venues</TabsTrigger></TabsList><TabsContent value="perform"><PerformTab data={data} /></TabsContent><TabsContent value="setlists"><SetlistsTab data={data} /></TabsContent><TabsContent value="venues"><VenuesTab data={data} /></TabsContent></Tabs></div>
}
