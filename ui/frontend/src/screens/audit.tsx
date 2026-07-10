import { useMemo, useState } from "react"
import { AlertOctagon, CheckCircle2, Clock3, Filter, History, Rocket, Search, ShieldCheck, SlidersHorizontal, Sparkles, XCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { EmptyState, PageHeader, StatusBadge } from "@/components/console-ui"
import type { ConsoleData } from "@/hooks/use-console-data"
import type { Dance } from "@/lib/api"
import { cn, fmtDate, fmtPercent, shortHash } from "@/lib/utils"

type AuditType = "outcome" | "incident" | "exam" | "promotion" | "deploy"
interface AuditEvent { id: string; at: number; type: AuditType; title: string; detail: string; danceId?: string; danceName?: string; status: string; second?: number; metrics?: string; hash?: string }

function deriveEvents(data: ConsoleData): AuditEvent[] {
  const events: AuditEvent[] = []
  data.shows.forEach((show) => {
    if (show.outcome) events.push({ id: `outcome-${show.id}`, at: show.outcome.at ?? show.created_at, type: show.outcome.result === "incident" ? "incident" : "outcome", title: `${show.outcome.result === "clean" ? "Clean" : show.outcome.result === "incident" ? "Incident" : "Aborted"} ${show.mode} run`, detail: show.outcome.notes || `${show.dance_name} run closed by ${show.operator}.`, danceId: show.dance_id, danceName: show.dance_name, status: show.outcome.result })
    if (show.deploy) events.push({ id: `deploy-${show.id}`, at: show.deploy.requested_at ?? show.created_at, type: "deploy", title: "Deploy authorized", detail: show.deploy.note || `Deploy request recorded for ${show.dance_name}.`, danceId: show.dance_id, danceName: show.dance_name, status: "deployed" })
  })
  data.dances.forEach((dance) => {
    if (dance.status === "show-ready") events.push({ id: `promotion-${dance.id}`, at: dance.updated_at, type: "promotion", title: "Promoted to show-ready", detail: `Policy ${shortHash(dance.policy_sha256)} pinned after ${dance.repeatability?.consecutive_clean ?? 0} clean signed exams.`, danceId: dance.id, danceName: dance.name, status: "show-ready", hash: dance.policy_sha256 ?? undefined })
    const history = dance.repeatability?.history ?? []
    history.forEach((run, index) => {
      const nominal = run.metrics?.nominal
      const push = run.metrics?.push
      events.push({ id: `exam-${dance.id}-${index}-${run.at}`, at: run.at ?? dance.updated_at, type: "exam", title: `Signed sim exam ${run.passed ? "passed" : "failed"}`, detail: `Held-out seed ${nominal?.held_out_seed ?? "—"}; nominal ${fmtPercent(nominal?.success_rate, 1)}, push ${fmtPercent(push?.success_rate, 1)}.`, danceId: dance.id, danceName: dance.name, status: run.passed ? "pass" : "fail", metrics: `MPKPE ${nominal?.mpkpe_m != null ? (nominal.mpkpe_m * 100).toFixed(1) + " cm" : "—"}`, hash: run.policy_sha256 })
    })
    if (dance.incident) events.push({ id: `incident-${dance.id}`, at: dance.incident.at ?? dance.updated_at, type: "incident", title: "Dance demoted after incident", detail: String(dance.incident.detail ?? dance.incident.message ?? "Incident recorded; verification reset."), danceId: dance.id, danceName: dance.name, status: "incident", second: typeof dance.incident.second === "number" ? dance.incident.second : undefined })
  })
  return events.sort((a, b) => b.at - a.at)
}

const iconFor = (event: AuditEvent) => event.type === "incident" ? AlertOctagon : event.type === "exam" ? ShieldCheck : event.type === "promotion" ? Sparkles : event.type === "deploy" ? Rocket : event.status === "clean" ? CheckCircle2 : Clock3

export function AuditScreen({ data }: { data: ConsoleData }) {
  const [type, setType] = useState<"all" | AuditType>("all")
  const [danceId, setDanceId] = useState("all")
  const [query, setQuery] = useState("")
  const events = useMemo(() => deriveEvents(data), [data])
  const filtered = events.filter((event) => (type === "all" || event.type === type) && (danceId === "all" || event.danceId === danceId) && (!query || `${event.title} ${event.detail} ${event.danceName}`.toLowerCase().includes(query.toLowerCase())))
  const incidents = events.filter((event) => event.type === "incident").length
  const failed = events.filter((event) => event.status === "fail").length
  return <div>
    <PageHeader eyebrow="Evidence" title="Audit log" description="A filterable chain of exams, promotions, deploys, outcomes, and incidents. Nothing safety-relevant is buried in a card modal." actions={<><Badge variant={incidents ? "destructive" : "success"}>{incidents} incidents</Badge><Badge variant={failed ? "warning" : "secondary"}>{failed} failed exams</Badge></>} />
    <Card className="mb-4"><CardContent className="grid gap-3 pt-5 md:grid-cols-[1fr_220px_220px]"><div className="relative"><Search className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><Input data-testid="audit-search" className="pl-9" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search outcome, verdict, operator note…" /></div><div className="relative"><Filter className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><select data-testid="audit-type-filter" className="h-10 w-full appearance-none rounded-md border border-input bg-background/70 pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-blue-500/20" value={type} onChange={(event) => setType(event.target.value as typeof type)}><option value="all">All event types</option><option value="outcome">Run outcomes</option><option value="incident">Incidents</option><option value="exam">Sim exams</option><option value="promotion">Promotions</option><option value="deploy">Deploys</option></select></div><div className="relative"><SlidersHorizontal className="absolute left-3 top-3 h-4 w-4 text-muted-foreground" /><select data-testid="audit-dance-filter" className="h-10 w-full appearance-none rounded-md border border-input bg-background/70 pl-9 pr-3 text-sm outline-none focus:ring-2 focus:ring-blue-500/20" value={danceId} onChange={(event) => setDanceId(event.target.value)}><option value="all">All dances</option>{data.dances.map((dance) => <option key={dance.id} value={dance.id}>{dance.name}</option>)}</select></div></CardContent></Card>

    <Card><CardHeader className="flex-row items-center justify-between space-y-0"><div><div className="panel-kicker"><History /> Timeline</div><CardTitle className="mt-2">Operational evidence</CardTitle></div><Badge variant="secondary">{filtered.length} shown</Badge></CardHeader><CardContent>{filtered.length ? <div className="relative"><div className="absolute bottom-3 left-[19px] top-3 w-px bg-border" />{filtered.map((event) => { const Icon = iconFor(event); const danger = event.type === "incident" || event.status === "fail"; return <div key={event.id} className="relative grid gap-3 pb-3 sm:grid-cols-[42px_150px_1fr]"><div className={cn("relative z-10 flex h-10 w-10 items-center justify-center rounded-full border bg-card", danger ? "border-red-500/35 text-red-300" : event.status === "pass" || event.status === "clean" || event.status === "show-ready" ? "border-emerald-500/30 text-emerald-300" : "border-blue-500/30 text-blue-300")}><Icon className="h-4 w-4" /></div><div className="pt-1"><div className="font-mono text-[10px] text-muted-foreground">{fmtDate(event.at)}</div><div className="mt-1 text-[10px] font-semibold text-blue-300">{event.danceName ?? "Global"}</div></div><div className={cn("rounded-lg border bg-background/25 p-3", danger ? "border-red-500/25" : "border-border")}><div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between"><div><div className="text-xs font-semibold">{event.title}</div><p className="mt-1 text-[11px] leading-5 text-muted-foreground">{event.detail}</p></div><StatusBadge status={event.status} /></div>{(event.second != null || event.metrics || event.hash) && <div className="mt-3 flex flex-wrap gap-2">{event.second != null && <Badge variant="destructive">at {event.second.toFixed(1)}s</Badge>}{event.metrics && <Badge variant="secondary">{event.metrics}</Badge>}{event.hash && <Badge variant="outline" className="font-mono">{shortHash(event.hash)}</Badge>}</div>}</div></div>})}</div> : <EmptyState icon={History} title="No matching audit events" body="Clear a filter or choose another dance. Events are derived from the current dance and show API records." />}</CardContent></Card>
  </div>
}
