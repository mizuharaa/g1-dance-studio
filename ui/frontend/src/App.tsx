import { lazy, Suspense, useEffect, useMemo, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { BarChart3, ChevronRight, CircleAlert, Command, Cpu, History, LayoutDashboard, Menu, Radio, RefreshCw, Shield, Square, Workflow, X } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import { api } from "@/lib/api"
import { cn, fmtMoney } from "@/lib/utils"
import { useConsoleData } from "@/hooks/use-console-data"
import { OverviewScreen } from "@/screens/overview"
import { CuteRobotMark } from "@/components/robot-preview"

const PipelineScreen = lazy(() => import("@/screens/pipeline").then((module) => ({ default: module.PipelineScreen })))
const DancesScreen = lazy(() => import("@/screens/dances").then((module) => ({ default: module.DancesScreen })))
const PerformScreen = lazy(() => import("@/screens/perform").then((module) => ({ default: module.PerformScreen })))
const AuditScreen = lazy(() => import("@/screens/audit").then((module) => ({ default: module.AuditScreen })))
const SystemScreen = lazy(() => import("@/screens/system").then((module) => ({ default: module.SystemScreen })))

type Screen = "overview" | "pipeline" | "dances" | "perform" | "audit" | "system"

const NAV: Array<{ id: Screen; label: string; short: string; icon: typeof LayoutDashboard; section: "mission" | "operate" | "evidence" }> = [
  { id: "overview", label: "Overview", short: "Overview", icon: LayoutDashboard, section: "mission" },
  { id: "pipeline", label: "Pipeline studio", short: "Pipeline", icon: Workflow, section: "mission" },
  { id: "dances", label: "Dances & stats", short: "Dances", icon: BarChart3, section: "mission" },
  { id: "perform", label: "Show mode", short: "Show mode", icon: Radio, section: "operate" },
  { id: "audit", label: "Audit log", short: "Audit", icon: History, section: "evidence" },
  { id: "system", label: "System", short: "System", icon: Cpu, section: "evidence" },
]

function Brand() {
  return <div className="flex items-center gap-3"><div className="relative flex h-10 w-10 items-center justify-center rounded-xl border border-blue-200 bg-blue-50 text-blue-600 shadow-sm"><CuteRobotMark className="h-7 w-7" /></div><div><div className="text-sm font-black tracking-tight text-slate-900">G1 Dance</div><div className="mt-0.5 text-[9px] font-semibold uppercase tracking-[.15em] text-blue-600">Operator console</div></div></div>
}

function Sidebar({ active, onNavigate, data }: { active: Screen; onNavigate: (screen: Screen) => void; data: ReturnType<typeof useConsoleData> }) {
  return <aside className="fixed inset-y-0 left-0 z-40 hidden w-[236px] flex-col border-r border-slate-200 bg-white px-3 py-4 lg:flex"><div className="px-2"><Brand /></div><div className="mt-8 flex-1">{(["mission", "operate", "evidence"] as const).map((section) => <div key={section} className="mb-6"><div className="mb-2 px-3 text-[9px] font-bold uppercase tracking-[.17em] text-slate-400">{section}</div><nav className="space-y-1">{NAV.filter((item) => item.section === section).map((item) => { const Icon = item.icon; const count = item.id === "dances" ? data.dances.length : item.id === "audit" ? data.shows.length : undefined; return <button key={item.id} onClick={() => onNavigate(item.id)} className={cn("group flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-xs font-semibold transition-all", active === item.id ? "bg-blue-50 text-blue-800 ring-1 ring-inset ring-blue-200 shadow-sm" : "text-slate-600 hover:-translate-y-px hover:bg-slate-50 hover:text-slate-900")}><Icon className={cn("h-4 w-4", active === item.id ? "text-blue-600" : "text-slate-400 group-hover:text-blue-500")} /><span className="min-w-0 flex-1">{item.label}</span>{count != null && <span className="rounded-full bg-slate-100 px-1.5 py-0.5 font-mono text-[9px] text-slate-500">{count}</span>}{active === item.id && <ChevronRight className="h-3 w-3 text-blue-500" />}</button>})}</nav></div>)}</div><div className="space-y-2 border-t border-slate-200 pt-3"><div className="rounded-lg bg-slate-950 p-3 text-white shadow-lg"><div className="flex items-center gap-2"><span className="text-[10px] font-semibold">GPU box</span><span className="ml-auto text-[9px] text-slate-400">{data.system?.reachable ? "Live" : "Offline"}</span></div><div className="mt-3 flex justify-between text-[9px] text-slate-400"><span>Cloud spend</span><span className={cn("font-mono font-semibold text-white", data.system?.cost?.over_cap && "text-red-300")}>{fmtMoney(data.system?.cost?.accrued_vnd)}</span></div><div className="mt-2 h-1 overflow-hidden rounded-full bg-slate-800"><div className={cn("h-full rounded-full", data.system?.cost?.over_cap ? "bg-red-500" : "bg-blue-500")} style={{ width: `${Math.min(100, (data.system?.cost?.cap_fraction ?? 0) * 100)}%` }} /></div></div><div className="flex items-center gap-2 px-2 text-[9px] text-slate-400"><Command className="h-3 w-3" /><span>Local engine • :8735</span></div></div></aside>
}

function MobileNav({ open, active, onNavigate, onClose }: { open: boolean; active: Screen; onNavigate: (screen: Screen) => void; onClose: () => void }) {
  if (!open) return null
  return <div className="fixed inset-0 z-50 bg-slate-950/45 backdrop-blur-sm lg:hidden" onClick={onClose}><div className="h-full w-[280px] border-r border-slate-200 bg-white p-4 shadow-2xl" onClick={(event) => event.stopPropagation()}><div className="flex items-center justify-between"><Brand /><Button variant="ghost" size="icon" onClick={onClose}><X /></Button></div><nav className="mt-8 space-y-1">{NAV.map((item) => { const Icon = item.icon; return <button key={item.id} onClick={() => { onNavigate(item.id); onClose() }} className={cn("flex w-full items-center gap-3 rounded-lg px-3 py-3 text-sm font-semibold", active === item.id ? "bg-blue-50 text-blue-800" : "text-slate-600")}><Icon className="h-4 w-4" />{item.label}</button>})}</nav></div></div>
}

function TopBar({ active, onMenu, data }: { active: Screen; onMenu: () => void; data: ReturnType<typeof useConsoleData> }) {
  const item = NAV.find((entry) => entry.id === active)!
  const [clock, setClock] = useState(new Date())
  useEffect(() => { const timer = window.setInterval(() => setClock(new Date()), 1_000); return () => window.clearInterval(timer) }, [])
  return <header className="sticky top-0 z-30 flex h-16 items-center border-b border-slate-200 bg-white/92 px-4 backdrop-blur-xl sm:px-6 lg:px-8"><Button variant="ghost" size="icon" className="mr-2 lg:hidden" onClick={onMenu} aria-label="Menu"><Menu /></Button><div className="hidden items-center gap-2 text-[11px] text-slate-500 sm:flex"><Shield className="h-3.5 w-3.5 text-blue-600" /><span>Operator console</span><ChevronRight className="h-3 w-3" /><span className="font-semibold text-slate-800">{item.label}</span></div><div className="text-sm font-semibold sm:hidden">{item.short}</div><div className="ml-auto flex items-center gap-2 sm:gap-3"><div className="hidden rounded-full border border-slate-200 bg-slate-50 px-3 py-1.5 text-[10px] text-slate-600 md:flex">Engine {data.error ? "degraded" : "online"}</div><div className="rounded-full border border-slate-200 bg-white px-3 py-1.5 font-mono text-[10px] text-slate-700 shadow-sm">{clock.toLocaleTimeString([], { hour12: false })}</div><Button variant="ghost" size="icon" onClick={() => data.refetchAll()} aria-label="Refresh data"><RefreshCw className="h-4 w-4" /></Button></div></header>
}

function GlobalStop({ data }: { data: ReturnType<typeof useConsoleData> }) {
  const queryClient = useQueryClient()
  const stop = useMutation({ mutationFn: () => api.send<{ detail: string }>("/api/shows/runs/current/stop", "POST"), onSuccess: (result) => { toast.warning(result.detail); queryClient.invalidateQueries({ queryKey: ["current-run"] }) }, onError: (error: Error) => toast.error(error.message) })
  if (!data.run.running) return null
  return <div className="fixed bottom-4 left-4 right-4 z-40 lg:left-[252px]" data-testid="global-stop"><div className="mx-auto flex max-w-5xl items-center gap-3 rounded-xl border border-red-500/40 bg-red-950/95 p-3 shadow-danger backdrop-blur-xl"><div className="hidden rounded-lg bg-red-500/15 p-2 text-red-300 sm:block"><Shield className="h-5 w-5" /></div><div className="hidden min-w-0 flex-1 sm:block"><div className="text-xs font-black text-red-100">SHOW RUNNING • REMOTE B-DAMP IN HAND</div><div className="mt-0.5 truncate text-[10px] text-red-200/55">Software STOP damps the robot soft. Keep the tether ready to take load.</div></div><Button variant="destructive" size="lg" className="h-12 w-full text-base font-black sm:w-auto" onClick={() => stop.mutate()}><Square className="fill-current" /> STOP SHOW</Button></div></div>
}

export default function App() {
  const [active, setActive] = useState<Screen>("overview")
  const [mobileOpen, setMobileOpen] = useState(false)
  const data = useConsoleData()
  const screen = useMemo(() => {
    if (active === "pipeline") return <PipelineScreen data={data} />
    if (active === "dances") return <DancesScreen data={data} />
    if (active === "perform") return <PerformScreen data={data} />
    if (active === "audit") return <AuditScreen data={data} />
    if (active === "system") return <SystemScreen data={data} />
    return <OverviewScreen data={data} onPerform={() => setActive("perform")} />
  }, [active, data])
  return <div className="min-h-screen"><Sidebar active={active} onNavigate={setActive} data={data} /><MobileNav open={mobileOpen} active={active} onNavigate={setActive} onClose={() => setMobileOpen(false)} /><div className="min-h-screen lg:pl-[236px]"><TopBar active={active} onMenu={() => setMobileOpen(true)} data={data} /><main className={cn("light-console mx-auto w-full max-w-[1680px] px-4 py-6 sm:px-6 lg:px-8 lg:py-8", data.run.running && "pb-28")}>
    {data.error && <div className="mb-5 flex items-center gap-3 rounded-lg border border-red-500/25 bg-red-500/[.07] p-3 text-xs text-red-200"><CircleAlert className="h-4 w-4" /><span>Some console data could not be loaded. Safety actions remain server-enforced.</span></div>}
    <Suspense fallback={<div className="flex min-h-[55vh] items-center justify-center"><div className="flex items-center gap-3 text-xs text-muted-foreground"><RefreshCw className="h-4 w-4 animate-spin text-blue-400" /> Loading console module…</div></div>}>{screen}</Suspense>
  </main></div><GlobalStop data={data} /></div>
}
