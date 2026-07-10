import type { ReactNode } from "react"
import { AlertTriangle, CheckCircle2, CircleDashed, Clock3, XCircle } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { cn } from "@/lib/utils"

export function PageHeader({ eyebrow, title, description, actions }: { eyebrow?: string; title: string; description: string; actions?: ReactNode }) {
  return (
    <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-end sm:justify-between">
      <div>
        {eyebrow && <div className="mb-2 text-[10px] font-bold uppercase tracking-[0.2em] text-blue-400">{eyebrow}</div>}
        <h1 className="text-2xl font-bold tracking-tight text-foreground sm:text-[30px]">{title}</h1>
        <p className="mt-1 max-w-3xl text-sm text-muted-foreground">{description}</p>
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2">{actions}</div>}
    </header>
  )
}

export function StatusBadge({ status, label }: { status?: string | null; label?: string }) {
  const normalized = (status || "unknown").toLowerCase()
  const success = ["done", "pass", "passed", "clean", "show-ready", "ready", "running"].includes(normalized)
  const warning = ["blocked", "aborted", "warning", "sim-verified", "arming", "launching"].includes(normalized)
  const danger = ["failed", "fail", "incident", "fall", "stopped"].includes(normalized)
  const Icon = success ? CheckCircle2 : warning ? Clock3 : danger ? XCircle : CircleDashed
  return <Badge variant={success ? "success" : warning ? "warning" : danger ? "destructive" : "secondary"} className="gap-1.5"><Icon className="h-3 w-3" />{label || normalized.replaceAll("-", " ")}</Badge>
}

export function EmptyState({ icon: Icon = CircleDashed, title, body, action }: { icon?: typeof CircleDashed; title: string; body: string; action?: ReactNode }) {
  return (
    <div className="flex min-h-48 flex-col items-center justify-center rounded-lg border border-dashed border-border bg-background/20 px-6 py-10 text-center">
      <div className="mb-3 rounded-full border border-border bg-muted p-3"><Icon className="h-5 w-5 text-muted-foreground" /></div>
      <h3 className="text-sm font-semibold text-foreground">{title}</h3>
      <p className="mt-1 max-w-md text-xs leading-5 text-muted-foreground">{body}</p>
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}

export function InlineAlert({ title, body, tone = "warning", className }: { title: string; body?: string; tone?: "warning" | "danger" | "info" | "success"; className?: string }) {
  const colors = tone === "danger" ? "border-red-200 bg-red-50 text-red-800" : tone === "success" ? "border-emerald-200 bg-emerald-50 text-emerald-800" : tone === "info" ? "border-blue-200 bg-blue-50 text-blue-800" : "border-amber-200 bg-amber-50 text-amber-900"
  return (
    <div className={cn("flex gap-3 rounded-lg border p-3 text-xs", colors, className)}>
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
      <div><div className="font-semibold">{title}</div>{body && <div className="mt-0.5 leading-5 opacity-75">{body}</div>}</div>
    </div>
  )
}

export function Metric({ label, value, detail, accent }: { label: string; value: ReactNode; detail?: ReactNode; accent?: "blue" | "green" | "red" | "amber" }) {
  const accentClass = accent === "green" ? "text-emerald-700" : accent === "red" ? "text-red-600" : accent === "amber" ? "text-amber-700" : "text-foreground"
  return <div><div className="metric-label">{label}</div><div className={cn("mt-2 text-2xl font-semibold tracking-tight", accentClass)}>{value}</div>{detail && <div className="mt-1 text-[11px] text-muted-foreground">{detail}</div>}</div>
}
