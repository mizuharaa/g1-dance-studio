import { Activity } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { InlineAlert } from "@/components/console-ui"
import type { SuccessEstimate } from "@/lib/api"
import { cn } from "@/lib/utils"

/** "Predicted show readiness" — the pre-training physics crosscheck computed by
 * the retarget stage BEFORE any GPU money is spent. Deliberately subdued: this
 * is a rough calibrated band, not a verified result. */
export function SuccessEstimateCard({ est, className }: { est: SuccessEstimate; className?: string }) {
  const lo = est.predicted_survival_lo_pct
  const hi = est.predicted_survival_hi_pct
  const windows = est.risk_windows ?? []
  return <Card className={className}>
    <CardHeader className="flex-row items-start justify-between space-y-0">
      <div>
        <div className="panel-kicker"><Activity /> Pre-training crosscheck</div>
        <CardTitle className="mt-2">Predicted show readiness</CardTitle>
      </div>
      <div className="text-right">
        <div className="font-mono text-2xl font-bold text-blue-300">{est.predicted_survival_pct_range ?? "—"}</div>
        <div className="text-[9px] uppercase tracking-wide text-muted-foreground">predicted nominal survival</div>
      </div>
    </CardHeader>
    <CardContent>
      <div className="relative h-1.5 overflow-hidden rounded-full bg-muted">
        {lo != null && hi != null && <div className="absolute inset-y-0 rounded-full bg-blue-500/70" style={{ left: `${Math.max(0, lo)}%`, width: `${Math.max(2, hi - lo)}%` }} />}
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-muted-foreground"><span>0%</span><span>100%</span></div>
      {!!est.hard_blockers?.length && <InlineAlert className="mt-3" tone="danger" title="Hard vet blockers" body={`${est.hard_blockers.join(" · ")} — training stays blocked until fixed; the band assumes a fixed motion.`} />}
      {windows.length
        ? <div className="mt-3"><div className="metric-label">Top risk windows</div><div className="mt-2 flex flex-wrap gap-1.5">{windows.map((w, i) => <span key={i} className="rounded-full border border-amber-500/25 bg-amber-500/[.07] px-2.5 py-1 font-mono text-[10px] text-amber-300">{w.start_s}–{w.end_s}s · {w.label ?? "high demand"}</span>)}</div></div>
        : <div className="mt-3 text-[10px] text-emerald-400/80">No sustained high-ankle-demand windows flagged.</div>}
      <div className={cn("mt-3 rounded-lg border border-border bg-background/25 p-2.5 text-[10px] leading-4 text-muted-foreground")}>
        Rough pre-training estimate from motion physics{est.confidence ? ` (${est.confidence})` : ""}; not a guarantee. The torque model historically over-estimated demand, so this leans conservative.
      </div>
    </CardContent>
  </Card>
}
