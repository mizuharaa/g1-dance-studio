import { useMutation, useQueryClient } from "@tanstack/react-query"
import { OctagonX, WifiOff, Wifi } from "lucide-react"
import { toast } from "sonner"
import { Button } from "@/components/ui/button"
import { CuteRobot } from "@/components/robot-preview"
import { api, type RunStatus } from "@/lib/api"
import { cn } from "@/lib/utils"

export type RobotState = "idle" | "arming" | "performing" | "damping" | "stopped" | "fall"

interface StateSpec {
  label: string
  detail: string
  tone: string      // ring + accent colour classes
  chip: string      // status chip classes
  figure: string    // .robot-figure-* animation modifier
  robotColor: string
}

const SPECS: Record<RobotState, StateSpec> = {
  idle: {
    label: "Standing by",
    detail: "No policy owns the robot. Onboard 'ai' or the remote is in control.",
    tone: "ring-emerald-400/40 text-emerald-600",
    chip: "border-emerald-200 bg-emerald-50 text-emerald-700",
    figure: "robot-figure--idle", robotColor: "text-emerald-600",
  },
  arming: {
    label: "Arming",
    detail: "Moving to the default pose and taking over control. Keep the remote in hand.",
    tone: "ring-amber-400/50 text-amber-600",
    chip: "border-amber-200 bg-amber-50 text-amber-800",
    figure: "", robotColor: "text-amber-500",
  },
  performing: {
    label: "Performing",
    detail: "The dance policy is actively controlling all joints.",
    tone: "ring-blue-400/50 text-blue-600",
    chip: "border-blue-200 bg-blue-50 text-blue-700",
    figure: "robot-figure--perform", robotColor: "text-blue-600",
  },
  damping: {
    label: "Damping / handoff",
    detail: "Ramping to soft damping and handing back to onboard. The robot may sag.",
    tone: "ring-slate-400/40 text-slate-500",
    chip: "border-slate-200 bg-slate-100 text-slate-600",
    figure: "robot-figure--idle", robotColor: "text-slate-500",
  },
  stopped: {
    label: "Stopped",
    detail: "The run ended or was stopped. Record the outcome before the next run.",
    tone: "ring-slate-400/40 text-slate-500",
    chip: "border-slate-200 bg-slate-100 text-slate-600",
    figure: "robot-figure--idle", robotColor: "text-slate-500",
  },
  fall: {
    label: "FALL / INSTABILITY",
    detail: "The fall detector tripped, or the robot cannot find stable ground. Damp it NOW — E-STOP + remote B-damp.",
    tone: "ring-red-500/60 text-red-600",
    chip: "border-red-300 bg-red-50 text-red-700",
    figure: "robot-figure--thrash", robotColor: "text-red-600",
  },
}

/** Coarse robot state derived from the live run status. Deliberately not detailed —
 *  it answers "what is the robot doing and is it safe" at a glance. */
export function robotStateFrom(run: RunStatus): RobotState {
  if (run.fall_detected) return "fall"
  switch (run.phase) {
    case "fall": return "fall"
    case "stopped": return "stopped"
    case "performing": return "performing"
    case "arming":
    case "launching": return "arming"
    case "ramp-to-damping": return "damping"
    case "ended": return "stopped"
    default: return "idle"
  }
}

export function RobotStateViz({ run, reachable, className }: { run: RunStatus; reachable?: boolean; className?: string }) {
  const state = robotStateFrom(run)
  const spec = SPECS[state]
  const alarm = state === "fall"
  return (
    <div className={cn("flex items-center gap-5 rounded-xl border p-5", alarm ? "border-red-300 bg-red-50" : "border-slate-200 bg-white", className)} data-testid="robot-state-viz" data-robot-state={state}>
      <div className={cn("relative flex h-28 w-24 shrink-0 items-center justify-center rounded-xl ring-2 ring-inset", spec.tone, alarm ? "bg-red-100/60" : "bg-slate-50")}>
        <CuteRobot className={cn("robot-figure h-24", spec.figure, spec.robotColor)} />
      </div>
      <div className="min-w-0 flex-1">
        <div className="flex flex-wrap items-center gap-2">
          <span className={cn("inline-flex items-center rounded-full border px-2.5 py-0.5 text-[11px] font-bold uppercase tracking-wide", spec.chip)}>{spec.label}</span>
          {run.running && <span className="inline-flex items-center gap-1 rounded-full border border-blue-200 bg-blue-50 px-2 py-0.5 text-[10px] font-semibold text-blue-700"><span className="h-1.5 w-1.5 animate-pulse rounded-full bg-blue-500" />live</span>}
          <span className={cn("inline-flex items-center gap-1 text-[10px] font-semibold", reachable ? "text-emerald-600" : "text-slate-500")}>
            {reachable ? <Wifi className="h-3 w-3" /> : <WifiOff className="h-3 w-3" />}
            PC2 {reachable == null ? "—" : reachable ? "reachable" : "no ping"}
          </span>
        </div>
        <p className={cn("mt-2 text-xs leading-5", alarm ? "font-semibold text-red-800" : "text-slate-600")}>{spec.detail}</p>
        <div className="mt-2 font-mono text-[10px] text-slate-500">phase: {run.phase}{run.mode ? ` • ${run.mode}` : ""}</div>
      </div>
    </div>
  )
}

/** The software E-STOP. Calls the emergency-kill endpoint (SIGTERM → deploy_runtime damps).
 *  Always available, even when the app thinks nothing is running — the backend replies
 *  honestly about what it could and could not reach. */
export function EStopButton({ compact = false, className }: { compact?: boolean; className?: string }) {
  const queryClient = useQueryClient()
  const estop = useMutation({
    mutationFn: () => api.send<{ stopped: boolean; detail: string }>("/api/safety/estop", "POST"),
    onSuccess: (result) => {
      result.stopped ? toast.warning(result.detail, { duration: 8_000 }) : toast.info(result.detail, { duration: 8_000 })
      queryClient.invalidateQueries({ queryKey: ["current-run"] })
      queryClient.invalidateQueries({ queryKey: ["safety-status"] })
    },
    onError: (error: Error) => toast.error(error.message),
  })
  if (compact) {
    return (
      <Button variant="destructive" size="sm" onClick={() => estop.mutate()} disabled={estop.isPending} data-testid="estop-compact" title="Emergency software stop — damps any app-launched policy run" className={cn("font-black tracking-wide estop-pulse", className)}>
        <OctagonX /> E-STOP
      </Button>
    )
  }
  return (
    <Button variant="destructive" size="lg" onClick={() => estop.mutate()} disabled={estop.isPending} data-testid="estop-button" className={cn("h-16 w-full text-lg font-black tracking-wide estop-pulse", className)}>
      <OctagonX className="h-6 w-6" /> {estop.isPending ? "STOPPING…" : "EMERGENCY STOP — DAMP ROBOT"}
    </Button>
  )
}
