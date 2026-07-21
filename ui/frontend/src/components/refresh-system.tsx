import { useEffect, useState } from "react"
import { useMutation, useQueryClient } from "@tanstack/react-query"
import { RefreshCw, Wifi, WifiOff } from "lucide-react"
import { toast } from "sonner"
import { api, type SystemStatus } from "@/lib/api"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

function agoLabel(ts: number | null): string {
  if (!ts) return "not synced yet"
  const s = Math.max(0, Math.round((Date.now() - ts) / 1000))
  if (s < 2) return "synced just now"
  if (s < 60) return `synced ${s}s ago`
  const m = Math.floor(s / 60)
  return `synced ${m}m ${s % 60}s ago`
}

/**
 * Global sync control for the operator console. Forces an immediate box
 * telemetry snapshot (POST /api/system/refresh, bypassing the cached 8 s loop)
 * AND refetches dances + jobs, so a newly published policy / finished run shows
 * up at once. Gives explicit feedback — spinner, a result toast, and a live
 * "synced Ns ago" timestamp — so it is never ambiguous whether it worked.
 */
export function RefreshSystemButton({ className, showTimestamp = true }:
  { className?: string; showTimestamp?: boolean }) {
  const queryClient = useQueryClient()
  const [lastSync, setLastSync] = useState<number | null>(null)
  const [, forceTick] = useState(0)

  // re-render every second so the "synced Ns ago" label stays live
  useEffect(() => {
    const t = window.setInterval(() => forceTick((n) => n + 1), 1000)
    return () => window.clearInterval(t)
  }, [])

  const sync = useMutation({
    mutationFn: async () => {
      const snap = await api.send<SystemStatus>("/api/system/refresh", "POST")
      // pull the rest of the console in parallel so everything is fresh at once
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: ["dances"] }),
        queryClient.invalidateQueries({ queryKey: ["jobs"] }),
      ])
      return snap
    },
    onSuccess: (snap) => {
      queryClient.setQueryData(["system"], snap)
      setLastSync(Date.now())
      if (snap?.reachable) {
        const gpu = snap.gpu?.utilization_pct
        const job = snap.jobs?.[0]
        toast.success("Synced", {
          description: job
            ? `Box connected · ${job.name} @ ${job.iteration?.toLocaleString() ?? "—"} iters${gpu != null ? ` · GPU ${Math.round(gpu)}%` : ""}`
            : `Box connected${gpu != null ? ` · GPU ${Math.round(gpu)}%` : ""}`,
        })
      } else {
        toast.warning("Box unreachable", {
          description: snap?.detail?.slice(0, 140) ?? "No response from the GPU box. Showing the last known state.",
        })
      }
    },
    onError: (e: Error) => toast.error("Sync failed", { description: e.message }),
  })

  const reachable = queryClient.getQueryData<SystemStatus>(["system"])?.reachable

  return (
    <div className={cn("flex items-center gap-2", className)}>
      {showTimestamp && (
        <span className="hidden items-center gap-1 text-[11px] text-muted-foreground sm:flex">
          {reachable ? <Wifi className="h-3 w-3 text-emerald-400" /> : <WifiOff className="h-3 w-3 text-amber-400" />}
          {agoLabel(lastSync)}
        </span>
      )}
      <Button
        variant="outline"
        size="sm"
        disabled={sync.isPending}
        onClick={() => sync.mutate()}
        title="Poll the GPU box now and refresh dances/jobs"
      >
        <RefreshCw className={cn("h-3.5 w-3.5", sync.isPending && "animate-spin")} />
        {sync.isPending ? "Syncing…" : "Sync"}
      </Button>
    </div>
  )
}
