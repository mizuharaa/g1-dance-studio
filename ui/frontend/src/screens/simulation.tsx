import { useEffect, useMemo, useState } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Clapperboard, GitCompare, Loader2, Play, VideoOff } from "lucide-react"
import { toast } from "sonner"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { EmptyState, PageHeader } from "@/components/console-ui"
import { PreviewPlayer } from "@/components/robot-preview"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type Dance } from "@/lib/api"
import { cn, fmtDate, fmtPercent } from "@/lib/utils"

interface SimVersion {
  sha: string
  url: string | null
  achieved?: number | null
  created_at?: number | null
  policy_sha256?: string | null
  status: string
}

function pct(v?: number | null) {
  return v == null ? "—" : fmtPercent(v)
}

function SimVideo({ v, tag }: { v: SimVersion; tag: string }) {
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-3 py-2">
        <div className="flex items-center gap-2 text-xs font-bold text-slate-900">
          {tag}
          <span className="font-mono text-[10px] font-medium text-slate-400">{v.sha}</span>
        </div>
        <Badge className={cn((v.achieved ?? 0) > 0.85 ? "bg-emerald-100 text-emerald-700" : "bg-amber-100 text-amber-700")}>
          dances {pct(v.achieved)} of the motion
        </Badge>
      </div>
      <div className="bg-slate-950 p-2">
        {v.url ? (
          <PreviewPlayer url={v.url} autoPlay={false} testId="sim-video" />
        ) : (
          <div className="flex h-56 items-center justify-center text-xs text-slate-400">
            <VideoOff className="mr-2 h-4 w-4" /> video unavailable — re-render
          </div>
        )}
      </div>
      <div className="px-3 py-2 text-[10px] text-slate-500">
        Left = reference (intended). Right = policy (what the robot actually does).
        {v.created_at ? ` • rendered ${fmtDate(v.created_at)}` : ""}
      </div>
    </div>
  )
}

export function SimulationScreen({ data }: { data: ConsoleData }) {
  const dances = useMemo(() => data.dances.filter((d) => d.policy_path), [data.dances])
  const [selectedId, setSelectedId] = useState<string>("")
  const [compare, setCompare] = useState(false)
  const qc = useQueryClient()

  useEffect(() => {
    if (!selectedId && dances.length) setSelectedId(dances[0].id)
  }, [dances, selectedId])

  const sims = useQuery({
    queryKey: ["sims", selectedId],
    queryFn: () => api.get<{ sims: SimVersion[] }>(`/api/dances/${selectedId}/sims`),
    enabled: !!selectedId,
    refetchInterval: (query) =>
      query.state.data?.sims.some((s) => s.status === "rendering") ? 4000 : false,
  })

  const render = useMutation({
    mutationFn: () => api.send(`/api/dances/${selectedId}/sim`, "POST"),
    onSuccess: () => {
      toast.success("Rendering simulation — this takes ~1–2 min")
      qc.invalidateQueries({ queryKey: ["sims", selectedId] })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const versions = sims.data?.sims ?? []
  const ready = versions.filter((v) => v.status === "ready" && v.url)
  const rendering = versions.some((v) => v.status === "rendering")
  const failed = versions.find((v) => v.status.startsWith("failed"))
  const selected = dances.find((d) => d.id === selectedId) as Dance | undefined

  if (!dances.length) {
    return (
      <div className="space-y-6">
        <PageHeader title="Simulation" description="Preview what the robot actually does vs the reference." />
        <EmptyState title="No trained dances yet"
          body="Train a dance in the Pipeline, then simulate it here to see what the robot will really do." />
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <PageHeader title="Simulation"
        description="Policy-in-the-loop preview — reference (intended) vs policy (actual robot). Every training keeps its own version, so you can compare before vs after." />

      <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-2.5 text-xs text-amber-800">
        ⚠️ Sim not yet calibrated to the training model — it currently <b>under-represents</b> the
        dance vs hardware. Use it to compare policies (before/after) and spot which joints move,
        not as an exact preview of the real robot.
      </div>

      <div className="flex flex-wrap items-center gap-2">
        {dances.map((d) => (
          <button key={d.id} onClick={() => { setSelectedId(d.id); setCompare(false) }}
            className={cn("rounded-lg border px-3 py-2 text-xs font-semibold transition-all",
              selectedId === d.id ? "border-blue-300 bg-blue-50 text-blue-800 shadow-sm"
                : "border-slate-200 bg-white text-slate-600 hover:bg-slate-50")}>
            {d.name}
          </button>
        ))}
      </div>

      <Card>
        <CardHeader className="flex flex-row items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Clapperboard className="h-4 w-4 text-blue-600" />
            {selected?.name ?? "Select a dance"}
            <Badge variant="outline" className="ml-1">{ready.length} version{ready.length === 1 ? "" : "s"}</Badge>
          </CardTitle>
          <div className="flex items-center gap-2">
            {ready.length >= 2 && (
              <Button size="sm" variant={compare ? "default" : "outline"} onClick={() => setCompare((c) => !c)}>
                <GitCompare className="h-4 w-4" /> Before / after
              </Button>
            )}
            <Button size="sm" onClick={() => render.mutate()} disabled={rendering || render.isPending}>
              {rendering ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
              {rendering ? "Rendering…" : ready.length ? "Re-render" : "Render simulation"}
            </Button>
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {failed && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              Render failed: {failed.status.replace("failed:", "")}
            </div>
          )}
          {rendering && (
            <div className="flex items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800">
              <Loader2 className="h-4 w-4 animate-spin" />
              Simulating the policy in MuJoCo (~1–2 min)… this page updates automatically.
            </div>
          )}
          {!ready.length && !rendering && (
            <EmptyState title="No simulation yet"
              body='Click "Render simulation" to run the current policy in the sandbox and preview it.' />
          )}

          {ready.length >= 2 && compare ? (
            <div className="grid gap-4 lg:grid-cols-2">
              <SimVideo v={ready[1]} tag="BEFORE (previous policy)" />
              <SimVideo v={ready[0]} tag="AFTER (current policy)" />
            </div>
          ) : (
            ready[0] && <SimVideo v={ready[0]} tag="Current policy" />
          )}

          {ready.length > 0 && (
            <div className="rounded-lg border border-slate-200 bg-slate-50 p-3">
              <div className="mb-2 text-[10px] font-bold uppercase tracking-wide text-slate-400">Version history</div>
              <div className="space-y-1">
                {ready.map((v, i) => (
                  <div key={v.sha} className="flex items-center justify-between text-xs">
                    <span className="font-mono text-slate-500">{v.sha}{i === 0 ? "  (current)" : ""}</span>
                    <span className="font-semibold text-slate-700">dances {pct(v.achieved)} of the motion</span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
