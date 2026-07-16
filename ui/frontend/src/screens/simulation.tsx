import { useEffect, useMemo, useRef, useState, type ComponentType, type ReactNode } from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { Clapperboard, Film, GitCompare, Layers, Loader2, Play, ScanFace, SquareStack, VideoOff } from "lucide-react"
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
  overlay_url?: string | null
  vs_original_url?: string | null
  achieved?: number | null
  created_at?: number | null
  policy_sha256?: string | null
  status: string
}

type ViewMode = "sbs" | "overlay" | "vsoriginal" | "landmark"

function pct(v?: number | null) {
  return v == null ? "—" : fmtPercent(v)
}

/** Keep two <video> elements time-locked: play/pause/seek on either mirrors to the other. */
function useTimeLock(a: HTMLVideoElement | null, b: HTMLVideoElement | null) {
  useEffect(() => {
    if (!a || !b) return
    let syncing = false
    const mirror = (from: HTMLVideoElement, to: HTMLVideoElement) => {
      if (syncing) return
      syncing = true
      if (Math.abs(to.currentTime - from.currentTime) > 0.2) to.currentTime = from.currentTime
      queueMicrotask(() => { syncing = false })
    }
    const onPlay = (from: HTMLVideoElement, to: HTMLVideoElement) => () => { to.play().catch(() => {}) }
    const onPause = (from: HTMLVideoElement, to: HTMLVideoElement) => () => { to.pause() }
    const onSeek = (from: HTMLVideoElement, to: HTMLVideoElement) => () => mirror(from, to)
    const aPlay = onPlay(a, b), aPause = onPause(a, b), aSeek = onSeek(a, b)
    const bPlay = onPlay(b, a), bPause = onPause(b, a), bSeek = onSeek(b, a)
    a.addEventListener("play", aPlay); a.addEventListener("pause", aPause); a.addEventListener("seeking", aSeek); a.addEventListener("timeupdate", aSeek)
    b.addEventListener("play", bPlay); b.addEventListener("pause", bPause); b.addEventListener("seeking", bSeek)
    return () => {
      a.removeEventListener("play", aPlay); a.removeEventListener("pause", aPause); a.removeEventListener("seeking", aSeek); a.removeEventListener("timeupdate", aSeek)
      b.removeEventListener("play", bPlay); b.removeEventListener("pause", bPause); b.removeEventListener("seeking", bSeek)
    }
  }, [a, b])
}

function VideoFrame({ url, label, caption, tone = "slate", autoPlay = false, onVideo }:
  { url: string; label: ReactNode; caption?: ReactNode; tone?: "slate" | "emerald" | "amber"; autoPlay?: boolean; onVideo?: (el: HTMLVideoElement | null) => void }) {
  const toneCls = tone === "emerald" ? "text-emerald-700" : tone === "amber" ? "text-amber-700" : "text-slate-900"
  return (
    <div className="overflow-hidden rounded-xl border border-slate-200 bg-white shadow-sm">
      <div className="flex items-center justify-between gap-2 border-b border-slate-100 px-4 py-2.5">
        <div className={cn("flex items-center gap-2 text-sm font-bold", toneCls)}>{label}</div>
      </div>
      <div className="bg-slate-950 p-2">
        <PreviewPlayer url={url} autoPlay={autoPlay} testId="sim-video" onVideo={onVideo} />
      </div>
      {caption && <div className="px-4 py-2.5 text-[11px] leading-4 text-slate-500">{caption}</div>}
    </div>
  )
}

function Unavailable({ text }: { text: string }) {
  return (
    <div className="flex h-64 items-center justify-center rounded-xl border border-dashed border-slate-300 bg-slate-50 text-sm text-slate-400">
      <VideoOff className="mr-2 h-4 w-4" /> {text}
    </div>
  )
}

function SegToggle({ value, onChange, options }:
  { value: string; onChange: (v: string) => void; options: { id: string; label: string; icon: ComponentType<{ className?: string }> }[] }) {
  return (
    <div className="inline-flex rounded-lg border border-slate-200 bg-slate-50 p-1">
      {options.map((o) => {
        const Icon = o.icon
        return (
          <button key={o.id} onClick={() => onChange(o.id)}
            className={cn("flex items-center gap-1.5 rounded-md px-3 py-1.5 text-xs font-semibold transition-all",
              value === o.id ? "bg-white text-blue-700 shadow-sm ring-1 ring-inset ring-blue-200" : "text-slate-500 hover:text-slate-800")}>
            <Icon className="h-3.5 w-3.5" /> {o.label}
          </button>
        )
      })}
    </div>
  )
}

export function SimulationScreen({ data }: { data: ConsoleData }) {
  const dances = useMemo(() => data.dances.filter((d) => d.policy_path), [data.dances])
  const [selectedId, setSelectedId] = useState<string>("")
  const [view, setView] = useState<ViewMode>("sbs")
  const [compare, setCompare] = useState(false)
  const qc = useQueryClient()
  const lmRef = useRef<HTMLVideoElement | null>(null)
  const simRef = useRef<HTMLVideoElement | null>(null)
  const [, force] = useState(0)
  useTimeLock(lmRef.current, simRef.current)

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

  const landmark = useQuery({
    queryKey: ["landmark", selectedId],
    queryFn: () => api.get<{ status: string; url?: string; reason?: string }>(`/api/dances/${selectedId}/landmark`),
    enabled: !!selectedId && view === "landmark",
    refetchInterval: (query) => query.state.data?.status === "rendering" ? 4000 : false,
  })

  const render = useMutation({
    mutationFn: () => api.send(`/api/dances/${selectedId}/sim`, "POST"),
    onSuccess: () => {
      toast.success("Rendering simulation — this takes ~1–2 min")
      qc.invalidateQueries({ queryKey: ["sims", selectedId] })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const renderLm = useMutation({
    mutationFn: () => api.send(`/api/dances/${selectedId}/landmark`, "POST"),
    onSuccess: () => {
      toast.success("Rendering landmark overlay — this takes ~1–2 min")
      qc.invalidateQueries({ queryKey: ["landmark", selectedId] })
    },
    onError: (e: Error) => toast.error(e.message),
  })

  const versions = sims.data?.sims ?? []
  const ready = versions.filter((v) => v.status === "ready" && v.url)
  const rendering = versions.some((v) => v.status === "rendering")
  const failed = versions.find((v) => v.status.startsWith("failed"))
  const selected = dances.find((d) => d.id === selectedId) as Dance | undefined
  const cur = ready[0]

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
        description="See what the robot actually does vs the dance you intended — side by side, overlaid in one scene, or against the pose-estimation landmarks." />

      <div className="rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
        <b>⚠️ Uncalibrated preview.</b> This sim uses a sandbox model that <b>under-represents hardware</b> —
        it currently shows less motion than the real robot. Use it to compare policies (before/after) and to
        spot which joints move or where the policy falls, not as an exact preview of the real robot.
        (A faithful mjlab model will be swapped in later.)
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
        <CardHeader className="flex flex-col gap-3 lg:flex-row lg:items-center lg:justify-between">
          <CardTitle className="flex items-center gap-2 text-base">
            <Clapperboard className="h-4 w-4 text-blue-600" />
            {selected?.name ?? "Select a dance"}
            <Badge variant="outline" className="ml-1">{ready.length} version{ready.length === 1 ? "" : "s"}</Badge>
          </CardTitle>
          <div className="flex flex-wrap items-center gap-2">
            <SegToggle value={view} onChange={(v) => setView(v as ViewMode)} options={[
              { id: "sbs", label: "Side by side", icon: SquareStack },
              { id: "overlay", label: "Overlay", icon: Layers },
              { id: "vsoriginal", label: "vs Original", icon: Film },
              { id: "landmark", label: "Landmarks", icon: ScanFace },
            ]} />
            {view === "sbs" && ready.length >= 2 && (
              <Button size="sm" variant={compare ? "default" : "outline"} onClick={() => setCompare((c) => !c)}>
                <GitCompare className="h-4 w-4" /> Before / after
              </Button>
            )}
            {view !== "landmark" && (
              <Button size="sm" onClick={() => render.mutate()} disabled={rendering || render.isPending}>
                {rendering ? <Loader2 className="h-4 w-4 animate-spin" /> : <Play className="h-4 w-4" />}
                {rendering ? "Rendering…" : ready.length ? "Re-render" : "Render simulation"}
              </Button>
            )}
          </div>
        </CardHeader>
        <CardContent className="space-y-4">
          {failed && view !== "landmark" && (
            <div className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-xs text-red-700">
              Render failed: {failed.status.replace("failed:", "")}
            </div>
          )}
          {rendering && view !== "landmark" && (
            <div className="flex items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800">
              <Loader2 className="h-4 w-4 animate-spin" />
              Simulating the policy in MuJoCo (~1–2 min)… this page updates automatically.
            </div>
          )}

          {/* ---- SIDE BY SIDE ---- */}
          {view === "sbs" && (
            !ready.length && !rendering ? (
              <EmptyState title="No simulation yet"
                body='Click "Render simulation" to run the current policy in the sandbox and preview it.' />
            ) : ready.length >= 2 && compare ? (
              <div className="grid gap-4 lg:grid-cols-2">
                <VideoFrame url={ready[1].url!} tone="amber"
                  label={<>BEFORE (previous policy) <span className="font-mono text-[10px] text-slate-400">{ready[1].sha}</span></>}
                  caption={<>Left = reference (intended) · Right = policy. dances {pct(ready[1].achieved)} of the motion.</>} />
                <VideoFrame url={ready[0].url!} tone="emerald"
                  label={<>AFTER (current policy) <span className="font-mono text-[10px] text-slate-400">{ready[0].sha}</span></>}
                  caption={<>Left = reference (intended) · Right = policy. dances {pct(ready[0].achieved)} of the motion.</>} />
              </div>
            ) : cur ? (
              <VideoFrame url={cur.url!} autoPlay={false}
                label={<><span className="text-slate-500">Reference</span> <span className="text-slate-300">|</span> <span className="text-blue-700">Policy</span></>}
                caption={<>
                  <b>Left = the intended (reference) dance. Right = what the policy actually does.</b>{" "}
                  dances {pct(cur.achieved)} of the motion.{cur.created_at ? ` · rendered ${fmtDate(cur.created_at)}` : ""}
                </>} />
            ) : null
          )}

          {/* ---- OVERLAY (same scene, color-coded) ---- */}
          {view === "overlay" && (
            !cur ? (
              <EmptyState title="No simulation yet"
                body='Click "Render simulation" to produce the overlay view.' />
            ) : cur.overlay_url ? (
              <VideoFrame url={cur.overlay_url} autoPlay={false}
                label={<><span className="text-emerald-600">● Reference (intended)</span> <span className="text-slate-300">+</span> <span className="text-blue-700">● Policy (actual)</span> — same scene</>}
                caption={<>
                  Both dances rendered in <b>one shared scene</b>: the <b className="text-emerald-600">green ghost</b> is
                  the intended reference, the <b className="text-blue-700">solid blue</b> robot is the actual policy.
                  Where they separate is exactly where the robot diverges from the choreography.
                </>} />
            ) : (
              <Unavailable text="Overlay not in this version — click Re-render to produce it." />
            )
          )}


          {/* ---- VS ORIGINAL (real dancer footage | robot) ---- */}
          {view === "vsoriginal" && (
            !cur ? (
              <EmptyState title="No simulation yet"
                body='Render a simulation first — the vs-Original view pairs it with the source video.' />
            ) : cur.vs_original_url ? (
              <VideoFrame url={cur.vs_original_url} autoPlay={false}
                label={<><span className="text-rose-600">Original dancer (source video)</span> <span className="text-slate-300">|</span> <span className="text-blue-700">Robot</span></>}
                caption={<>
                  <b>Left = the real dancer from the uploaded video. Right = the robot performing the same choreography.</b>{" "}
                  Timing alignment is approximate (the robot's motion includes a standing lead-in).
                </>} />
            ) : (
              <Unavailable text="No vs-Original comparison for this policy yet — it's generated for dances with source footage." />
            )
          )}

          {/* ---- LANDMARK (pose-estimation debug) ---- */}
          {view === "landmark" && (
            <div className="space-y-3">
              <div className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-2.5 text-xs text-slate-600">
                Pose-estimation debugging: the skeleton GVHMR extracted from your uploaded video, drawn back onto it.
                If the skeleton doesn’t follow the dancer, the motion was garbage-in — fix it here before trusting anything downstream.
              </div>
              {landmark.data?.status === "unavailable" ? (
                <Unavailable text={landmark.data.reason || "No pose-estimation output — this dance wasn’t video-sourced."} />
              ) : landmark.data?.status === "rendering" ? (
                <div className="flex items-center gap-3 rounded-lg border border-blue-200 bg-blue-50 px-4 py-3 text-sm text-blue-800">
                  <Loader2 className="h-4 w-4 animate-spin" /> Rendering landmark overlay (~1–2 min)… updates automatically.
                </div>
              ) : landmark.data?.status === "ready" && landmark.data.url ? (
                <div className="grid gap-4 lg:grid-cols-2">
                  <VideoFrame url={landmark.data.url} autoPlay={false} tone="emerald"
                    label={<><ScanFace className="h-4 w-4" /> Pose landmarks on your video</>}
                    caption="Green dots = tracked joints, bones = skeleton. Should hug the dancer."
                    onVideo={(el) => { lmRef.current = el; force((n) => n + 1) }} />
                  {cur?.url ? (
                    <VideoFrame url={cur.url} autoPlay={false}
                      label={<>Robot preview (reference | policy)</>}
                      caption="Time-locked with the landmark video — play or scrub either."
                      onVideo={(el) => { simRef.current = el; force((n) => n + 1) }} />
                  ) : (
                    <Unavailable text="Render a simulation to compare against the robot." />
                  )}
                </div>
              ) : (
                <div className="flex flex-col items-start gap-3">
                  <p className="text-sm text-slate-500">No landmark overlay rendered yet for this dance’s source video.</p>
                  <Button size="sm" onClick={() => renderLm.mutate()} disabled={renderLm.isPending}>
                    {renderLm.isPending ? <Loader2 className="h-4 w-4 animate-spin" /> : <ScanFace className="h-4 w-4" />}
                    Render landmark overlay
                  </Button>
                </div>
              )}
            </div>
          )}

          {ready.length > 0 && view !== "landmark" && (
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
