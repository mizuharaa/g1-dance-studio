import { useState } from "react"
import { Box, Expand, Play, VideoOff } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog"
import { cn, fmtDuration } from "@/lib/utils"

export function CuteRobot({ className }: { className?: string }) {
  return (
    <svg className={cn("cute-robot", className)} viewBox="0 0 150 220" fill="none" aria-hidden="true">
      <ellipse cx="75" cy="204" rx="45" ry="8" fill="currentColor" opacity=".12" />
      <g className="cute-robot__body">
        <rect x="51" y="22" width="48" height="40" rx="16" fill="#f8fbff" stroke="#3469c9" strokeWidth="4" />
        <path d="M47 37h5M98 37h5" stroke="#3469c9" strokeWidth="6" strokeLinecap="round" />
        <circle cx="66" cy="40" r="4" fill="#17385e" /><circle cx="84" cy="40" r="4" fill="#17385e" />
        <path d="M68 50c4 4 10 4 14 0" stroke="#3469c9" strokeWidth="2.5" strokeLinecap="round" />
        <rect x="47" y="70" width="56" height="70" rx="19" fill="#fff" stroke="#7698bd" strokeWidth="4" />
        <path d="M58 85h34" stroke="#d9e7f5" strokeWidth="8" strokeLinecap="round" />
        <circle cx="75" cy="112" r="10" fill="#2f6ee5" /><path d="M71 112h8M75 108v8" stroke="white" strokeWidth="2" strokeLinecap="round" />
        <g className="cute-robot__arm cute-robot__arm--left"><rect x="30" y="76" width="14" height="60" rx="7" fill="#d9e7f5" stroke="#7698bd" strokeWidth="3" /><circle cx="37" cy="140" r="9" fill="#f8fbff" stroke="#7698bd" strokeWidth="3" /></g>
        <g className="cute-robot__arm cute-robot__arm--right"><rect x="106" y="76" width="14" height="60" rx="7" fill="#d9e7f5" stroke="#7698bd" strokeWidth="3" /><circle cx="113" cy="140" r="9" fill="#f8fbff" stroke="#7698bd" strokeWidth="3" /></g>
        <g className="cute-robot__leg cute-robot__leg--left"><rect x="54" y="143" width="17" height="52" rx="8" fill="#d9e7f5" stroke="#7698bd" strokeWidth="3" /><path d="M48 195h25v9H48z" fill="#f8fbff" stroke="#7698bd" strokeWidth="3" strokeLinejoin="round" /></g>
        <g className="cute-robot__leg cute-robot__leg--right"><rect x="79" y="143" width="17" height="52" rx="8" fill="#d9e7f5" stroke="#7698bd" strokeWidth="3" /><path d="M77 195h25v9H77z" fill="#f8fbff" stroke="#7698bd" strokeWidth="3" strokeLinejoin="round" /></g>
      </g>
    </svg>
  )
}

export function CuteRobotMark({ className }: { className?: string }) {
  return <svg className={className} viewBox="0 0 40 40" fill="none" aria-hidden="true"><rect x="7" y="8" width="26" height="22" rx="9" fill="currentColor" /><path d="M12 8V5M28 8V5" stroke="currentColor" strokeWidth="3" strokeLinecap="round" /><circle cx="15" cy="18" r="2.5" fill="white" /><circle cx="25" cy="18" r="2.5" fill="white" /><path d="M15 24c3 2 7 2 10 0" stroke="white" strokeWidth="2" strokeLinecap="round" /><path d="M4 18h3M33 18h3" stroke="currentColor" strokeWidth="3" strokeLinecap="round" /></svg>
}

export function RobotPreview({ url, title, duration, compact = false, className }: { url?: string | null; title: string; duration?: number | null; compact?: boolean; className?: string }) {
  const [open, setOpen] = useState(false)
  const [failed, setFailed] = useState(false)
  const available = Boolean(url) && !failed
  const Stage = available ? "button" : "div"
  return <>
    <Stage data-testid={available ? "preview-open" : "preview-unavailable"} type={available ? "button" : undefined} onClick={available ? () => setOpen(true) : undefined} className={cn("robot-stage group relative w-full overflow-hidden rounded-xl border border-slate-200 bg-[#eaf4ff] text-left shadow-sm", available && "cursor-pointer hover-lift focus:outline-none focus-visible:ring-2 focus-visible:ring-blue-500", compact ? "h-32" : "h-56", className)}>
      <div className="robot-stage__sky" />
      <div className="robot-stage__floor" />
      <div className="robot-stage__ring robot-stage__ring--one" /><div className="robot-stage__ring robot-stage__ring--two" />
      <CuteRobot className={cn("absolute left-1/2 top-1/2 -translate-x-1/2 -translate-y-[46%] text-blue-700", compact ? "h-28" : "h-44")} />
      <div className="absolute left-3 top-3 flex items-center gap-2"><Badge variant="outline" className="border-white/80 bg-white/90 text-slate-700 shadow-sm"><Box className="mr-1 h-3 w-3 text-blue-600" />MuJoCo environment</Badge>{duration != null && <Badge variant="secondary" className="bg-slate-900 text-white">{fmtDuration(duration)}</Badge>}</div>
      <div className="absolute inset-x-3 bottom-3 flex items-center justify-between gap-3 rounded-lg border border-white/80 bg-white/90 p-2.5 shadow-lg backdrop-blur"><div className="min-w-0"><div className="truncate text-xs font-bold text-slate-900">{title}</div><div className="mt-0.5 text-[10px] text-slate-500">{available ? "Click to watch the robot simulation" : "Preview video not rendered yet"}</div></div><div className={cn("flex h-9 w-9 shrink-0 items-center justify-center rounded-full", available ? "bg-blue-600 text-white shadow-md transition-transform group-hover:scale-110" : "bg-slate-200 text-slate-500")}>{available ? <Play className="ml-0.5 h-4 w-4 fill-current" /> : <VideoOff className="h-4 w-4" />}</div></div>
    </Stage>
    <Dialog open={open} onOpenChange={setOpen}><DialogContent className="max-w-5xl overflow-hidden border-slate-200 bg-white p-0 text-slate-900"><DialogHeader className="border-b border-slate-200 px-6 py-4"><DialogTitle className="flex items-center gap-2"><Expand className="h-4 w-4 text-blue-600" />{title}</DialogTitle><DialogDescription>Reference-aligned robot simulation preview. Review the full motion before approving training or performance.</DialogDescription></DialogHeader><div className="bg-slate-950 p-3 sm:p-5">{url && <video data-testid="preview-video" src={url} controls autoPlay playsInline className="max-h-[70vh] w-full rounded-lg bg-black" onError={() => setFailed(true)} />}</div><div className="flex items-center justify-between px-6 py-3 text-xs text-slate-500"><span>Use the timeline to inspect foot contact, balance, and sharp transitions.</span>{url && <Button asChild size="sm" variant="outline"><a href={url} target="_blank" rel="noreferrer">Open original</a></Button>}</div></DialogContent></Dialog>
  </>
}
