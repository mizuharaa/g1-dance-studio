import { useQuery } from "@tanstack/react-query"
import { AlertTriangle, Footprints, Gamepad2, Hand, Power, ShieldAlert, Wrench } from "lucide-react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { PageHeader } from "@/components/console-ui"
import { EStopButton, RobotStateViz } from "@/components/robot-state"
import type { ConsoleData } from "@/hooks/use-console-data"
import { api, type RunStatus } from "@/lib/api"

interface SafetyStatus { robot_reachable: boolean; run: RunStatus }

// The reminders that matter before switching the robot from damping into a policy — the
// top one is the direct fix for the "thrashes 360° with limbs flying" failure the operator
// hit: a suspended / barely-grounded robot cannot find the ground its balancer keeps
// reaching for.
const REMINDERS: Array<{ icon: typeof Footprints; title: string; body: string; tone: "danger" | "warn" }> = [
  {
    icon: Footprints, tone: "danger",
    title: "Feet flat and fully loaded on the ground BEFORE you arm",
    body: "A robot on a hoist, hung in the air, or barely touching the floor will thrash violently when it enters a policy/walk mode — its balancer keeps driving the legs to find ground it can’t reach. Only leave damping for a policy with both feet planted on flat, solid ground.",
  },
  {
    icon: Hand, tone: "danger",
    title: "Physical damping remote in hand for the whole run",
    body: "This tetherless G1 has NO torque-cut hardware e-stop. The remote’s B-damp is the primary hard stop. Keep it in your hand from arm to hand-off.",
  },
  {
    icon: AlertTriangle, tone: "warn",
    title: "If it starts thrashing: E-STOP here + B-damp on the remote, immediately",
    body: "Don’t wait for it to settle — it won’t. Hit the software E-STOP and the remote B-damp together, then lift/steady the robot before trying again from a stable, grounded start pose.",
  },
  {
    icon: Wrench, tone: "warn",
    title: "Start upright, on a clear flat 2 m area",
    body: "The runtime refuses to start if the torso isn’t upright (>~32° tilt). Stand the robot straight, clear the area, and confirm the venue radius before every run.",
  },
]

export function SafetyScreen({ data }: { data: ConsoleData }) {
  const status = useQuery({
    queryKey: ["safety-status"],
    queryFn: () => api.get<SafetyStatus>("/api/safety/status"),
    refetchInterval: 8_000,
  })
  const reachable = status.data?.robot_reachable
  const run = data.run

  return (
    <div>
      <PageHeader eyebrow="Operate" title="Safety & E-stop"
        description="Live robot state, the pre-arm reminders that keep a run safe, and an always-available software emergency stop." />

      <div className="grid items-start gap-4 xl:grid-cols-[minmax(0,1.1fr)_minmax(320px,.9fr)]">
        {/* left: state + the E-STOP */}
        <div className="space-y-4">
          <Card>
            <CardHeader className="pb-3"><div className="panel-kicker"><ShieldAlert className="text-blue-500" /> Robot state</div><CardTitle className="mt-2">What the robot is doing now</CardTitle></CardHeader>
            <CardContent className="space-y-4">
              <RobotStateViz run={run} reachable={reachable} />
              <div className="rounded-xl border-2 border-red-300 bg-red-50 p-4">
                <div className="flex items-start gap-3">
                  <div className="rounded-lg bg-red-600 p-2 text-white"><Power className="h-5 w-5" /></div>
                  <div className="min-w-0">
                    <div className="text-sm font-black text-red-900">Software emergency stop</div>
                    <p className="mt-1 text-[11px] leading-5 text-red-800/80">Damps any policy run this app launched (SIGTERM → the runtime damps soft). The robot goes limp and may sag — keep the remote in hand.</p>
                  </div>
                </div>
                <div className="mt-3"><EStopButton /></div>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader className="pb-3"><div className="panel-kicker"><Gamepad2 className="text-blue-500" /> What this can and can’t stop</div><CardTitle className="mt-2">Honest scope</CardTitle></CardHeader>
            <CardContent className="space-y-2 text-xs leading-5 text-slate-600">
              <div className="flex gap-2"><span className="font-bold text-emerald-600">CAN</span><span>damp a dance/show that you started from this app — the tracked run and any stray deploy process.</span></div>
              <div className="flex gap-2"><span className="font-bold text-red-600">CAN’T</span><span>stop the robot when it’s driven from the hand remote or onboard ‘ai’ — the app has no channel to those. Use the remote B-damp or the power switch.</span></div>
              <div className="flex gap-2"><span className="font-bold text-slate-500">ALWAYS</span><span>the hand-held remote B-damp is the primary hard stop. This button is a second, software stop — not a replacement.</span></div>
            </CardContent>
          </Card>
        </div>

        {/* right: pre-arm reminders */}
        <Card className="border-amber-300/60">
          <CardHeader className="pb-3"><div className="panel-kicker text-amber-600"><AlertTriangle /> Before you arm</div><CardTitle className="mt-2">Pre-arm safety reminders</CardTitle></CardHeader>
          <CardContent className="space-y-3">
            {REMINDERS.map((reminder) => {
              const Icon = reminder.icon
              const danger = reminder.tone === "danger"
              return (
                <div key={reminder.title} className={`rounded-xl border p-3 ${danger ? "border-red-200 bg-red-50" : "border-amber-200 bg-amber-50"}`}>
                  <div className="flex items-start gap-3">
                    <div className={`mt-0.5 shrink-0 rounded-lg p-2 ${danger ? "bg-red-600 text-white" : "bg-amber-500 text-white"}`}><Icon className="h-4 w-4" /></div>
                    <div className="min-w-0">
                      <div className={`text-xs font-bold ${danger ? "text-red-900" : "text-amber-900"}`}>{reminder.title}</div>
                      <p className={`mt-1 text-[11px] leading-5 ${danger ? "text-red-800/80" : "text-amber-900/80"}`}>{reminder.body}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </CardContent>
        </Card>
      </div>

      {!!run.last_lines?.length && (
        <Card className="mt-4">
          <CardHeader className="pb-2"><div className="panel-kicker"><Wrench className="text-blue-500" /> Runtime log</div><CardTitle className="mt-2">Last lines from the live run</CardTitle></CardHeader>
          <CardContent>
            <ScrollArea className="h-28 rounded-lg border border-slate-200 bg-slate-950 p-3"><pre className="whitespace-pre-wrap font-mono text-[10px] leading-5 text-slate-300">{run.last_lines.join("\n")}</pre></ScrollArea>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
