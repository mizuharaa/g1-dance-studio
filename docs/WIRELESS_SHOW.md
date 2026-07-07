# Wireless Show — untethered bring-up + latency/jitter preflight

**Read `docs/ROBOT_DAY_PLAN.md` first.** Its safety truths are absolute here: this
tether-free G1 has **NO torque-cutting hardware e-stop** — your only stops are the
remote's B-damping (in your hand) and the power switch. Going wireless removes the one
thing that has always been reliable — the wired link — so it earns its own gate.

This doc covers taking the laptop→robot control link **off the ethernet cable and onto
WiFi** for an untethered show, and the preflight that must pass before you do.

---

## Why WiFi is a real risk (not just an inconvenience)

Today all control DDS traffic — `rt/lowcmd` (commands out), `rt/lowstate` (joint/IMU
state in), `rt/odommodestate` (onboard odometry) — runs on the `192.168.123.x` **wired**
ethernet. `~/robot/RUNBOOK.md` §0 is blunt: *"Laptop ↔ ROBOT = the Ethernet cable. Always
keep it plugged."* The teleop and our deploy runtime both bind CycloneDDS to that wired
interface (`enp0s31f6`).

The laptop-side balance loop (`pipeline/deploy_runtime.py`) runs at **50 Hz** and reads
`LowState` **every single tick** (20 ms budget) to build the observation and command the
next joint targets. That is a hard real-time loop feeding a 35 kg machine. Over WiFi:

- **Jitter / dropout → stale reads.** A read that arrives late means the policy acts on
  an old robot state; a late command can destabilize a balancing robot.
- **The comms-loss deadman fires.** The run loop calls `read_state(sub, timeout_s=0.5)`.
  A single read that exceeds **0.5 s** raises, and the mode's `except/finally` **damps the
  robot** (soft motors, handed back to onboard). That is the safety net — but a robot that
  drops to damping mid-dance while standing untethered can topple. The net catches a comms
  failure; it does **not** make one safe.

Wired baseline (measured on hardware, recorded in `deploy_runtime.py`): `LowState`
staleness **p95 ≈ 1.75 ms**. That is the bar WiFi has to get anywhere near.

**Conclusion: WiFi must be latency/jitter-validated over a sustained window before any
untethered wireless run.** That is what `tools/wireless_preflight.py` measures.

---

## The two networks change on WiFi (don't mix them up)

| | Wired (today) | Wireless (show) |
|---|---|---|
| Laptop ↔ robot control | ethernet `enp0s31f6`, subnet `192.168.123.x` | laptop WiFi iface, the router's subnet (e.g. `192.168.1.x`) |
| Robot's control address | PC2 = `192.168.123.164` | PC2's **WiFi** IP on the router (DHCP-assigned, different) |

Both the laptop WiFi and the robot PC2 join **your own 5 GHz router** (same band the Quest
uses per `~/robot/RUNBOOK.md` §1 — *not* company WiFi). CycloneDDS discovery is L2/multicast
on that shared subnet, so **the laptop WiFi iface and the robot's WiFi IP must be on the
same subnet.** The preflight checks this — and it deliberately fails if you point it at the
old wired `192.168.123.164`, which is a common footgun.

---

## Bring-up

### 1. Robot PC2 onto the 5 GHz router WiFi (user does this on the robot)

The robot must actually join the router WiFi and publish DDS over it:

1. Power on the router (5 GHz, no internet cable needed). Power on the robot.
2. SSH to PC2 over the wire first (`ssh unitree@192.168.123.164`, answer ROS prompt `1`).
3. Join the robot PC2's WiFi to your 5 GHz SSID (NetworkManager on PC2, e.g.
   `nmcli device wifi connect <SSID> password <PSK>`, or the Jetson's network UI).
4. Note the robot's **WiFi IP** on the router: on PC2, `hostname -I` / `ip -o -4 addr`.
   This is the `<robot-wifi-ip>` you pass to the preflight and to the runtime.
5. Keep the wired cable plugged for now — it is your fallback and your bring-up path.

> The robot side is the user's job; we cannot SSH-configure the robot from here. Everything
> below (laptop side) is what we own.

### 2. Point the laptop at the WiFi interface

Find the laptop's WiFi interface name (it is **not** `enp0s31f6`):

```bash
ip -o link            # or: nmcli device        (look for the wlp*/wlan* device that's connected)
```

The runtime reads the interface from the environment — `IFACE = os.environ.get("ROBOT_IFACE",
"enp0s31f6")` — and also accepts `--iface`. For a wireless run:

```bash
export ROBOT_IFACE=wlp0s20f3        # <-- your WiFi iface
```

`tools/show_run.sh` and `pipeline/deploy_runtime.py` both honor `ROBOT_IFACE`, so setting it
in the environment is all that's needed to move the whole show onto WiFi.

### 3. CycloneDDS notes

- The unitree SDK binds CycloneDDS to the interface passed to
  `ChannelFactoryInitialize(0, iface)` (our `make_dds(iface)`); the teleop does the same with
  `--network-interface`. Set `ROBOT_IFACE` to the WiFi iface and DDS will run over WiFi — no
  XML config needed.
- The DDS staleness measurement needs the same environment the runtime needs: the **`tv`
  conda env** (has `unitree_sdk2py` + CycloneDDS + numpy), with `CYCLONEDDS_HOME` and
  `LD_LIBRARY_PATH` exported exactly as the teleop scripts do
  (`export CYCLONEDDS_HOME="$HOME/robot/cyclonedds/install"`,
  `export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"`).
- Run the preflight **alone** — do not run it at the same time as `deploy_runtime`; both call
  `ChannelFactoryInitialize` and would contend for the interface.

---

## Run the preflight

The preflight is **read-only** with respect to the robot: it pings, and it *subscribes* to
`rt/lowstate` — it never publishes a command.

```bash
# link + ping only — quick, but NOT sufficient to authorize a wireless run:
ROBOT_IFACE=wlp0s20f3 PYTHONPATH=. python tools/wireless_preflight.py <robot-wifi-ip>

# FULL check with the real staleness window (do this before any untethered run), in tv env:
conda activate tv
export CYCLONEDDS_HOME="$HOME/robot/cyclonedds/install"
export LD_LIBRARY_PATH="$CYCLONEDDS_HOME/lib:$LD_LIBRARY_PATH"
ROBOT_IFACE=wlp0s20f3 PYTHONPATH=. python tools/wireless_preflight.py <robot-wifi-ip> \
    --dds --dds-secs 60
```

It runs three checks, escalating in fidelity:

1. **LINK** — the WiFi iface is `up` and shares the robot's subnet (else DDS never discovers
   the robot and control never starts).
2. **PING** — N ICMP pings → RTT min/avg/max/mdev + packet loss. A coarse transport proxy.
3. **STALENESS** (`--dds`, **the real metric**) — init CycloneDDS on the WiFi iface, subscribe
   to `rt/lowstate` exactly like the runtime, and measure the gap distribution between
   successive **fresh** `LowState` samples (de-duplicated by the robot's `tick`) over a
   sustained window. This is precisely the staleness the 50 Hz loop experiences.

Exit code: **0 = GO, 1 = NO-GO, 2 = error.** JSON via `--json`.

**Ping alone is never a GO.** RTT is a proxy; the loop lives or dies on `LowState`
staleness, so a wireless run is only authorized after the `--dds` window passes. A ping-only
run therefore returns NO-GO by design.

---

## GO / NO-GO thresholds — and why

All thresholds derive from one fact: the loop wants a **fresh `LowState` every 20 ms tick,
with margin, and must never approach the 0.5 s deadman.** (`tools/wireless_preflight.py`,
`class Thresholds`; each is CLI-overridable, e.g. `--max-p99-ms`.)

| Check | Threshold | Why |
|---|---|---|
| Packet loss | **0%** | A dropped packet is a missed tick / stale read. Zero tolerance over the window. |
| Ping RTT avg / max / mdev | **≤ 10 / 50 / 10 ms** | Coarse proxy; kept under half a tick (avg/jitter) with a hard worst-case ceiling. |
| **DDS staleness p99** | **≤ 10 ms** | **The real gate.** p99 under *half* a tick → almost every tick has a current sample. Compare the wired p95 ≈ 1.75 ms. |
| DDS staleness max | **≤ 20 ms** | No single gap may exceed one full control tick. |
| DDS deadman trips (gap ≥ 0.5 s) | **0** | A single trip is a live comms-loss damp — categorically unacceptable untethered. |
| DDS read misses | **0** | Any read returning no sample within the poll window is early evidence of dropout. |
| DDS window length | **≥ 200 gaps** | A short sample can't be trusted at the tail; use `--dds-secs 60` (longer = better) so p99/max reflect sustained behavior, not a lucky second. |

A GO requires **all** of: iface up, same subnet, ping clean, and the DDS staleness window
within thresholds. Anything else is NO-GO. Tune thresholds only with a written justification
(measurement discipline, per `CLAUDE.md`).

### The comms-loss deadman is the net, not the plan

`read_state(timeout_s=0.5)` → damp is the backstop for a link that fails *during* a run. The
preflight exists so you never rely on it: you validate that the link stays far away from
0.5 s (p99 under 10 ms) across a sustained window **before** trusting the robot to it. If the
preflight can't clear that bar, the WiFi is not show-ready — full stop.

---

## Standing recommendation (do not skip)

1. **Validate on the tether first.** Bring the robot up wireless while it is still on the
   gantry/tether, run the preflight, and do a tethered wireless rehearsal. Earn the wireless
   gate the same way every other gate in `docs/ROBOT_DAY_PLAN.md` is earned.
2. **Keep the wired cable as a fallback.** Do not unplug ethernet until wireless has passed
   the preflight *and* a tethered wireless run. If anything looks marginal, plug back in.
3. **Never run wireless untethered until the preflight passes over a sustained window**
   (`--dds --dds-secs 60`, GO). A single clean second is not a pass.
4. **Remote (B-damping) in hand, abort at the first twitch/sag/buzz** — the one safety truth
   from `docs/ROBOT_DAY_PLAN.md` does not relax on WiFi; it matters more.

A passing preflight is **necessary, not sufficient**. It says the link *was* good over the
window you measured; RF conditions change (people, microwaves, other 5 GHz traffic). Re-run
it at the venue, on the day, on the actual channel — not just at the bench.
