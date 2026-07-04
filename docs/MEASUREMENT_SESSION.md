# Robot Measurement Session — the numbers the audit needs (run in this order)

**Why this session exists:** the first-principles audit (docs/first_principles_audit.md)
found every load-bearing hardware number is single-sample, uncommitted, or assumed —
ankle ~15 Nm (§6.2, one dirty 5 s window), "onboard stands at ~0 Nm ankle" (§6.4,
never measured), thermal 22.5 C/min (§6.7, one 34 s datapoint), obs latency (§4 #4,
never measured), tau_est semantics (§6.1), knee delivery 0.2–0.4x (§6.3). This
session captures all of them cleanly with committed tooling. Total robot time
~30–40 min plus cooldowns. **These numbers gate the sim2real retrain config —
do this before spending GPU-hours (audit §4: "do all six before training").**

Everything below records to `data/telemetry/` automatically and each capture prints
its own ANALYSIS SUMMARY + writes a `.analysis.json` — nothing depends on you
copying numbers by hand mid-session.

---

## Safety preamble (read aloud, every session)

- **Human present the whole time; remote (B-damping) in your hand** whenever motors
  can be energized. It is your ONLY trusted stop (`docs/ROBOT_DAY_PLAN.md`).
- **No motion mode runs without `export CONFIRMED_BY_HUMAN=alois`** and
  `--i-will-watch-the-robot`. The read-only capture (Steps 1) needs neither — it
  sends nothing, ever.
- **The robot always ends soft.** Every deploy_runtime mode damps on any exit path
  (Ctrl-C, `timeout` SIGTERM, crash). If it ever doesn't: power switch.
- Motion steps (2–3) are **tethered** — gantry/tether rigged, feet on ground, line
  ready to catch a fall. Clear space around the robot.
- **Abort criteria, all steps:** any motor >70 C (heartbeat prints max temp) → stop
  and cool; buzzing/oscillation/lurch → remote-damp immediately; any weirdness →
  stop, the partial capture is still saved and analyzed.
- Laptop wired on `enp0s31f6` (192.168.123.x), `tv` conda env
  (`conda activate tv`), `cd ~/g1-dance`.

---

## Step 0 — Weigh the robot as-deployed (audit §3.2a, mechanism #4)

The model says 33.34 kg; ~35 kg is assumed (Inspire hands +1.1–1.3 kg, battery,
covers). The retrain wants the **measured nominal mass**, not a guess.

- Robot **powered off** (or damped limp), bathroom/luggage scale, exactly the
  deploy configuration: battery in, hands on, covers on.
- Record: total kg, battery model, what was attached.
- Write the number into `PROJECT_STATE.md` and below in the results table.
- If no scale available: note that explicitly — do NOT let 35 kg keep masquerading
  as a measurement.

**Expected:** 34–36 kg. If it's ≥1 kg from whatever the retrain config assumes,
that's a config change, not a footnote.

---

## Step 1 — Onboard standby capture, 3 min, READ-ONLY (audit §4 exp #4 + #5, §6.1/§6.4/§6.5)

Robot standing in **normal onboard 'ai' standby** (the colleague's controller, the
exact "stands cool all day" reference condition). No tether needed — this is the
robot's normal parked state. You send NOTHING: the tool is subscribers-only.

```bash
conda activate tv && cd ~/g1-dance
python deploy/capture_stage0.py --minutes 3 --label onboard-standby
```

Heartbeat prints every 10 s (ankle tau L/R, max temp, odom presence). Ctrl-C stops
early and still saves + analyzes.

**What the ANALYSIS SUMMARY answers:**

| Section | Question | Expected if healthy | Red flag |
|---|---|---|---|
| [a] staleness | does obs latency beyond ~2 ms exist at all? (§4 #4) | p95 staleness ≤ ~2–3 ms, tick unit ~1.00 ms | p95 ≫ 5 ms, repeated-tick reads ≫ few % → center latency DR on the measured value |
| [b] torque | "onboard ~0 Nm ankle" — assumed, never measured (§6.4) | ankle_pitch mean 3–6 Nm (audit's healthy prediction; ~0 also possible) | ≥10 Nm means even the vendor controller pays a big ankle bill → posture target for the retrain changes |
| [c] thermal | baseline heating under the vendor controller (§6.5/§6.7) | < 2 C/min all motors, hottest motor named | any motor climbing fast at standby |
| tau_est semantics (§6.1) | calibrates tau_est against a known stance | leg tau_est consistent with stance statics | ankle values wildly inconsistent with any static posture |

**Abort/skip criteria:** `no LowState within 2s` → wrong iface / LAN down, fix
network first (ROBOT_DAY_PLAN First-30-min table). Odom absent is **non-fatal** and
is itself recorded.

Also note in the results table: robot posture (knee bend visible?), floor surface,
ambient temp if known.

---

## Step 2 — Tethered stand-hold at trained gains: ANKLE_TRIM_DEG sweep 0 / +3 / −3 (audit §4 exp #6, §6.2/§6.5/§6.7)

Maps **posture → ankle torque → heat** with three 2-min points. This is the exact
test class that produced the unauditable "20 Nm continuous" number — now every run
records telemetry automatically (`data/telemetry/<stamp>_stand-hold.npz`, ankle
trim recorded in `run_meta_json`).

Setup: robot tethered (line taut enough to catch, robot bearing its own weight),
feet on ground, remote in hand.

```bash
export CONFIRMED_BY_HUMAN=alois
cd ~/g1-dance && conda activate tv

# Run A — no trim (baseline):
timeout 130 python -m pipeline.deploy_runtime --mode stand-hold --i-will-watch-the-robot
# (timeout's SIGTERM is a supported stop path: the runtime damps, saves telemetry, exits.)

# ... COOL DOWN (see below) ...

# Run B — +3 deg ankle_pitch trim:
ANKLE_TRIM_DEG=3 timeout 130 python -m pipeline.deploy_runtime --mode stand-hold --i-will-watch-the-robot

# ... COOL DOWN ...

# Run C — -3 deg trim:
ANKLE_TRIM_DEG=-3 timeout 130 python -m pipeline.deploy_runtime --mode stand-hold --i-will-watch-the-robot
```

- When trim ≠ 0 the runtime prints a loud `!!` banner — **if you don't see it, the
  env var didn't take; stop and check.** Trim is clamped to ±6 deg in code.
- **Cool down between runs:** watch the previous run's thermal section; resume when
  ankle temps are back within ~2 C of their pre-run start (typically 3–5 min).
  Don't stack heat — it corrupts the C/min comparison.
- **Abort:** lean you don't like, oscillation, tether taking load → Ctrl-C /
  remote-damp. Partial runs still save.
- Note per run: tether slack or taut, and any visible posture difference.

**Expected:** three (trim, ankle tau mean/RMS, C/min) points. If −3 deg (or +3,
sign to be learned!) drops ankle RMS toward the ≤8 Nm/leg sustainable band (audit
§4 #6), a **deploy-side posture trim alone** is a real thermal lever and the
retrain's posture gate gets a measured target. If torque is trim-insensitive, the
sag story needs revisiting.

---

## Step 3 — ONE tethered ground-run-legodom rerun with the new deploy fixes (audit §3.7, §6.2, §6.6)

The previous 15 Nm run predates the yaw-align + torso-anchor fixes (**both now
default-ON** in deploy_runtime: `YAW_ALIGN=1`, `TORSO_ANCHOR=1`). This rerun
decontaminates the 15 Nm figure (per-tick stage-tagged telemetry, no blind
`sleep 5.2` window) and tests how much of the "2x hardware excess" the free obs
fixes bought back.

```bash
export CONFIRMED_BY_HUMAN=alois
python -m pipeline.deploy_runtime --mode ground-run-legodom --max-secs 30 --i-will-watch-the-robot
```

- Watch the startup log: it must print `reference yaw-aligned to robot heading
  (offset ... deg)` — **record that offset number** (it's the audit's never-logged
  boot heading, §6.6).
- Telemetry auto-records every tick (`<stamp>_ground-run-legodom.npz`).
- **Slack-vs-taut tether A/B:** if the first 30 s segment is clean, do ONE more
  30 s run with the tether visibly slacker (still catch-capable) and note which run
  was which — the audit says the taut tether biases ankle torque LOW (§2, §6.2).
  If the first run was NOT clean, skip the A/B; don't push a bad day.
- **Abort:** first lurch/lean/fault → remote-damp. The run always ends soft.
- Between the two runs: same cool-down rule as Step 2.

**Expected:** ankle_pitch mean somewhere in the 9–31 Nm decontaminated band —
now as a clean, stage-tagged number. Lower than ~15 suggests the obs fixes helped.

---

## Step 4 — Pull data + offline analysis (no robot needed)

Every capture already wrote its analysis JSON. Now run the kp*err identity check on
the motion-run files (audit §4 exp #3 — the "knee delivers 0.2–0.4x" question):

```bash
cd ~/g1-dance && conda activate tv    # any env with numpy works for --analyze
ls -t data/telemetry/*.npz | head
python deploy/capture_stage0.py --analyze data/telemetry/<stamp>_stand-hold.npz
python deploy/capture_stage0.py --analyze data/telemetry/<stamp>_ground-run-legodom.npz
```

Section [d] prints the per-joint **delivery ratio** (tau_est vs kp·err − kd·dq
regression slope; gains auto-corrected for stand-hold's approach scale). Read it as:

- ankle_pitch ratio ~1.0 → tau_est is joint-space consistent (closes §6.1).
- knee ratio < 0.6 with good R² and real commanded RMS → the delivery deficit is
  REAL → the retrain's actuator DR ranges must widen (audit §3.2c).
- knee ratio ~1.0 → the 0.2–0.4x datapoint was tether contamination → drop it,
  keep effort DR at 0.80–1.00.
- `ratio --  (cmd ~0)` → that joint wasn't commanded hard enough to identify;
  not evidence of anything.

Then: copy the `.analysis.json` numbers into the results table below, update
`PROJECT_STATE.md` (decision log: which audit unknowns are now closed), and commit
`data/telemetry/*.npz + *.analysis.json` — **provenance is the whole point**
(audit process note: "commit raw measurement scripts + outputs").

---

## Results table (fill in during the session)

| # | Measurement | Audit ref | Result | File |
|---|---|---|---|---|
| 0 | Robot mass as-deployed (kg) | §3.2a | | (photo of scale) |
| 1a | Obs staleness p50/p95/max (ms); >2 ms? | §4 #4 | | `_stage0_onboard-standby` |
| 1b | Onboard ankle_pitch tau mean L/R (Nm) | §4 #5, §6.4 | | " |
| 1c | Onboard thermal rate, hottest motor | §6.5 | | " |
| 2 | Stand-hold ankle tau + C/min at trim 0/+3/−3 | §4 #6, §6.7 | | `_stand-hold` x3 |
| 3 | Legodom rerun ankle mean; yaw-align offset; slack-vs-taut delta | §6.2, §6.6 | | `_ground-run-legodom` |
| 4 | Delivery ratio: ankle / knee / hip | §6.1, §6.3 | | `.analysis.json` |

## What NOT to do this session

- Do not run any motion mode without the tether + `CONFIRMED_BY_HUMAN=alois`.
- Do not "quickly try" gain scales, GRAVITY_FF, or new policies — this session is
  for MEASUREMENT; every knob change contaminates a number.
- Do not skip cool-downs to save time; a heat-stacked thermal slope is worthless.
- Do not hand-transcribe numbers instead of committing the npz + json files.
