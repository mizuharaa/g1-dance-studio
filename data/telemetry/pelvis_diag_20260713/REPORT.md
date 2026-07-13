# G1 EDU Ultimate — Pelvis / Full-Body Fault Diagnosis (fishy / burning smell)

**Date:** 2026-07-13  **Robot state:** POWERED OFF, battery physically REMOVED (SAFE).
**Task type:** read-only analysis + research. No power, no commands, no network contact was made.

> **Bottom line up front.** The robot's entire recorded thermal history **exonerates the
> motors — including the three waist motors and the hips.** The only anomalous heat is the
> **pelvis-mounted IMU at 80 °C**, which is **+21 °C hotter than the hottest motor ever
> recorded** on this robot (59 °C) and **+16 °C above today's hottest motor** (ankle, 64 °C).
> A MEMS IMU dissipates only milliwatts and **cannot self-heat to 80 °C**, so that reading is
> a *thermometer* for a hot **pelvis/waist power-electronics board**, not a failing sensor.
> Combined with a **fishy / amine** smell — the textbook signature of a **vented electrolytic
> capacitor, an overheating power connector, or a scorched PCB / DC-DC stage** — the fault is
> almost certainly **electrical/power in the pelvis-waist zone, not a motor winding.**

---

## 0. Evidence base (every load-bearing number is traceable to a raw file)

- **Today's live DDS scan** — `data/telemetry/thermal_20260713/lowstate_thermal_scan.txt`:
  IMU (pelvis) **80 °C**; hottest motor **left_ankle_pitch 64 °C**; no motor ≥ 75 °C.
  The "IMU temp" field is `LowState_.imu_state.temperature` (see `deploy/robot_thermal_diag.py:65-67`).
- **Historical per-run telemetry** — 54 `.npz` runs under `data/telemetry/`, spanning
  **2026-07-04 → 2026-07-10** (last real robot run: `20260710-145111_ground-run-legodom.npz`;
  no run recorded 07-13 — today produced only the read-only scan above). Each `.npz` stores
  per-tick per-**motor** temperature for all 29 joints, `tau_est`, `q/dq`, IMU quat/gyro
  (`pipeline/deploy_runtime.py`, `Telemetry`, ~line 400).
- **This diagnosis's script + full output:**
  `data/telemetry/pelvis_diag_20260713/pelvis_thermal_trend.py`
  `data/telemetry/pelvis_diag_20260713/pelvis_thermal_trend.out.txt`

**Critical measurement caveat.** The `.npz` history contains **no IMU / pelvis-electronics
temperature channel** (only `imu_quat` + `gyro`). Today's 80 °C IMU therefore has **no
historical time series** — it can only be compared against the motor history. A hot
**battery / BMS** is likewise **not** in `rt/lowstate`, so an overheating pack would never
have shown up in any of this telemetry.

---

## 1. Telemetry history analysis — motors are clean, no pelvis bind

### 1a. Motor thermal history is benign and stable
Across all 54 runs / 9 days, motor temperatures spanned only **34–59 °C** (no NaNs, no
dropouts). The **hottest motor ever recorded was `right_shoulder_pitch` at 59 °C**
(`20260704-233131_stage0_limp-hanging-zerocal.npz`) — consistent with the *known* Unitree
G1 + Inspire-FTP-hand **shoulder-pitch overheat** signature (GitHub `unitree_sdk2_python`
issue #129, also logged in `docs/research-findings.md`). **No motor has ever approached the
~75–85 °C motor-casing alarm line.** Today's live scan agrees: motors 32–64 °C.

### 1b. Waist motors are NOT trending hot (no developing bind)
Run-over-run **peak** temperature for the three waist motors (runs ≥25 s only, for
comparable warm-up):

| waist motor | first long-run | last long-run (07-10) | min | max | slope |
|---|---|---|---|---|---|
| waist_yaw   | 40 °C | 39 °C | 35 | 41 | +0.01 °C/run |
| waist_roll  | 48 °C | 49 °C | 43 | 53 | +0.05 °C/run |
| waist_pitch | 47 °C | 49 °C | 43 | 56 | +0.09 °C/run |

The tiny positive slopes sit **inside normal session/ambient scatter**. The peak waist_pitch
(56 °C) and waist_roll (53 °C) both occurred on **2026-07-07** during a long back-to-back
session (`...-172942_...`), **6 days before today** — not on the most recent run, which fell
back to 49–50 °C. **There is no monotonic creep toward the present** → no evidence of a
developing waist-motor / bearing bind. (Full per-run table: `pelvis_thermal_trend.out.txt` §2.)

### 1c. Hips: only a minor, non-alarming warm-up on the last run
The most recent run (`20260710-145111`) ran the hips slightly warm — `right_hip_roll` 43 °C,
`left_hip_roll` 41 °C, ~2–4 °C above their usual 37–40 °C. Within scatter, far below any
limit; noted for completeness only.

### 1d. Sustained-torque / bind detector — high torque is load-bearing, not a bind
`left_hip_roll` holds |τ|>5 Nm for **99.7 %** of ticks (p95 25.7 Nm) in standby
(`20260706-094401_stage0_onboard-standby.npz`); the waist joints hold high torque while
dancing (waist_pitch 76 % of ticks >5 Nm, p95 22 Nm). **This is expected** — hips bear the
standing load, the torso works during choreography. A *bind* would show high torque with the
joint **stationary AND hot**; these joints stayed 37–56 °C. So the high torque is normal load,
not a fault. (Prior *resolved* motor event: the 2026-07-04 gain-boosted-PD 20 Nm ankle
"thermal wall," fixed by a 100× ankle-torque reduction — a closed motor issue, unrelated.)

### 1e. 20-minute thermal soak — the lower torso reaches steady state, no runaway
Best long-duration proxy, `20260706-115959_stage0_post-session-watch.npz` (20 min, 60 000
rows): the waist **motors** hold **45–48 °C** and slightly **cool** over the soak
(≈ −0.1 °C/min). The lower-torso motors do not thermally run away.

### 1f. The anomaly is entirely outside the motor dataset
Today's IMU at **80 °C** is **+21 °C above the all-time motor max (59 °C)** and **+16 °C
above today's hottest motor (64 °C)**. Nothing in the motor thermal history explains it. An
IMU inertial sensor dissipates only milliwatts, so it cannot self-heat to 80 °C — the reading
reflects a **hot pelvis board / enclosure / adjacent power electronics**. This is the
telemetry's single strongest signal and points squarely at **pelvis electronics / power**,
matching a "burning with no hot motor" smell.

---

## 2. Unitree G1 EDU pelvis / waist internals (sourced)

**Source-quality note.** Unitree publishes almost no PCB-level pelvis documentation. High-
confidence items come from **official model files** (URDF/MJCF) and integrator docs; internal
component IDs come from **third-party teardowns** (flagged). Some "repair" blogs (reboot-hub,
roboticscenter, robozaps) appear AI/SEO-generated and are treated as **low-reliability /
illustrative only** where cited.

| Component | Finding | Confidence | Source |
|---|---|---|---|
| **IMU — in PELVIS** | Confirmed physically pelvis-mounted. Official MJCF: `<site name="imu_in_pelvis" pos="0.04525 0 -0.08339"/>` on the `pelvis` body (pelvis gyro+accel sensors). Matches our own model note `docs/exam_physics_fix.md:23`. | **High** (official) | unitree_rl_gym `g1_29dof.xml` |
| **IMU — 2nd, in TORSO** | A second IMU on `torso_link` (`imu_in_torso`), exposed as `rt/secondary_imu`. | High (official) | same XML; DeepWiki unitree_ros2 |
| **IMU chip part #** | **NOT disclosed** ("6-axis IMU"). MEMS comparables (QMI8658C / MPU-6050 class) are rated **+85 °C max**. | Low (chip), High (rating class) | QST/AnalogDevices datasheets |
| **Main compute (base)** | Rockchip **RK3588/S** SoC, described mounted **back/torso** (heat-fin + air cooling); not confirmed strictly pelvis. | Medium (teardown) | robotopian, chinabizinsider |
| **EDU AI compute (PC2)** | **NVIDIA Jetson Orin NX** module (EDU/Ultimate). | Medium (teardown) | chinabizinsider, RoboStore |
| **Power distribution / DC-DC** | Carrier board exposes **58 V/5 A raw battery pass-through** + regulated **24 V** and **12 V** rails → on-board **DC-DC step-down converters**. | Medium-High (integrator) | Weston Robot G1 dev guide |
| **Power decoupling** | "**Distributed capacitor buffer networks** at the power-supply ends of arm & leg joints"; **XT30** power connectors; **CAN bus** joint comms. | Medium (teardown) | LinkedIn G1 teardown |
| **Waist motors** | 3 actuators: idx **12 waist_yaw** (±2.618 rad), **13 waist_roll** (±0.52), **14 waist_pitch** (±0.52). | High | Weston Robot, DeepWiki |
| **Waist motor drivers** | Unitree **"4-in-1" integrated actuators** (motor + reducer + encoder + **driver board** in one housing) — driver electronics inside each joint, not a central board. | Medium-High (teardown) | chinabizinsider |
| **Cooling — waist zone** | **Hybrid.** **ACTIVE small centrifugal fans on the main control board AND the waist joint**, pulling air through the drive circuits; passive VC plate / copper spreaders elsewhere. Automatic **thermal derating**; ~1–2 h continuous before derating/depletion. | Medium (teardown) | LinkedIn teardown, chinabizinsider |
| **Temp sensors** | NTC thermistors on SoC + **battery BMS**; per-motor dual temp (casing + winding, warning ~85 °C, winding critical 120 °C). | Medium (teardown) / High (SDK) | LinkedIn teardown; DeepWiki SDK |

**Battery / BMS.** 13S **Li-ion**, **46.8 V nominal** (~54.6 V full), **9000 mAh ≈ 421 Wh**,
**≈2.5 kg**, quick-release pack mounted on the **back/torso**; Unitree BMS with over-charge /
-discharge / short / charge-temperature protection + balancing. (robotopian, RoboStore,
Robots International.) Our robot's as-deployed mass **34.6 kg** with "standard battery"
(`PROJECT_STATE.md:1528`) = the stock pack. A low-reliability blog reports an **aftermarket**
pack overheating (one cell 82 °C → BMS cutoff → deformation); **no credible report of a stock
G1 pack producing a fishy/burning smell** was found. NMC vs other chemistry and cell vendor:
**unverified.**

**Fishy / amine / burning-electronics smell — what produces it.** A distinctly **fishy /
ammonia-amine** odor from hot electronics is caused by **amines** released as materials
overheat. Ranked classic sources:
1. **Electrolytic capacitors (most classic).** Electrolyte contains amine/alkanolamine
   compounds; an overheated/vented/leaking cap emits a strong **fishy** smell.
2. **Overheated power connector / terminal with a high-resistance contact.** The housing/resin
   emits **amine gas — a fishy smell that is often the last warning before ignition** (highly
   relevant to XT30 / battery terminals carrying tens of amps).
3. **PCB substrate / brominated flame-retardants / flux** degrade and smell fishy when strongly
   heated (a scorched-hot board near a failing DC-DC or a shorting trace).
4. **Overheated insulation / varnish / potting resin** (PVC jackets, magnet-wire enamel, epoxy).

**Not typically fishy:** dry transformers/inductors and MOSFETs failing smell **acrid /
ozone / scorched-resin**, not fishy. So a *fishy* note tilts the diagnosis toward a **vented
electrolytic cap, a hot power connector, or a scorched PCB region** rather than a bare MOSFET.

*(Full sourced brief with all URLs and the explicit "could-not-verify" list is preserved in
this report's §7.)*

---

## 3. Correlated hypothesis ranking

Given: **pelvis IMU 80 °C**, **no hot motor** (telemetry-confirmed), **fishy/amine smell**,
robot in an **actively fan-cooled waist zone**. The IMU is the *thermometer*, not the patient.

| # | Hypothesis | Reasoning | Confidence |
|---|---|---|---|
| **1** | **Vented / failing electrolytic capacitor** on the pelvis-waist power-distribution / DC-DC board (incl. the "distributed capacitor buffer networks"). | Fishy/amine smell is the **single most textbook** cap-failure signature; caps sit right on the hot pelvis board next to the IMU. Explains smell + local 80 °C. | **High** it's a prime candidate; **Medium** it is the specific root cause |
| **2** | **DC-DC step-down stage (58 V → 24 V/12 V) running too hot / failing.** | A stressed buck converter cooks its inductor, caps, MOSFETs and the surrounding FR-4 → 80 °C at the adjacent IMU + scorched-PCB/amine smell. This is the classic power-electronics failure point in that board. | **Medium-High** |
| **3** | **Degraded / stalled waist-zone cooling fan or blocked vent (possible ROOT cause).** | The waist joint + main board are *actively fan-cooled*; a dead/clogged centrifugal fan lets the whole zone soak to 80 °C, which then cooks caps/connectors → the smell. May be the **upstream root** with #1/#2 as the consequence. | **Medium-High** |
| **4** | **High-resistance / arcing power connector** (XT30 joint-power, or the main battery interface) under load. | Tens of amps at 48–54 V; a loose/corroded/partially-melted connector runs very hot and emits an **amine/fishy "last-warning" smell**. Battery interface sits in torso/back near the pelvis. | **Medium** |
| **5** | **Battery pack / BMS thermal fault** (cell overheat/swelling). | **Invisible to this telemetry** (BMS temp not in `rt/lowstate`). Highest-severity outcome, so must be excluded even though Li-ion venting smells more solventy/sweet than fishy. Elevated risk if any non-stock pack is in use. | **Low-Medium** for smell match, **but mandatory to inspect (safety)** |
| **6** | **Waist actuator internal driver board (MOSFET) overheat.** | 4-in-1 waist actuators contain driver electronics — but telemetry shows waist **motor** temps normal (49–56 °C historically), and MOSFET failure smells acrid/ozone, not fishy; a driver fault would likely also raise casing temp. | **Low** |
| **7** | **Main compute (RK3588 / Jetson Orin NX) overheat.** | Runs hot but is thermally managed and on the **back/torso**, not pelvis; SoC overheat rarely smells fishy unless an adjacent regulator/cap fails. Would explain a warm torso, not the specific fishy odor. | **Low** |

**Motors (all 29, incl. waist): effectively ruled OUT as the smell source** by §1 — none hot,
no bind, no run-over-run creep. Shoulder-pitch is the historically hottest motor (≤59 °C, the
known FTP-hand issue) but is nowhere near a smell/burn threshold.

---

## 4. Prioritized PHYSICAL INSPECTION checklist (robot OPEN, battery OUT, unpowered)

> Precondition: keep it powered off and the battery out for all of this. Do **not** recharge a
> battery you suspect. Work under good light + magnification; **photograph** everything before
> touching. This extends the operator's existing "burning smell → power off → inspect" rule
> (`docs/ROBOT_DAY_RUNBOOK.md` FMEA "Overheat → power off; cool; inspect").

**Priority 1 — localize the smell + inspect the pelvis/waist power board (most likely source):**
1. **Nose-localize the smell to a zone** *before* touching anything — pelvis power board vs
   battery bay vs a specific connector. A fresh vs. faded smell tells you if it is active.
2. **Inspect the pelvis / waist power-distribution + control PCB** under magnification for:
   **bulged / domed / vented electrolytic capacitor tops**, brown electrolyte crust or residue,
   **discolored (brown/black) scorch marks** on the FR-4, lifted/charred traces, or a
   localized burnt spot. This is the #1 suspect region.
3. **Inspect the DC-DC converter area** (58 V → 24 V / 12 V step-downs): look for **heat-tint /
   discoloration** on the PCB around inductors and MOSFETs, melted or deformed components, and
   compare against any identical rail that looks pristine.

**Priority 2 — connectors and the battery interface (high-current, classic fishy source):**
4. **Inspect every load-bearing power connector** — the **XT30 joint-power connectors** and
   especially the **main battery-interface connector/terminals**: melted / deformed /
   discolored housings, **blackened or pitted (arced) contacts**, loose or backed-out crimps.
   Gently wiggle-test for looseness (battery out).
5. **Inspect the battery pack + its terminals** (safety-critical): **swelling / bulging** of
   the pack or a deformed housing, discolored/corroded terminals, any electrolyte residue or
   lingering hot smell, and the **BMS connector**. If anything looks off, **isolate the pack,
   do not recharge, and treat as a hazard.** Note whether it is the stock Unitree pack or an
   aftermarket one.

**Priority 3 — cooling + wiring (likely root cause / propagation path):**
6. **Verify the waist-zone / main-board cooling fan(s):** spin each by hand — must turn
   **freely, no grinding/clog**; clear dust/debris from the fan and its vents; confirm the
   **fan power connector is seated**. A dead or clogged fan in this fan-cooled zone is a strong
   root-cause candidate.
7. **Trace the power + CAN harness through the pelvis/waist** for **chafing / pinch points at
   the moving waist joint**, melted insulation, or a conductor rubbing a moving part (an
   arcing/chafed power cable is both a smell source and a shock/fire risk).

**Priority 4 — confirm the thermometer + measurements (multimeter only, still unpowered):**
8. **Check the IMU/pelvis board itself** for any local scorch (treat the IMU as the
   thermometer, not necessarily the failed part).
9. **Measure contact resistance / continuity** across each suspect connector — should read
   **near-0 Ω**; an elevated reading pinpoints the high-resistance (heat-generating) joint.
10. **Battery pack open-circuit voltage** via the BMS SoC display if it is safe to read; flag a
    low / imbalanced cell group. **Do not disassemble a swollen pack** — return it to Unitree.

---

## 5. Usage / configuration mitigations

- **Do not power on** until the smell source is found and cleared (robot is already off — good).
  A burning smell + 80 °C pelvis IMU is a **warranty/safety event — contact Unitree support**
  with the scan file and photos.
- **If a fan is dead/clogged:** replace/clean it *before* any further operation — the
  fan-cooled waist zone has little thermal margin.
- **Enforce a run/cool duty cycle.** A field report notes the G1 **overheats after ~15 min
  continuous and needs ~45 min to cool.** For the 2–3-min show target, add mandatory cool-downs
  between rehearsals and never chain long back-to-back runs (the hottest historical session,
  2026-07-07 17:2x, was exactly such a back-to-back block).
- **Avoid sustained high-torque holds.** Standby held hip-roll at ~25 Nm and the old ankle
  "thermal wall" burned 20 Nm continuously — these keep the power stages loaded. Keep leg/ankle
  standby torque low (the 100× ankle reduction already in place is the right direction).
- **Improve ventilation:** keep waist/torso vents unobstructed; add ambient cooling between runs.
- **Instrument the blind spot.** BMS/battery temperature is **not** in `rt/lowstate`, so add a
  **mid-show battery voltage/temperature watchdog** (already flagged HIGH in
  `docs/safety_review_findings.md`) reading the battery API, plus enforce the 30 % SoC floor.
- **Re-baseline after repair.** Re-run the read-only thermal scan
  (`deploy/robot_thermal_diag.py`); a healthy pelvis IMU should sit **well below 80 °C** — in
  the ~40–60 °C band of the boards around it — not 20 °C above every motor. Establish that as
  the go/no-go pre-show thermal baseline.

---

## 6. Confidence & limits of this diagnosis

- **Strong / high-confidence:** motors (incl. waist) are thermally normal with no bind and no
  run-over-run creep (direct telemetry, 54 runs); the pelvis IMU is genuinely anomalous at
  80 °C (live scan) and is board-level, not sensor self-heating (physics + MEMS rating);
  a fishy smell + no hot motor localizes the fault to pelvis/waist **power electronics**.
- **Medium / inferred:** *which* pelvis component (cap vs DC-DC vs connector vs fan) — the
  ranking is reasoned from failure-mode signatures and G1 layout, not from direct measurement
  of the (now-off) board. Physical inspection (§4) is required to confirm.
- **Could not verify:** exact IMU chip, precise pelvis-PCB power topology/part numbers, whether
  `LowState.imu_state` maps to the pelvis or torso IMU, battery cell chemistry/vendor, and the
  DC-DC/MOSFET "repair-code" claims (single low-reliability source). No stock-G1 burning-smell
  precedent was found online.

---

## 7. Appendix — full sourced hardware-research brief

The following is the raw hardware-research brief (14+ web searches) that §2 summarizes.
Confidence and source URLs are preserved; low-reliability sources are flagged inline.

### 7.1 Component inventory — pelvis / waist / lower-torso

- **Main compute board (base G1):** Rockchip **RK3588/S** SoC (4×A76+4×A55, Mali-G610, 6-TOPS
  NPU), mounted **back/torso** with a local heat-fin + air cooling; not confirmed strictly
  pelvis. *Medium.* robotopian.com/blogs/news/unitree-g1-humanoid-robot-teardown ;
  chinabizinsider.com/unitree-g1-teardown-...
- **EDU AI compute (PC2-type):** **NVIDIA Jetson Orin NX** ("100 TOPS") on EDU/Ultimate.
  *Medium.* chinabizinsider ; robostore.com/blogs/news/unitree-g1-edu-ultimate-technical-specifications
- **RAM/storage:** BIWIN 8 GB; Longsys 64 GB. *Medium.* robotopian.
- **Power distribution / DC-DC:** developer/carrier board exposes **58 V/5 A raw pass-through**
  + regulated **24 V** and **12 V** rails (on-board DC-DC step-downs). *Medium-High.*
  docs.westonrobot.com/tutorial/unitree/g1_dev_guide/
- **Power decoupling:** "distributed capacitor buffer networks at the power-supply ends of arm
  and leg joints"; **XT30** connectors; **CAN** joint comms. *Medium.*
  linkedin.com/pulse/teardown-unitree-g1-humanoid-robot-wiring-harness-integration-jlncc
- **IMU — pelvis (HIGH):** official MJCF `<site name="imu_in_pelvis" pos="0.04525 0 -0.08339"/>`
  on `pelvis` body. raw.githubusercontent.com/unitreerobotics/unitree_rl_gym/main/resources/robots/g1_description/g1_29dof.xml
- **IMU — torso, 2nd (HIGH):** `<site name="imu_in_torso" pos="-0.03959 -0.00224 0.13792"/>`;
  `rt/secondary_imu`. same XML ; deepwiki.com/unitreerobotics/unitree_ros2/4.2.1-g1-overview
- **IMU chip part #:** NOT disclosed ("6-axis"). *Low.*
- **Waist motors (HIGH):** 12 waist_yaw (±2.618 rad), 13 waist_roll (±0.52), 14 waist_pitch
  (±0.52). Weston Robot ; DeepWiki.
- **Waist motor drivers:** Unitree **"4-in-1" integrated actuators** (motor+reducer+encoder+
  driver board in one housing). *Medium-High.* chinabizinsider.
- **Temp sensors:** NTC thermistors on SoC + battery BMS; per-motor dual temp (casing warning
  ~85 °C, winding critical 120 °C). *Medium/High.* LinkedIn teardown ;
  deepwiki.com/unitreerobotics/unitree_sdk2/3-g1-humanoid-robot

### 7.2 Battery / BMS
13S Li-ion, **46.8 V nominal** (~54.6 V full), **9000 mAh ≈ 421 Wh**, ~2.5 kg, quick-release
pack on the **back/torso**; charger 54 V/5 A; runtime ~1–2 h. Unitree BMS: over-charge/
-discharge/short/charge-temperature protection, balancing, self-discharge protection, SoC
display. robotopian ; robostore.com/products/unitree-g1-humanoid-high-performance-battery ;
robotsinternational.com/Unitree-G1-BATTERY-G1-Battery.htm . Chemistry (NMC?) and cell vendor
**unverified.** A low-reliability blog (reboot-hub.com/blogs/support-learning/unitree-g1-repair-guide-2026-05)
reports aftermarket-pack overheat (one cell 82 °C → BMS cutoff → deformation) — treat specifics
as unreliable (its "6S/4S" contradicts the real 13S), failure *mode* plausible. **No credible
stock-pack burning-smell report found.**

### 7.3 Cooling / ventilation
Hybrid: **active small centrifugal fans on the main control board AND the waist joint** pulling
air through drive circuits; passive VC plate on compute (EDU), copper heat-spreader on knees;
base uses aluminum heatsink + fan. Automatic **thermal derating**; continuous ~1–2 h before
derating/depletion. LinkedIn teardown ; chinabizinsider. → the pelvis/waist is one of the few
**actively fan-cooled** zones; a fan failure/blocked vent there overheats the waist drive
circuits and the adjacent pelvis IMU board.

### 7.4 Known issues / failure modes
- **Motor/actuator overheating is dominant:** field report — overheats after ~15 min, ~45 min
  to cool. x.com/VaderResearch/status/2062825101817782667
- **Shoulder-pitch motor overheat with Inspire dexterous hand:**
  github.com/unitreerobotics/unitree_sdk2_python/issues/129
- **Actuator thermal shutdown:** driver cuts power to protect windings; casing warning ~85 °C,
  winding critical 120 °C. DeepWiki SDK.
- **Cable fatigue (hip/knee), calibration drift; DC-DC-converter faults on main controller and
  MOSFET failures on joint driver boards** — from reboot-hub (**low-reliability**, unverified;
  classic power-electronics failure points, useful as hypotheses only).
- **G1 EDU arm hardware malfunctioning** — forum.mybotshop.de/t/...1323 (could not fetch; topic
  confirmed by snippet only).
- **No first-hand "burning/fishy smell" report for a stock G1** found.

### 7.5 Is ~80 °C at the IMU alarming?
80 °C is **high and worth investigating, but it is the board/environment, not the IMU chip
failing.** MEMS IMUs are milliwatt parts and cannot self-heat to 80 °C; the die-temp tracks the
substrate. Typical MEMS IMUs (QMI8658C, MPU-6050 class) are rated **+85 °C max operating**
(qstcorp.com QMI8658C datasheet; analog.com MEMS-selection article) — so 80 °C is **at the top
of the rated envelope**, with near-zero margin and expected **bias drift**. Electronics boards
are not meant to run this hot (unlike the metal actuators at 85/120 °C), so 80 °C on a pelvis
control/IMU board indicates a **local heat source** (a DC-DC stage, a stressed connector, or the
fan-cooled waist drive circuitry) or **degraded cooling** — exactly what a burning smell
corroborates.

### 7.6 Fishy / amine / burning-electronics smell — component analysis
Fishy/ammonia-amine odor = **amines** (e.g. trimethylamine) released as materials overheat.
Ranked: (1) **electrolytic capacitors** — electrolyte contains amine/alkanolamine; vented/leaking
cap = textbook fishy smell (bmorton.com; badcaps.net; dtelectriccompany.com). (2) **overheated
connectors/terminals** with high-resistance contact — housing/resin emits amine gas, "last
warning before ignition" (dtelectriccompany.com; gacservices.com). (3) **PCB substrate /
brominated flame-retardants / flux** degrade fishy when strongly heated (bmorton.com).
(4) **overheated insulation/varnish/potting** (inlightec.com.au). **Not typically fishy:** dry
transformers/inductors and MOSFETs → acrid/ozone/scorched-resin. → a *fishy* note tilts toward a
**vented electrolytic cap, an overheating power connector, or a scorched PCB/DC-DC region.**

### 7.7 UNCERTAIN / COULD NOT VERIFY
IMU chip part#/vendor/axis count; which IMU is "primary" in the SDK (`LowState.imu_state` vs
`rt/secondary_imu` → pelvis vs torso mapping unconfirmed); exact physical location of the RK3588
board (teardowns say back/torso) and pelvis DC-DC topology/part numbers; DC-DC/MOSFET fault codes
(reboot-hub only — possibly fabricated); battery cell chemistry/vendor (only "13S Li-ion"
corroborated); any official Unitree pelvis-PCB/IMU thermal spec (none found); airoboticdaily.com
(DNS fail) and the MyBotShop forum thread (connection refused) known only from snippets.
