# Independent external audit — G1 Dance

**Audit date:** 2026-07-21

**Repository baseline:** `d60ac3c3609a1bd3868ff206709e0c48ddb684d5` (`main`)

**Auditor access:** full repository and local CPU environment; no GPU box, robot, or
credentials

**Pinned training package checked:** `mjlab==1.5.0` wheel SHA-256
`93aa539d1c7d8e984a34b8855967d304dd14a01e60c95aa03d7ac71c228f070c`

## 1. Executive summary

**Verdict: do not use the current two-box sweep to choose an ankle recipe, do not
train further on the present v12 artifact, and do not launch a robot show through
the current one-button endpoint.** These are evidence failures, not marginal tuning
concerns.

The five mechanisms most responsible for slow progress and weak sim-to-real trust are:

1. **The recent “ankle-only” effort fix applies to every actuator group.** mjlab's
   `effort_limits` selects `actuator_ids`, while the repository supplies
   `joint_names`. The final 0.52–0.76 event therefore wins globally. Arms train at
   13–19 Nm instead of 25, wrists at 2.6–3.8 instead of 5, and knee/hip-roll motors
   at 72.3–105.6 instead of 139. This invalidates the current control-vs-tau sweep's
   stated hypothesis.
2. **The v12 training reference is not the artifact its scorecard certifies.** Its
   current SHA does not match the recorded SHA; the recorded source lived in a
   deleted `/tmp` path; the post-scorecard cleaner changed 692 rows. The current
   motion also has 163 mm of apparent floor-height drift and re-scores at maximum
   binding ratio 2.865 rather than 1.179.
3. **Evaluation is self-contaminated.** “Nominal” is built from `play=False`, so it
   retains startup DR, randomized initialization, 0–20 ms command delay, and up to
   one control-step observation delay. Conditions use different seeds and are not
   isolated. The full-motion fix also runs ten control steps beyond the reference;
   pinned mjlab teleports surviving environments back to frame zero at that point.
4. **Artifact identity is not end-to-end.** The three available v8/v10/v11 ONNX
   policies take 770 observations, while their copied metadata says 160 and names
   estimator-dependent terms. The show endpoint does not reauthorize or rehash the
   selected policy/motion/meta/NPZ immediately before spawn, permits an explicitly
   unsigned `free` bundle, and silently falls back to runtime defaults when a dance
   bundle is incomplete.
5. **The “faithful training model” preview is not the pinned training model.** It is
   a useful alternate/hardware-leaning model, but it uses a different mass
   distribution and collision scene. Calling it an exact training replay turns a
   potentially valuable stress test into false corroboration.

What to change first: freeze/discount the current sweep; fix and runtime-assert
effort ranges; rebuild and content-address v12; generate policy-specific metadata;
make evaluation conditions explicit and stop at the last motion frame; then run one
small GPU smoke that dumps realized configuration before another full curriculum.
Independently, make the show launch fail closed on one immutable, just-reauthorized
bundle before any robot is powered for deployment.

The strongest positive result from the previous audit is narrow but important:
the shared `HistoryStacker`'s term-major, oldest-to-newest, first-frame-backfilled
layout is statically correct against the exact pinned package. It is not yet a
validated robot deployment path because the box verifier and artifact metadata are
broken.

## 2. Scope, method, and evidence quality

I followed the prescribed reading order, traced the v3→v11 inheritance chain,
motion preparation, gate, preview, deploy runtime, show authorization, and relevant
git history. I treated prior reports as claims. Independent CPU/static checks and
raw results are committed beside this report:

- `crosscheck.py` / `crosscheck.json`: wheel semantics and hashes, effort scope,
  ONNX-vs-metadata contracts, v12 history/hash/grounding, eval horizon, synthetic
  flight, model comparison, incomplete show-bundle fallback, and signed-verdict
  replay.
- `PINNED_MJLAB_EVIDENCE.md`: cited excerpts from the exact wheel, with upstream
  member line numbers.
- `v12_dynamics.json`: a fresh run of the repository's current feasibility tool on
  the current v12 CSV.
- `TEST_RESULTS.md`: full CPU-suite result.

The full suite is **604 passed, 7 failed out of 611 collected**. The failures are
baseline fixture/expectation drift rather than changes made by this audit, but the
repository currently has no green full-suite baseline
(`experiments/external_audit_20260721/TEST_RESULTS.md:1-30`).

GPU-dependent outcomes—actual reward trade-offs, learned policy behavior, realized
Warp scene values, and performance changes—are explicitly marked as needing GPU.
Physical safety, system identification, drift sensing, thermal behavior, and show
repeatability need the repaired robot. No claim here substitutes same-engine sim for
hardware evidence.

## 3. Verification of the previous audit's A–L fixes

| Fix | Audit result | Evidence / remaining gap |
|---|---|---|
| A, velocity-aware cleaning | **Partial** | Joint reblending landed (`tools/motion_quality.py:280-327`), but the cleaned v12 bytes were not re-scorecarded; root quaternion is still always smoothed without an angular-velocity retention guard (`tools/motion_quality.py:329-335`). |
| B, stale tempo NPZ | **Landed statically** | The launcher deletes/rebuilds failed conversions and checks each NPZ's frame count (`cloud/run_attempt9.sh:33-71`). It still does not bind the NPZ to a source CSV hash. |
| C, policy-own preview motion | **Partial** | Own staged NPZ/CSV is preferred, but a shared-motion fallback remains allowed (`pipeline/publish_policy.py:154-171`). Exact lineage is not a hard requirement. |
| D, preview yaw | **Landed statically** | Reference and policy panes receive the same yaw transform (`tools/sim_studio.py:88-124`). |
| E, deploy history | **Core implementation correct; validation incomplete** | `HistoryStacker` matches pinned mjlab semantics (`pipeline/deploy_runtime.py:736-781`; `experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:101-137`). The purported box gate uses the wrong config API and buffer API and is not called by attempt 9 (`cloud/verify_obs_layout.py:43-76`; `cloud/run_attempt9.sh:79-99`). |
| F, ankle effort clamp/DR | **Broken, high impact** | The selector is for joints, but mjlab's event selects actuators. Confirmed finding F1 below. |
| G, ankle barrier | **Formula landed; needs GPU** | Threshold 16 Nm now produces gradient in the measured operating band (`cloud/sim2real_task_v8.py:207-215,282-301`). Its learned trade-off is unknown, and the current sweep cannot isolate it because F is broken. |
| H, foot-flat residual | **Landed statically** | Tilt is now a robot-minus-reference residual (`cloud/sim2real_task_v10.py:245-260`). Needs GPU behavior, not another code rewrite. |
| I, saturation thresholds | **Landed statically** | Per-class thresholds are explicit and name-resolved (`cloud/sim2real_task_v10.py:152-181,263-287`). Reward incidence still needs a runtime histogram. |
| J, recorded gate bars | **Landed** | Current outputs include the gate configuration; downstream comparisons should reject old artifacts without it. |
| K, drift statistic label | **Landed** | Picker uses episode-max p95, with an explicit legacy fallback (`cloud/pick_checkpoint.py:51-73`). |
| L, held-out full motion | **Partial, introduced end-wrap** | Full duration is set (`cloud/heldout_eval.py:89-99`), but `+0.2 s` runs past the reference and invokes mjlab's frame-zero state write. Confirmed finding F4 below. |

## 4. Confirmed findings

### F1 — CRITICAL: the “ankle-only” effort randomization is global

**Evidence.** v8 re-scopes all three effort events with
`SceneEntityCfg(..., joint_names=...)`, and inserts the composed 0.52–0.76 event
last (`cloud/sim2real_task_v8.py:448-478`). v10 and v11 inherit that function
unchanged (`cloud/sim2real_task_v10.py:290-293`; `cloud/sim2real_task_v11.py:83-86`).
Pinned mjlab resolves joint and actuator selectors separately, while
`effort_limits` reads only `asset_cfg.actuator_ids`, iterates high-level actuator
groups, and writes from pristine defaults
(`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:11-73`). Startup
events preserve insertion order, so the last event overwrites the earlier two on
all groups (`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:75-88`).

**Mechanism and quantified effect.** `joint_names` does not narrow
`actuator_ids`, whose default remains `slice(None)`. The intended final ankle scale
therefore applies to all six G1 actuator groups. The independent calculation records
the following realized ranges from nominal limits
(`experiments/external_audit_20260721/crosscheck.json:7-36`):

| nominal class | nominal | current training range |
|---|---:|---:|
| wrist pitch/yaw | 5 Nm | 2.6–3.8 Nm |
| shoulders/elbows/wrist roll | 25 Nm | 13–19 Nm |
| ankles + waist roll/pitch | 50 Nm | 26–38 Nm |
| hip pitch/yaw + waist yaw | 88 Nm | 45.8–66.9 Nm |
| knee + hip roll | 139 Nm | 72.3–105.6 Nm |

This is direct ML-weight distortion: every rollout trains under globally reduced
authority, while the recipe and selfcheck report only ankle reduction. It can
explain upper-body/leg under-reach and makes the current tau-16-vs-tau-12 sweep a
comparison under an unintended shared intervention, not a validation of fix F.

**Fix specification.** Replace the generic event with a custom, tested event that
resolves the exact target control IDs for each high-level actuator group, applies
the stock 0.80–1.00 band to non-ankles and 0.52–0.76 only to four ankle controls,
and then emits a 29-joint realized `[min,max]` table after startup. Do not merely
swap `joint_names` for `actuator_names` without first asserting the six group names
and their `target_names`; one high-level group can own multiple controls. Make the
64-env preflight fail unless every non-ankle and ankle control is in its intended
range. The current selfcheck only verifies event presence and intended constants
(`cloud/sim2real_task_v8.py:624-636`), so replace that claim-level check with a
realized-model check.

**Blast radius:** v8/v10/v11 training configuration, all new policies and current
sweep conclusions; no motion or deploy-runtime change.

**Confidence:** very high. CPU/static semantics are exact; one GPU-box scene dump is
required to confirm the expanded Warp model before spending on training. Robot is
not required to validate the bug.

### F2 — HIGH: v12 is neither provenance-bound nor physically re-certified

**Evidence.** Attempt 9 consumes `thriller_v12_full.csv` and checks only existence
and derived frame counts (`cloud/run_attempt9.sh:23-24,55-76`). Its scorecard records
repaired SHA `2ee0…`, source SHA `151b…`, and a source under a transient `/tmp`
scratchpad (`data/motions/thriller/thriller_v12_full_scorecard.json:3227-3231`). The
current CSV SHA is `1bf7…`; `hash_matches_current=false`
(`experiments/external_audit_20260721/crosscheck.json:381-397`). Commit `57e4b2c`
changed 692 of 1,554 rows, 27 joint columns, up to 0.1004 rad, after the scorecard;
root xyz/quaternion were unchanged
(`experiments/external_audit_20260721/crosscheck.json:388-394`).

The current pipeline says training/show inputs use per-frame grounding
(`pipeline/grounding.py:18-26`), but the current v12 lowest-geometry height ranges
0–163.2 mm, has a 90.0 mm median, and exceeds 100 mm in 41.1% of frames. A known
per-frame-grounded comparator has only 18.9 mm range and 0% above 100 mm
(`experiments/external_audit_20260721/crosscheck.json:363-379`). A fresh feasibility
run reports ankle max 53.79 Nm, 2.51% any-joint-over frames, maximum binding ratio
2.865, 74.84% floaty feet, and 66.92% stepping-required
(`experiments/external_audit_20260721/v12_dynamics.json:11-16,176-181`). The stale
scorecard says 47.16 Nm, 1.93%, and 1.179.

**Mechanism and effect.** The launcher can train bytes that no retained report
certifies. Fix A changed joint dynamics without updating its scorecard, while the
pre-existing root-height drift contradicts the claimed grounded lineage. The
stance schedule is relative to whichever foot is lower, so both floating feet can
still be labeled “stance” (`cloud/sim2real_task_v10.py:216-225`); training then
faithfully learns a vertically drifting reference. The missing immutable source
also makes the recorded style similarity 0.905 irreproducible.

The absolute contact-height helper uses geometry centers and all robot geoms, so it
is not a perfect sole-surface measurement (`pipeline/grounding.py:53-68`). The
same-helper 163-vs-19 mm comparison and the independent floaty-feet result are large
enough that this caveat does not reverse the finding.

**Fix specification.** Stop using the current v12 in training. Choose an immutable
committed input (the parent artifact is at least recoverable from git), fix the
flight-aware grounding issue in F8 first, then regenerate cleaning, grounding,
feasibility, CSV→NPZ, and a compact scorecard in one command. The manifest must bind
source CSV, output CSV, each tempo NPZ, cleaning config, model hash, and scorecard by
full SHA-256. Make attempt 9 verify those hashes, not just frame count. Add per-joint
and per-window fidelity retention; a group mean or one groupwise peak can hide one
blunted joint.

**Blast radius:** current v12 motion, its tempo NPZs, all policies trained from it,
and comparisons to its scorecard.

**Confidence:** high for identity and floor drift; medium-high for the feasibility
model's absolute torque numbers because it is itself an approximation. CPU can
validate the rebuilt artifact; GPU is needed to quantify learned-policy impact.

### F3 — HIGH: the 770-observation deployment artifact is not a self-consistent bundle

**Evidence.** The shared metadata declares the stock task, estimator-dependent
`motion_anchor_pos_b` and `base_lin_vel`, and a 160-wide observation
(`data/policies/thriller/policy_meta.json:2-16,54-62`). `publish_policy` copies that
file into every policy directory when absent (`pipeline/publish_policy.py:127-150`),
and the export script writes only ONNX (`cloud/export_ckpt_onnx.py:54-69`). CPU ONNX
inspection of available v8, v10, and v11 exports finds `[1,770]` for all three while
each metadata file says `[1,160]`; none sets `requires_ground_contact`
(`experiments/external_audit_20260721/crosscheck.json:227-334`).

The ground runtime correctly refuses metadata that still requests estimator-only
terms (`pipeline/deploy_runtime.py:681-703`), so copied metadata blocks a v11 ground
run. If terms are omitted to use the fallback, the absent contact flag disables the
contact-loss guard (`pipeline/deploy_runtime.py:361-369,1455-1459`). The sandbox
sidesteps metadata by inferring `obs_dim // 154` directly from ONNX
(`tools/sim_sandbox.py:99-110`), so a preview may look functional while deployment
refuses or loses a safety guard.

The previous audit's live layout verifier is also nonfunctional: it uses attribute
access on dict-backed `cfg.commands` and `cfg.observations`, then treats a
`CircularBuffer` lag lookup as a `[history,dim]` array
(`cloud/verify_obs_layout.py:43-76`; pinned API evidence at
`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:90-137`). Attempt 9
never invokes it (`cloud/run_attempt9.sh:79-99`). Finally, ONNX width is inspected
only after `_release_motion_service()`, and any non-multiple silently degrades to
history 1 (`pipeline/deploy_runtime.py:1427-1444,761-768`). An incompatible policy
therefore can be discovered only after onboard control has been released.

**Mechanism and effect.** Policy, observation schema, contact semantics, motion, and
model are assembled from independent files and heuristics. There is no single
artifact that proves the exact 770-vector the network expects. Fix E's stacker is
correct, but the surrounding integration can fail closed before motion, fail late
after release, or disable contact safety depending on which stale fields are
present.

**Fix specification.** Export one policy-specific manifest from the live task cfg:
ordered terms and widths, history length/layout, ONNX I/O, joint order, gains,
action scale, `requires_ground_contact`, motion CSV/NPZ hashes, task/config hash,
and policy hash. Validate it against ONNX and a live actor observation before
signing. Fix the box verifier to use `cfg.commands["motion"]`,
`cfg.observations["actor"]`, and `buffer.buffer[0]`; call it from the launcher. At
runtime, resolve and strictly validate the whole bundle before DDS setup or human
confirmation; reject a dynamic/nonmultiple width rather than returning history 1.

**Blast radius:** exporter, publish path, preview, v8+ ground deployment, and show
bundle schema.

**Confidence:** very high for the file/ONNX mismatch and verifier bugs. GPU is
needed for the live actor byte comparison; robot validation remains mandatory after
preflight is fixed.

### F4 — HIGH: gate conditions are not isolated, and “full motion” wraps to frame zero

**Evidence.** Every condition loads `play=False` (`cloud/sim_gap_check.py:216-237`).
The v5 recipe defaults to four physics-step command delay and one control-step
observation delay (`cloud/sim2real_task_v5.py:56-60`), which the base task applies to
actuators and observations (`cloud/sim2real_task.py:218-230`). The gate overrides
command delay only when the advertised delay is greater than zero and never clears
observation delay (`cloud/sim_gap_check.py:241-250`). It also leaves startup mass,
CoM, gains, friction, effort, armature, encoder and RSI randomization active, and
uses a different seed for every condition (`cloud/sim_gap_check.py:252-255`). Yet
the output labels nominal delay as zero (`cloud/sim_gap_check.py:409-425`).

The gate and held-out evaluator set the episode to motion duration plus 0.2 s
(`cloud/sim_gap_check.py:453-454`; `cloud/heldout_eval.py:89-99`). The v11 motion is
2,464 frames at 50 Hz, while the nominal artifact ran 2,474 steps—exactly ten extra
(`experiments/external_audit_20260721/crosscheck.json:37-49`). At the reference end,
pinned `MotionCommand` resets the clock and writes frame-zero root/joint state into
the simulator (`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:139-169`).

**Mechanism and effect.** The nominal 98.4% result is a DR/latency robustness result,
not clean nominal tracking. Delay rows differ in seed, action delay, noise, and
unreported observation delay, so their deltas cannot estimate latency sensitivity.
The ten-step wrap physically teleports survivors and then scores 0.2 s of the start
again. It does cover most of the dance, but it does not test a natural final pose or
handoff and can bias survival, drift, and torque summaries.

**Fix specification.** Build explicit evaluation configs rather than mutating the
training config in place:

- clean deterministic baseline: no DR, RSI, noise, delay, or push;
- one-factor rows for action delay, observation delay, noise, push, model DR, and
  their deliberately selected composites;
- common random numbers for paired comparisons and a recorded realized-parameter
  manifest per environment;
- exactly `T` policy steps, `auto_reset=False`, and explicit completion before
  `time_steps >= T`; test entry and exit handoffs as separate scenarios.

Keep the current randomized suite as a robustness campaign, but stop calling it
nominal or treating rows as an orthogonal latency curve.

**Blast radius:** checkpoint picker, gap/held-out metrics, historical comparisons,
and show verdicts.

**Confidence:** very high statically; GPU is needed to remeasure how much each
metric changes. No robot is needed to fix the harness.

### F5 — HIGH: the preview model is mislabeled as an exact training replay

**Evidence.** The sandbox and studio call the generated model the mjlab training
model (`tools/sim_sandbox.py:34-66`; `tools/sim_studio.py:43-50`). Its builder starts
from the official Unitree XML, not the pinned mjlab XML
(`tools/assets/g1_faithful/build_faithful_model.py:18-30`). The reconciliation table
calls 35.11 kg “faithful (= mjlab-aligned)” and says 33.34 kg is menagerie
(`experiments/g1_model_reconciliation.md:46-74`), while the exact pinned mjlab wheel
contains a 33.341 kg raw G1 model with a 7.818 kg torso. The preview is 35.112 kg
with a 9.598 kg torso, 1.771 kg heavier overall and 1.780 kg heavier at the torso
(`experiments/external_audit_20260721/crosscheck.json:85-153`).

Training then randomizes the pinned torso by only 1.00–1.12 and adds 0.4–0.7 kg at
each wrist (`cloud/sim2real_task.py:271-302`). Thus the training total spans roughly
34.14–35.68 kg, but its torso tops out near 8.76 kg; the preview's correct-looking
total hides a materially different mass distribution. Pinned config uses full-body
collisions and 0.6 base foot friction before the task's 0.3–1.2 friction DR
(`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:191-225`;
`third_party/mjlab_mdp_ref/tracking_env_cfg.py:193-201`), whereas the preview inherits
the official feet-only/1.0 setup documented in its own reconciliation table.

**Mechanism and effect.** Armature matching alone does not make two floating-base
contact systems the same. A policy rollout on this scene is valuable as an alternate
model stress test and may be closer to hardware mass, but it cannot corroborate
same-training-model behavior. The current labels encourage treating two different
simulations as agreement when one is neither an exact training replay nor validated
against more than one real policy.

**Fix specification.** Maintain two explicitly named scenes:

1. an exact-training scene exported/dumped from the pinned live mjlab environment,
   including expanded actuators, collision filters, friction and runtime parameter
   realization; and
2. a hardware-uncertainty scene based on official Unitree/identified robot values.

Hash and record the scene for every preview/gate result. Never call the second one
“training faithful.” Compare both against the anchor telemetry and future repaired-
robot trials; disagreement is useful evidence, not a rendering defect.

**Blast radius:** preview labels, sim-studio interpretation, calibration reports,
and any gate that treats this as independent sim2sim.

**Confidence:** high that the current label is false; medium on the dynamic size of
the gap until a live mjlab model dump and robot system identification are available.

### F6 — CRITICAL: one-button show launch can execute an unsigned or substituted bundle

**Evidence.** The live endpoint checks dance status, audio, ping, single-run lock,
typed phrase, operator and mode, then spawns (`ui/server.py:962-1007`). It does not
call the available pre-show checklist, whose policy step rehashes the file
(`pipeline/preshow.py:84-112`), and does not hash policy, metadata, CSV or NPZ itself.
Promotion rehashes only the policy and only at promotion time
(`pipeline/shows.py:346-371`); the `Dance` record stores no motion hash
(`pipeline/shows.py:112-127`).

When the selected dance bundle is missing any member, `_dance_policy_args` returns
an empty list (`pipeline/show_runner.py:75-94`), causing deploy-runtime's hard-coded
Thriller defaults to be used (`pipeline/deploy_runtime.py:60-62,1844-1846`). A
temporary-directory execution confirms that a present policy with missing metadata
and NPZ produces no CLI overrides and therefore selects those defaults
(`experiments/external_audit_20260721/crosscheck.json:337-345`). The
`free` path always selects a fixed standtail bundle and explicitly admits it is not
signed (`pipeline/show_runner.py:43-72`); the API permits `free=true` even for live
mode (`ui/server.py:990-1007`). `begin_run` creates the show and spawns either path
without artifact authorization (`pipeline/show_runner.py:235-266`).

**Mechanism and effect.** “Show-ready” authenticates an earlier policy/CSV pair,
but the process launch resolves a mutable policy/meta/first-matching-NPZ tuple later.
Post-promotion mutation, a missing file, an extra lexicographically earlier NPZ, or
the `free` flag can run bytes the verdict never evaluated. The typed phrase is real
human consent, but it is consent to an artifact the server has not resolved or
authorized. On an e-stop-less 35 kg humanoid this is a critical fail-open seam.

**Fix specification.** Before creating a show or spawning a process, resolve exactly
one immutable bundle ID and rehash **policy + metadata + deploy NPZ + source motion +
runtime config**. Require one signed verdict that names those hashes, the scene hash,
and unique evaluation campaign. Evaluate the entire server-side pre-show checklist
under the same lock and issue a short-lived launch nonce tied to operator and bundle.
Reject incomplete/ambiguous bundles; never fall back to defaults. Remove the `free`
path until that exact bundle is signed and staged through the same gate.

**Blast radius:** show records/schema, promotion, pre-show API, process spawn, and
operator UI. It does not require changing the policy loop.

**Confidence:** very high from direct control-flow tracing. CPU tests can validate
fail-closed behavior; tethered robot rehearsal is required only after the gate is
fixed.

### F7 — HIGH: the show-ready verdict accepts an arbitrary “875 N” label and replayed evidence

**Evidence.** The verifier converts a direct 0.5 m/s root-velocity change into 875 N
by assuming 35 kg and an arbitrary 20 ms interval (`pipeline/mjlab_verify.py:26-36`),
then stores that number in `force_n` (`pipeline/mjlab_verify.py:79-97`). The
authorization layer accepts any signed push phase at or above 150 N
(`pipeline/exam_verdict.py:23-33,93-110`). Pinned mjlab states that this event is an
instantaneous, mass-independent velocity overwrite that ignores inertia and contact
dynamics (`experiments/external_audit_20260721/PINNED_MJLAB_EVIDENCE.md:171-189`).

The same signed verdict can also be submitted repeatedly. Recording increments the
consecutive-clean counter on every submission and uses `verdict.get("at")` as its
exam ID (`pipeline/shows.py:273-315`), but `build_verdict` creates no `at` or unique
evaluation ID (`pipeline/mjlab_verify.py:56-103`). Submitting one verdict three times
therefore satisfies the separate three-clean-run promotion target with three null
exam IDs. An isolated-store execution reproduces credits `[1,2,3]` from the same
authentically signed verdict (`experiments/external_audit_20260721/crosscheck.json:346-360`).

**Mechanism and quantified effect.** The simulated disturbance is valid as a
delta-velocity stress test, but it has no unique Newton-force magnitude. Choosing 5
ms would label the same state edit 3,500 N; choosing 100 ms would label it 175 N.
The 150 N gate therefore tests an arbitrary unit conversion, not physical push
strength. Replay does not erase the underlying 128/256 simulated episodes, but it
defeats the intended evidence that three distinct campaigns were run.

**Fix specification.** Define the schema honestly as
`disturbance_kind=delta_velocity` with sampled `delta_v`, axes, interval schedule and
seed. If a physical force/impulse floor is required, run an actual finite-duration
force or impulse event and record N, seconds and N·s separately. Add a signed unique
`eval_id`, seed set, config hash, model hash and creation time; deduplicate at record
time and derive repeatability only from distinct IDs. Keep same-engine held-out
evaluation labeled as such and require alternate-sim plus staged hardware evidence
for show authorization.

**Blast radius:** verdict schema, tests, stored dance history and promotion logic;
old verdicts require migration or re-exam.

**Confidence:** very high statically. GPU is required to rerun a corrected push
campaign; robot trials are required to calibrate a useful physical threshold.

### F8 — MEDIUM: per-frame grounding deletes sustained airborne phases

**Evidence.** The ground line is a nine-frame Savitzky–Golay fit to the same contact
height being classified. Flight is then defined as `c - g > 0.08 m` for at least
four frames, and root z subtracts `g` (`pipeline/grounding.py:121-187`). For a
synthetic two-second, 0.20 m airborne plateau, the low-pass follows the plateau,
detects zero flight frames, and leaves median plateau height effectively zero
(`experiments/external_audit_20260721/crosscheck.json:50-61`). The helper also takes
the minimum center height of **all** robot geoms despite documenting a support foot
(`pipeline/grounding.py:53-68`).

**Mechanism and effect.** A sustained jump looks like a new baseline to a short
low-pass. Only a transition edge can exceed `c-g`; the plateau itself cannot. The
algorithm therefore subtracts genuine whole-body flight just as it subtracts floor
drift. This does not explain current Thriller—its feasibility report contains no
flight—but it will corrupt the project's intended general video-to-dance product,
especially longer choreography with hops, kneels or a low hand/head.

**Fix specification.** Resolve explicit sole collision geoms and use their surface
height, not every geometry center. Estimate a slowly varying floor only during
confident single/double support; hold it through intervals where both feet lose
support, using pre/post support anchors plus root/CoM vertical velocity. Add synthetic
unit cases for short hops, two-second jumps, alternating stance, crouch, hand-to-
floor and true camera-floor drift. Run grounding before any balance/torque scorecard.

**Blast radius:** grounding, vetting, motion scorecards and all future motions with
airborne/floorwork content.

**Confidence:** high for the algorithmic failure; CPU-only to fix and validate.

## 5. Optimization opportunities

### Stop paying for invalid hypotheses

The current curriculum is 9,500 iterations (`cloud/train_v11_curriculum.sh:1-12`).
The measured throughput/cost is about 2,040 iterations/hour and 8,900 VND per 1,000
iterations (`logs/jobs.md:55-57`). Two full candidates therefore consume roughly
**9.3 GPU-hours and 169,000 VND (~$6.5 using the repository's own conversion)**.
F1 makes that entire wave incapable of validating the intended ankle-only change.
A seconds-long realized-config dump would have prevented it.

Required CPU/static gates before any trainer starts:

1. exact wheel/config semantics and realized actuator ranges;
2. source/output/NPZ hash manifest;
3. current scorecard regenerated from current bytes;
4. live actor observation vs generated metadata/ONNX shape;
5. clean 64-env, three-iteration smoke with finite rewards and per-term summaries.

### Use a funnel instead of two full retrains per knob

After one corrected base checkpoint exists, screen reward/DR changes as paired
fine-tunes from that identical checkpoint:

- 2–4 candidates × 1,000–1,500 iterations, same seed and motion;
- top two on three distinct seeds;
- only the winner gets the full curriculum and 11-condition stress suite.

Compared with two independent 9.5k runs, a 3k screen plus one 9.5k confirmation is
about **34% fewer training iterations**; early rejection at 1k can save 50%+. This is
an estimate, not a promise: large motion changes may require a fresh base. Always
separate “does the intervention have signal?” from “does the final recipe generalize?”

### Make every run answer one question

Current gate rows change seed and retain hidden DR/delay. Use paired seeds and an
explicit factorial manifest. Report:

- clean tracking/survival;
- action-delay-only and observation-delay-only curves;
- noise-only, push-only, parameter-DR-only;
- selected realistic composites;
- per-reward raw value, weighted contribution, nonzero incidence and percentile;
- realized actuator/model parameters;
- per-joint, per-phrase fidelity and torque, not only aggregate MPKPE/p95.

This turns a failed run into a diagnosis. Today, inherited terms can conflict or be
inactive without leaving enough evidence to decide what to delete.

### Reduce checkpoint-screen cost without reducing rigor

The picker evaluates six checkpoints over two full-motion conditions at 64 envs
(`cloud/train_v11_curriculum.sh:116-127`). First evaluate all six on a deterministic
clean rollout plus a short high-risk phrase; advance only the top two to full-motion
paired screening, then run the full gate once on the winner. Preserve the current
good behavior of re-exporting the selected checkpoint (`cloud/train_v11_curriculum.sh:130-153`).

### Restore a green test baseline

Update the fake-box fixture for the remote-directory preflight and reconcile the
three deliberate show behavior changes and five-second video tolerance. Seven known
red tests train humans to ignore CI, which is particularly dangerous around show
launch and cloud-stage changes. This is small work with disproportionate iteration
value; it should land with the first CPU-only group.

## 6. Architecture-level recommendations

### Keep motion-tracking RL, but simplify the evidence architecture

BeyondMimic/mjlab tracking is still a reasonable core: the anchor policy worked on
the real robot, and v11 improved root-relative fidelity and leg reach in sim. The
evidence does not justify switching algorithms before fixing reference identity,
actuator configuration and evaluation. A new learner on the same broken contracts
would only reset progress.

The bigger architectural change should be a **content-addressed dance bundle**:

```text
bundle_id
 ├─ source video / extraction identity
 ├─ source CSV → cleaned/grounded/repaired CSV hashes
 ├─ every tempo NPZ hash
 ├─ task + reward + DR + mjlab wheel/scene hashes
 ├─ checkpoint + ONNX + generated observation/action metadata
 ├─ evaluation campaign IDs and raw outputs
 └─ hardware calibration/rehearsal records
```

Every preview, promotion and launch should accept only that ID. No copying shared
metadata, globbing the first NPZ, or default-policy fallback.

### Use a three-tier validation ladder

1. **Exact training environment:** clean and DR campaigns in pinned mjlab, with
   realized model dumps.
2. **Independent hardware-uncertainty simulator:** official/identified G1 scene,
   velocity-dependent torque envelope, alternate contacts/friction/latency. Treat
   disagreement as a risk signal.
3. **Robot:** gantry → tethered ground → short phrase → full piece, with explicit
   human confirmation and unique telemetry IDs.

The current same-engine “sim2sim” cannot detect mjlab-specific exploitation, and
the alternate scene is mislabeled. Separating their roles makes both useful.

### Handle 2–3 minute dances as phrases with explicit transitions

A single long tracker compounds drift, thermal load, reference failures and poor
credit assignment. Segment at musically and dynamically safe boundaries; train or
fine-tune phrase policies with explicit entry/exit standing or transition states;
then certify each transition and the composed set. A phase/phrase command can help
long-horizon disambiguation, but only after the metadata/deploy contract can encode
and verify it. Do not enable the existing optional phase input while artifact
generation still assumes a shared 160/770 schema.

### Reintroduce position information as a bounded safety/localization channel

Dropping raw position-dependent actor terms was defensible after the leg-odometry
fling and the reported frozen onboard odometry. Feeding `rt/odommodestate` directly
back into the actor now would be unjustified. Instead:

- build a separate stage-local drift observer from calibrated foot contact,
  kinematics and, later, external/visual localization;
- train with dropout, bias, resets and frozen-estimator faults;
- bound its correction rate/magnitude and keep an estimator-free actor fallback;
- use it first for slow recentering/safety termination between phrases, not fast
  balance control.

This attacks 0.5–0.9 m drift without giving an untrusted estimator authority over
fast policy actions.

### Calibrate to more than one hardware point

The one anchor policy is valuable but cannot identify mass distribution, contact,
latency, torque envelope and estimator errors simultaneously. After repair, collect
matched telemetry for the anchor, v11 or its corrected successor, a quiet standing
sequence, controlled steps, and bounded pushes. Use identical bundle/config IDs in
sim and robot logs. At least two materially different policies/motions are needed
before tuning a gate to claim sim-to-real prediction rather than one-point fitting.

## 7. Prioritized action plan

### Land together A — CPU-only training-integrity gate (highest session savings)

1. Replace the effort event with exact control-ID targeting and 29-joint realized
   assertions (F1).
2. Fix grounding flight/support semantics, rebuild v12 from an immutable git input,
   rerun feasibility/fidelity, and bind CSV + tempo NPZ hashes (F2/F8).
3. Generate policy-specific metadata and repair/invoke the live observation-layout
   gate (F3).
4. Restore all 611 CPU tests green.

**Exit criterion:** one command produces a hashed motion/config bundle; a box smoke
dumps the intended per-joint effort ranges and byte-compares one live actor
observation to the deploy stack. Do not start a full trainer before this passes.

### Land together B — evaluation and model truthfulness (CPU code; GPU rerun needed)

1. Split clean baseline from DR/stress rows, pair seeds and log realized parameters.
2. Stop exactly at the final motion frame; add separate entry/exit tests.
3. Rename/rebuild exact-training and hardware-uncertainty scenes; hash both.
4. Change push schema to delta-v or run a real finite-duration impulse; add unique,
   deduplicated evaluation IDs.

**Exit criterion:** a tiny known policy produces reproducible paired results with no
motion wrap, and every result names policy, motion, metadata, task, wheel, scene,
seed and condition hashes.

### Land together C — show launch safety (CPU-only implementation, then tethered robot)

1. Replace mutable dance paths with one immutable bundle ID.
2. Reauthorize every bundle member immediately before spawn under the run lock.
3. Make the full server-side checklist and short-lived human confirmation nonce
   mandatory.
4. Fail closed on missing/ambiguous files; remove unsigned `free` and default fallback.

**Exit criterion:** mutation/removal/addition of any bundle file, verdict replay,
missing checklist step, or stale confirmation makes process spawn impossible in
tests. Only then rehearse tethered.

### GPU campaign — after A and B

1. 64-env, three-iteration smoke: dump realized effort/gains/mass/friction, actor
   observation bytes, reward nonzero incidence and finite tensors.
2. Corrected base run or checkpoint adaptation on certified v12.
3. Paired short tau/weight screens; three distinct seeds only for finalists.
4. Full clean + isolated stress + selected-composite gate on the winner, exact
   horizon, followed by alternate-scene stress.

The present F-tainted sweep may be retained as a diagnostic of globally weakened
actuation, but it must not select the ankle recipe or be compared as if fix F worked.

### Robot campaign — after hardware repair and C

1. Revalidate remote damping, publisher watchdog, contact guards and all runtime
   thresholds without policy motion.
2. Exact signed bundle: gantry observation replay, then tethered ready/entry/short
   phrase, then full dance.
3. Capture unique telemetry and thermal/endurance evidence; compare to both sims.
4. Require three genuinely distinct clean rehearsals before live, not three API
   submissions of one verdict.

No recent sim-only policy should be called show-ready before this campaign.

## 8. Watchlist — plausible but not yet decisive

1. **Reward-stack conflict and double-counting.** v5 and v11 add scoped arm/leg
   position+orientation rewards on top of whole-body position+orientation, while
   stance and saturation terms also act on legs (`cloud/sim2real_task_v5.py:73-89`;
   `cloud/sim2real_task_v11.py:83-107`). This may be appropriate prioritization, but
   there are no per-term incidence/contribution distributions. Settle with GPU logs
   and one-term ablations; do not delete by inspection alone.
2. **Unprotected root-quaternion smoothing.** Root xyz/joints get a velocity-aware
   reblend, but root quaternion is always smoothed (`tools/motion_quality.py:323-335`).
   Measure per-window torso angular-velocity/acceleration retention against the
   immutable raw source; protect genuine fast yaw/torso hits if retention falls.
3. **Grounding contact identity.** `per_contact_height` uses all geometry centers.
   Log which geom wins each frame and compare center vs sole-surface height; test
   hand/floorwork clips before general use.
4. **Exact compiled mjlab scene.** The wheel/source comparison is decisive about
   provenance and config, but local CPU cannot instantiate the full Warp-expanded
   scene. Dump body mass/inertia, actuator targets/limits, geom collision/friction
   and solver settings from one live box and diff it against both local scenes.
5. **Drift estimator choice.** The current position-blind actor avoids a known bad
   estimator but cannot station-keep globally. Settle only with repaired-robot logs
   comparing onboard odometry, leg/contact integration and an external reference.
6. **Two-to-three-minute endurance.** No recent policy, transition system, thermal
   envelope or show gate is validated at the product's target duration. Settle with
   phrase-composed sim campaigns followed by staged robot thermal/battery telemetry;
   a 50-second Thriller pass is not extrapolatable.
7. **Current cloud state.** `PROJECT_STATE.md` says v12 was stopped while later
   `logs/jobs.md` says a two-box sweep launched. Without box access, neither current
   billing nor run completion is verified. Settle from provider status/logs; this
   does not affect the code findings above.

## Final disposition

The next gains will not come from another reward-weight guess. They come from making
the motion, actuator model, policy schema, evaluation condition and deployed bundle
the **same named, hashed object** from preprocessing through robot launch. Once that
chain is enforced, a GPU run can finally answer a scientific question and a sim pass
can become meaningful evidence rather than another internally consistent claim.
