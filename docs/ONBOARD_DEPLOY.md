# Onboard policy deploy (the wireless show) — status + debug runbook

**Why onboard:** running the 50 Hz balance loop on the laptop over wifi is a fall risk
(jitter/dropout → stale commands). The correct design (what Unitree does) is to run the
policy **onboard PC2** (the Jetson) on the local control net (eth1), so the real-time loop
never touches wifi; wifi/tailscale carries only the trigger (start/stop). This removes the
wireless-control risk entirely.

## What is already SET UP on PC2 (2026-07-07)
- `teleimager` conda env (Python 3.10) has `unitree_sdk2py` + `cyclonedds` + numpy 1.26.
- **onnxruntime 1.23.2 installed** into `teleimager` (aarch64 wheel — the tiny MLP runs on CPU
  in sub-ms; the Jetson is plenty).
- **Code + policy bundled** to `~/g1-dance` on PC2: `pipeline/{__init__,deploy_runtime,leg_odometry}.py`
  + `data/policies/thriller_standtail_candidate/{policy.onnx,policy_meta.json,thriller_deploy.npz}`.
  deploy_runtime is nearly standalone (numpy at top; SDK lazy-imported) so this minimal set runs it.
- `IFACE` is env-configurable (`ROBOT_IFACE`); onboard we use **eth1** (192.168.123.164 = the local
  control net). wlan0 (192.168.21.237) + tailscale are available for the trigger.

## The BLOCKER (needs a session with the operator at the robot)
Running the onboard subscriber fails at topic creation:
```
CYCLONEDDS_URI=/home/kc_ws/cyclonedds.xml \
  ~/miniconda3/envs/teleimager/bin/python -m pipeline.deploy_runtime --mode read --iface eth1 ...
-> cyclonedds.core.DDSException: [DDS_RETCODE_PRECONDITION_NOT_MET]
   Occurred upon initialisation of a cyclonedds.topic.Topic  (rt/lowstate)
```
`PRECONDITION_NOT_MET` on a Topic almost always = **a topic of that name already exists with an
incompatible type descriptor**. PC2's onboard `master_service` (running) owns `rt/lowstate` with the
type from ITS SDK build. Our Python subscriber uses PC2's **`kc_ws` SDK (sha 58c3f62)**, whose
`LowState_` IDL evidently does not match.

**Key evidence pointing at SDK/IDL version, not config:**
- The **laptop reads `rt/lowstate` fine** over ethernet using `~/robot/unitree_sdk2_python` — so that
  SDK's `LowState_` type IS compatible with `master_service`. PC2's `kc_ws` SDK is a *different* build.
- Matching the robot's own DDS XML (`CYCLONEDDS_URI=/home/kc_ws/cyclonedds.xml`) did **not** fix it
  (and that XML even names `eth0` while the control net is `eth1` — likely not master_service's actual
  config anyway).

## ROOT CAUSE FOUND (2026-07-07, six approaches tried)
It is **not** the SDK version and **not** the `rt/lowstate` type specifically. A minimal probe
subscribing to a **benign topic name** (`rt/PROBE_benign_xyz`) with the `LowState_` type **also**
fails `PRECONDITION_NOT_MET`. `ChannelFactoryInitialize` succeeds; the FIRST `ChannelSubscriber.Init()`
(topic/type creation) fails. So it is a **domain-level TYPE-registration conflict**: `master_service`
(C++) has already registered the `unitree_hg` `LowState_` type in the DDS domain, and our co-located
Python participant registering the same fully-qualified type name with a different sertype is rejected —
for ANY topic name. The laptop avoids this because it is a separate host whose own domain instance
negotiates the type over the wire (XTypes), never re-registering into `master_service`'s registry.

Tried and FAILED (all same PRECONDITION_NOT_MET): default DDS config; `/home/kc_ws/cyclonedds.xml`;
the laptop's working `~/robot` SDK on PC2; an explicit eth1 + SharedMemory-off config; benign topic
name; both SDKs. => A **parallel Python DDS subscriber co-located with `master_service` cannot register
the type.** This is the wrong integration shape, not a tuning problem.

## THE RIGHT PATH (needs Unitree's onboard method + operator; this is how their demos work)
Unitree's onboard policy demos do NOT stand up a competing participant. Options, best first:
1. **Run the policy inside the robot's control framework/container** (the motion_tracking_controller /
   `qiayuanl/unitree:jazzy` image referenced in docs/architecture.md), which already owns the compatible
   `LowState_` type and the control loop — deploy the ONNX + our obs/action glue there. Needs `sudo`
   docker access + Unitree's onboard-deploy docs.
2. **Ask Unitree** for the supported way to read `rt/lowstate` from a second onboard process (a
   CycloneDDS XTypes / type-discovery config that permits type coexistence, or a shared type library).
3. **Fallback if wireless is required sooner:** laptop-in-the-loop but WIRELESS — see docs/WIRELESS_SHOW.md
   + tools/wireless_preflight.py. HARD constraint: the control net (192.168.123.x) is physically the
   ethernet/eth1; wifi does not reach it without a bridge, and tailscale adds VPN latency unfit for 50 Hz.
   So this needs the robot to BRIDGE the control net onto wifi (or a wifi AP on the control subnet) AND
   the preflight (RTT + DDS staleness p99 < ~10 ms, 0 loss, sustained) to PASS first. Higher risk than
   onboard; the comms-loss deadman is the only backstop.

## Superseded debug plan (kept for history — the SDK-swap hypothesis, now DISPROVEN)
1. **Align the SDK.** Put the laptop's WORKING SDK (`~/robot/unitree_sdk2_python`, the one whose
   `LowState_` matches `master_service`) onto PC2 and import IT instead of `kc_ws`'s (PYTHONPATH or a
   venv install). Re-run `--mode read --iface eth1`. This is the leading hypothesis: same SDK the
   laptop uses → same type descriptor → topic compatible. (unitree_sdk2py is pure-Python IDL, so
   arch-independent; only cyclonedds is native and already present.)
2. If still failing, **inspect the live type**: which SDK/commit built `master_service` (ask Unitree
   / the robot image docs), and match `unitree_sdk2py` to it. Compare the `LowState_` IDL hash.
3. **SHM/iceoryx check:** confirm whether `iox-roudi` is running (co-located SHM transport). If so, a
   type mismatch is fatal over SHM; align types OR force network transport via a CYCLONEDDS_URI that
   disables shared memory, and retest.
4. **Domain/participant:** confirm the domain id `master_service` uses; our participant must join the
   same one. The kc_ws XML uses `Domain id="any"`.

## Once `--mode read` works onboard (still no motor commands)
- It prints finite/bounded actions from the real onboard `rt/lowstate` → the onboard policy path is
  proven. Then, and only then, with the OPERATOR PRESENT + remote + tether:
  - onboard `--mode ground-run-legodom` (the full safety spine — entry catch, fall detector, exit
    stand handoff, start-pose guard — is in the bundled deploy_runtime), tethered first, exactly like
    the laptop staircase we already validated.
- **Trigger:** wrap the onboard run in a small script on PC2; fire it wirelessly (ssh over tailscale/
  wlan0, or map a remote button). The trigger is not real-time; only the local eth1 control loop is.

## Safety notes
- This G1 has no torque-cut e-stop; the remote's B-damp + power switch are the only hard stops.
- Onboard motion is a first-of-its-kind run for this project — tether-first, operator-present, and
  the comms path in the loop is now LOCAL (eth1), which is the whole point (no wifi jitter in control).
- Do the SDK/DDS debugging near the live control service ONLY with the operator aware.
