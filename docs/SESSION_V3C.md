# SESSION PLAN — v3c tethered validation + audio checks (~60 min)

Everything below is staged and offline-verified (2026-07-07 morning): candidate ONNX
loads + infers in the `tv` env, input signature identical to the proven s2r-b policy
(obs[1,160] + time_step), motion npz BYTE-IDENTICAL to the show policy's, policy_meta
PD spec unchanged (v3c = reward-deltas-only recipe). The robot hour is pure execution.

**Candidate:** `data/policies/thriller_v3c_candidate/` — sim: arm RMS 8.75° (s2r-b:
13.81°), gate PASS (drift 0.71 m), 3× signed 100% held-out. Sim predicts hardware arm
RMS well under your measured 13.2°.
**Show policy `data/policies/thriller/` is NOT touched by anything below.**

## 0. Before touching the robot (3 min)
- [ ] Watch `data/previews/rollout_v3c.mp4` — render sign-off (arms crisp?). If NO → stop,
      session becomes audio-only (Step 3).
- [ ] Tether rigged to catch, 2 m area clear, **damping remote in hand**, robot-lan up
      (`ping 192.168.123.164`).

## 1. v3c tethered runs — staged, telemetry auto-records (25 min)
Same staging as the s2r-b session (5 s → 15 s → full). All commands from repo root.
The env `CONFIRMED_BY_HUMAN=alois` + `--i-will-watch-the-robot` are required by design.

```bash
# stage A: 5 s (through the ramp)
CONFIRMED_BY_HUMAN=alois ~/miniconda3/envs/tv/bin/python -m pipeline.deploy_runtime \
  --mode ground-run-legodom --max-secs 5 --i-will-watch-the-robot \
  --policy data/policies/thriller_v3c_candidate/policy.onnx \
  --meta   data/policies/thriller_v3c_candidate/policy_meta.json \
  --motion-npz data/policies/thriller_v3c_candidate/thriller_deploy.npz

# stage B: 15 s (stepping-window entry) — same command, --max-secs 15
# stage C: full dance — same command, --max-secs 52
```
- Gate between stages: previous run ended clean (no STOP line, no cap trips, robot stable).
- Remember the known handoff quirk: end-of-run damp→restore can catch-step ~1–1.5 m
  rightward — keep fence clearance at run end.
- Abort = remote B-damping, or signal the PYTHON pid (`pgrep -f "python.*deploy_runtime"`).

## 2. A/B read-out (5 min, laptop — I do this)
Telemetry lands in `data/telemetry/<stamp>_ground-run-legodom.npz` per run. I compute
hardware arm RMS + leg 2–10 Hz band from the full-dance run and compare against s2r-b's
13.2° / 0.10 rad/s. Decision input for promotion, alongside how it LOOKED to you.

## 3. Robot audio + LED cue (15 min) — docs/SHOW_AUDIO.md checklist
Robot in DAMP (no motion commands anywhere in this step):
- [ ] 30 s speaker smoke test (16 kHz PCM PlayStream), volume check.
- [ ] Measure `AUDIO_LATENCY_COMP`: film screen+speaker, read offset per SHOW_AUDIO.md.
- [ ] LED cue check: blue T-3/-2/-1 → GREEN.
Note: `data/audio/thriller/music.wav` is still the placeholder click track — fine for
latency/smoke. Bring the real song file if you have it → `tools/attach_music.py <file>`.

## 4. (Optional, if A-C were clean) dress rehearsal with cue
```bash
AUDIO_MODE=led CONFIRMED_BY_HUMAN=alois bash tools/show_run.sh \
  --policy data/policies/thriller_v3c_candidate/policy.onnx \
  --meta   data/policies/thriller_v3c_candidate/policy_meta.json \
  --motion-npz data/policies/thriller_v3c_candidate/thriller_deploy.npz
```
(`show_run.sh` passes the policy args through; music/LED cue fires at tick0+4.0 s.)

## 5. Promotion decision (you)
If the run was clean AND the arms visibly better AND hardware numbers confirm:
say the word and I run the guarded promotion machinery (attach_policy → 3× verdict
ingest → promote; candidate becomes the show policy, s2r-b archived as fallback).

## Also ready for you
- Backflip story: `data/previews/rollout_acro1.mp4` (refuses) / `rollout_acro2.mp4`
  (tries, physics says no) + finding in docs/DYNAMIC_SKILLS.md §6. No hardware question.
- v3e (sharp-reference × winning recipe) still training on the box — verdict this
  afternoon; decides the per-joint clamp pipeline default.
- After v3e: I pull the retention checkpoints, then the box is yours to DELETE in the
  GreenNode console (stops the ~18k VND/h meter).
