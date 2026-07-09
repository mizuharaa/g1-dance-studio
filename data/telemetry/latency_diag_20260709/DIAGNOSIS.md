# 2026-07-09 fall diagnosis — sim2real LATENCY gap

Run 20260709-192740-ba7ccc (dance thriller_csv_ankle_penalty, ground-run-legodom, wired
enp0s31f6). Robot danced ~44 s, drifted side/back/forward, then fully fell at tick 2222
(~45 s); fall-detector damped + handed to onboard; tether caught it. NOT the exit-fix,
NOT a false trigger (robot was on the ground), NOT the tether.

## Evidence (4 independent signals agree)
1. sim gap_check.json: 0 falls at nominal/noise/10ms/20ms delay; MANY falls at 40ms delay
   (3/30/44/1 across dance sections). Policy robust <=20ms, collapses at 40ms.
2. Hardware fall signature (imu): tilt in normal <=16 deg envelope through 40 s, then climbs
   >16 deg from ~42 s, right knee buckles to 2.35 rad at ~45 s, torso -0.24 m -> abort.
3. Telemetry cross-correlation (tools/measure_latency_from_telemetry.py): effective
   command->response lag = leg median 80 ms (40-121 ms); light arm joints (near-perfect
   tracking) lag 60-100 ms => pure sensorimotor latency >=40 ms. See cmd_response_lag.txt.
4. Comms ruled out: wired, ping RTT 0.16 ms, DDS staleness ~2 ms baseline. The 80 ms is
   intrinsic to the actuation + leg-odometry estimation pipeline (wired or wireless).

## Root cause (config)
cloud/sim2real_task.py trains latency DR only to 20 ms (CMD_DELAY_MAX_LAG=4,
OBS_DELAY_MAX_LAG=1); 40 ms was left EVAL-ONLY. Real hardware latency is 40-80 ms ->
outside the trained range. "Latency DR is hygiene, not headline" assumption was wrong
for this stack.

## Fix
Retrain with latency DR extended to cover ~40-80 ms (+ margin) and push robustness;
extend gap_check to eval 60/80 ms. Recipe applied in the same file.
