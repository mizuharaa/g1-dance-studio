# Frontend API gaps

Recorded by Agent C during the React operator-console revamp. These are display/data
gaps only; the frontend does not change backend endpoint semantics.

## Dance statistics

`GET /api/dances` does not expose the following per-dance provenance required by the
Stats screen:

- training iterations, wall time, run name, checkpoint, and attributed cloud cost;
- `sim_gap_check` condition results, especially 40/60/80 ms survival, MPKPE, falls,
  and ankle-torque statistics;
- a stable artifact link or embedded summary for the gap-check JSON.

The new UI labels these values "Not exposed by API" instead of inventing them.

Suggested additive shape: `dance.training` and `dance.latency_gate.conditions[]`.

## Audit history

There is no canonical global event endpoint. The UI currently derives a timeline from
the dance records (`repeatability.history`, current promotion status/updated time) and
show records (`deploy`, `outcome`). That cannot faithfully reconstruct every historical
promotion, demotion, rollback, or artifact attachment after records are overwritten.

Incident timing is also unstructured in older records. To answer "at which second did
the failure occur?" reliably, incidents need a numeric `performance_second` plus the
verdict/policy identifiers that were active at that instant.

Suggested additive endpoint: `GET /api/audit?dance_id=&type=&before=` returning immutable
events with `at`, `performance_second`, `event_type`, `dance_id`, `show_id`, `policy_sha`,
and structured metrics.

## Live run progress

`GET /api/shows/runs/current` exposes a coarse log-derived phase but no current tick,
total ticks, performance time, or explicit entry/stand-handback substate. The UI maps
the existing phases into the requested state machine and estimates progress from
`started_at + dance.duration_s`.

Suggested additive fields: `state`, `tick`, `total_ticks`, `performance_second`, and
`handoff_owner`. The existing `phase` and STOP semantics should remain unchanged.
