# B3 — Fix F6: show launch fails CLOSED on one re-authorized bundle (CRITICAL)

**Finding:** REPORT §F6 (verified in review). `show_runner._dance_policy_args`
returns `[]` on ANY missing bundle member → deploy_runtime falls back to
hardcoded Thriller defaults; the NPZ is `sorted(glob)[0]` (first alphabetical,
not policy-bound); promotion hashes only the policy, only at promotion time; the
live endpoint (`ui/server.py:962-1007`) never rehashes anything before spawn and
permits the self-described-unsigned `free` bundle in live mode.

**Owned files:** `pipeline/show_runner.py`, `pipeline/shows.py`,
`pipeline/preshow.py`, `ui/server.py` (show endpoints), `ui/frontend/**` (only
if an error surface needs it), `tests/test_show_failclosed.py` (new),
existing show tests. Damp spine / deploy loop untouched.

## Spec
1. **Dance record gains bundle identity (CONVENTIONS §5):** at PROMOTION time
   compute and store `policy_sha256` (exists), `meta_sha256`, `npz_sha256`,
   `motion_sha256`, and `bundle_id` (from the colocated `bundle.json` when
   present — artifacts.verify_manifest must pass; if no manifest, hash the three
   files directly and store `bundle_id: null` with a `legacy_bundle: true`
   flag). Additive JSON, old records load with `.get()`.
2. **`_dance_policy_args` → `resolve_bundle(dance)`:** returns the EXACT three
   paths by re-hashing each against the stored hashes. ANY missing/mismatched
   member → raise `ShowBundleError` (named member + expected/actual sha). The
   NPZ is the RECORDED one (by stored path+hash), never a glob. Returning `[]`
   / default-fallback is DELETED — a dance without a complete verified bundle
   cannot spawn.
3. **Pre-spawn re-authorization under the run lock** (`begin_run` path): resolve
   → rehash → run the server-side `preshow` checklist items that are
   machine-checkable → only then build the command. Human typed-phrase consent
   happens AFTER validation (consent to a verified artifact), and the endpoint
   response on refusal names the failed member (422).
4. **`free` path:** REMOVED from live mode entirely (409 with explanation).
   Trial mode may keep it ONLY behind `G1_ALLOW_UNSIGNED_FREE=1` env (default
   absent) AND the same resolve/rehash flow against the standtail bundle —
   otherwise it also refuses. Document in the task MD's API delta.
5. **Tests (the audit's exit criterion):** mutation of any member after
   promotion → spawn impossible; deletion of the NPZ → impossible; an EXTRA
   lexicographically-earlier npz in the dir → ignored (recorded one used);
   `free` in live → 409; incomplete legacy dance → 422 naming the member;
   happy path still launches (mock spawn).

## Acceptance
No code path from any API endpoint to `spawn_show_process` that skips
resolve+rehash; no default-policy fallback reachable from a dance launch;
existing show tests updated to the new contract (coordinate with B6);
`tests/test_show_failclosed.py` green.

## API delta
POST show-run endpoints: new 422 (bundle member failed, names it) and 409
(free-in-live) responses; Dance JSON gains the §5 fields.
