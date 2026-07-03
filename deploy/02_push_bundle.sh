#!/usr/bin/env bash
# Push a generated bundle (deploy/bundles/<dance>/) to PC2.
# DRY-RUN by default. Real execution needs BOTH:
#   export CONFIRMED_BY_HUMAN=alois
#   ./02_push_bundle.sh --dance <name> --yes-push
# Copies files only — never starts a controller.
set -euo pipefail
cd "$(dirname "$0")"
# shellcheck source=lib.sh
source ./lib.sh

DANCE="" DRY_RUN=1
while [ $# -gt 0 ]; do
    case "$1" in
        --dance) DANCE="$2"; shift 2 ;;
        --yes-push) DRY_RUN=0; shift ;;
        *) die "unknown arg $1" ;;
    esac
done
[ -n "$DANCE" ] || die "usage: 02_push_bundle.sh --dance <name> [--yes-push]"

require_human
BUNDLE="bundles/${DANCE}"
[ -f "${BUNDLE}/bundle.json" ] || die "no bundle at deploy/${BUNDLE} — run gen_config.py first (it enforces the exam gate)"

# A rehearsal bundle (built with gen_config.py --rehearsal) is never pushable.
if [ -f "${BUNDLE}/REHEARSAL_ONLY" ]; then
    die "refusing: ${BUNDLE} is a REHEARSAL bundle (no exam authorization) — rebuild without --rehearsal once show-ready"
fi

# integrity re-check against the manifest before anything leaves the laptop.
# Matches gen_config.py's manifest exactly: exam.authorized (NOT a 'verdict' string),
# FULL 64-hex sha256 (finding #32), and EVERY hash-pinned file (findings #8/#19).
python3 - "$BUNDLE" <<'PYEOF'
import hashlib, json, sys
from pathlib import Path
b = Path(sys.argv[1])
man = json.loads((b / "bundle.json").read_text())
assert man.get("rehearsal") is not True, "rehearsal bundle is not deployable"
scope = man.get("scope", "full")
# A pushable bundle is either show-ready ("full", exam.authorized) OR "gantry-only"
# (signed+bound but not show-ready — feet-off-ground testing; 10_gantry_test.sh will
# still refuse --stage ground for it). Anything else must not leave the laptop.
if scope == "full":
    assert man["exam"]["authorized"] is True, "manifest exam is not authorized"
elif scope == "gantry-only":
    assert man["exam"].get("gantry_authorized") is True, "gantry-only bundle not gantry_authorized"
    print("*** GANTRY-ONLY bundle: valid for FEET-OFF-GROUND testing only, NOT ground/show ***")
else:
    raise AssertionError(f"unknown bundle scope {scope!r} — refusing to push")
for name, want in man["files_sha256"].items():
    got = hashlib.sha256((b / name).read_bytes()).hexdigest()
    assert got == want, f"{name}: sha mismatch\n  got  {got}\n  want {want}"
for part in ("policy", "motion"):
    f = b / man[part]["file"]
    got = hashlib.sha256(f.read_bytes()).hexdigest()
    assert got == man[part]["sha256"], f"{f}: sha mismatch {got} != {man[part]['sha256']}"
print("bundle integrity ok (exam authorized, all files match full sha256)")
PYEOF

[ "$DRY_RUN" = "0" ] && check_robot_reachable
log "mode: $([ "$DRY_RUN" = 1 ] && echo DRY-RUN || echo LIVE) pushing ${BUNDLE} -> ${PC2_USER}@${PC2_HOST}:${PC2_WORKDIR}/bundles/${DANCE}"

if [ "$DRY_RUN" = "1" ]; then
    log "DRY-RUN scp> ${BUNDLE}/* ${PC2_USER}@${PC2_HOST}:${PC2_WORKDIR}/bundles/${DANCE}/"
else
    pc2 "mkdir -p ${PC2_WORKDIR}/bundles/${DANCE}"
    scp -o BatchMode=yes "${BUNDLE}"/* "${PC2_USER}@${PC2_HOST}:${PC2_WORKDIR}/bundles/${DANCE}/"
fi
log "bundle push issued. NOTHING was started. Next: 10_gantry_test.sh (robot day only)"
