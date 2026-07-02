#!/usr/bin/env bash
# Send one command to the browser pilot and wait for its result.
# Usage: pilot_cmd.sh '{"action":"goto","url":"https://example.com"}'
set -euo pipefail
PILOT=~/g1-dance/.secrets/pilot
ID=$(date +%s%N)
echo "$1" | python3 -c "import json,sys; c=json.load(sys.stdin); c['id']=$ID; print(json.dumps(c))" > "$PILOT/cmd.json"
for _ in $(seq 1 120); do
  sleep 0.5
  if [ -f "$PILOT/result.json" ] && python3 -c "import json,sys; r=json.load(open('$PILOT/result.json')); sys.exit(0 if r.get('id')==$ID else 1)" 2>/dev/null; then
    cat "$PILOT/result.json"; echo; exit 0
  fi
done
echo '{"ok": false, "error": "timeout waiting for pilot"}'; exit 1
