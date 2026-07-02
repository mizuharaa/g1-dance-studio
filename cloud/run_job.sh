#!/usr/bin/env bash
# tmux-wrapped long-run job pattern for the GreenNode notebook.
# Jobs survive browser tabs, laptop reboots and SSH drops; each writes a
# status file the laptop polls over the cloud transport.
#
#   bash run_job.sh start <name> -- <command...>   # launch in tmux
#   bash run_job.sh status <name>                  # print status JSON
#   bash run_job.sh tail <name> [n]                # last n log lines (default 40)
#   bash run_job.sh list                           # all jobs
#   bash run_job.sh stop <name>                    # SIGTERM the job's tmux session
#
# Status file: $NB_DATA/jobs/<name>.status.json
#   {"name","state": running|done|failed, "rc", "started_at", "finished_at", "log"}
cd "$(dirname "$0")" || exit 1
# shellcheck source=lib.sh
. ./lib.sh
layout

JOBS="$NB_DATA/jobs"
usage() { grep '^#   ' "$0" | sed 's/^#   //'; exit 1; }

cmd="${1:-}"; shift || true
case "$cmd" in
start)
    name="${1:?job name required}"; shift
    [ "${1:-}" = "--" ] && shift
    [ $# -gt 0 ] || usage
    status="$JOBS/$name.status.json"
    logf="$JOBS/$name.log"
    if tmux has-session -t "job-$name" 2>/dev/null; then
        die "job '$name' is already running (tmux session job-$name)"
    fi
    printf '{"name":"%s","state":"running","rc":null,"started_at":"%s","finished_at":null,"log":"%s"}\n' \
        "$name" "$(date -Is)" "$logf" > "$status"
    # the inner script marks done/failed by rewriting the status file on exit
    tmux new-session -d -s "job-$name" \
        "bash -c '
            set -o pipefail
            ( $* ) >>\"$logf\" 2>&1
            rc=\$?
            state=done; [ \$rc -ne 0 ] && state=failed
            printf \"{\\\"name\\\":\\\"$name\\\",\\\"state\\\":\\\"\$state\\\",\\\"rc\\\":\$rc,\\\"started_at\\\":\\\"$(date -Is)\\\",\\\"finished_at\\\":\\\"\$(date -Is)\\\",\\\"log\\\":\\\"$logf\\\"}\n\" > \"$status\"
        '"
    log "job '$name' started (tmux session job-$name, log $logf)"
    ;;
status)
    cat "$JOBS/${1:?job name required}.status.json"
    ;;
tail)
    tail -n "${2:-40}" "$JOBS/${1:?job name required}.log"
    ;;
list)
    ls "$JOBS"/*.status.json >/dev/null 2>&1 || { echo "no jobs"; exit 0; }
    cat "$JOBS"/*.status.json
    ;;
stop)
    tmux kill-session -t "job-${1:?job name required}" \
        && log "job '$1' stopped (status file keeps last state)"
    ;;
*)
    usage
    ;;
esac
