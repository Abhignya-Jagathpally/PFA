#!/usr/bin/env bash
# Emits a wake sentinel whenever a job drops a new *.done marker in logs/.
dir="$(cd "$(dirname "$0")/.." && pwd)/logs"
seen=""
while true; do
  for f in "$dir"/*.done; do
    [ -e "$f" ] || continue
    case " $seen " in *" $f "*) continue ;; esac
    seen="$seen $f"
    echo "AGENT_LOOP_WAKE_cafa6 {\"prompt\":\"job finished: $(basename "$f"); check its log, record results in NOTES.md, continue the CAFA-6 goal\"}"
  done
  sleep 30
done
