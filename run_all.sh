#!/usr/bin/env bash
# Rebuild the site (landing, Teams and Players pages) end to end. Pass --force to re-download this season's data first (do this each week).
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
$PY pipeline/download_data.py "$@"
for step in prep seasons fit2026 calibrate calibrate_players qbstats arch dataset prep_skill skill project teams build; do
  echo "== $step"
  $PY "pipeline/$step.py"
done
