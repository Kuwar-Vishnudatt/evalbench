#!/usr/bin/env bash
# Runs the bundled golden set against both the good and deliberately
# degraded example targets, then compares them — demonstrating the full
# pipeline (concurrent evaluation, caching, bootstrap statistics) end to
# end with no API keys or external services required.
set -euo pipefail
cd "$(dirname "$0")/.."

rm -f /tmp/evalbench-demo-cache.db /tmp/evalbench-baseline.json /tmp/evalbench-candidate.json

echo "--- running baseline (good target) ---"
./bin/evalbench run -cases=testdata/golden.json -target=good \
  -cache=/tmp/evalbench-demo-cache.db -out=/tmp/evalbench-baseline.json

echo
echo "--- running candidate (degraded target: truncated context) ---"
./bin/evalbench run -cases=testdata/golden.json -target=degraded -degraded-truncate-words=3 \
  -cache=/tmp/evalbench-demo-cache.db -judge-id=fake-v1-degraded -out=/tmp/evalbench-candidate.json

echo
echo "--- comparing ---"
./bin/evalbench compare -baseline=/tmp/evalbench-baseline.json -candidate=/tmp/evalbench-candidate.json || true
