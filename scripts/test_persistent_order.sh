#!/usr/bin/env bash
# Diagnose QNN HTP concurrent context allocations by replaying selected
# saved graphs, in the requested order, in one process. No new model inputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/test_persistent_order.sh GRAPH [GRAPH ...]
GRAPH: prefill | decode | head  (1 to 3 unique names; order matters)

Example: bash ../gemma4-cli-stage1/scripts/test_persistent_order.sh decode head

Uses persistent_jobs_all3.txt produced by test_persistent_three.sh.
Does not change QNN binaries or saved reference outputs.
Environment: GEMMA4_MODEL_DIR, GEMMA4_QNN_RUNNER, PERSISTENT_MANIFEST,
             PERSISTENT_ORDER_LOG (optional).
EOF
}

if (( $# < 1 || $# > 3 )); then usage >&2; exit 2; fi
REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GEMMA4_MODEL_DIR:-$HOME/gemma4_qnn_test}"
RUNNER="${GEMMA4_QNN_RUNNER:-$REPO_DIR/build/gemma4-qnn-run}"
MANIFEST="${PERSISTENT_MANIFEST:-persistent_jobs_all3.txt}"

fail() { printf 'PERSISTENT_ORDER_ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$MODEL_DIR" ]] || fail "Missing model directory: $MODEL_DIR"
[[ -x "$RUNNER" ]] || fail "Missing compiled gemma4-qnn-run: $RUNNER"
cd "$MODEL_DIR"
[[ -f "$MANIFEST" ]] || fail "Missing manifest $MANIFEST; run test_persistent_three.sh once to generate it"
[[ -f sdk247/libQnnHtp.so && -f sdk247/libQnnSystem.so ]] ||
  fail "Missing SDK 2.47 libraries in sdk247"

declare -A entries
while IFS= read -r line || [[ -n "$line" ]]; do
  [[ "$line" =~ ^[[:space:]]*$ ]] && continue
  [[ "$line" =~ ^[[:space:]]*# ]] && continue
  read -r binary graph input_list ref extra <<< "$line"
  [[ -n "${binary:-}" && -n "${graph:-}" &&
     -n "${input_list:-}" && -n "${ref:-}" &&
     -z "${extra:-}" ]] ||
    fail "Malformed manifest line: $line"
  [[ -z "${entries[$graph]:-}" ]] || fail "Duplicate graph in manifest: $graph"
  [[ -f "$binary" ]] || fail "Missing binary: $binary"
  [[ -f "$input_list" ]] || fail "Missing input list: $input_list"
  [[ -d "$ref" ]] || fail "Missing matching reference directory: $ref"
  entries["$graph"]="$line"
done < "$MANIFEST"

declare -A seen
selected=""
total_outputs=0
manifest_suffix=""
OUT_MANIFEST=""
for name in "$@"; do
  case "$name" in
    prefill) graph=qnn_partition_2; outputs=30 ;;
    decode)  graph=qnn_partition_0; outputs=31 ;;
    head)    graph=qnn_partition_1; outputs=1  ;;
    *) fail "Unknown graph '$name'; expected prefill|decode|head" ;;
  esac
  [[ -z "${seen[$graph]:-}" ]] || fail "Duplicate graph selection: $name"
  [[ -n "${entries[$graph]:-}" ]] || fail "Graph $name missing from $MANIFEST"
  seen["$graph"]=1
  manifest_suffix="${manifest_suffix:+${manifest_suffix}_}$name"
  total_outputs=$((total_outputs + outputs))
done
OUT_MANIFEST="persistent_jobs_${manifest_suffix}.txt"
LOG="${PERSISTENT_ORDER_LOG:-persistent_${manifest_suffix}.log}"

# All preflight checks are complete before touching the NPU.
: > "$OUT_MANIFEST"
for name in "$@"; do
  case "$name" in
    prefill) graph=qnn_partition_2 ;;
    decode) graph=qnn_partition_0 ;;
    head) graph=qnn_partition_1 ;;
  esac
  printf '%s\n' "${entries[$graph]}" >> "$OUT_MANIFEST"
done

export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
printf 'PERSISTENT_ORDER: %s\n' "$*"
printf 'PERSISTENT_EXPECT: %s graphs, %s output tensors\n' "$#" "$total_outputs"
cat "$OUT_MANIFEST"
free -h

set +e
"$RUNNER" \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --jobs "$OUT_MANIFEST" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

if (( status != 0 )); then
  printf 'PERSISTENT_ORDER_FAILED: %s exit=%s log=%s\n' "$manifest_suffix" "$status" "$PWD/$LOG" >&2
  free -h >&2
  exit "$status"
fi
grep -Fqx "ALL_JOBS_REFERENCES_MATCH: $# graphs $total_outputs output tensors" "$LOG" ||
  fail "Missing expected successful comparison in $LOG"
printf 'PERSISTENT_ORDER_VERIFIED: %s graphs=%s outputs=%s\n' \
  "$manifest_suffix" "$#" "$total_outputs"
