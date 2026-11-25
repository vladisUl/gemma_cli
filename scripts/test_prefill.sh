#!/usr/bin/env bash
# Replay a known qnn-net-run Prefill-128 sample via direct QNN, then compare
# all 30 output tensors byte for byte. Run on Radxa Q8B (QAIRT 2.47).
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/test_prefill.sh INPUT_LIST_OR_DIR REFERENCE_RESULT_0 [OUTPUT_RESULT_0]

Run from any directory. Relative sample/reference/output paths are resolved
against GEMMA4_MODEL_DIR (default: ~/gemma4_qnn_test).
INPUT_LIST_OR_DIR: a one-line qnn-net-run input list, or a directory containing
                   all 36 named input .raw tensors.
REFERENCE_RESULT_0: Result_0 directory from qnn-net-run for THE SAME sample.
OUTPUT_RESULT_0:    destination, default direct_result_prefill_128/Result_0.

Optional environment: GEMMA4_MODEL_DIR, GEMMA4_QNN_RUNNER, GEMMA4_PREFILL_LOG.
This script does not generate substitute inputs or modify QNN .bin files.
EOF
}

if (( $# < 2 || $# > 3 )); then
  usage >&2
  exit 2
fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GEMMA4_MODEL_DIR:-$HOME/gemma4_qnn_test}"
RUNNER="${GEMMA4_QNN_RUNNER:-$REPO_DIR/build/gemma4-qnn-run}"
INPUT_SOURCE="$1"
REFERENCE_DIR="$2"
OUTPUT_DIR="${3:-direct_result_prefill_128/Result_0}"
LOG="${GEMMA4_PREFILL_LOG:-prefill_reference_test.log}"

fail() { printf 'PREFILL_TEST_ERROR: %s\n' "$*" >&2; exit 1; }

[[ -d "$MODEL_DIR" ]] || fail "Model directory not found: $MODEL_DIR"
[[ -x "$RUNNER" ]] || fail "Build gemma4-qnn-run first: $RUNNER"
cd "$MODEL_DIR"

[[ -f sdk247/libQnnSystem.so ]] || fail "Missing sdk247/libQnnSystem.so"
[[ -f sdk247/libQnnHtp.so ]] || fail "Missing sdk247/libQnnHtp.so"
[[ -f qnn_partition_2_2.bin ]] || fail "Missing qnn_partition_2_2.bin"
[[ -d "$REFERENCE_DIR" ]] || fail "Reference directory not found: $REFERENCE_DIR"

if [[ -f "$INPUT_SOURCE" ]]; then
  INPUT_ARGS=(--input-list "$INPUT_SOURCE")
elif [[ -d "$INPUT_SOURCE" ]]; then
  INPUT_ARGS=(--inputs-dir "$INPUT_SOURCE")
else
  fail "Need an existing input list or input directory: $INPUT_SOURCE"
fi

# Run against one known QAIRT version. Do not inherit Qwen/other QAIRT libs:
# a mismatched DSP skel/stub can fail at deviceCreate before graph execution.
for required in libQnnHtp.so libQnnSystem.so libQnnHtpV68Stub.so libQnnHtpV68Skel.so; do
  [[ -f "sdk247/$required" ]] || fail "Required QAIRT 2.47 library missing: sdk247/$required"
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
printf 'QAIRT_LD_LIBRARY_PATH: %s\nQAIRT_ADSP_LIBRARY_PATH: %s\n' \
  "$LD_LIBRARY_PATH" "$ADSP_LIBRARY_PATH"

COMMON=(
  --system "$PWD/sdk247/libQnnSystem.so"
  --binary "$PWD/qnn_partition_2_2.bin"
  --graph qnn_partition_2
  "${INPUT_ARGS[@]}"
  --outputs-dir "$OUTPUT_DIR"
)

printf 'MODEL_DIR: %s\nINPUT_SOURCE: %s\nREFERENCE_DIR: %s\nOUTPUT_DIR: %s\n' \
  "$PWD" "$INPUT_SOURCE" "$REFERENCE_DIR" "$OUTPUT_DIR"

# Validate all 36 inputs (metadata order and exact byte sizes), without NPU.
"$RUNNER" "${COMMON[@]}" --check-inputs

# QNN execute and strict comparison against all 30 original output files.
# Include stderr in the log; otherwise DSP transport failures are lost.
# Preserve the runner status even though its output passes through tee.
set +e
"$RUNNER" "${COMMON[@]}" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --reference-dir "$REFERENCE_DIR" 2>&1 | tee "$LOG"
runner_status=${PIPESTATUS[0]}
set -e

if (( runner_status != 0 )); then
  if grep -Eq 'QNN error 14001|QNN_TRANSPORT_CONFIG|Failed to load skel|Transport layer setup failed' "$LOG"; then
    printf '%s\n' \
      'HTP_TRANSPORT_ERROR: failed before successful QNN graph execution.' \
      'Verify matching QAIRT 2.47 libQnnHtp.so / V68 Stub / V68 Skel and DSP state.' \
      'Diagnostic: see the DSP error lines in the log above.' >&2
  fi
  printf 'PREFILL_TEST_FAILED: runner_exit=%s log=%s\n' "$runner_status" "$PWD/$LOG" >&2
  exit "$runner_status"
fi

grep -Fxq 'ALL_REFERENCES_MATCH: 30 output tensors' "$LOG" ||
  fail "Expected 30 matching Prefill outputs. See $PWD/$LOG"

printf 'PREFILL_VERIFIED: 30/30 outputs byte-identical\n'
