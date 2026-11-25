#!/usr/bin/env bash
# Stage 4: replay Prefill, Decode, LM Head using 3 persistent QNN contexts.
# Use only saved inputs AND the matching qnn-net-run reference outputs.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/test_persistent_three.sh
Runs on Radxa Q8B from ~/gemma4_qnn_test, using saved tests.

Optional environment to select exact successful reference pairs:
  GEMMA4_MODEL_DIR         default: $HOME/gemma4_qnn_test
  GEMMA4_QNN_RUNNER        default: <repo>/build/gemma4-qnn-run
  PREFILL_LIST             default: prefill_real_input_list.txt
  PREFILL_REF              default: result_prefill_real_prompt/Result_0
  DECODE_LIST              default: auto_decode_step3_list.txt
  DECODE_REF               default: auto_result_decode_3/Result_0
  HEAD_INPUT               default: real_result_decode_17/Result_0/<actual hidden tensor>
  HEAD_REF                 default: real_result_head_17/Result_0
  PERSISTENT_THREE_LOG     default: persistent_three.log

No dummy inputs. Any missing saved sample/reference stops BEFORE NPU init.
EOF
}

if (( $# != 0 )); then usage >&2; exit 2; fi

REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${GEMMA4_MODEL_DIR:-$HOME/gemma4_qnn_test}"
RUNNER="${GEMMA4_QNN_RUNNER:-$REPO_DIR/build/gemma4-qnn-run}"
PREFILL_LIST="${PREFILL_LIST:-prefill_real_input_list.txt}"
PREFILL_REF="${PREFILL_REF:-result_prefill_real_prompt/Result_0}"
DECODE_LIST="${DECODE_LIST:-auto_decode_step3_list.txt}"
DECODE_REF="${DECODE_REF:-auto_result_decode_3/Result_0}"
HEAD_INPUT="${HEAD_INPUT:-real_result_decode_17/Result_0/LanguageModel_decode_graph_LanguageModel_transformer_stack_transformer_transformer_final_norm_composite_quantized_native.raw}"
HEAD_REF="${HEAD_REF:-real_result_head_17/Result_0}"
LOG="${PERSISTENT_THREE_LOG:-persistent_three.log}"

fail() {
  printf 'PERSISTENT_THREE_ERROR: %s\n' "$*" >&2
  exit 1
}
check_file() {
  [[ -f "$1" ]] || fail "Missing file: $1"
}
check_dir() {
  [[ -d "$1" ]] || fail "Missing directory: $1"
}
check_no_space() {
  [[ ! "$1" =~ [[:space:]] ]] || fail "Manifest cannot encode whitespace in path: $1"
}
count_references() {
  local dir="$1" expected="$2" found
  check_dir "$dir"
  found="$(find "$dir" -maxdepth 1 -type f -name '*_native.raw' | wc -l)"
  [[ "$found" -eq "$expected" ]] ||
    fail "Expected $expected reference outputs in $dir, found $found"
}

check_dir "$MODEL_DIR"
[[ -x "$RUNNER" ]] || fail "Runner missing: $RUNNER (build it first)"
cd "$MODEL_DIR"
for lib in libQnnHtp.so libQnnSystem.so libQnnHtpV68Stub.so libQnnHtpV68Skel.so; do
  check_file "sdk247/$lib"
done
for bin in qnn_partition_2_2.bin qnn_partition_0_0.bin qnn_partition_1_1.bin; do
  check_file "$bin"
done

check_file "$PREFILL_LIST"
check_file "$DECODE_LIST"
check_file "$HEAD_INPUT"
count_references "$PREFILL_REF" 30
count_references "$DECODE_REF" 31
count_references "$HEAD_REF" 1

[[ "$(wc -c < "$HEAD_INPUT")" -eq 1536 ]] ||
  fail "LM Head input must be 1536 bytes: $HEAD_INPUT"
check_file "$HEAD_REF/StatefulPartitionedCall_30_native.raw"
[[ "$(wc -c < "$HEAD_REF/StatefulPartitionedCall_30_native.raw")" -eq 262144 ]] ||
  fail "LM Head reference must be 262144 bytes: $HEAD_REF"

# --jobs parser intentionally accepts only whitespace-separated paths.
for p in "$PREFILL_LIST" "$PREFILL_REF" "$DECODE_LIST" "$DECODE_REF" "$HEAD_INPUT" "$HEAD_REF"; do
  check_no_space "$p"
done

# LM Head input from the decode step is byte-for-byte the same pair used in Stage 2.
HEAD_LIST="persistent_head_input_list.txt"
MANIFEST="persistent_jobs_all3.txt"
check_no_space "$HEAD_LIST"
printf '%s\n' "$HEAD_INPUT" > "$HEAD_LIST"
{
  printf 'qnn_partition_2_2.bin qnn_partition_2 %s %s\n' "$PREFILL_LIST" "$PREFILL_REF"
  printf 'qnn_partition_0_0.bin qnn_partition_0 %s %s\n' "$DECODE_LIST" "$DECODE_REF"
  printf 'qnn_partition_1_1.bin qnn_partition_1 %s %s\n' "$HEAD_LIST" "$HEAD_REF"
} > "$MANIFEST"

export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"

printf 'STAGE4_MANIFEST_READY: %s\n' "$PWD/$MANIFEST"
cat "$MANIFEST"
printf '\nMemory before context loading:\n'
free -h
printf '\nLoading all three QNN contexts; this may require substantial RAM/DSP memory.\n'
set +e
"$RUNNER" \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --jobs "$MANIFEST" 2>&1 | tee "$LOG"
status=${PIPESTATUS[0]}
set -e

if (( status != 0 )); then
  printf 'PERSISTENT_THREE_FAILED: exit=%s log=%s\n' "$status" "$PWD/$LOG" >&2
  free -h >&2
  exit "$status"
fi
grep -Fqx 'ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors' "$LOG" ||
  fail "Missing successful 3-graph/62-output comparison in $LOG"
printf 'PERSISTENT_THREE_VERIFIED: 3 graphs, 62 output tensors byte-identical\n'
