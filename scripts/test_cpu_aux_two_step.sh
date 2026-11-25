#!/usr/bin/env bash
# Two saved Decode steps: compute CPU AUX in RAM for both, KV feedback in RAM,
# run QNN Decode+LM Head persistently; verify exact step-4 QNN references.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
if [[ ! -d "$WORK" ]]; then
  printf "CPU_AUX_TEST_ERROR: missing test directory: %s\n" "$WORK" >&2
  exit 2
fi
cd "$WORK"
MODEL="$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"
for p in "$MODEL" "$PWD/sdk247/libQnnSystem.so" "$PWD/sdk247/libQnnHtp.so" \
         "$PWD/persistent_jobs_decode_prefill_head.txt" \
         "$PWD/auto_decode_step4_list.txt" \
         "$PWD/auto_result_decode_4/Result_0" "$PWD/auto_result_head_4/Result_0"; do
  if [[ ! -e "$p" ]]; then
    echo "CPU_AUX_TEST_ERROR: required file or directory missing: $p" >&2
    exit 2
  fi
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
echo "CPU_AUX_TWO_STEP_BEGIN: CPU RoPE/mask for pos=3 and pos=4, token probe=2"
echo "CPU_AUX_TWO_STEP_NOTE: ID 2 is a non-padding probe, not the identified original token"
"$ROOT/build/gemma4-qnn-run" \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --jobs persistent_jobs_decode_prefill_head.txt \
  --chain-decode-head \
  --cpu-aux-model "$MODEL" \
  --cpu-aux-pos 3 --cpu-aux-token 2 \
  --kv-auto-pos --kv-next-list auto_decode_step4_list.txt \
  --kv-run-next \
  --cpu-aux-next-pos 4 --cpu-aux-next-token 2 \
  --kv-next-reference-dir auto_result_decode_4/Result_0 \
  --kv-next-head-reference-dir auto_result_head_4/Result_0 \
  2>&1 | tee cpu_aux_two_step.log
echo "CPU_AUX_TWO_STEP_EXIT_OK: all requested comparisons completed"
