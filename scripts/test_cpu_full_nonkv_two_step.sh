#!/usr/bin/env bash
# Stage 5.9: all eight Decode non-KV inputs calculated in RAM from Section0/1/2.
# Saved steps 3/4 are strict byte references. Initial 30 K/V still from step3 files.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
if [[ ! -d "$WORK" ]]; then
  echo "CPU_FULL_NONKV_ERROR: missing test directory: $WORK" >&2
  exit 2
fi
cd "$WORK"
MODEL0="$PWD/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
MODEL1="$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"
MODEL2="$PWD/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"
for path in "$MODEL0" "$MODEL1" "$MODEL2" \
            "$PWD/sdk247/libQnnSystem.so" \
            "$PWD/sdk247/libQnnHtp.so" \
            "$PWD/persistent_jobs_decode_prefill_head.txt" \
            "$PWD/auto_decode_step4_list.txt" \
            "$PWD/auto_result_decode_4/Result_0" \
            "$PWD/auto_result_head_4/Result_0"; do
  if [[ ! -e "$path" ]]; then
    echo "CPU_FULL_NONKV_ERROR: required model/reference missing: $path" >&2
    exit 2
  fi
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
echo "CPU_FULL_NONKV_TWO_STEP_BEGIN: Section0/2 token=568 for both, Section1 probe=2"
echo "CPU_FULL_NONKV_REFERENCE_NOTE: all eight non-KV inputs compared to saved QNN inputs before replacement"
"$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --chain-decode-head \
    --cpu-aux-model "$MODEL1" \
    --cpu-aux-pos 3 --cpu-aux-token 2 \
    --cpu-embed-section0 "$MODEL0" \
    --cpu-embed-section2 "$MODEL2" \
    --cpu-embed-token 568 \
    --kv-auto-pos --kv-next-list auto_decode_step4_list.txt \
    --kv-run-next \
    --cpu-aux-next-pos 4 --cpu-aux-next-token 2 \
    --cpu-embed-next-token 568 \
    --kv-next-reference-dir auto_result_decode_4/Result_0 \
    --kv-next-head-reference-dir auto_result_head_4/Result_0 \
    2>&1 | tee cpu_full_nonkv_two_step.log
echo "CPU_FULL_NONKV_TWO_STEP_OK: CPU Section0/1/2 all non-KV inputs + in-RAM KV feedback verified"
