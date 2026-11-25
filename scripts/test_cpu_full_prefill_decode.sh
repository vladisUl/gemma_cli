#!/usr/bin/env bash
# Stage 6.4: all eight Prefill and all eight Decode non-KV inputs from CPU RAM.
# Saved data remain strict byte references. Initial Prefill 28 KV and Decode 30 KV
# are still replayed from QNN input lists; two-step Decode KV feedback stays in RAM.
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
            "$PWD/derived_prefill_128_token_ids_i32.raw" \
            "$PWD/auto_result_decode_4/Result_0" \
            "$PWD/auto_result_head_4/Result_0"; do
  if [[ ! -e "$path" ]]; then
    echo "CPU_FULL_NONKV_ERROR: required model/reference missing: $path" >&2
    exit 2
  fi
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
[[ "$(stat -c %s derived_prefill_128_token_ids_i32.raw)" == 512 ]] || { echo "PREFILL_TOKENS_INVALID: expected 512 bytes" >&2; exit 2; }
echo "CPU_FULL_PREFILL_DECODE_BEGIN: Prefill-128 tokens from verified 512-byte ID file; Decode steps3/4 token=568"
echo "CPU_FULL_PREFILL_REFERENCE_NOTE: all eight non-KV inputs for both Prefill and Decode byte-compared before QNN"
PREFILL_KV_OPTIONS=()
case "${GEMMA4_PREFILL_KV_PROBE:-0}" in
    0) ;;
    1) PREFILL_KV_OPTIONS+=(--prefill-kv-probe) ;;
    *) echo "CPU_FULL_PREFILL_ERROR: GEMMA4_PREFILL_KV_PROBE must be 0 or 1" >&2; exit 2 ;;
esac
case "${GEMMA4_PREFILL_KV_RUN:-0}" in
    0) ;;
    1)
      if [[ "${GEMMA4_PREFILL_KV_PROBE:-0}" != 1 ]]; then
        echo "CPU_FULL_PREFILL_ERROR: link run requires KV probe=1" >&2; exit 2
      fi
      PREFILL_KV_OPTIONS+=(--prefill-kv-run --prefill-kv-token 568)
      ;;
    *) echo "CPU_FULL_PREFILL_ERROR: GEMMA4_PREFILL_KV_RUN must be 0 or 1" >&2; exit 2 ;;
esac
LOG=cpu_full_prefill_decode.log
if [[ "${GEMMA4_PREFILL_KV_RUN:-0}" == 1 ]]; then
  LOG=cpu_prefill_link_stage6_6.log
fi
"$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --chain-decode-head \
    "${PREFILL_KV_OPTIONS[@]}" \
    --cpu-prefill-tokens "$PWD/derived_prefill_128_token_ids_i32.raw" \
    --cpu-prefill-section0 "$MODEL0" \
    --cpu-prefill-section2 "$MODEL2" \
    --cpu-prefill-aux-model "$MODEL1" \
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
    2>&1 | tee "$LOG"

for marker in \
    "CPU_PREFILL_ALL_NONKV_QNN_REPLACED: tensors=8" \
    "ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors" \
    "KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical" \
    "CPU_EMBED_QNN_REPLACED: scope=first token=568 tensors=2" \
    "CPU_EMBED_QNN_REPLACED: scope=next token=568 tensors=2" \
    "DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical" \
    "DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical"; do
    grep -Fq "$marker" "$LOG" || { echo "CPU_FULL_PREFILL_DECODE_MISSING_MARKER: $marker" >&2; exit 1; }
done
[[ "$(grep -c 'CPU_PREFILL_INPUT_MATCH:' "$LOG")" == 8 ]] || { echo "CPU_FULL_PREFILL_DECODE_EXPECTED_8_INPUT_MATCHES" >&2; exit 1; }
echo "CPU_FULL_PREFILL_DECODE_OK: Prefill-128 eight CPU inputs, 3 QNN graphs 62 outputs, two Decode steps and LM Head byte-identical"
if [[ "${GEMMA4_PREFILL_KV_RUN:-0}" == 1 ]]; then
  for marker in \
      "PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=17 input_token=568 saved_decode_inputs_used=0" \
      "PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=17 input_token=568" \
      "PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1 no_saved_decode_reference_claim=1"; do
    grep -Fq "$marker" "$LOG" || { echo "PREFILL_LINK_STAGE6_6_MISSING: $marker" >&2; exit 1; }
  done
  echo "PREFILL_LINK_STAGE6_6_OK: Prefill 30 KV blocks (17 rows) -> real Decode(pos17, token568) -> real LM Head, no independent reference claim"
fi
