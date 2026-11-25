#!/usr/bin/env bash
# Stage 6.15: same Decode(pos16,4368), compare Decode-only vs Prefill KV prefix.
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
# Dedicated Stage 6.11 test: zero-history first prompt token.
PREFILL_KV_OPTIONS=(
    --prefill-bos-only-compare
    --prefill-first-input-compare
    --decode-only-prompt
    --compare-kv-histories
    --prefill-kv-probe
    --prefill-kv-run
    --prefill-seed-first-prompt
    --prefill-kv-diagnose
)
LOG=cpu_kv_history_stage6_15.log
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
for marker in \
    "PREFILL_FIRST_PROMPT_SEED: prompt_tokens=17 replay_position=0 replay_token=2 kv_rows_from_prefill=0 all_KV_raw_zero=1" \
    "KV_PREFILL_BRIDGE_PROBE_OK: tensors=30 layers=15 valid_tokens=17 padding_rows_ignored=111 copied_rows=0 next_decode_position=0 last_prompt_replay=0 first_prompt_replay=1" \
    "PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=0 input_token=2 saved_decode_inputs_used=0" \
    "PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=0 input_token=2" \
    "PREFILL_FIRST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216" \
    "PREFILL_BOS_ONLY_INPUTS_READY: valid_tokens=1 original_tokens=17 padded_tokens=127" \
    "PREFILL_BOS_ONLY_NPU_OK: graph=qnn_partition_2 outputs=30" \
    "PREFILL_BOS_ONLY_KV_TOTAL: tensors=30 bytes=9216" \
    "PREFILL_BOS_ONLY_STAGE6_12_OK: diagnostic_complete=1" \
    "PREFILL_FIRST_INPUT_TOTAL: tensors=8 shared_bytes=51714" \
    "PREFILL_FIRST_INPUT_STAGE6_13_OK: diagnostic_complete=1" \
    "DECODE_ONLY_BEGIN: prompt_tokens=17 first_token=2" \
    "DECODE_ONLY_PROMPT_KV_READY: positions=17 cached_slices=510" \
    "DECODE_ONLY_FIRST_GENERATION_CANDIDATES: prompt_tokens=17 next_position=17" \
    "DECODE_ONLY_STAGE6_14_OK: steps=17" \
    "KV_HISTORY_CONTROL_BEGIN: pos=16 token_id=4368" \
    "KV_HISTORY_PREFILL_PREFIX_DECODE_OK: pos=16 token_id=4368" \
    "KV_HISTORY_PREFILL_PREFIX_HEAD_OK: logits_bytes=262144" \
    "KV_HISTORY_COMPARE: pos=16 token_id=4368" \
    "KV_HISTORY_STAGE6_15_OK: read_only=1" \
    "PREFILL_KV_DIAG_SUMMARY: tensors=30 compared_bytes=9216" \
    "PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1" \
    "PREFILL_LINK_MULTI_STEP_OK: steps=1 first_position=0 last_position=0 first_token=2 last_prompt_replayed=0 first_prompt_replayed=1"; do
    grep -Fq "$marker" "$LOG" || {
        echo "PREFILL_KV_FIRST_ROW_MISSING: $marker" >&2; exit 1
    }
done
num_tensors="$(grep -c '^PREFILL_KV_DIAG_TENSOR:' "$LOG" || true)"
[[ "$num_tensors" == 30 ]] || {
    echo "PREFILL_KV_FIRST_ROW_DIAG_INCOMPLETE: expected 30, got $num_tensors" >&2
    exit 1
}
grep -Fq 'no_relabel_or_patch_applied=1 diagnostic_only=1' "$LOG" || {
    echo "PREFILL_KV_FIRST_ROW_DIAG_NOT_READONLY" >&2; exit 1
}
[[ "$(grep -c '^PREFILL_BOS_ONLY_KV_TENSOR:' "$LOG")" == 30 ]] || {
    echo "PREFILL_BOS_ONLY_TENSOR_COUNT_INVALID" >&2
    exit 1
}
[[ "$(grep -c '^PREFILL_BOS_ONLY_KV_GROUP:' "$LOG")" == 4 ]] || {
    echo "PREFILL_BOS_ONLY_GROUP_COUNT_INVALID" >&2
    exit 1
}
[[ "$(grep -c '^PREFILL_FIRST_INPUT_TENSOR:' "$LOG")" == 8 ]] || {
    echo "PREFILL_FIRST_INPUT_TENSOR_COUNT_INVALID" >&2
    exit 1
}
[[ "$(grep -c '^DECODE_ONLY_KV_POSITION:' "$LOG")" == 17 ]] || {
    echo "DECODE_ONLY_POSITION_COUNT_INVALID" >&2
    exit 1
}
[[ "$(grep -c '^DECODE_ONLY_STEP_OK:' "$LOG")" == 16 ]] || {
    echo "DECODE_ONLY_EXECUTION_COUNT_INVALID" >&2
    exit 1
}
[[ "$(grep -c '^KV_HISTORY_LOGIT_MAX:' "$LOG")" == 2 ]] || {
    echo "KV_HISTORY_MAX_COUNT_INVALID" >&2
    exit 1
}
echo "KV_HISTORY_STAGE6_15_SCRIPT_OK: same-graph Decode histories compared"
grep -E '^(DECODE_ONLY_FIRST_GENERATION_CANDIDATES:|KV_HISTORY_CONTROL_BEGIN:|KV_HISTORY_LOGIT_MAX:|KV_HISTORY_COMPARE:|KV_HISTORY_STAGE6_15_OK:|DECODE_ONLY_STAGE6_14_OK:)' "$LOG"
if [[ -f "$WORK/cpu_prefill_kv_diagnostic_stage6_10.log" ]]; then
  python3 "$ROOT/scripts/compare_prefill_kv_diagnostics.py" \
    --first "$WORK/$LOG" \
    --last "$WORK/cpu_prefill_kv_diagnostic_stage6_10.log"
else
  echo "PREFILL_KV_POSITION_COMPARE_NOTE: previous Stage6.10 diagnostic log not present"
fi
