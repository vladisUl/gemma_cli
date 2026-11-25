#!/usr/bin/env bash
# Stage 6.9: 3 linked Decode steps with explicit INT8 tie policy and candidate listing.
# Saved data remain strict byte references. Initial Prefill 28 KV and Decode 30 KV
# are still replayed from QNN input lists; two-step Decode KV feedback stays in RAM.
set -euo pipefail
# Explicit self-contained opt-in for experimental prompt-last seed.
export GEMMA4_PREFILL_KV_PROBE=1 GEMMA4_PREFILL_KV_RUN=1
# This standalone test passes --prefill-kv-steps 3 explicitly.
# Prevent an inherited stage-6.7 environment value from duplicating the CLI flag.
export GEMMA4_PREFILL_KV_STEPS=1
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
      PREFILL_KV_OPTIONS+=(--prefill-kv-run --prefill-seed-last-prompt --prefill-kv-steps 3 --prefill-tie-policy lowest-id)
      case "${GEMMA4_PREFILL_KV_STEPS:-1}" in
          1) ;;
          2|3|4)
            PREFILL_KV_OPTIONS+=(--prefill-kv-steps "${GEMMA4_PREFILL_KV_STEPS}")
            ;;
          *)
            echo "CPU_FULL_PREFILL_ERROR: GEMMA4_PREFILL_KV_STEPS must be 1..4" >&2
            exit 2
            ;;
      esac
      ;;
    *) echo "CPU_FULL_PREFILL_ERROR: GEMMA4_PREFILL_KV_RUN must be 0 or 1" >&2; exit 2 ;;
esac
LOG=cpu_prefill_tie_stage6_9.log
if [[ "${GEMMA4_PREFILL_KV_RUN:-0}" == 1 ]]; then
  LOG=cpu_prefill_tie_stage6_9.log
  if [[ "${GEMMA4_PREFILL_KV_STEPS:-1}" != 1 ]]; then
    LOG=cpu_prefill_multistep_stage6_7.log
  fi
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
if [[ "${GEMMA4_PREFILL_KV_RUN:-0}" != 1 ||
      "${GEMMA4_PREFILL_KV_PROBE:-0}" != 1 ]]; then
  echo "PREFILL_FIRST_TOKEN_ERROR: requires GEMMA4_PREFILL_KV_PROBE=1 GEMMA4_PREFILL_KV_RUN=1" >&2
  exit 2
fi
for marker in \
    "PREFILL_LAST_PROMPT_SEED: prompt_tokens=17 replay_position=16 replay_token=4368 kv_rows_from_prefill=16" \
    "PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216" \
    "PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=16 input_token=4368 saved_decode_inputs_used=0" \
    "PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=16 input_token=4368" \
    "PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1" \
    "PREFILL_FIRST_TOKEN_FROM_PROMPT: token_id=" \
    "PREFILL_LINK_KV_FEEDBACK_OK: executed_step=1 position=16 cache_tensors=30 inserted_bytes=9216" \
    "PREFILL_LINK_NEXT_DECODE_OK: step=2 position=17" \
    "PREFILL_LINK_NEXT_HEAD_OK: step=2 logits_bytes=262144 linked_KV_feedback=1" \
    "PREFILL_LINK_MULTI_STEP_OK: steps=3 first_position=16 last_position=18 first_token=4368 last_prompt_replayed=1"; do
  grep -Fq "$marker" "$LOG" || {
    echo "PREFILL_FIRST_TOKEN_STAGE6_8_MISSING: $marker" >&2
    exit 1
  }
done
mapfile -t FIRST_IDS < <(sed -n 's/^PREFILL_FIRST_TOKEN_FROM_PROMPT: token_id=\([0-9]\+\) .*/\1/p' "$LOG")
[[ "${#FIRST_IDS[@]}" == 1 ]] || { echo "PREFILL_FIRST_TOKEN_STAGE6_8_ID_COUNT_INVALID" >&2; exit 1; }
FIRST_TOKEN="${FIRST_IDS[0]}"
[[ "$FIRST_TOKEN" =~ ^[0-9]+$ ]] || { echo "PREFILL_FIRST_TOKEN_STAGE6_8_ID_INVALID" >&2; exit 1; }
grep -Fq "PREFILL_LINK_NEXT_INPUTS_READY: step=2 position=17 token=$FIRST_TOKEN kv_from_feedback_ram=30 nonkv_from_cpu=8 saved_decode_inputs_used=0" "$LOG" || {
  echo "PREFILL_FIRST_TOKEN_STAGE6_8_FEEDBACK_TOKEN_MISMATCH" >&2; exit 1
}
echo "PREFILL_FIRST_TOKEN_STAGE6_9_SEED_OK: last_prompt_id=4368 replay_pos=16 first_generated_id=$FIRST_TOKEN next_decode_pos=17 saved_decode_inputs_used=0"

# Stage 6.8 showed two equal INT8 maxima at q=79 from the second
# Prefill-linked Decode. Diagnose both IDs and explicitly select the lower
# ID only for this experiment; it is NOT a full-precision greedy ranking.
for marker in \
    "CHAIN_LINKED_INT8_MAX_TOKENS: max_logit_q=79 ties=2 token_ids=" \
    "PREFILL_LINK_GREEDY_CANDIDATES: executed_step=2 max_logit_q=79 ties=2 candidate_token_ids=" \
    "PREFILL_LINK_TIE_RESOLUTION: executed_step=2 policy=lowest-id" \
    "PREFILL_LINK_KV_FEEDBACK_OK: executed_step=2 position=17 cache_tensors=30 inserted_bytes=9216" \
    "PREFILL_LINK_NEXT_DECODE_OK: step=3 position=18" \
    "PREFILL_LINK_NEXT_HEAD_OK: step=3 logits_bytes=262144 linked_KV_feedback=1"; do
  grep -Fq "$marker" "$LOG" || {
    echo "PREFILL_TIE_STAGE6_9_MISSING: $marker" >&2
    exit 1
  }
done
mapfile -t TIE_LINES < <(grep '^PREFILL_LINK_GREEDY_CANDIDATES: executed_step=2 ' "$LOG")
[[ "${#TIE_LINES[@]}" == 1 ]] || {
    echo "PREFILL_TIE_STAGE6_9_CANDIDATE_COUNT_INVALID" >&2; exit 1
}
TIE_IDS="${TIE_LINES[0]##*candidate_token_ids=}"
IFS=, read -r -a IDS <<< "$TIE_IDS"
[[ "${#IDS[@]}" == 2 ]] || { echo "PREFILL_TIE_STAGE6_9_EXPECTED_TWO_IDS" >&2; exit 1; }
[[ "${IDS[0]}" =~ ^[0-9]+$ && "${IDS[1]}" =~ ^[0-9]+$ ]] || {
    echo "PREFILL_TIE_STAGE6_9_IDS_INVALID" >&2; exit 1
}
(( IDS[0] < IDS[1] )) || {
    echo "PREFILL_TIE_STAGE6_9_CANDIDATE_ORDER_INVALID" >&2; exit 1
}
grep -Fq "PREFILL_LINK_TIE_RESOLUTION: executed_step=2 policy=lowest-id chosen_token_id=${IDS[0]} tied_ids=$TIE_IDS" "$LOG" || {
  echo "PREFILL_TIE_STAGE6_9_POLICY_MISMATCH" >&2; exit 1
}
grep -Fq "PREFILL_LINK_NEXT_INPUTS_READY: step=3 position=18 token=${IDS[0]} kv_from_feedback_ram=30 nonkv_from_cpu=8 saved_decode_inputs_used=0" "$LOG" || {
  echo "PREFILL_TIE_STAGE6_9_NEXT_INPUT_MISMATCH" >&2; exit 1
}
grep -Fq "PREFILL_LINK_MULTI_STEP_OK: steps=3 first_position=16 last_position=18 first_token=4368 last_prompt_replayed=1 int8_tie_policy=lowest-id-experimental" "$LOG" || {
  echo "PREFILL_TIE_STAGE6_9_COMPLETION_MISSING" >&2; exit 1
}
echo "PREFILL_TIE_STAGE6_9_OK: first_generated_id=$FIRST_TOKEN tied_candidates=$TIE_IDS experimental_selected=${IDS[0]} third_decode_position=18 real_KV_feedback=1 not_FP32_greedy=1"
