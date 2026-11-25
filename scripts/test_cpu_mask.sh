#!/usr/bin/env bash
# Run Section1 decode_mask for sample step 3 and compare to saved QNN inputs.
# Token ID 2 is a nonpadding probe, NOT the verified actual token at step 3.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
TOKEN="${GEMMA4_MASK_TOKEN:-2}"
STEP="${GEMMA4_MASK_STEP:-3}"
if [[ ! "$TOKEN" =~ ^-?[0-9]+$ || ! "$STEP" =~ ^[0-9]+$ ]]; then
  echo "CPU_MASK_TEST_ERROR: token must be signed integer, step must be nonnegative" >&2
  exit 2
fi
if [[ ! -d "$WORK" ]]; then
  echo "CPU_MASK_TEST_ERROR: missing working directory: $WORK" >&2
  exit 2
fi
cd "$WORK"
MODEL="$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"
LIST="$PWD/auto_decode_step3_list.txt"
if [[ ! -f "$MODEL" || ! -f "$LIST" ]]; then
  echo "CPU_MASK_TEST_ERROR: missing Section1 model or Decode step3 list" >&2
  exit 2
fi
OUT="$(mktemp -d "$PWD/cpu_mask_pos3.XXXXXX")"
echo "CPU_MASK_TEST_INPUTS: token_id=$TOKEN time_step=$STEP"
echo "CPU_MASK_TEST_NOTE: nonpadding probe; actual saved token ID not established"
echo "CPU_MASK_TEST_OUTPUT: $OUT"
"$ROOT/build/gemma4-cpu-run" \
  --model "$MODEL" --signature decode_mask \
  --i32 "input_tokens=$TOKEN" --i32 "time_step=$STEP" \
  --output-dir "$OUT"
python3 "$ROOT/scripts/verify_cpu_aux.py" \
  --kind mask --cpu-dir "$OUT" --decode-list "$LIST"
echo "CPU_MASK_STEP3_VERIFIED: outputs byte-identical to saved QNN inputs"
