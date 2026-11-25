#!/usr/bin/env bash
# Run original Section0 or Section2 TFLite CPU embedding signature on Q8B.
# Output files are diagnostics; token ID 2 is not the identified saved-step token.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
SECTION="${GEMMA4_EMBEDDER_SECTION:-0}"
TOKEN="${GEMMA4_EMBEDDER_TOKEN:-2}"
if [[ "$SECTION" != "0" && "$SECTION" != "2" ]]; then
  echo "CPU_EMBED_TEST_ERROR: GEMMA4_EMBEDDER_SECTION must be 0 or 2" >&2
  exit 2
fi
if [[ ! "$TOKEN" =~ ^[0-9]+$ ]] || (( ${#TOKEN} > 7 )) || (( 10#$TOKEN > 262143 )); then
  echo "CPU_EMBED_TEST_ERROR: GEMMA4_EMBEDDER_TOKEN must be 0..262143" >&2
  exit 2
fi
if [[ "$SECTION" == 0 ]]; then
  MODEL="$WORK/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
  SIGNATURE="decode_embedder"
  EXPECTED=6144
else
  MODEL="$WORK/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"
  SIGNATURE="decode_per_layer_embedder"
  EXPECTED=35840
fi
if [[ ! -f "$MODEL" ]]; then
  echo "CPU_EMBED_TEST_MODEL_MISSING: $MODEL" >&2
  exit 2
fi
if [[ ! -x "$ROOT/build/gemma4-cpu-run" ]]; then
  echo "CPU_EMBED_TEST_RUNNER_MISSING: build/gemma4-cpu-run" >&2
  exit 2
fi
mkdir -p "$WORK"
OUT="$(mktemp -d "$WORK/cpu_embed_section${SECTION}.XXXXXX")"
echo "CPU_EMBED_TEST_START: section=$SECTION token_probe=$TOKEN"
echo "CPU_EMBED_TEST_NOTE: token is a probe; do not compare to QNN saved Decode unless token ID is known"
echo "CPU_EMBED_TEST_OUTPUT_DIR: $OUT"
"$ROOT/build/gemma4-cpu-run" \
    --model "$MODEL" --signature "$SIGNATURE" \
    --i32 "token_ids=$TOKEN" --output-dir "$OUT"
RESULT="$OUT/embeddings.raw"
if [[ ! -f "$RESULT" ]]; then
  echo "CPU_EMBED_TEST_OUTPUT_MISSING: $RESULT" >&2
  exit 1
fi
ACTUAL="$(stat -c %s "$RESULT")"
if [[ "$ACTUAL" != "$EXPECTED" ]]; then
  echo "CPU_EMBED_TEST_SIZE_ERROR: expected=$EXPECTED actual=$ACTUAL" >&2
  exit 1
fi
echo "CPU_EMBED_TEST_OK: section=$SECTION output_bytes=$ACTUAL token_probe=$TOKEN"
