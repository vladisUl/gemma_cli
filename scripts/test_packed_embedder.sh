#!/usr/bin/env bash
# Experimental no-TensorFlow C++ packed INT2/INT4 Gemma decode embedder.
# Does not modify model files, existing saved QNN inputs or SDK libraries.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
SECTION="${GEMMA4_PACKED_SECTION:-0}"
TOKEN="${GEMMA4_PACKED_TOKEN:-2}"
if [[ "$SECTION" != 0 && "$SECTION" != 2 ]]; then
  echo "PACKED_TEST_ERROR: GEMMA4_PACKED_SECTION must be 0 or 2" >&2
  exit 2
fi
if [[ ! "$TOKEN" =~ ^[0-9]+$ ]] || (( ${#TOKEN} > 7 )) || (( 10#$TOKEN > 262143 )); then
  echo "PACKED_TEST_ERROR: GEMMA4_PACKED_TOKEN must be 0..262143" >&2
  exit 2
fi
if [[ "$SECTION" == 0 ]]; then
  MODEL="$WORK/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
  EXPECTED=6144
else
  MODEL="$WORK/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"
  EXPECTED=35840
fi
if [[ ! -f "$MODEL" || ! -x "$ROOT/build/gemma4-packed-embed" ]]; then
  echo "PACKED_TEST_ERROR: missing model or build/gemma4-packed-embed" >&2
  exit 2
fi
OUTDIR="$(mktemp -d "$WORK/packed_embed_section${SECTION}.XXXXXX")"
echo "PACKED_TEST_BEGIN: section=$SECTION token_probe=$TOKEN"
echo "PACKED_TEST_DIR: $OUTDIR"
"$ROOT/build/gemma4-packed-embed" --section "$SECTION" \
  --model "$MODEL" --token "$TOKEN" \
  --output "$OUTDIR/embeddings.raw"
BYTES="$(stat -c %s "$OUTDIR/embeddings.raw")"
if [[ "$BYTES" != "$EXPECTED" ]]; then
  echo "PACKED_TEST_ERROR: size mismatch actual=$BYTES expected=$EXPECTED" >&2
  exit 1
fi
echo "PACKED_TEST_OK: section=$SECTION output_bytes=$BYTES (not yet reference-verified)"
