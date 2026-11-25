#!/usr/bin/env bash
# Stage 7.2: sample-free dynamic QNN Prefill+Decode on Q8B.
# Golden end-to-end control first, then the requested text.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/gemma4_qnn_test"
if [[ -n "$(printenv GEMMA4_TEST_ROOT || true)" ]]; then
  WORK="$(printenv GEMMA4_TEST_ROOT)"
fi
TEXT="Сколько будет 3 + 5?"
if [[ $# -ge 1 ]]; then TEXT="$1"; fi
cd "$WORK"
for f in \
  "$PWD/sdk247/libQnnSystem.so" "$PWD/sdk247/libQnnHtp.so" \
  "$PWD/persistent_jobs_decode_prefill_head.txt" \
  "$PWD/tokenizer.json" "$PWD/derived_prefill_128_token_ids_i32.raw" \
  "$PWD/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite" \
  "$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite" \
  "$PWD/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"; do
    [[ -f "$f" ]] || { echo "STAGE7_2_REQUIRED_FILE_MISSING: $f" >&2; exit 2; }
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
TMP="$(mktemp -d "$WORK/stage7_2_new_inputs.XXXXXXXX")"
cleanup() { rm -f "$TMP/known.raw" "$TMP/user.raw"; rmdir "$TMP" || true; }
trap cleanup EXIT

run_dynamic() {
  local raw="$1" log="$2"
  "$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --dynamic-prefill \
    --dynamic-prompt-tokens "$raw" \
    --dynamic-section0 "$PWD/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite" \
    --dynamic-section1 "$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite" \
    --dynamic-section2 "$PWD/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite" \
    --dynamic-max-tokens 16 \
    2>&1 | tee "$log"
}

decode_log() {
  local log="$1" scope="$2"
  python3 - "$log" "$scope" <<'PY' | tee -a "$log"
import json,re,sys
from pathlib import Path

log=Path(sys.argv[1]).read_text(errors="replace")
scope=sys.argv[2]
model=json.loads(Path("tokenizer.json").read_text(encoding="utf-8"))
vocab={int(i):word for word,i in model["model"]["vocab"].items()}
for t in model.get("added_tokens",[]):
    vocab[int(t["id"])]=t["content"]
lines=[s for s in log.splitlines() if s.startswith("DYNAMIC_GENERATED_TOKEN:")]
if not lines:raise SystemExit("STAGE7_2_NO_GENERATED_TOKENS")
ids=[]
out=bytearray()
for line in lines:
    m=re.search(r"\bid=(\d+)\b",line)
    q=re.search(r"\bties=(\d+)\b",line)
    if not m or not q:raise SystemExit("STAGE7_2_BAD_TOKEN_MARKER")
    if int(q.group(1))!=1:raise SystemExit("STAGE7_2_AMBIGUOUS_TIES")
    tid=int(m.group(1))
    ids.append(tid)
    if tid in (1,106,50):continue
    raw=vocab.get(tid)
    if raw is None:raise SystemExit(f"STAGE7_2_UNKNOWN_TOKEN_{tid}")
    raw=raw.replace("▁"," ")
    esc=re.fullmatch(r"<0x([0-9A-Fa-f]{2})>",raw)
    if esc:out.append(int(esc.group(1),16))
    else:out.extend(raw.encode("utf-8"))
response=out.decode("utf-8",errors="replace")
print(f"STAGE7_2_RESULT: scope={scope} tokens={ids} text={response!r}")
print(f"STAGE7_2_TEXT_PREVIEW: scope={scope} {response!r}")
if scope=="known":
    expected=[236778,900,236743,236778,16632,5213,236812,84750,106]
    if ids!=expected or response!="2 + 2 будет **4**.":
        raise SystemExit(f"STAGE7_2_GOLDEN_INFERENCE_MISMATCH: {ids}, {response!r}")
    if "DYNAMIC_GENERATION_OK:" not in log or "stop_reason=eos" not in log:
        raise SystemExit("STAGE7_2_GOLDEN_INCOMPLETE")
    print("STAGE7_2_GOLDEN_INFERENCE_OK: matched_9_ids=1"
          " expected_text_match=1 EOS106=1"
          " sample_inputs_and_refs_read=0")
PY
}

echo "STAGE7_2_BEGIN: golden control before user text, sample-free dynamic QNN"
python3 "$ROOT/scripts/gemma4_bpe.py" \
  --text "Привет! Сколько будет 2+2?" \
  --output-new-raw "$TMP/known.raw"
run_dynamic "$TMP/known.raw" "$WORK/cpu_dynamic_golden_stage7_2.log"
decode_log "$WORK/cpu_dynamic_golden_stage7_2.log" "known"

echo "STAGE7_2_USER_BEGIN: $TEXT"
python3 "$ROOT/scripts/gemma4_bpe.py" \
  --text "$TEXT" \
  --output-new-raw "$TMP/user.raw"
run_dynamic "$TMP/user.raw" "$WORK/cpu_dynamic_user_stage7_2.log"
decode_log "$WORK/cpu_dynamic_user_stage7_2.log" "user"
echo "STAGE7_2_OK: golden_inference_verified=1"
echo "STAGE7_2_NOTE: user text is experimental; INT8 ties may limit response"
