#!/usr/bin/env bash
# Stage 6.16: test the missing newline at the end of Gemma 4 model header.
# Original 17-token Prefill and all saved QNN RAW files remain read-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/gemma4_qnn_test"
if [[ -n "$(printenv GEMMA4_TEST_ROOT || true)" ]]; then
  WORK="$(printenv GEMMA4_TEST_ROOT)"
fi
[[ -d "$WORK" ]] || { echo "STAGE6_16_WORKDIR_MISSING: $WORK" >&2; exit 2; }
cd "$WORK"
MODEL0="$PWD/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
MODEL1="$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"
MODEL2="$PWD/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"
for p in "$MODEL0" "$MODEL1" "$MODEL2" \
    "$PWD/sdk247/libQnnSystem.so" "$PWD/sdk247/libQnnHtp.so" \
    "$PWD/persistent_jobs_decode_prefill_head.txt" \
    "$PWD/derived_prefill_128_token_ids_i32.raw" \
    "$PWD/auto_decode_step4_list.txt" \
    "$PWD/auto_result_decode_4/Result_0" \
    "$PWD/auto_result_head_4/Result_0" \
    "$PWD/tokenizer.json"; do
  [[ -e "$p" ]] || { echo "STAGE6_16_FILE_MISSING: $p" >&2; exit 2; }
done
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
python3 - <<'PY'
import json,struct
from pathlib import Path
info=json.loads(Path("tokenizer.json").read_text(encoding="utf-8"))
v={int(i):w for w,i in info["model"]["vocab"].items()}
for t in info.get("added_tokens",[]):v[int(t["id"])]=t["content"]
raw=Path("derived_prefill_128_token_ids_i32.raw").read_bytes()
assert len(raw)==512,"STAGE6_16_BAD_TOKEN_FILE"
ids=struct.unpack("<128i",raw)
assert ids[0]==2 and ids[16]==4368 and all(x==0 for x in ids[17:]),"STAGE6_16_BAD_PROMPT"
assert v[107]=="\n" and v[4368]=="model","STAGE6_16_BAD_TOKENIZER"
print("STAGE6_16_PROMPT_VERIFIED: tokens=17 last_id=4368 model_header_newline_id=107")
PY
LOG=cpu_correct_header_stage6_16.log
echo "STAGE6_16_BEGIN: Prefill17 -> Decode(pos17,token107 LF) -> first response logits"
"$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --chain-decode-head \
    --prefill-kv-probe --prefill-kv-run \
    --prefill-kv-token 107 --prefill-kv-steps 1 \
    --cpu-prefill-tokens "$PWD/derived_prefill_128_token_ids_i32.raw" \
    --cpu-prefill-section0 "$MODEL0" \
    --cpu-prefill-section2 "$MODEL2" \
    --cpu-prefill-aux-model "$MODEL1" \
    --cpu-aux-model "$MODEL1" \
    --cpu-aux-pos 3 --cpu-aux-token 2 \
    --cpu-embed-section0 "$MODEL0" \
    --cpu-embed-section2 "$MODEL2" \
    --cpu-embed-token 568 \
    --kv-auto-pos --kv-next-list auto_decode_step4_list.txt --kv-run-next \
    --cpu-aux-next-pos 4 --cpu-aux-next-token 2 \
    --cpu-embed-next-token 568 \
    --kv-next-reference-dir auto_result_decode_4/Result_0 \
    --kv-next-head-reference-dir auto_result_head_4/Result_0 \
    2>&1 | tee "$LOG"
for marker in \
    "ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors" \
    "DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical" \
    "DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical" \
    "KV_PREFILL_BRIDGE_PROBE_OK: tensors=30 layers=15 valid_tokens=17" \
    "PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=17 input_token=107" \
    "PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=17 input_token=107" \
    "PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1" \
    "PREFILL_LINK_MULTI_STEP_OK: steps=1 first_position=17 last_position=17 first_token=107"; do
    grep -Fq "$marker" "$LOG" || {
      echo "STAGE6_16_MISSING_MARKER: $marker" >&2
      exit 1
    }
done
python3 - "$LOG" <<'PY' | tee -a "$LOG"
import json,re,sys
from pathlib import Path
lines=Path(sys.argv[1]).read_text(errors="replace").splitlines()
heads=[line for line in lines if line.startswith("CHAIN_DECODE_HEAD_OK:")]
assert len(heads)>=3,"STAGE6_16_EXPECTED_BASELINES_AND_LINKED_HEAD"
last=heads[-1]
mid=re.search(r"\bint8_argmax_id=(\d+)\b",last)
ties=re.search(r"\bargmax_ties=(\d+)\b",last)
assert mid and ties,"STAGE6_16_MISSING_ARGMAX_FIELDS"
tid=int(mid.group(1))
info=json.loads(Path("tokenizer.json").read_text(encoding="utf-8"))
v={int(i):w for w,i in info["model"]["vocab"].items()}
for t in info.get("added_tokens",[]):v[int(t["id"])]=t["content"]
print("STAGE6_16_FIRST_RESPONSE_TOKEN:",f"id={tid}",f"ties={ties.group(1)}",f"raw_piece={v.get(tid)!r}","response_position=18")
print("STAGE6_16_OK: completed_model_header_LF=1 no_original_data_modified=1")
PY
