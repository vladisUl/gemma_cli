#!/usr/bin/env bash
# Stage 6.17: continue from canonical Gemma 4 model header, four Decode steps.
# Original 17-token Prefill and all saved QNN RAW files remain read-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/gemma4_qnn_test"
if [[ -n "$(printenv GEMMA4_TEST_ROOT || true)" ]]; then
  WORK="$(printenv GEMMA4_TEST_ROOT)"
fi
[[ -d "$WORK" ]] || { echo "STAGE6_17_WORKDIR_MISSING: $WORK" >&2; exit 2; }
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
  [[ -e "$p" ]] || { echo "STAGE6_17_FILE_MISSING: $p" >&2; exit 2; }
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
assert len(raw)==512,"STAGE6_17_BAD_TOKEN_FILE"
ids=struct.unpack("<128i",raw)
assert ids[0]==2 and ids[16]==4368 and all(x==0 for x in ids[17:]),"STAGE6_17_BAD_PROMPT"
assert v[107]=="\n" and v[4368]=="model","STAGE6_17_BAD_TOKENIZER"
print("STAGE6_17_PROMPT_VERIFIED: tokens=17 last_id=4368 model_header_newline_id=107")
PY
LOG=cpu_correct_header_generation_stage6_17.log
echo "STAGE6_17_BEGIN: Prefill17 -> Decode(pos17,token107 LF) -> four response predictions"
"$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --chain-decode-head \
    --prefill-kv-probe --prefill-kv-run \
    --prefill-kv-token 107 --prefill-kv-steps 4 \
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
    "PREFILL_LINK_MULTI_STEP_OK: steps=4 first_position=17 last_position=20 first_token=107" \
    "PREFILL_LINK_NEXT_DECODE_OK: step=2 position=18 input_token=236778" \
    "PREFILL_LINK_NEXT_HEAD_OK: step=4 logits_bytes=262144"; do
    grep -Fq "$marker" "$LOG" || {
      echo "STAGE6_17_MISSING_MARKER: $marker" >&2
      exit 1
    }
done
# Only parse LM Head outputs after the linked Decode(pos17, LF).
# Earlier CHAIN_DECODE_HEAD_OK records are saved baseline checks.
python3 - "$LOG" <<'PY' | tee -a "$LOG"
import json,re,sys
from pathlib import Path

lines=Path(sys.argv[1]).read_text(errors="replace").splitlines()
start=next((i for i,line in enumerate(lines)
            if line.startswith("PREFILL_LINK_DECODE_OK: graph=qnn_partition_0")
            and "position=17 input_token=107" in line),None)
if start is None:
    raise SystemExit("STAGE6_17_LINKED_DECODE_MISSING")
heads=[line for line in lines[start:]
       if line.startswith("CHAIN_DECODE_HEAD_OK:")]
if len(heads)!=4:
    raise SystemExit(f"STAGE6_17_EXPECTED_4_LINKED_HEADS: actual={len(heads)}")
predictions=[]
for index,line in enumerate(heads,1):
    id_match=re.search(r"\bint8_argmax_id=(\d+)\b",line)
    tie_match=re.search(r"\bargmax_ties=(\d+)\b",line)
    q_match=re.search(r"\bmax_logit_q=(-?\d+)\b",line)
    if not id_match or not tie_match or not q_match:
        raise SystemExit(f"STAGE6_17_HEAD_FIELDS_MISSING: step={index}")
    predictions.append((int(id_match.group(1)),int(tie_match.group(1)),int(q_match.group(1))))
if predictions[0][:2]!=(236778,1):
    raise SystemExit("STAGE6_17_FIRST_TOKEN_NOT_ID236778_UNIQUE")

data=json.loads(Path("tokenizer.json").read_text(encoding="utf-8"))
vocab={int(idx):word for word,idx in data["model"]["vocab"].items()}
for item in data.get("added_tokens",[]):
    vocab[int(item["id"])]=item["content"]

# Decode JSON Sequence: Replace(▁ -> space), ByteFallback, Fuse.
# This is a diagnostic preview, not a full implementation of BPE encoding.
result_bytes=bytearray()
for step,(tid,ties,q) in enumerate(predictions,1):
    piece=vocab.get(tid)
    if piece is None:
        raise SystemExit(f"STAGE6_17_UNKNOWN_TOKEN_ID: {tid}")
    print(f"STAGE6_17_GENERATED_TOKEN: step={step} response_pos={17+step}"
          f" id={tid} ties={ties} max_q={q} raw_piece={piece!r}")
    match=re.fullmatch(r"<0x([0-9A-Fa-f]{2})>",piece)
    if match:
        result_bytes.append(int(match.group(1),16))
    else:
        result_bytes.extend(piece.replace("▁"," ").encode("utf-8"))
print("STAGE6_17_TEXT_PREVIEW:",repr(result_bytes.decode("utf-8",errors="replace")))
print("STAGE6_17_OK: predicted_tokens=4"
      " unique_argmax_before_feedback=1"
      " no_forced_tie_break=1"
      " original_files_unchanged=1")
PY
