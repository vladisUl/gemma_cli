#!/usr/bin/env bash
# Stage 6.18: longer canonical Gemma 4 continuation with EOS detection.
# Original 17-token Prefill, tokenizer and saved QNN RAW files remain read-only.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
WORK="$HOME/gemma4_qnn_test"
if [[ -n "$(printenv GEMMA4_TEST_ROOT || true)" ]]; then
  WORK="$(printenv GEMMA4_TEST_ROOT)"
fi
[[ -d "$WORK" ]] || { echo "STAGE6_18_WORKDIR_MISSING: $WORK" >&2; exit 2; }
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
  [[ -e "$p" ]] || { echo "STAGE6_18_FILE_MISSING: $p" >&2; exit 2; }
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
assert len(raw)==512,"STAGE6_18_BAD_TOKEN_FILE"
ids=struct.unpack("<128i",raw)
assert ids[0]==2 and ids[16]==4368 and all(x==0 for x in ids[17:]),"STAGE6_18_BAD_PROMPT"
assert v[107]=="\n" and v[4368]=="model","STAGE6_18_BAD_TOKENIZER"
print("STAGE6_18_PROMPT_VERIFIED: tokens=17 last_id=4368 model_header_newline_id=107")
PY
LOG=cpu_correct_header_generation_stage6_18.log
echo "STAGE6_18_BEGIN: Prefill17 -> Decode(pos17,token107 LF) -> <=16 response predictions, EOS aware"
"$ROOT/build/gemma4-qnn-run" \
    --system "$PWD/sdk247/libQnnSystem.so" \
    --backend "$PWD/sdk247/libQnnHtp.so" \
    --jobs persistent_jobs_decode_prefill_head.txt \
    --chain-decode-head \
    --prefill-kv-probe --prefill-kv-run \
    --prefill-kv-token 107 --prefill-kv-steps 16 --prefill-stop-on-eos \
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
    "PREFILL_LINK_MULTI_STEP_OK: steps=" \
    "PREFILL_LINK_NEXT_DECODE_OK: step=2 position=18 input_token=236778" \
    "PREFILL_LINK_NEXT_HEAD_OK: step=2 logits_bytes=262144"; do
    grep -Fq "$marker" "$LOG" || {
      echo "STAGE6_18_MISSING_MARKER: $marker" >&2
      exit 1
    }
done
# Inspect both the greedy candidates and LM Head outputs. A previous
# baseline CHAIN_DECODE_HEAD_OK belongs to a different run; exclude those.
python3 - "$LOG" <<'PY' | tee -a "$LOG"
import json,re,sys
from pathlib import Path

def require(condition,message):
    if not condition:
        raise SystemExit(message)

lines=Path(sys.argv[1]).read_text(errors="replace").splitlines()
start=next((i for i,line in enumerate(lines)
    if line.startswith("PREFILL_LINK_DECODE_OK: graph=qnn_partition_0")
    and "position=17 input_token=107" in line),None)
require(start is not None,"STAGE6_18_LINKED_DECODE_MISSING")
linked=lines[start:]

heads=[line for line in linked if line.startswith("CHAIN_DECODE_HEAD_OK:")]
greedy=[line for line in linked if line.startswith("PREFILL_LINK_GREEDY_CANDIDATES:")]
summary=[line for line in linked if line.startswith("PREFILL_LINK_MULTI_STEP_OK:")]
eos=[line for line in linked if line.startswith("PREFILL_LINK_STOP_EOS:")]
limit=[line for line in linked if line.startswith("PREFILL_LINK_STOP_LIMIT:")]

require(len(summary)==1,"STAGE6_18_FINAL_SUMMARY_MISSING")
m=re.search(r"\bsteps=(\d+)\b",summary[0])
require(m is not None,"STAGE6_18_SUMMARY_STEPS_INVALID")
n=int(m.group(1))
require(1<=n<=16 and len(heads)==n and len(greedy)==n,
        f"STAGE6_18_BAD_STEP_COUNTS: steps={n} heads={len(heads)} greedy={len(greedy)}")
require((len(eos)==1 and not limit) or (len(limit)==1 and not eos),
        "STAGE6_18_EXPECTED_EXACTLY_ONE_STOP_CAUSE")

data=json.loads(Path("tokenizer.json").read_text(encoding="utf-8"))
vocab={int(idx):piece for piece,idx in data["model"]["vocab"].items()}
for item in data.get("added_tokens",[]):
    vocab[int(item["id"])]=item["content"]

def number(line,name):
    match=re.search(r"\b"+re.escape(name)+r"=(-?\d+)\b",line)
    require(match is not None,f"STAGE6_18_MISSING_{name}")
    return int(match.group(1))

ids=[]
out=bytearray()
stops={1,106,50} # google/gemma-4-E2B-it generation_config.json
for step,(head,candidates) in enumerate(zip(heads,greedy),1):
    token_id=number(head,"int8_argmax_id")
    ties=number(head,"argmax_ties")
    q=number(head,"max_logit_q")
    require(ties==1,"STAGE6_18_AMBIGUOUS_INT8_MAX")
    require(number(candidates,"executed_step")==step,
            "STAGE6_18_CANDIDATE_STEP_MISMATCH")
    require(number(candidates,"ties")==ties and
            number(candidates,"max_logit_q")==q and
            candidates.split("candidate_token_ids=",1)[-1].strip()==str(token_id),
            "STAGE6_18_CANDIDATE_LOGIT_MISMATCH")
    piece=vocab.get(token_id)
    require(piece is not None,f"STAGE6_18_UNKNOWN_ID_{token_id}")
    ids.append(token_id)
    print(f"STAGE6_18_GENERATED_TOKEN: step={step} response_pos={17+step}"
          f" id={token_id} max_q={q} ties={ties} raw_piece={piece!r}")
    if token_id in stops:
        require(step==n and len(eos)==1,
                "STAGE6_18_EOS_NOT_FINAL_PREDICTION")
        continue
    hexbyte=re.fullmatch(r"<0x([0-9A-Fa-f]{2})>",piece)
    if hexbyte:
        out.append(int(hexbyte.group(1),16))
    else:
        out.extend(piece.replace("▁"," ").encode("utf-8"))
require(ids[:4]==[236778,900,236743,236778],
        f"STAGE6_18_FIRST_FOUR_REGRESSION: {ids[:4]}")
if eos:
    eos_id=number(eos[0],"eos_id")
    require(ids[-1]==eos_id and eos_id in stops,
            "STAGE6_18_EOS_MARKER_ID_MISMATCH")
    reason="eos"
else:
    require(n==16,"STAGE6_18_LIMIT_BEFORE_16")
    reason="limit"
print("STAGE6_18_TEXT_PREVIEW:",repr(out.decode("utf-8",errors="replace")))
print(f"STAGE6_18_STOP_REASON: reason={reason} predictions={n}"
      f" eos_id={ids[-1] if reason=='eos' else 'none'}")
print("STAGE6_18_OK: no_tie_break=1"
      " original_QNN_RAW_unchanged=1" 
      " stop_token_not_submitted_to_Decode=1")
PY
