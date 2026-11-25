#!/usr/bin/env python3
"""Parse Stage 7.2 dynamic Decode logs into text, using tokenizer.json.

This parser is deliberately a SEPARATE PROCESS from QNN execution, so the
~32MB BPE JSON (and its Python dictionaries) do not stay resident while the
large QNN contexts are loaded on the Q8B.

Output: one compact JSON object on stdout, nonzero exit for malformed logs.
No tokenizer package, QNN hardware, reference samples, or file writes.
"""
import argparse
import json
import re
import sys
from pathlib import Path

STOP_IDS = {1, 106, 50}


class GenerationParseError(ValueError):
    pass


def field(line, name):
    match = re.search(r"\b" + re.escape(name) + r"=(-?\d+)\b", line)
    if match is None:
        raise GenerationParseError("Missing " + name + " in " + line[:220])
    return int(match.group(1))


def decode_ids(ids, reverse_vocab):
    output = bytearray()
    for tid in ids:
        if tid in STOP_IDS:
            continue
        if tid not in reverse_vocab:
            raise GenerationParseError("Unknown token ID " + str(tid))
        piece = reverse_vocab[tid].replace("▁", " ")
        raw = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
        if raw:
            output.append(int(raw.group(1), 16))
        else:
            output.extend(piece.encode("utf-8"))
    return output.decode("utf-8", errors="replace")


def parse_dynamic(log, reverse_vocab):
    """Never include the arbitrary first argmax if its INT8 maximum was tied."""
    lines = log.splitlines()
    required = ("DYNAMIC_PROMPT_READY:", "DYNAMIC_PREFILL_NPU_OK:",
                "DYNAMIC_KV_BRIDGE_OK:")
    for marker in required:
        if not any(line.startswith(marker) for line in lines):
            raise GenerationParseError("Missing " + marker)
    candidates = [
        line for line in lines if line.startswith("DYNAMIC_GENERATED_TOKEN:")
    ]
    if not candidates:
        raise GenerationParseError("No DYNAMIC_GENERATED_TOKEN lines")
    ids = []
    tie_step = None
    tie_count = None
    for index, line in enumerate(candidates, 1):
        if field(line, "step") != index:
            raise GenerationParseError("Nonconsecutive Decode step")
        tid = field(line, "id")
        ties = field(line, "ties")
        if ties < 1 or tid < 0 or tid >= 262144:
            raise GenerationParseError("Invalid INT8 token candidate")
        if ties != 1:
            if index != len(candidates):
                raise GenerationParseError("Decode ran after ambiguous INT8 argmax")
            tie_step, tie_count = index, ties
            break
        ids.append(tid)
        if tid in STOP_IDS and index != len(candidates):
            raise GenerationParseError("Decode ran after EOS")

    stops = [line for line in lines if line.startswith("DYNAMIC_STOP_EOS:")]
    ambiguous = [line for line in lines if line.startswith("DYNAMIC_STOP_TIE:")]
    summaries = [line for line in lines if line.startswith("DYNAMIC_GENERATION_OK:")]

    if tie_step is not None:
        if len(ambiguous) != 1 or summaries or stops:
            raise GenerationParseError("Ambiguous prediction lacks stop marker")
        if field(ambiguous[0], "step") != tie_step:
            raise GenerationParseError("Wrong tie-stop position")
        if field(ambiguous[0], "ties") != tie_count:
            raise GenerationParseError("Tie count disagreement")
        return {
            "text": decode_ids(ids, reverse_vocab),
            "ids": ids, "stop_reason": "tie",
            "steps": len(candidates), "tie_count": tie_count,
            "complete": False
        }

    if len(summaries) != 1 or len(ambiguous) != 0:
        raise GenerationParseError("Missing unique final generation summary")
    stop_reason = re.search(r"\bstop_reason=(eos|limit)\b", summaries[0])
    if not stop_reason:
        raise GenerationParseError("Invalid stop_reason")
    reason = stop_reason.group(1)
    if field(summaries[0], "npu_decode_steps") != len(candidates):
        raise GenerationParseError("NPU executed step count mismatch")
    if field(summaries[0], "text_tokens") != sum(t not in STOP_IDS for t in ids):
        raise GenerationParseError("Text token count mismatch")
    if reason == "eos":
        if len(stops) != 1 or ids[-1] not in STOP_IDS:
            raise GenerationParseError("EOS stop not supported by token ID")
        if field(stops[0], "eos_id") != ids[-1]:
            raise GenerationParseError("EOS stop ID mismatch")
    else:
        if stops or any(t in STOP_IDS for t in ids):
            raise GenerationParseError("Limit stop has EOS")
    return {
        "text": decode_ids(ids, reverse_vocab),
        "ids": ids, "stop_reason": reason,
        "steps": len(candidates), "complete": reason == "eos"
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log", required=True, type=Path)
    p.add_argument("--tokenizer", required=True, type=Path)
    args = p.parse_args()
    data = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise GenerationParseError("Not a BPE tokenizer")
    reverse = {int(i): text for text, i in model["vocab"].items()}
    for extra in data.get("added_tokens", []):
        tid = int(extra["id"])
        if tid in reverse and reverse[tid] != extra["content"]:
            raise GenerationParseError("Added token ID conflict")
        reverse[tid] = extra["content"]
    response = parse_dynamic(args.log.read_text(errors="replace"), reverse)
    print(json.dumps(response, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print("GEMMA4_DECODE_LOG_ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
