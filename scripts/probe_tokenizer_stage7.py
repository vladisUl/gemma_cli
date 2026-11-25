#!/usr/bin/env python3
"""Stage 7.0: inspect the *actual* Q8B HF BPE tokenizer contract, stdlib only.

This is a read-only probe, NOT a replacement for HuggingFace BPE encoding.
It verifies the exact already-working prompt and output token sequences and
prints the normalizer, pre-tokenizer, decoder and BPE merge formats required
before writing an accurate native text -> IDs implementation.
"""
import argparse
import json
import re
import struct
import sys
from pathlib import Path

EXPECTED_PROMPT = (
    2, 105, 2364, 107, 116130, 236888, 217598, 16632, 236743,
    236778, 236862, 236778, 236881, 106, 107, 105, 4368
)
EXPECTED_ANSWER = (236778, 900, 236743, 236778, 16632,
                   5213, 236812, 84750, 106)
EXPECTED_TEXT = "2 + 2 будет **4**."


def describe(title, value):
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True)
    if len(payload) > 6000:
        print(f"TOKENIZER_STAGE7_{title}: {payload[:6000]}...(truncated)")
    else:
        print(f"TOKENIZER_STAGE7_{title}: {payload}")


def decoder_sequence(info):
    decoder = info.get("decoder")
    if not isinstance(decoder, dict) or decoder.get("type") != "Sequence":
        raise ValueError("Unexpected decoder: expected Sequence")
    steps = decoder.get("decoders", [])
    expected = ["Replace", "ByteFallback", "Fuse"]
    if [d.get("type") for d in steps] != expected:
        raise ValueError(f"Unexpected decoder pipeline (not {expected}): {steps}")
    replace = steps[0]
    pattern = replace.get("pattern", {})
    if pattern.get("String") != "▁" or replace.get("content") != " ":
        raise ValueError("Unexpected decoder Replace rules")


def decode_ids(ids, vocab, skip_special=False, special_ids=frozenset()):
    """Implements this JSON decoder's simple Replace/ByteFallback/Fuse path."""
    raw = bytearray()
    for token_id in ids:
        if skip_special and token_id in special_ids:
            continue
        if token_id not in vocab:
            raise ValueError(f"Unknown token id {token_id}")
        piece = vocab[token_id].replace("▁", " ")
        b = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
        if b:
            raw.append(int(b.group(1), 16))
        else:
            raw.extend(piece.encode("utf-8"))
    return raw.decode("utf-8", errors="replace")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path.home() / "gemma4_qnn_test" / "tokenizer.json",
    )
    parser.add_argument(
        "--prompt-ids",
        type=Path,
        default=Path.home() / "gemma4_qnn_test" /
                "derived_prefill_128_token_ids_i32.raw",
    )
    args = parser.parse_args()

    data = json.loads(args.tokenizer.read_text(encoding="utf-8"))
    model = data.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError("Expected BPE model, got " + repr(model.get("type")))
    vocab_forward = model.get("vocab")
    if not isinstance(vocab_forward, dict) or len(vocab_forward) != 262144:
        raise ValueError("Expected 262144 BPE vocab entries")
    vocab = {int(token_id): piece
             for piece, token_id in vocab_forward.items()}
    if len(vocab) != len(vocab_forward):
        raise ValueError("Duplicate vocab IDs")
    special = set()
    for token in data.get("added_tokens", []):
        token_id = int(token["id"])
        if token_id in vocab and vocab[token_id] != token["content"]:
            raise ValueError(f"Conflicting added token {token_id}")
        vocab[token_id] = token["content"]
        if token.get("special", False):
            special.add(token_id)

    merges = model.get("merges", [])
    if not isinstance(merges, list):
        raise ValueError("Expected BPE merges list")
    decoder_sequence(data)
    describe("MODEL", {
        "type": model["type"], "vocab_entries": len(vocab),
        "merges": len(merges), "first_merges": merges[:5],
        "first_merge_type": type(merges[0]).__name__ if merges else None,
        "unk_token": model.get("unk_token"),
        "byte_fallback": model.get("byte_fallback"),
        "fuse_unk": model.get("fuse_unk"),
        "continuing_subword_prefix": model.get("continuing_subword_prefix"),
        "end_of_word_suffix": model.get("end_of_word_suffix"),
        "dropout": model.get("dropout"),
    })
    describe("NORMALIZER", data.get("normalizer"))
    describe("PRE_TOKENIZER", data.get("pre_tokenizer"))
    describe("POST_PROCESSOR", data.get("post_processor"))
    describe("DECODER", data.get("decoder"))
    describe("ADDED_TOKENS", [
        {"id": t["id"], "content": t["content"],
         "special": bool(t.get("special"))}
        for t in data.get("added_tokens", [])[:35]
    ])
    describe("CONFIG", {
        "padding": data.get("padding"),
        "truncation": data.get("truncation"),
        "model_max_length": data.get("model_max_length"),
    })

    prompt_raw = args.prompt_ids.read_bytes()
    if len(prompt_raw) != 128 * 4:
        raise ValueError("Expected 512-byte, 128-int32 prompt file")
    all_ids = struct.unpack("<128i", prompt_raw)
    n = next((i for i, x in enumerate(all_ids) if x == 0), 128)
    if tuple(all_ids[:n]) != EXPECTED_PROMPT or any(all_ids[n:]):
        raise ValueError("Original Q8B prompt does not match golden 17 IDs")
    if vocab[4368] != "model" or vocab[107] != "\n":
        raise ValueError("Expected 4368=model and 107=LF")
    if vocab[106] != "<turn|>":
        raise ValueError("Expected Gemma 4 turn-end ID106")
    original = decode_ids(EXPECTED_PROMPT, vocab)
    canon = decode_ids(EXPECTED_PROMPT + (107,), vocab)
    answer = decode_ids(EXPECTED_ANSWER[:-1], vocab)
    if answer != EXPECTED_TEXT:
        raise ValueError(f"Expected {EXPECTED_TEXT!r}, got {answer!r}")
    if not canon.endswith("model\n"):
        raise ValueError("Canonical model header does not end with LF")
    describe("PROMPT", {
        "prompt_token_count": n,
        "raw_text": original,
        "correct_header_text": canon,
        "correct_header_token_ids": list(EXPECTED_PROMPT + (107,)),
        "final_added_token": 107,
    })
    describe("ANSWER", {
        "raw_token_ids": list(EXPECTED_ANSWER),
        "text": answer,
        "stop_id": EXPECTED_ANSWER[-1],
    })
    print("TOKENIZER_STAGE7_GOLDEN_OK: source=on_device_tokenizer_json"
          " known_prompt_ids=17 model_header_LF=107"
          " generated_tokens=9 expected_text_match=1"
          " input_files_read_only=1")
    print("TOKENIZER_STAGE7_NEXT: inspect normalizer/pre_tokenizer/merges"
          " before implementing arbitrary text-to-BPE IDs")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, struct.error) as exc:
        print(f"TOKENIZER_STAGE7_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
