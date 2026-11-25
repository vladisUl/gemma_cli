#!/usr/bin/env python3
"""Stage 7.1: strict, stdlib-only Gemma 4 BPE encoder (experimental).

The implementation follows the BPE ID-pair ranking used by Hugging Face
tokenizers; this is not a complete reimplementation of its AddedVocabulary
alignment/offset APIs. User text is ordinary content, while the Gemma 4 chat
template is inserted explicitly as verified special token IDs.

The verified on-device golden prompt MUST match before this program can
generate a new padded token-ID file. Original RAW files are never modified.
"""
import argparse
import json
import os
import re
import struct
import sys
from pathlib import Path

GOLDEN_TEXT = "Привет! Сколько будет 2+2?"
GOLDEN_USER_IDS = (116130, 236888, 217598, 16632, 236743,
                   236778, 236862, 236778, 236881)
GOLDEN_PROMPT_IDS = (
    2, 105, 2364, 107, *GOLDEN_USER_IDS, 106, 107, 105, 4368
)
GOLDEN_CANONICAL_IDS = GOLDEN_PROMPT_IDS + (107,)
DEFAULT_JSON = Path.home() / "gemma4_qnn_test" / "tokenizer.json"
DEFAULT_GOLDEN_RAW = (Path.home() / "gemma4_qnn_test" /
                      "derived_prefill_128_token_ids_i32.raw")


class ContractError(ValueError):
    pass


class GemmaBPE:
    def __init__(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        model = data.get("model", {})
        if model.get("type") != "BPE":
            raise ContractError("Expected a BPE tokenizer")
        normalizer = data.get("normalizer")
        if normalizer != {"type": "Replace",
                          "pattern": {"String": " "},
                          "content": "▁"}:
            raise ContractError("Unsupported normalizer")
        pre = data.get("pre_tokenizer")
        if pre != {"type": "Split",
                   "pattern": {"String": " "},
                   "behavior": "MergedWithPrevious",
                   "invert": False}:
            raise ContractError("Unsupported pre-tokenizer")
        post = data.get("post_processor")
        if not isinstance(post, dict) or post.get("type") != "TemplateProcessing":
            raise ContractError("Unexpected post-processor")
        if post.get("single") != [{"Sequence": {"id": "A", "type_id": 0}}]:
            raise ContractError("Expected no automatic BOS in single post-processing")
        decoder = data.get("decoder", {})
        if [d.get("type") for d in decoder.get("decoders", [])] != [
                "Replace", "ByteFallback", "Fuse"]:
            raise ContractError("Unexpected decoder")
        if (model.get("dropout") not in (None, 0) or
                model.get("ignore_merges") not in (None, False) or
                model.get("continuing_subword_prefix") is not None or
                model.get("end_of_word_suffix") is not None or
                model.get("byte_fallback") is not True or
                model.get("fuse_unk") is not True):
            raise ContractError("Unsupported BPE model parameters")
        self.forward = model.get("vocab")
        if not isinstance(self.forward, dict) or len(self.forward) != 262144:
            raise ContractError("Expected 262144 BPE vocabulary entries")
        self.reverse = {int(i): piece for piece, i in self.forward.items()}
        if len(self.reverse) != len(self.forward):
            raise ContractError("Nonunique BPE vocabulary IDs")
        self.special = []
        for token in data.get("added_tokens", []):
            id_ = int(token["id"])
            value = token["content"]
            if id_ in self.reverse and self.reverse[id_] != value:
                raise ContractError("AddedVocabulary ID conflicts with BPE model")
            self.reverse[id_] = value
            self.special.append(value)
        if (self.forward.get("<unk>") != 3 or
                self.forward.get("▁") != 236743 or
                self.forward.get("\n") != 107):
            raise ContractError("Special/base token IDs differ from the verified Q8B model")
        merges = model.get("merges")
        if not isinstance(merges, list):
            raise ContractError("Expected BPE merge pairs")
        # HuggingFace tokenizers maps (id_a,id_b) to (rank,new_id).
        # If the same ID pair occurs more than once, the final entry wins.
        self.ranks = {}
        for rank, pair in enumerate(merges):
            if not isinstance(pair, list) or len(pair) != 2:
                raise ContractError(f"Bad merge pair at index {rank}")
            left, right = pair
            try:
                a = self.forward[left]
                b = self.forward[right]
                target = self.forward[left + right]
            except KeyError as exc:
                raise ContractError(f"Merge {rank} token missing: {exc}") from exc
            self.ranks[(a, b)] = (rank, target)
        print("BPE_STAGE7_1_MODEL_OK:"
              f" vocab={len(self.forward)} merges={len(merges)}"
              f" unique_pairs={len(self.ranks)}"
              " unicode_codepoints=1 byte_fallback=1"
              " pretokenizer_split_after_normalization=1")

    def encode_plain(self, text):
        if not isinstance(text, str):
            raise TypeError("Input must be Unicode text")
        # AddedVocabulary detection differs from BPE processing. Reject
        # literals whose special treatment we have not implemented.
        for special in self.special:
            if special and special in text:
                raise ContractError(
                    f"AddedVocabulary literal inside user text unsupported: {special!r}")
        # Replace ASCII spaces with U+2581 BEFORE the "Split on ASCII space"
        # pretokenizer; after normalization, the Split has no matches.
        normalized = text.replace(" ", "▁")
        symbols = []
        for char in normalized:
            token_id = self.forward.get(char)
            if token_id is not None:
                symbols.append(token_id)
                continue
            utf8 = char.encode("utf-8")
            fallback = [self.forward.get(f"<0x{value:02X}>")
                        for value in utf8]
            if all(i is not None for i in fallback):
                symbols.extend(fallback)
            else:
                # HF BPE with fuse_unk=true merges consecutive *unknown*
                # codepoints into a single unknown ID.
                if not symbols or symbols[-1] != 3:
                    symbols.append(3)
        # For max. 127 prompt tokens, ordinary O(n^2) pair selection is
        # sufficient; this is a correctness prototype, not a speed claim.
        while len(symbols) > 1:
            best_rank = len(self.ranks) + 10000000
            best_position = -1
            best_new_id = None
            for pos in range(len(symbols) - 1):
                result = self.ranks.get((symbols[pos], symbols[pos + 1]))
                if result is not None and result[0] < best_rank:
                    best_rank, best_new_id = result
                    best_position = pos
            if best_position < 0:
                break
            symbols[best_position:best_position + 2] = [best_new_id]
        return symbols

    def decode_ids(self, ids):
        data = bytearray()
        for token_id in ids:
            if token_id not in self.reverse:
                raise ContractError(f"ID {token_id} not in vocabulary")
            piece = self.reverse[token_id].replace("▁", " ")
            raw = re.fullmatch(r"<0x([0-9A-Fa-f]{2})>", piece)
            if raw:
                data.append(int(raw.group(1), 16))
            else:
                data.extend(piece.encode("utf-8"))
        return data.decode("utf-8", errors="replace")

    def encode_chat(self, user_text):
        return [2, 105, 2364, 107, *self.encode_plain(user_text),
                106, 107, 105, 4368, 107]


def verify_golden(tokenizer, path):
    raw = Path(path).read_bytes()
    if len(raw) != 512:
        raise ContractError("Golden prompt must contain exactly 128 int32 IDs")
    saved = struct.unpack("<128i", raw)
    if tuple(saved[:17]) != GOLDEN_PROMPT_IDS or any(saved[17:]):
        raise ContractError("Golden Q8B prompt bytes do not match verified 17 IDs")
    encoded = tokenizer.encode_plain(GOLDEN_TEXT)
    print("BPE_STAGE7_1_GOLDEN_USER:",
          f"actual={encoded}", f"expected={list(GOLDEN_USER_IDS)}")
    if encoded != list(GOLDEN_USER_IDS):
        first = next((i for i, (a, b) in enumerate(
            zip(encoded, GOLDEN_USER_IDS)) if a != b),
                     min(len(encoded), len(GOLDEN_USER_IDS)))
        raise ContractError(
            f"Golden BPE mismatch at position {first}: "
            "do not use this encoder for live inference")
    actual = tokenizer.encode_chat(GOLDEN_TEXT)
    if actual != list(GOLDEN_CANONICAL_IDS):
        raise ContractError("Canonical 18-token chat template mismatch")
    if tokenizer.decode_ids(encoded) != GOLDEN_TEXT:
        raise ContractError("Golden BPE decode roundtrip mismatch")
    print("BPE_STAGE7_1_GOLDEN_OK: known_user_tokens=9"
          " verified_chat_tokens=18 corrected_model_header_LF=107"
          " roundtrip=1 no_package_dependencies=1")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--golden-raw", type=Path, default=DEFAULT_GOLDEN_RAW)
    parser.add_argument("--text", help="A plain user message to BPE-encode")
    parser.add_argument("--output-new-raw", type=Path,
                        help="Create a NEW 512-byte raw ID file, never overwrite")
    args = parser.parse_args()

    if args.output_new_raw and args.text is None:
        parser.error("--output-new-raw requires --text")
    tok = GemmaBPE(args.tokenizer)
    verify_golden(tok, args.golden_raw)

    if args.text is None:
        return
    user_ids = tok.encode_plain(args.text)
    chat_ids = tok.encode_chat(args.text)
    if len(chat_ids) >= 128:
        raise ContractError(
            f"Prompt has {len(chat_ids)} tokens; require <128 for padding")
    if any(i < 0 or i >= 262144 for i in chat_ids):
        raise ContractError("Unexpected out-of-range token ID")
    print("BPE_STAGE7_1_USER_TEXT:", repr(args.text))
    print("BPE_STAGE7_1_USER_IDS:", json.dumps(user_ids))
    print("BPE_STAGE7_1_CHAT_IDS:", json.dumps(chat_ids))
    print("BPE_STAGE7_1_CHAT_TOKENS:",
          f"real={len(chat_ids)} padding={128-len(chat_ids)}"
          " canonical_model_header_LF=1")
    print("BPE_STAGE7_1_USER_ROUNDTRIP:",
          repr(tok.decode_ids(user_ids)))
    if args.output_new_raw:
        raw = struct.pack("<128i", *(chat_ids + [0] * (128-len(chat_ids))))
        # "xb" is exclusive: existing reference files/symlinks are not
        # overwritten, even if requested by mistake.
        with args.output_new_raw.open("xb") as handle:
            handle.write(raw)
        print("BPE_STAGE7_1_NEW_RAW_CREATED:",
              str(args.output_new_raw), "bytes=512"
              " existing_files_not_overwritten=1")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, TypeError, UnicodeError, struct.error) as exc:
        print("BPE_STAGE7_1_ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
