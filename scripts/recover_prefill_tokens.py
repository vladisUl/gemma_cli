#!/usr/bin/env python3
"""Recover 128 Prefill token IDs from saved Section0 FLOAT32 QNN embeddings.

Use the original Section0 packed INT2 table to shortlist candidates by the
signs of 32 dimensions, then prove each token by byte-identical 6144-byte
FLOAT32 output of our verified native decode embedding executor.

This script is read-only w.r.t. original TFLite and saved QNN input data.
It writes one separate verified token_ids INT32[128] file only on success.
No TensorFlow, numpy, or third-party Python packages are required.
"""
import argparse
import collections
import math
import mmap
import os
import pathlib
import shlex
import struct
import subprocess
import sys
import tempfile

from inspect_tflite_ops import FB
from verify_packed_embeddings import locate_int2_table

VOCAB = 262144
TOKENS = 128
EMBED_COLS = 1536
EMBED_BYTES = EMBED_COLS * 4
ROW_BYTES = EMBED_COLS // 4
QNN_INPUT_INDEX = 32
QNN_INPUT_NAME = "prefill_128_embeddings_0"

# Each pair of packed bits decodes to signed INT2 {-2,-1,0,1}.
# This maps each code to the 2-bit SIGN label: zero=0, plus=1, minus=2.
SIGN_BYTE = tuple(sum(
    (0 if (b >> (2 * i)) & 3 == 0 else
     1 if (b >> (2 * i)) & 3 == 1 else 2) << (2 * i)
    for i in range(4)
) for b in range(256))


def read_manifest(work, filename):
    path = work / filename
    if not path.is_file():
        raise ValueError(f"PREFILL_INPUT_LIST_MISSING: {path}")
    lines = [line.strip() for line in path.read_text(encoding="utf8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise ValueError(f"PREFILL_INPUT_LIST: expected 1 sample, got {len(lines)}")
    inputs = shlex.split(lines[0], comments=False)
    if len(inputs) != 36:
        raise ValueError(f"PREFILL_INPUT_LIST: expected 36 inputs, got {len(inputs)}")
    entry = inputs[QNN_INPUT_INDEX]
    if ":=" in entry:
        name, filepath = entry.split(":=", 1)
        if name != QNN_INPUT_NAME:
            raise ValueError(f"PREFILL_INPUT_NAME: expected {QNN_INPUT_NAME}, got {name}")
    else:
        filepath = entry
    reference = pathlib.Path(filepath)
    if not reference.is_absolute():
        reference = work / reference
    if not reference.is_file() or reference.stat().st_size != TOKENS * EMBED_BYTES:
        raise ValueError(f"PREFILL_SAVED_EMBEDDINGS_SIZE_MISMATCH: {reference}")
    return reference


def sign_key(row):
    if len(row) != EMBED_BYTES:
        raise ValueError("Expected 1536 FLOAT32 values in one Prefill row")
    values = struct.unpack_from("<32f", row)
    key = 0
    for i, value in enumerate(values):
        if not math.isfinite(value):
            raise ValueError("Non-finite saved Prefill embeddings")
        kind = 1 if value > 0 else 2 if value < 0 else 0
        key |= kind << (2 * i)
    return key


def scan_table(table, data_start, needed):
    # 262144 * 8 direct mmap byte accesses. One scan for the entire Prefill
    # sample, NOT 128 full-vocabulary searches and not 262144 subprocesses.
    match = {key: [] for key in needed}
    for token in range(VOCAB):
        start = data_start + token * ROW_BYTES
        key = (
            SIGN_BYTE[table[start]] |
            (SIGN_BYTE[table[start+1]] << 8) |
            (SIGN_BYTE[table[start+2]] << 16) |
            (SIGN_BYTE[table[start+3]] << 24) |
            (SIGN_BYTE[table[start+4]] << 32) |
            (SIGN_BYTE[table[start+5]] << 40) |
            (SIGN_BYTE[table[start+6]] << 48) |
            (SIGN_BYTE[table[start+7]] << 56)
        )
        if key in match:
            match[key].append(token)
    return match


def compute_native(exe, model, token, tempdir):
    output = tempdir / f"token_{token}.raw"
    cmd = [str(exe), "--section", "0", "--model", str(model),
           "--token", str(token), "--output", str(output)]
    run = subprocess.run(cmd, capture_output=True, text=True)
    if run.returncode:
        raise ValueError(
            f"PACKED_EMBED_SUBPROCESS_FAILED: token={token}\n"
            + run.stdout + run.stderr
        )
    if not output.is_file() or output.stat().st_size != EMBED_BYTES:
        raise ValueError(f"PACKED_EMBED_RESULT_INVALID: token={token}")
    return output.read_bytes()


def save_ids(path, ids):
    data = struct.pack("<128i", *ids)
    if path.exists():
        if not path.is_file() or path.read_bytes() != data:
            raise ValueError(f"REFUSING_TO_OVERWRITE_DIFFERENT_TOKENS: {path}")
        print(f"PREFILL_TOKENS_ALREADY_MATCH: {path}", flush=True)
        return
    with path.open("xb") as output:
        output.write(data)
    print(f"PREFILL_TOKENS_SAVED: {path} bytes={len(data)}", flush=True)


def run(args):
    work = args.work.resolve(strict=True)
    exe = args.exe.resolve(strict=True)
    model = work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
    if not model.is_file() or not exe.is_file():
        raise ValueError("Section0 model or native gemma4-packed-embed missing")
    reference = read_manifest(work, args.list)
    raw = reference.read_bytes()
    rows = [raw[i * EMBED_BYTES:(i+1) * EMBED_BYTES] for i in range(TOKENS)]
    by_key = collections.defaultdict(set)
    for index, row in enumerate(rows):
        by_key[sign_key(row)].add(index)

    with model.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            if mapped[4:8] != b"TFL3":
                raise ValueError("Section0 file is not an original TFLite FlatBuffer")
            start = locate_int2_table(FB(mapped))
            matches = scan_table(mapped, start, by_key.keys())

    unique_rows = len(set(rows))
    print(f"PREFILL_TOKEN_SEARCH_READY: rows={TOKENS} "
          f"unique_float32_rows={unique_rows} sign_patterns={len(by_key)} "
          f"model={model.name}", flush=True)

    token_cache = {}
    identified = [None] * TOKENS
    with tempfile.TemporaryDirectory(prefix="prefill_token_verify_", dir=work) as tmp:
        temporary = pathlib.Path(tmp)
        for key, positions in by_key.items():
            candidates = matches[key]
            if not candidates:
                raise ValueError(
                    f"PREFILL_TOKEN_NO_CANDIDATES: positions={sorted(positions)}")
            if len(candidates) > 200:
                raise ValueError(
                    f"PREFILL_TOKEN_AMBIGUOUS_SIGN_PATTERN: "
                    f"positions={sorted(positions)} candidates={len(candidates)}")
            print(f"PREFILL_TOKEN_CANDIDATES: positions={len(positions)} "
                  f"sign_candidates={len(candidates)}", flush=True)
            for pos in sorted(positions):
                exact = []
                for token in candidates:
                    if token not in token_cache:
                        token_cache[token] = compute_native(
                            exe, model, token, temporary)
                    if token_cache[token] == rows[pos]:
                        exact.append(token)
                if len(exact) != 1:
                    raise ValueError(
                        f"PREFILL_TOKEN_NOT_UNIQUE: position={pos} "
                        f"full_6144_byte_matches={exact} "
                        f"sign_candidates={len(candidates)}")
                identified[pos] = exact[0]
                print(f"PREFILL_TOKEN_REFERENCE_MATCH: position={pos} "
                      f"token={exact[0]} bytes={EMBED_BYTES}", flush=True)

    if any(token is None for token in identified):
        raise ValueError("Incomplete Prefill token identification")
    counts = collections.Counter(identified)
    output = args.output or work / "derived_prefill_128_token_ids_i32.raw"
    output = output.resolve()
    save_ids(output, identified)
    print(f"PREFILL_TOKEN_IDS: {','.join(map(str, identified))}", flush=True)
    print(f"PREFILL_TOKENS_VERIFIED: count={len(identified)} "
          f"unique={len(counts)} reference_bytes={len(raw)} "
          f"first={identified[0]} last={identified[-1]}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path.home() / "gemma4_qnn_test")
    parser.add_argument("--exe", type=pathlib.Path,
                        default=pathlib.Path.home() /
                        "gemma4-cli-stage1/build/gemma4-packed-embed")
    parser.add_argument("--list", default="prefill_real_input_list.txt")
    parser.add_argument("--output", type=pathlib.Path, default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, struct.error) as exc:
        print(f"PREFILL_TOKEN_RECOVERY_ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
