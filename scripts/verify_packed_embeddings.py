#!/usr/bin/env python3
"""Identify saved Decode token IDs from original INT2 embeddings and strictly
compare native packed Section0/2 outputs to BOTH saved QNN Decode inputs.

Read-only FlatBuffer mmap; no TensorFlow, numpy, or third-party Python packages.
Input-list relative paths are resolved against the test working directory.
Never writes to the original TFLite models or saved QNN input files.
"""
import argparse
import mmap
import os
import pathlib
import struct
import subprocess
import sys
import tempfile

from inspect_tflite_ops import FB
from verify_cpu_aux import parse_manifest

VOCAB = 262144
COLS = 1536
ROW_BYTES = COLS // 4  # INT2: four signed values per byte
PROBE_COLS = 32
IDX_SECTION0 = 33
IDX_SECTION2 = 7
NAMES = {IDX_SECTION0: "decode_embeddings_0",
         IDX_SECTION2: "decode_per_layer_embeddings_0"}
SIZES = {IDX_SECTION0: 6144, IDX_SECTION2: 35840}


def load_reference(entries, index):
    name, filename = entries[index]
    if name is not None and name != NAMES[index]:
        raise ValueError(f"Unexpected QNN input[{index}] name {name!r}")
    if not filename.is_file():
        raise ValueError(f"Saved QNN input[{index}] missing: {filename}")
    if filename.stat().st_size != SIZES[index]:
        raise ValueError(f"Unexpected saved QNN input[{index}] size: {filename}")
    return filename, filename.read_bytes()


def locate_int2_table(fb):
    root = fb.u32(0)
    if fb.scalar(root, 0, "I") != 3:
        raise ValueError("Section0 is not a schema-v3 TFLite FlatBuffer")
    graphs = fb.table_vector(root, 2)
    if len(graphs) != 2:
        raise ValueError("Unexpected Section0 subgraphs")
    graph = graphs[0]
    codes = fb.table_vector(root, 1)
    tensors = fb.table_vector(graph, 0)
    buffers = fb.table_vector(root, 4)
    matches = []
    for op in fb.table_vector(graph, 3):
        index = fb.scalar(op, 0, "I")
        code = codes[index]
        builtin = max(fb.scalar(code, 0, "b"), fb.scalar(code, 3, "i"))
        version = fb.scalar(code, 2, "i", 1)
        if builtin != 7:
            continue
        if version != 4:
            raise ValueError("Section0 EMBEDDING_LOOKUP is not v4")
        inputs = fb.int_vector(op, 1)
        if len(inputs) != 2:
            raise ValueError("Invalid EMBEDDING_LOOKUP inputs")
        tensor = tensors[inputs[1]]
        if fb.scalar(tensor, 1, "b") != 19:
            raise ValueError("Expected original INT2 embedding weights")
        if fb.int_vector(tensor, 0) != [VOCAB, COLS]:
            raise ValueError("Unexpected Section0 weight shape")
        bid = fb.scalar(tensor, 2, "I")
        data_start, data_length = fb.vector(buffers[bid], 0)
        if data_length != VOCAB * ROW_BYTES:
            raise ValueError("Unexpected INT2 packed weight buffer length")
        quant_ptr = fb.table_field(tensor, 4)
        if quant_ptr is None:
            raise ValueError("Missing Section0 row quantization")
        quant = fb.indirect(quant_ptr)
        _, nscales = fb.vector(quant, 2)
        if nscales != VOCAB or fb.scalar(quant, 6, "i") != 0:
            raise ValueError("Section0 must have per-row quantization")
        matches.append(data_start)
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one Section0 lookup: {len(matches)}")
    return matches[0]


def search_token_candidates(mm, data_start, reference):
    if len(reference) != SIZES[IDX_SECTION0]:
        raise ValueError("Expected exactly 1536 FLOAT32 embeddings")
    floats = struct.unpack_from("<" + "f" * PROBE_COLS, reference)
    signs = [(1 if value > 0 else (-1 if value < 0 else 0))
             for value in floats]
    if not any(signs):
        raise ValueError("Reference first 32 embedding floats all zero; cannot identify token by signs")

    # Signed INT2 encodes -2=0b10, -1=0b11, 0=0b00, 1=0b01.
    # Nonnegative quantization scale and the positive Section0 MUL scalar
    # preserve sign. Each 2-bit nibble allows an exact zero/positive match
    # or matches just the sign bit for either negative code.
    masks, patterns = [], []
    for byte_index in range(PROBE_COLS // 4):
        mask = 0
        expected = 0
        for slot in range(4):
            sign = signs[4 * byte_index + slot]
            shift = 2 * slot
            if sign < 0:
                mask |= 2 << shift
                expected |= 2 << shift
            else:
                mask |= 3 << shift
                if sign > 0:
                    expected |= 1 << shift
        masks.append(mask)
        patterns.append(expected)

    candidates = []
    for token in range(VOCAB):
        start = data_start + token * ROW_BYTES
        for i, (mask, pattern) in enumerate(zip(masks, patterns)):
            if mm[start + i] & mask != pattern:
                break
        else:
            candidates.append(token)
    return candidates


def compare_outputs(actual, expected, label):
    if len(actual) != len(expected):
        raise ValueError(f"{label}: different output size")
    if actual == expected:
        print(f"PACKED_EMBED_REFERENCE_MATCH: {label} bytes={len(actual)}", flush=True)
        return True

    diff = next(i for i in range(len(actual)) if actual[i] != expected[i])
    float_count = len(actual) // 4
    current = struct.unpack("<" + "f" * float_count, actual)
    original = struct.unpack("<" + "f" * float_count, expected)
    differs = sum(a != b for a, b in zip(current, original))
    max_abs = max(abs(a - b) for a, b in zip(current, original))
    print(f"PACKED_EMBED_REFERENCE_MISMATCH: {label} first_byte={diff} "
          f"floats_differ={differs}/{float_count} max_abs_diff={max_abs:.9g}",
          flush=True)
    return False


def execute(args, section, token, output):
    model = (args.work / "cpu_models" /
             ("Section0_TFLiteModel_tf_lite_embedder.tflite" if section == 0
              else "Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"))
    if not model.is_file():
        raise ValueError(f"CPU source model missing: {model}")
    cmd = [str(args.exe), "--model", str(model),
           "--section", str(section), "--token", str(token),
           "--output", str(output)]
    subprocess.run(cmd, check=True)
    return output.read_bytes()


def verify_step(args, step):
    entries = parse_manifest(args.work / f"auto_decode_step{step}_list.txt")
    refs = {index: load_reference(entries, index) for index in NAMES}
    ref0 = refs[IDX_SECTION0][1]

    model0 = args.work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"
    with model0.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            fb = FB(mm)
            data_start = locate_int2_table(fb)
            candidates = search_token_candidates(mm, data_start, ref0)

    if not candidates:
        raise ValueError(f"Step {step}: no token matches saved Section0 INT2 sign pattern")
    if len(candidates) > 32:
        raise ValueError(f"Step {step}: ambiguous token pattern ({len(candidates)} matches)")
    print(f"PACKED_TOKEN_CANDIDATES: step={step} count={len(candidates)} "
          f"ids={candidates}", flush=True)

    with tempfile.TemporaryDirectory(prefix=f"packed_check_step{step}.",
                                     dir=args.work) as scratch:
        directory = pathlib.Path(scratch)
        exact = []
        approx = []
        for token in candidates:
            raw = execute(args, 0, token, directory / f"section0_token{token}.raw")
            if raw == ref0:
                exact.append(token)
            else:
                approx.append(token)
        if len(exact) != 1:
            print(f"PACKED_TOKEN_NOT_VERIFIED: step={step} "
                  f"exact_matches={exact} sign_candidates={candidates}",
                  flush=True)
            for token in approx[:3]:
                compare_outputs(
                    (directory / f"section0_token{token}.raw").read_bytes(),
                    ref0, f"step={step} section=0 token={token}")
            raise ValueError(f"Step {step}: cannot identify a byte-exact Section0 token")
        token = exact[0]
        print(f"PACKED_TOKEN_IDENTIFIED: step={step} token={token} "
              "proof=Section0_6144_bytes_exact", flush=True)
        compare_outputs(
            (directory / f"section0_token{token}.raw").read_bytes(),
            ref0, f"step={step} section=0 token={token}")
        raw2 = execute(args, 2, token, directory / f"section2_token{token}.raw")
        if not compare_outputs(raw2, refs[IDX_SECTION2][1],
                               f"step={step} section=2 token={token}"):
            raise ValueError(f"Step {step}: Section2 native output != saved QNN input")
        print(f"PACKED_STEP_VERIFIED: step={step} token={token} "
              "sections=0,2 total_bytes=41984", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path.home() / "gemma4_qnn_test")
    parser.add_argument("--exe", type=pathlib.Path,
                        default=pathlib.Path.home() /
                        "gemma4-cli-stage1/build/gemma4-packed-embed")
    parser.add_argument("--steps", type=int, nargs="+", choices=(3, 4),
                        default=[3, 4])
    args = parser.parse_args()
    args.work = args.work.resolve(strict=True)
    args.exe = args.exe.resolve(strict=True)
    if not args.exe.is_file():
        raise ValueError(f"Packed runner missing: {args.exe}")
    os.chdir(args.work)
    for step in args.steps:
        verify_step(args, step)
    print(f"PACKED_ALL_STEPS_VERIFIED: steps={args.steps} "
          "sections=0,2 exact_reference_match=1", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(f"PACKED_VERIFY_ERROR: {exc}", file=sys.stderr, flush=True)
        sys.exit(1)
