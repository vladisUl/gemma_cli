#!/usr/bin/env python3
"""Compare locally computed Section1 AUX tensors with saved QNN Decode inputs.

The QNN input indices below are taken from the verified Gemma 4 E2B
qnn_partition_0 input metadata (38 inputs), NOT guessed from file names.
All paths inside a qnn-net-run input list are relative to the current
working directory, matching gemma4-qnn-run.
"""
import argparse
import pathlib
import shlex
import sys

# (CPU signature output, QNN graph-input index, QNN input name, bytes)
MAPS = {
    "rope": (
        ("pos_emb_cos", 3, "decode_pos_emb_cos_0", 512),
        ("pos_emb_sin", 5, "decode_pos_emb_sin_0", 512),
        ("pos_emb_local_sin", 6, "decode_pos_emb_local_sin_0", 256),
        ("pos_emb_local_cos", 18, "decode_pos_emb_local_cos_0", 256),
    ),
    "mask": (
        ("mask_local", 22, "decode_mask_local_0", 4097),
        ("mask_global", 27, "decode_mask_global_0", 4097),
    ),
}
GRAPH_INPUTS = 38


def parse_manifest(path):
    with path.open(encoding="utf-8") as handle:
        samples = [
            line.strip()
            for line in handle
            if line.strip() and not line.lstrip().startswith("#")
        ]
    if len(samples) != 1:
        raise ValueError(
            f"QNN_INPUT_LIST_ERROR: expected one sample, got {len(samples)}"
        )
    entries = shlex.split(samples[0], comments=False)
    if len(entries) != GRAPH_INPUTS:
        raise ValueError(
            f"QNN_INPUT_LIST_ERROR: expected {GRAPH_INPUTS} paths, "
            f"got {len(entries)}"
        )
    parsed = []
    for entry in entries:
        if ":=" in entry:
            name, filename = entry.split(":=", 1)
            if not name or not filename:
                raise ValueError(f"QNN_INPUT_LIST_ERROR: invalid {entry!r}")
        else:
            name, filename = None, entry
        if not filename:
            raise ValueError("QNN_INPUT_LIST_ERROR: empty path")
        parsed.append((name, pathlib.Path(filename)))
    return parsed


def verify(kind, cpu_dir, manifest):
    inputs = parse_manifest(manifest)
    for short_name, index, qnn_name, expected_bytes in MAPS[kind]:
        listed_name, qnn_path = inputs[index]
        if listed_name is not None and listed_name != qnn_name:
            raise ValueError(
                f"QNN_INPUT_NAME_MISMATCH: input[{index}] expected "
                f"{qnn_name}, got {listed_name}"
            )
        cpu_path = cpu_dir / (short_name + ".raw")
        if not cpu_path.is_file():
            raise ValueError(f"CPU_AUX_OUTPUT_NOT_FOUND: {cpu_path}")
        if not qnn_path.is_file():
            raise ValueError(f"QNN_INPUT_NOT_FOUND: {qnn_path}")

        cpu_size = cpu_path.stat().st_size
        qnn_size = qnn_path.stat().st_size
        if (cpu_size, qnn_size) != (expected_bytes, expected_bytes):
            raise ValueError(
                f"CPU_AUX_SIZE_MISMATCH: {short_name} expected="
                f"{expected_bytes} CPU={cpu_size} QNN={qnn_size}"
            )
        actual = cpu_path.read_bytes()
        expected = qnn_path.read_bytes()
        if actual != expected:
            first = next(i for i, (a, b) in enumerate(zip(actual, expected))
                         if a != b)
            raise ValueError(
                f"CPU_AUX_MISMATCH: {short_name} qnn_index={index} "
                f"first_byte={first} CPU=0x{actual[first]:02x} "
                f"QNN=0x{expected[first]:02x} "
                f"(check that input_pos and saved QNN sample match)"
            )
        print(
            f"CPU_AUX_REFERENCE_MATCH: {short_name} "
            f"qnn_input_index={index} bytes={expected_bytes} "
            f"reference={qnn_path}",
            flush=True,
        )
    print(
        f"CPU_AUX_ALL_MATCH: kind={kind} tensors={len(MAPS[kind])} "
        f"bytes={sum(row[3] for row in MAPS[kind])}",
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=sorted(MAPS), required=True)
    parser.add_argument("--cpu-dir", type=pathlib.Path, required=True)
    parser.add_argument("--decode-list", type=pathlib.Path, required=True)
    args = parser.parse_args()
    verify(args.kind, args.cpu_dir, args.decode_list)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as exc:
        print(f"CPU_AUX_VERIFY_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
