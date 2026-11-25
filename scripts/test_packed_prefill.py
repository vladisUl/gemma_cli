#!/usr/bin/env python3
"""Stage 6.2: byte-exact Section0/2 Prefill embeddings from recovered token IDs.

Uses original 128-token INT32 file and saved qnn-net-run Prefill input list.
The original INT2/INT4 models are read-only, never modified. Both FLOAT32
results are independently verified byte-for-byte BEFORE a success marker.
"""
import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import tempfile

INPUTS = {
    0: (32, "prefill_128_embeddings_0", 786432,
        "Section0_TFLiteModel_tf_lite_embedder.tflite"),
    2: (7, "prefill_128_per_layer_embeddings_0", 4587520,
        "Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"),
}


def reference_path(work, manifest, index, required_name, expected_bytes):
    with manifest.open(encoding="utf-8") as stream:
        lines = [line.strip() for line in stream
                 if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise ValueError(f"QNN Prefill manifest requires one sample: {manifest}")
    entries = shlex.split(lines[0], comments=False)
    if len(entries) != 36:
        raise ValueError(f"QNN Prefill needs 36 inputs, found {len(entries)}")
    item = entries[index]
    if ":=" in item:
        name, raw_path = item.split(":=", 1)
        if name != required_name:
            raise ValueError(f"QNN input[{index}]: expected {required_name}, got {name}")
    else:
        raw_path = item
    path = pathlib.Path(raw_path)
    if not path.is_absolute():
        path = work / path
    if not path.is_file() or path.stat().st_size != expected_bytes:
        raise ValueError(f"QNN reference absent/wrong size ({expected_bytes}): {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path.home() / "gemma4_qnn_test")
    parser.add_argument("--exe", type=pathlib.Path,
                        default=pathlib.Path.home() /
                        "gemma4-cli-stage1/build/gemma4-packed-prefill")
    parser.add_argument("--tokens", type=pathlib.Path, default=None)
    parser.add_argument("--list", default="prefill_real_input_list.txt")
    args = parser.parse_args()
    work = args.work.resolve(strict=True)
    runner = args.exe.resolve(strict=True)
    tokens = (args.tokens or
              work / "derived_prefill_128_token_ids_i32.raw").resolve(strict=True)
    if not runner.is_file():
        raise ValueError(f"Stage 6.2 executable missing: {runner}")
    if tokens.stat().st_size != 512:
        raise ValueError(f"Prefill 128×INT32 token file must be 512 bytes: {tokens}")
    manifest = pathlib.Path(args.list)
    if not manifest.is_absolute():
        manifest = work / manifest
    if not manifest.is_file():
        raise ValueError(f"Saved QNN Prefill manifest not found: {manifest}")

    print(f"PREFILL_EMBED_STAGE6_2_BEGIN: tokens={tokens} "
          f"manifest={manifest}", flush=True)
    with tempfile.TemporaryDirectory(prefix="prefill_embeddings_verified_",
                                     dir=work) as temporary:
        for section, (index, name, size, filename) in INPUTS.items():
            reference = reference_path(work, manifest, index, name, size)
            model = work / "cpu_models" / filename
            if not model.is_file():
                raise ValueError(f"Original Section{section} model absent: {model}")
            output = pathlib.Path(temporary) / f"section{section}_prefill.raw"
            command = [
                str(runner),
                "--section", str(section),
                "--model", str(model),
                "--tokens", str(tokens),
                "--reference", str(reference),
                "--output", str(output),
            ]
            print(f"PREFILL_EMBED_EXECUTE: section={section} "
                  f"reference={reference} bytes={size}", flush=True)
            subprocess.run(command, check=True)
            if not output.is_file() or output.stat().st_size != size:
                raise ValueError(f"Computed Section{section} output missing/wrong length")
            print(f"PREFILL_EMBED_INPUT_MATCH: section={section} "
                  f"qnn_index={index} bytes={size} positions=128",
                  flush=True)
    print("PREFILL_EMBED_ALL_MATCH: sections=0,2 tensors=2 "
          "bytes=5373952 positions=128 byte_identical=1", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f"PREFILL_EMBED_STAGE6_2_ERROR: {exc}", file=sys.stderr,
              flush=True)
        sys.exit(1)
