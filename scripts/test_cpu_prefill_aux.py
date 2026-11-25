#!/usr/bin/env python3
"""Stage 6.3: compute original Section1 Prefill-128 RoPE and masks on CPU.

Build arguments from actual TFLite signature metadata, never hard-code tensor
shapes; check all six results byte-for-byte against the original 36-input QNN
Prefill manifest. No NPU execution, no changes to original/reference .raw.
"""
import argparse
import mmap
import pathlib
import shlex
import struct
import subprocess
import sys
import tempfile

from inspect_tflite_ops import FB

SIGNATURE_OUTPUTS = {
    "prefill_rope_128": {
        "pos_emb_cos": (3, "prefill_128_pos_emb_cos_0", 65536),
        "pos_emb_sin": (5, "prefill_128_pos_emb_sin_0", 65536),
        "pos_emb_local_sin": (6, "prefill_128_pos_emb_local_sin_0", 32768),
        "pos_emb_local_cos": (18, "prefill_128_pos_emb_local_cos_0", 32768),
    },
    "prefill_mask_128": {
        "mask_local": (21, "prefill_128_mask_local_0", 540672),
        "mask_global": (26, "prefill_128_mask_global_0", 540672),
    },
}
TOKENS = 128
GRAPH_INPUTS = 36


def signature_info(model_path):
    with model_path.open("rb") as source:
        with mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if data[4:8] != b"TFL3":
                raise ValueError("Section1 is not a TFLite FlatBuffer")
            fb = FB(data)
            root = fb.u32(0)
            graphs = fb.table_vector(root, 2)
            found = {}
            for sig in fb.table_vector(root, 7):
                name = fb.string(sig, 2)
                if name not in SIGNATURE_OUTPUTS:
                    continue
                graph_index = fb.scalar(sig, 4, "I", 0)
                if graph_index >= len(graphs):
                    raise ValueError(f"CPU signature {name} has invalid subgraph")
                graph = graphs[graph_index]
                tensors = fb.table_vector(graph, 0)
                mapping = {}
                for direction, field in (("input", 0), ("output", 1)):
                    names = {}
                    for alias in fb.table_vector(sig, field):
                        alias_name = fb.string(alias, 0)
                        tid = fb.scalar(alias, 1, "I")
                        if tid >= len(tensors) or alias_name in names:
                            raise ValueError(f"Bad {name} {direction} tensor alias")
                        tensor = tensors[tid]
                        dtype = fb.scalar(tensor, 1, "b", 0)
                        shape = fb.int_vector(tensor, 0)
                        if any(dim < 0 for dim in shape):
                            raise ValueError(f"Unexpected dynamic tensor {alias_name}")
                        elements = 1
                        for dim in shape:
                            elements *= dim
                        names[alias_name] = (dtype, shape, elements)
                        print(f"PREFILL_AUX_SIGNATURE_IO: signature={name} "
                              f"direction={direction} name={alias_name} "
                              f"dtype={dtype} shape={shape} elements={elements}",
                              flush=True)
                    mapping[direction] = names
                if name in found:
                    raise ValueError(f"Repeated Section1 signature {name}")
                found[name] = mapping
            if set(found) != set(SIGNATURE_OUTPUTS):
                raise ValueError("Missing prefill Section1 signatures; "
                                 f"found={sorted(found)}")
            return found


def manifest_references(work, list_name):
    manifest = pathlib.Path(list_name)
    if not manifest.is_absolute():
        manifest = work / manifest
    lines = [line.strip() for line in manifest.read_text(encoding="utf-8").splitlines()
             if line.strip() and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise ValueError(f"Prefill manifest must have exactly one sample: {manifest}")
    entries = shlex.split(lines[0], comments=False)
    if len(entries) != GRAPH_INPUTS:
        raise ValueError(f"Expected {GRAPH_INPUTS} QNN inputs, got {len(entries)}")
    references = {}
    for signature, outputs in SIGNATURE_OUTPUTS.items():
        for alias, (index, qnn_name, size) in outputs.items():
            entry = entries[index]
            if ":=" in entry:
                name, filename = entry.split(":=", 1)
                if name != qnn_name:
                    raise ValueError(f"QNN index[{index}]: got {name!r}, expected {qnn_name!r}")
            else:
                filename = entry
            path = pathlib.Path(filename)
            if not path.is_absolute():
                path = work / path
            if not path.is_file() or path.stat().st_size != size:
                raise ValueError(f"Missing/wrong reference size for {qnn_name}: {path}")
            references[(signature, alias)] = (path, size, index)
    return references


def get_input_payload(alias, dtype, shape, elements, tokens, pos_start, time_step):
    if dtype != 2:
        raise ValueError(f"Only INT32 Section1 prefill inputs supported, got dtype={dtype} for {alias}")
    if alias == "input_tokens":
        if elements != TOKENS:
            raise ValueError(f"input_tokens expects 128 elements, got {shape}")
        values = tokens
    elif alias in ("input_pos", "input_positions", "positions"):
        if elements == TOKENS:
            values = list(range(pos_start, pos_start + TOKENS))
        elif elements == 1:
            values = [pos_start]
        else:
            raise ValueError(f"Unexpected input_pos shape {shape}")
    elif alias in ("time_step", "time_steps"):
        if elements == 1:
            values = [time_step]
        elif elements == TOKENS:
            values = list(range(time_step, time_step + TOKENS))
        else:
            raise ValueError(f"Unexpected time_step shape {shape}")
    else:
        raise ValueError(
            f"Unknown Section1 Prefill input {alias!r}, shape={shape}; "
            "no guessed input data supplied"
        )
    if len(values) != elements or any(v < -(1 << 31) or v >= (1 << 31) for v in values):
        raise ValueError(f"Invalid input vector for {alias}")
    return struct.pack("<" + "i" * elements, *values)


def compare_bytes(cpu_file, reference, expected_size, signature, name):
    if not cpu_file.is_file() or cpu_file.stat().st_size != expected_size:
        raise ValueError(f"Section1 output missing/wrong size: {cpu_file}")
    actual = cpu_file.read_bytes()
    original = reference.read_bytes()
    if actual != original:
        first = next(i for i, (a, b) in enumerate(zip(actual, original)) if a != b)
        raise ValueError(
            f"PREFILL_AUX_REFERENCE_MISMATCH: {signature} {name} "
            f"byte={first} actual={actual[first]} expected={original[first]}"
        )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--work", type=pathlib.Path,
                   default=pathlib.Path.home() / "gemma4_qnn_test")
    p.add_argument("--exe", type=pathlib.Path,
                   default=pathlib.Path.home() / "gemma4-cli-stage1/build/gemma4-cpu-run")
    p.add_argument("--list", default="prefill_real_input_list.txt")
    p.add_argument("--model", type=pathlib.Path, default=None)
    p.add_argument("--tokens", type=pathlib.Path, default=None)
    p.add_argument("--pos-start", type=int, default=0,
                   help="Prefill input_pos start; default 0 for initial prompt")
    p.add_argument("--time-step", type=int, default=0,
                   help="Mask time_step; default 0 for initial prompt")
    args = p.parse_args()

    work = args.work.resolve(strict=True)
    exe = args.exe.resolve(strict=True)
    model = (args.model or work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite").resolve(strict=True)
    token_path = (args.tokens or work / "derived_prefill_128_token_ids_i32.raw").resolve(strict=True)
    raw = token_path.read_bytes()
    if len(raw) != TOKENS * 4:
        raise ValueError(f"Expected 512-byte token array, got {len(raw)}")
    tokens = list(struct.unpack("<128i", raw))
    if any(token < 0 or token >= 262144 for token in tokens):
        raise ValueError("Recovered token IDs out of vocabulary range")
    if args.pos_start < 0 or args.time_step < 0:
        raise ValueError("Negative prefill positions/time_step are invalid")
    metadata = signature_info(model)
    references = manifest_references(work, args.list)
    with tempfile.TemporaryDirectory(prefix="prefill_aux_verified_", dir=work) as tmp:
        temp = pathlib.Path(tmp)
        byte_total = 0
        for signature, expected_outputs in SIGNATURE_OUTPUTS.items():
            actual_outputs = metadata[signature]["output"]
            if set(actual_outputs) != set(expected_outputs):
                raise ValueError(
                    f"Unexpected outputs for {signature}: {sorted(actual_outputs)}"
                )
            cmd = [str(exe), "--model", str(model), "--signature", signature]
            for alias, (dtype, shape, elements) in metadata[signature]["input"].items():
                payload = get_input_payload(alias, dtype, shape, elements, tokens,
                                            args.pos_start, args.time_step)
                path = temp / f"{signature}_{alias}.i32.raw"
                path.write_bytes(payload)
                cmd.extend(["--input", f"{alias}={path}"])
            output_dir = temp / signature
            cmd.extend(["--output-dir", str(output_dir)])
            for alias, (index, qnn_name, size) in expected_outputs.items():
                dtype, shape, elements = actual_outputs[alias]
                if dtype != 9 or elements != size:
                    raise ValueError(f"TFLite {signature}/{alias} not INT8/{size} bytes")
                reference, _, _ = references[(signature, alias)]
                cmd.extend(["--expect", f"{alias}={reference}"])
            print(f"PREFILL_AUX_CPU_INVOKE_BEGIN: signature={signature} "
                  f"pos_start={args.pos_start} time_step={args.time_step}",
                  flush=True)
            subprocess.run(cmd, check=True, cwd=work)
            for alias, (index, qnn_name, size) in expected_outputs.items():
                saved, _, _ = references[(signature, alias)]
                cpu = output_dir / (alias + ".raw")
                compare_bytes(cpu, saved, size, signature, alias)
                byte_total += size
                print(f"PREFILL_AUX_INPUT_MATCH: signature={signature} "
                      f"qnn_index={index} name={qnn_name} bytes={size}",
                      flush=True)
            print(f"PREFILL_AUX_SIGNATURE_OK: {signature} "
                  f"outputs={len(expected_outputs)}", flush=True)
    if byte_total != 1277952:
        raise ValueError(f"Expected 1277952 total INT8 bytes, got {byte_total}")
    print(f"PREFILL_AUX_ALL_MATCH: tensors=6 bytes={byte_total} "
          f"pos_start={args.pos_start} time_step={args.time_step} "
          "verified_against_saved_qnn=1", flush=True)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, subprocess.CalledProcessError, struct.error) as error:
        print(f"PREFILL_AUX_ERROR: {error}", file=sys.stderr, flush=True)
        sys.exit(1)
