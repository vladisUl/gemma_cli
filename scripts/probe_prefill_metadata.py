#!/usr/bin/env python3
"""Inspect original Gemma 4 Prefill-128 CPU signatures and saved QNN inputs.

No TFLite runtime, SDK headers, numpy or third-party dependencies. Uses
read-only mmap for original Section0/1/2 FlatBuffers. No QNN graph execution.
This probes model interfaces only; it does not establish CPU→NPU equivalence.
"""
import argparse
import mmap
import os
import pathlib
import shlex
import sys

from inspect_tflite_ops import FB, TYPES

QNN_NON_KV = {
    3: ("prefill_128_pos_emb_cos_0", 65536),
    5: ("prefill_128_pos_emb_sin_0", 65536),
    6: ("prefill_128_pos_emb_local_sin_0", 32768),
    7: ("prefill_128_per_layer_embeddings_0", 4587520),
    18: ("prefill_128_pos_emb_local_cos_0", 32768),
    21: ("prefill_128_mask_local_0", 540672),
    26: ("prefill_128_mask_global_0", 540672),
    32: ("prefill_128_embeddings_0", 786432),
}
MODEL_FILES = {
    0: "Section0_TFLiteModel_tf_lite_embedder.tflite",
    1: "Section1_TFLiteModel_tf_lite_aux.tflite",
    2: "Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite",
}


def alias_info(fb, graph, alias):
    tensors = fb.table_vector(graph, 0)
    tid = fb.scalar(alias, 1, "I")
    if tid >= len(tensors):
        raise ValueError(f"Invalid TensorMap tensor index: {tid}")
    tensor = tensors[tid]
    name = fb.string(alias, 0)
    dtype = fb.scalar(tensor, 1, "b", 0)
    dims = fb.int_vector(tensor, 0)
    elements = 1
    for dim in dims:
        if dim < 0:
            raise ValueError("Dynamic prefill tensor dimension")
        elements *= dim
    return (name, TYPES.get(dtype, f"TYPE_{dtype}"), dims, elements)


def inspect_model(path, section):
    if not path.is_file():
        raise ValueError(f"PREFILL_CPU_MODEL_MISSING: section={section} path={path}")
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            if mm[4:8] != b"TFL3":
                raise ValueError(f"Model is not a TFL3 FlatBuffer: {path}")
            fb = FB(mm)
            root = fb.u32(0)
            graphs = fb.table_vector(root, 2)
            signatures = fb.table_vector(root, 7)
            count = 0
            for sig in signatures:
                key = fb.string(sig, 2)
                if not key.startswith("prefill"):
                    continue
                graph_index = fb.scalar(sig, 4, "I", 0)
                if graph_index >= len(graphs):
                    raise ValueError(f"Signature {key} references missing subgraph")
                graph = graphs[graph_index]
                print(f"PREFILL_CPU_SIGNATURE: section={section} "
                      f"name={key} graph={graph_index}")
                for direction, index in (("input", 0), ("output", 1)):
                    for alias in fb.table_vector(sig, index):
                        name, dtype, dims, elements = alias_info(fb, graph, alias)
                        print(f"PREFILL_CPU_IO: section={section} signature={key} "
                              f"direction={direction} name={name} type={dtype} "
                              f"shape={dims} elements={elements}")
                count += 1
            if not count:
                raise ValueError(f"Section{section} has no prefill signatures")
            print(f"PREFILL_CPU_MODEL_OK: section={section} bytes={len(mm)} "
                  f"signatures={count}")
            return True


def inspect_saved_list(path):
    if not path.is_file():
        print(f"PREFILL_QNN_LIST_MISSING: {path}")
        return False
    with path.open(encoding="utf-8") as stream:
        lines = [line.strip() for line in stream if line.strip()
                 and not line.lstrip().startswith("#")]
    if len(lines) != 1:
        raise ValueError(f"Expected one QNN input sample; found {len(lines)}")
    entries = shlex.split(lines[0], comments=False)
    if len(entries) != 36:
        raise ValueError(f"Expected 36 QNN Prefill inputs; got {len(entries)}")
    count = 0
    for index, (expected_name, expected_size) in sorted(QNN_NON_KV.items()):
        entry = entries[index]
        if ":=" in entry:
            name, filename = entry.split(":=", 1)
            if name != expected_name:
                raise ValueError(f"QNN input[{index}] name {name!r}, "
                                 f"expected {expected_name!r}")
        else:
            filename = entry
        input_path = pathlib.Path(filename)
        if not input_path.is_file():
            raise ValueError(f"Missing saved QNN Prefill input[{index}]: {input_path}")
        n = input_path.stat().st_size
        if n != expected_size:
            raise ValueError(f"QNN Prefill input[{index}] wrong byte count: "
                             f"expected {expected_size}, actual {n}")
        print(f"PREFILL_QNN_NONKV: index={index} name={expected_name} "
              f"bytes={n} file={input_path}", flush=True)
        count += 1
    print(f"PREFILL_QNN_LIST_OK: inputs={len(entries)} "
          f"non_kv={count} kv={len(entries)-count} "
          f"source={path}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path.home() / "gemma4_qnn_test")
    parser.add_argument("--list", type=pathlib.Path,
                        default=pathlib.Path("prefill_real_input_list.txt"))
    args = parser.parse_args()
    args.work = args.work.resolve(strict=True)
    os.chdir(args.work)
    for section, filename in MODEL_FILES.items():
        inspect_model(args.work / "cpu_models" / filename, section)
    inspect_saved_list(args.list)
    print("PREFILL_INTERFACE_PROBE_OK: metadata inspection completed")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, IndexError, UnicodeError) as exc:
        print(f"PREFILL_INTERFACE_PROBE_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
