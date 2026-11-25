#!/usr/bin/env python3
"""Inspect TFLite FlatBuffer operator versions without TensorFlow or third-party Python packages.

Uses read-only mmap: even multi-GB embedding models do not get loaded into RAM.
Prints builtin opcode IDs, required versions, subgraph usage and signature names.
"""
import argparse
import collections
import mmap
import pathlib
import struct
import sys


NAMES = {
    0: "ADD", 2: "CONCATENATION", 6: "DEQUANTIZE", 7: "EMBEDDING_LOOKUP",
    9: "FULLY_CONNECTED", 18: "MUL", 22: "RESHAPE",
    23: "RESIZE_BILINEAR", 36: "GATHER", 53: "CAST",
    58: "LESS", 62: "GREATER_EQUAL", 84: "LOGICAL_OR",
    86: "LOGICAL_AND", 114: "QUANTIZE", 123: "SELECT_V2",
    150: "GELU", 32: "CUSTOM",
}
TYPES = {
    0: "FLOAT32", 1: "FLOAT16", 2: "INT32", 3: "UINT8",
    4: "INT64", 6: "BOOL", 7: "INT16", 9: "INT8", 10: "FLOAT64",
    15: "UINT32", 16: "UINT16", 17: "INT4", 18: "BFLOAT16", 19: "INT2", 20: "UINT4",
}


class FB:
    def __init__(self, data):
        self.data = data
        self.size = len(data)

    def unpack(self, fmt, pos):
        size = struct.calcsize(fmt)
        if pos < 0 or pos + size > self.size:
            raise ValueError(f"FlatBuffer offset outside model: {pos} + {size}")
        return struct.unpack_from("<" + fmt, self.data, pos)[0]

    def u8(self, p):
        return self.unpack("B", p)

    def i8(self, p):
        return self.unpack("b", p)

    def u16(self, p):
        return self.unpack("H", p)

    def u32(self, p):
        return self.unpack("I", p)

    def i32(self, p):
        return self.unpack("i", p)

    def table_field(self, table, index):
        vtable = table - self.i32(table)
        vsize = self.u16(vtable)
        pos = vtable + 4 + 2 * index
        if pos + 2 > vtable + vsize:
            return None
        offset = self.u16(pos)
        return table + offset if offset else None

    def indirect(self, p):
        return p + self.u32(p)

    def scalar(self, table, field_index, kind, default=0):
        p = self.table_field(table, field_index)
        return self.unpack(kind, p) if p is not None else default

    def vector(self, table, field_index):
        field = self.table_field(table, field_index)
        if field is None:
            return None, 0
        p = self.indirect(field)
        length = self.u32(p)
        return p + 4, length

    def table_vector(self, table, field_index, limit=100000):
        pos, count = self.vector(table, field_index)
        if count > limit:
            raise ValueError(f"Unreasonable FlatBuffer table count: {count}")
        return [self.indirect(pos + 4 * i) for i in range(count)] if pos is not None else []

    def int_vector(self, table, field_index):
        pos, count = self.vector(table, field_index)
        if count > 1000000:
            raise ValueError(f"Unreasonable integer vector: {count}")
        return [self.i32(pos + 4 * i) for i in range(count)] if pos is not None else []

    def string(self, table, field_index):
        field = self.table_field(table, field_index)
        if field is None:
            return ""
        p = self.indirect(field)
        length = self.u32(p)
        if length > 4096 or p + 4 + length > self.size:
            raise ValueError("Invalid or oversized FlatBuffer string")
        return self.data[p + 4:p + 4 + length].decode("utf-8", "replace")


def inspect(path):
    if not path.is_file():
        raise ValueError(f"CPU_MODEL_NOT_FOUND: {path}")
    with path.open("rb") as stream:
        with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
            if len(data) < 12 or data[4:8] != b"TFL3":
                raise ValueError("Not a TFL3 TensorFlow Lite FlatBuffer")
            fb = FB(data)
            model = fb.u32(0)
            print(f"TFLITE_MODEL: {path} bytes={len(data)} schema_version={fb.scalar(model,0,'I')}")
            raw_codes = fb.table_vector(model, 1)
            codes = []
            for n, code in enumerate(raw_codes):
                deprecated = fb.scalar(code, 0, "b", 0)
                builtin = fb.scalar(code, 3, "i", 0)
                number = max(deprecated, builtin)
                version = fb.scalar(code, 2, "i", 1)
                if version < 1 or version > 9999:
                    raise ValueError(f"Implausible builtin op version: {version}")
                label = NAMES.get(number, f"BUILTIN_{number}")
                if number == 32:
                    label += ":" + fb.string(code, 1)
                codes.append((number, version, label))
                print(f"TFLITE_OPCODE: index={n} builtin={number} name={label} version={version}")

            graphs = fb.table_vector(model, 2)
            for g, graph in enumerate(graphs):
                operators = fb.table_vector(graph, 3, limit=200000)
                counts = collections.Counter()
                for op in operators:
                    index = fb.scalar(op, 0, "I", 0)
                    if index >= len(codes):
                        raise ValueError(f"Invalid opcode_index {index} in subgraph {g}")
                    counts[index] += 1
                print(f"TFLITE_SUBGRAPH: index={g} name={fb.string(graph,4)!r} ops={len(operators)}")
                for index, count in sorted(counts.items()):
                    number, version, name = codes[index]
                    print(f"TFLITE_GRAPH_OP: graph={g} name={name} version={version} count={count}")

                tensors = fb.table_vector(graph, 0, limit=200000)
                buffers = fb.table_vector(model, 4, limit=200000)
                for op_number, op in enumerate(operators):
                    opcode = fb.scalar(op, 0, "I", 0)
                    if codes[opcode][0] != 7:
                        continue
                    inputs = fb.int_vector(op, 1)
                    for slot, tid in enumerate(inputs):
                        if tid < 0 or tid >= len(tensors):
                            raise ValueError("Invalid embedding op tensor index")
                        tensor = tensors[tid]
                        dtype = fb.scalar(tensor, 1, "b", 0)
                        shape = fb.int_vector(tensor, 0)
                        buffer_index = fb.scalar(tensor, 2, "I", 0)
                        if buffer_index >= len(buffers):
                            raise ValueError("Invalid embedding tensor buffer index")
                        buf = buffers[buffer_index]
                        _, buffer_bytes = fb.vector(buf, 0)
                        quant_field = fb.table_field(tensor, 4)
                        scales = 0
                        if quant_field is not None:
                            _, scales = fb.vector(fb.indirect(quant_field), 2)
                        print(
                            f"TFLITE_EMBED_INPUT: graph={g} op={op_number} "
                            f"slot={slot} name={fb.string(tensor,3)!r} "
                            f"type={TYPES.get(dtype, f'TYPE_{dtype}')} "
                            f"shape={shape} buffer_bytes={buffer_bytes} "
                            f"quant_scales={scales}"
                        )
                for direction, field_index in (("input", 1), ("output", 2)):
                    for idx in fb.int_vector(graph, field_index):
                        if idx < 0 or idx >= len(tensors):
                            raise ValueError(f"Invalid {direction} tensor index {idx} in subgraph {g}")
                        tensor = tensors[idx]
                        dtype = fb.scalar(tensor, 1, "b", 0)
                        dims = fb.int_vector(tensor, 0)
                        print(f"TFLITE_IO: graph={g} direction={direction} "
                              f"tensor={fb.string(tensor,3)!r} "
                              f"type={TYPES.get(dtype, f'TYPE_{dtype}')} shape={dims}")

            for signature in fb.table_vector(model, 7):
                key = fb.string(signature, 2)
                graph = fb.scalar(signature, 4, "I", 0)
                input_names = [fb.string(t, 0) for t in fb.table_vector(signature, 0)]
                output_names = [fb.string(t, 0) for t in fb.table_vector(signature, 1)]
                print(f"TFLITE_SIGNATURE: key={key} subgraph={graph} "
                      f"inputs={input_names} outputs={output_names}")
            print(f"TFLITE_INSPECT_OK: operator_codes={len(codes)} subgraphs={len(graphs)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("models", nargs="+", type=pathlib.Path)
    args = parser.parse_args()
    for path in args.models:
        inspect(path)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError) as exc:
        print(f"TFLITE_INSPECT_ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
