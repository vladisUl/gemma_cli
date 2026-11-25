#!/usr/bin/env python3
"""Diagnose a Section1 Prefill-128 mask mismatch without guessing a fix.

Read-only original .tflite / saved QNN references. Re-run only Section1's
prefill_mask_128 signature with a small controlled matrix of time_step and
padding-token representations. Do not modify true recovered token IDs, saved
QNN .raw or the verified Decode/Prefill embeddings.

Print byte-exact/diagnostic mismatch counts for BOTH mask outputs; even if a
candidate matches, no production input selection is changed automatically.
"""
import argparse
import collections
import pathlib
import struct
import subprocess
import sys
import tempfile

from test_cpu_prefill_aux import (
    SIGNATURE_OUTPUTS,
    TOKENS,
    get_input_payload,
    manifest_references,
    signature_info,
)

MASK = "prefill_mask_128"
ROW_BYTES = 4224
NUM_ROWS = 128


def compare(actual, reference):
    if len(actual) != len(reference) or len(actual) != ROW_BYTES * NUM_ROWS:
        raise ValueError(
            f"Unexpected comparison size CPU={len(actual)} reference={len(reference)}"
        )
    differences = [i for i, (a, b) in enumerate(zip(actual, reference)) if a != b]
    if not differences:
        return {
            "matches": True,
            "diff_bytes": 0,
            "first": -1,
            "row": -1,
            "column": -1,
            "prefix_diff": 0,
            "cache_diff": 0,
            "window_diff": 0,
            "bad_rows": 0,
            "actual": -1,
            "expected": -1,
        }
    first = differences[0]
    return {
        "matches": False,
        "diff_bytes": len(differences),
        "first": first,
        "row": first // ROW_BYTES,
        "column": first % ROW_BYTES,
        "prefix_diff": sum(1 for n in differences if n // ROW_BYTES < 17),
        "cache_diff": sum(1 for n in differences if n % ROW_BYTES < 4096),
        "window_diff": sum(1 for n in differences if n % ROW_BYTES >= 4096),
        "bad_rows": len({n // ROW_BYTES for n in differences}),
        "actual": actual[first],
        "expected": reference[first],
    }


def payload_candidates(base_tokens):
    nonzero_end = next(
        (i for i, token in enumerate(base_tokens) if token == 0), TOKENS
    )
    if any(token != 0 for token in base_tokens[nonzero_end:]):
        print(
            "PREFILL_MASK_DIAGNOSTIC_NOTE: zero IDs also occur inside non-padding "
            "span; suffix-only replacement is not a full validity mask",
            flush=True,
        )
    print(
        f"PREFILL_MASK_PADDING_BOUNDARY: first_zero={nonzero_end} "
        f"zero_suffix={TOKENS - nonzero_end}", flush=True
    )
    yield "original", base_tokens
    if nonzero_end < TOKENS:
        # Controlled probes only. They are not claimed to be original IDs.
        # All nonzero candidates may yield the same mask if the signature
        # interprets input_tokens as padding flags rather than embedding IDs.
        yield "padding_to_2", (
            base_tokens[:nonzero_end] + [2] * (TOKENS - nonzero_end)
        )
        yield "padding_to_1", (
            base_tokens[:nonzero_end] + [1] * (TOKENS - nonzero_end)
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=pathlib.Path,
                        default=pathlib.Path.home() / "gemma4_qnn_test")
    parser.add_argument("--exe", type=pathlib.Path,
                        default=pathlib.Path.home() /
                        "gemma4-cli-stage1/build/gemma4-cpu-run")
    parser.add_argument("--model", type=pathlib.Path, default=None)
    parser.add_argument("--tokens", type=pathlib.Path, default=None)
    parser.add_argument("--list", default="prefill_real_input_list.txt")
    parser.add_argument("--time-steps", type=int, nargs="+",
                        default=[0, 1, 17, 128],
                        help="Candidate time_step probe values; not auto-accepted")
    args = parser.parse_args()
    work = args.work.resolve(strict=True)
    exe = args.exe.resolve(strict=True)
    model = (
        args.model or work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"
    ).resolve(strict=True)
    token_path = (
        args.tokens or work / "derived_prefill_128_token_ids_i32.raw"
    ).resolve(strict=True)
    data = token_path.read_bytes()
    if len(data) != 512:
        raise ValueError(f"Expected 512-byte verified token file: {token_path}")
    tokens = list(struct.unpack("<128i", data))
    if any(t < 0 or t >= 262144 for t in tokens):
        raise ValueError("Invalid recovered token ID")
    if not args.time_steps or any(t < 0 or t >= 4096 for t in args.time_steps):
        raise ValueError("time_steps must be 0..4095")
    meta = signature_info(model)[MASK]
    output_info = meta["output"]
    if set(output_info) != set(SIGNATURE_OUTPUTS[MASK]):
        raise ValueError(f"Unexpected mask output aliases: {list(output_info)}")
    references = manifest_references(work, args.list)
    original = {
        alias: references[(MASK, alias)][0].read_bytes()
        for alias in SIGNATURE_OUTPUTS[MASK]
    }
    print(
        "PREFILL_MASK_DIAGNOSTIC_BEGIN: no QNN execution; no files modified; "
        "compare_both_outputs=1", flush=True
    )
    ranking = []
    with tempfile.TemporaryDirectory(prefix="prefill_mask_diagnostic_", dir=work) as temp:
        tempdir = pathlib.Path(temp)
        for padding_name, candidate_tokens in payload_candidates(tokens):
            for time_step in dict.fromkeys(args.time_steps):
                tag = f"{padding_name}_time{time_step}"
                inputs_dir = tempdir / tag
                inputs_dir.mkdir()
                command = [
                    str(exe), "--model", str(model),
                    "--signature", MASK,
                ]
                for alias, (dtype, shape, elements) in meta["input"].items():
                    payload = get_input_payload(
                        alias, dtype, shape, elements, candidate_tokens,
                        0, time_step,
                    )
                    path = inputs_dir / f"{alias}.raw"
                    path.write_bytes(payload)
                    command += ["--input", f"{alias}={path}"]
                output_dir = inputs_dir / "out"
                command += ["--output-dir", str(output_dir)]
                run = subprocess.run(
                    command, cwd=work, capture_output=True, text=True,
                    check=False,
                )
                if run.returncode:
                    print(
                        f"PREFILL_MASK_PROBE_INVOKE_FAILED: candidate={padding_name} "
                        f"time_step={time_step} exit_code={run.returncode} "
                        f"details={(run.stderr or run.stdout)[-700:]!r}",
                        flush=True,
                    )
                    continue
                total = 0
                both_exact = True
                for alias, (index, name, expected_size) in SIGNATURE_OUTPUTS[MASK].items():
                    dtype, shape, elements = output_info[alias]
                    if dtype != 9 or elements != expected_size:
                        raise ValueError(f"Unexpected metadata for {alias}: {shape}")
                    result = (output_dir / f"{alias}.raw").read_bytes()
                    data = compare(result, original[alias])
                    total += data["diff_bytes"]
                    both_exact &= data["matches"]
                    print(
                        f"PREFILL_MASK_PROBE: candidate={padding_name} "
                        f"time_step={time_step} output={alias} "
                        f"diff_bytes={data['diff_bytes']} "
                        f"first_byte={data['first']} "
                        f"first_row={data['row']} first_column={data['column']} "
                        f"actual={data['actual']} expected={data['expected']} "
                        f"prefix_rows_diff={data['prefix_diff']} "
                        f"cache_columns_diff={data['cache_diff']} "
                        f"current_window_diff={data['window_diff']} "
                        f"bad_rows={data['bad_rows']}",
                        flush=True,
                    )
                ranking.append((total, padding_name, time_step))
                if both_exact:
                    print(
                        f"PREFILL_MASK_PROBE_EXACT: candidate={padding_name} "
                        f"time_step={time_step} both_masks=1 "
                        f"bytes={2 * ROW_BYTES * NUM_ROWS}",
                        flush=True,
                    )
    for total, padding_name, time_step in sorted(ranking)[:5]:
        print(
            f"PREFILL_MASK_PROBE_RANK: candidate={padding_name} "
            f"time_step={time_step} total_mismatching_bytes={total}",
            flush=True,
        )
    if ranking and min(score for score, _, _ in ranking) == 0:
        print(
            "PREFILL_MASK_DIAGNOSTIC_EXACT_FOUND: both mask outputs matched "
            "for a tested candidate; still requires semantic confirmation",
            flush=True,
        )
    else:
        print(
            "PREFILL_MASK_DIAGNOSTIC_INCONCLUSIVE: no tested candidate "
            "matched both saved mask outputs byte-for-byte",
            flush=True,
        )
    print("PREFILL_MASK_DIAGNOSTIC_DONE: original input and references unchanged")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, OSError, subprocess.CalledProcessError, KeyError) as err:
        print(f"PREFILL_MASK_DIAGNOSTIC_ERROR: {err}", file=sys.stderr, flush=True)
        sys.exit(1)
