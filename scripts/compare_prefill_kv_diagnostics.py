#!/usr/bin/env python3
"""Read-only comparison of Stage 6.10(pos16) and 6.11(pos0) QNN KV diagnostics.

No inference, no guessed source weights, no mutation of raw data or logs.
Only aggregate measurements from actual PREFILL_KV_DIAG_TENSOR records.
"""
import argparse
import pathlib
import sys


def parse_report(path, expected_pos):
    records = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("PREFILL_KV_DIAG_TENSOR: "):
            continue
        fields = dict(part.split("=", 1) for part in line.split()[1:]
                      if "=" in part)
        name = fields["mapped_output"]
        if name in records:
            raise ValueError(f"Duplicate diagnostic tensor {name} in {path}")
        for key in ("expected_row", "channels", "mismatched_bytes",
                    "abs_delta1", "abs_delta2", "abs_delta_gt2",
                    "sum_abs_delta", "max_abs_delta",
                    "best_same_tensor_row",
                    "best_same_tensor_row_mismatches",
                    "best_compatible_source_mismatches"):
            fields[key] = int(fields[key])
        if fields["expected_row"] != expected_pos:
            raise ValueError(f"Wrong row {fields['expected_row']} for {name}")
        if fields["kind"] not in ("K", "V"):
            raise ValueError(f"Unsupported KV type in {name}")
        if fields["mismatched_bytes"] != (fields["abs_delta1"] +
                                          fields["abs_delta2"] +
                                          fields["abs_delta_gt2"]):
            raise ValueError(f"Delta histogram inconsistent for {name}")
        records[name] = fields
    if len(records) != 30:
        raise ValueError(f"Expected 30 tensors in {path}, got {len(records)}")
    return records


def summarize(records, where):
    for kind in ("K", "V"):
        for channels in (256, 512):
            group = [r for r in records.values()
                     if r["kind"] == kind and r["channels"] == channels]
            if not group:
                raise ValueError(f"Missing {kind}-{channels} tensors at {where}")
            count = sum(x["channels"] for x in group)
            changed = sum(x["mismatched_bytes"] for x in group)
            abs_sum = sum(x["sum_abs_delta"] for x in group)
            print(f"PREFILL_KV_POSITION_GROUP: row={where} kind={kind} "
                  f"channels={channels} tensors={len(group)} "
                  f"bytes={count} different_bytes={changed} "
                  f"mismatch_percent={100*changed/count:.2f} "
                  f"avg_abs_int8_delta={abs_sum/count:.3f} "
                  f"max_abs_delta={max(x['max_abs_delta'] for x in group)}")
    total = sum(x["channels"] for x in records.values())
    changed = sum(x["mismatched_bytes"] for x in records.values())
    print(f"PREFILL_KV_POSITION_TOTAL: row={where} tensors=30 bytes={total} "
          f"different_bytes={changed} mismatch_percent={100*changed/total:.2f} "
          f"best_expected_row={sum(x['best_same_tensor_row']==where for x in records.values())} "
          f"best_expected_source={sum(x['best_compatible_source']==name for name,x in records.items())}")
    return changed


def main():
    p = argparse.ArgumentParser(description=__doc__)
    default = pathlib.Path.home() / "gemma4_qnn_test"
    p.add_argument("--first", type=pathlib.Path,
                   default=default / "cpu_prefill_first_row_stage6_11.log")
    p.add_argument("--last", type=pathlib.Path,
                   default=default / "cpu_prefill_kv_diagnostic_stage6_10.log")
    args = p.parse_args()
    first = parse_report(args.first.resolve(strict=True), 0)
    last = parse_report(args.last.resolve(strict=True), 16)
    if set(first) != set(last):
        raise ValueError("Stage 6.10 and 6.11 tensor names are not identical")
    for name in first:
        if (first[name]["kind"], first[name]["channels"]) != (
                last[name]["kind"], last[name]["channels"]):
            raise ValueError(f"KV kind/channels changed for {name}")
    a = summarize(first, 0)
    b = summarize(last, 16)
    print(f"PREFILL_KV_POSITION_COMPARISON_OK: mapped_tensors=30 "
          f"row0_mismatches={a} row16_mismatches={b} "
          f"delta_mismatches={b-a:+d} "
          f"zero_history_diagnostics_completed=1 "
          f"no_semantic_equivalence_claim=1")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, ZeroDivisionError) as err:
        print(f"PREFILL_KV_POSITION_COMPARISON_ERROR: {err}", file=sys.stderr)
        raise SystemExit(1)
