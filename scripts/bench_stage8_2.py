#!/usr/bin/env python3
"""Stage 8.2: controlled Q8B RAW versus RPCmem MEMHANDLE KV experiment.

First prove exact graph-output checksums RAW == MEMHANDLE on an identical
short synthetic Decode stream. Only then run 256-step A/B without QNN
profiling, and optionally profile sampled steps in a separate process.

NO model/AOT/RAW/reference modifications. Not a 256/256/1024 test:
Prefill-128 is retained and Decode KV capacity remains 4096.
"""
import argparse
import json
import math
import re
import statistics
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

from bench_stage8_1 import (
    DEFAULT_WORK, ROOT, BenchError, check_models, runner_ready,
    GOLDEN_TEXT, GOLDEN_IDS, GOLDEN_ANSWER, run_once, run_fixed
)

CHECKSUM_RE = re.compile(
    r"^STAGE82_CHECKSUM: step=(\d+) decode_fnv=(\d+)"
    r" head_fnv=(\d+) comparable_raw_memhandle=1$"
)


def get_digests(log, steps):
    records = {}
    for line in log.splitlines():
        if not line.startswith("STAGE82_CHECKSUM:"):
            continue
        match = CHECKSUM_RE.fullmatch(line)
        if match is None:
            raise BenchError("Malformed Stage 8.2 graph checksum: " + line[:160])
        step, decode, head = map(int, match.groups())
        if step in records:
            raise BenchError("Duplicate Stage 8.2 graph checksum")
        records[step] = (decode, head)
    if list(records) != list(range(1, steps + 1)):
        raise BenchError("Missing or reordered Stage 8.2 graph checksums")
    return records


def assert_registration(log, steps, enabled):
    if enabled:
        for marker in (
            "STAGE82_MEM_REGISTERED: tensors=30",
            "STAGE82_KV_MIRROR_OK: stage=initial full_cache_equal=1 tensors=30",
            "STAGE82_KV_MIRROR_OK: stage=final full_cache_equal=1 tensors=30",
            "kv_binding=memhandle",
            "STAGE82_MEM_REGISTERED:",
        ):
            if marker not in log:
                raise BenchError("MEMHANDLE run missing contract: " + marker)
    elif "STAGE82_MEM_REGISTERED:" in log or "kv_binding=raw" not in log:
        raise BenchError("RAW run has wrong KV binding markers")
    if f"BENCH81_OK: executed_steps={steps}" not in log:
        raise BenchError("Incomplete Stage 8.1 trace")


def compare_checksums(raw_log, mem_log, steps):
    raw = get_digests(raw_log, steps)
    mem = get_digests(mem_log, steps)
    for step in range(1, steps + 1):
        if raw[step] != mem[step]:
            raise BenchError(
                f"RAW/MEMHANDLE graph outputs differ at step={step}: "
                f"RAW Decode/Head={raw[step]}, MEMHANDLE={mem[step]}"
            )
    return len(raw)


def check_profile(log, steps, level):
    if "STAGE82_PROFILE_EVENT:" not in log:
        raise BenchError("QNN backend returned no reported profile events")
    calls = [
        line for line in log.splitlines()
        if line.startswith("STAGE82_PROFILE_CALL:")
    ]
    events = [
        line for line in log.splitlines()
        if line.startswith("STAGE82_PROFILE_EVENT:")
    ]
    sample = [i for i in range(1, steps + 1) if
              i in {1, 2, 16, 64, 128, steps}]
    expected = len(sample) * 2
    if len(calls) != expected:
        raise BenchError(f"Expected {expected} QNN profile calls, found {len(calls)}")
    for step in sample:
        for name in ("decode", "head"):
            if not any(f"graph={name} step={step} " in line for line in calls):
                raise BenchError(f"No profile for {name} at step {step}")
    return {
        "level": level,
        "sampled_steps": sample,
        "graph_calls": len(calls),
        "event_records": len(events),
        "profile_event_types": sorted({
            int(m.group(1)) for line in events if (
                m := re.search(r"\btype=(\d+)\b", line))
        }),
    }


def median(runs, key):
    return statistics.median(row[key] for row in runs)


def present(raw_runs, mem_runs, profile, folder, preflight_verified):
    print("\nSTAGE82_RESULT: RAW vs RPCmem MEMHANDLE (same AOT, KV4096)")
    for label, results in (("RAW", raw_runs), ("MEMHANDLE", mem_runs)):
        for index, run in enumerate(results, 1):
            print(
                f"STAGE82_RUN: backend={label} run={index} "
                f"steps={run['steps']} "
                f"decode_steps_s={run['decode_steps_per_s']:.3f} "
                f"npu_decode_ms_per_step={run['decode_npu_ms_per_step']:.3f} "
                f"head_ms_per_step={run['lm_head_graph_npu_ms_per_step']:.3f} "
                f"kv_feedback_ms_per_step={run['kv_feedback_ms_per_step']:.3f}"
            )
    if raw_runs and mem_runs:
        baseline=median(raw_runs,"decode_steps_per_s")
        candidate=median(mem_runs,"decode_steps_per_s")
        print(
            f"STAGE82_AB_MEDIAN: RAW={baseline:.3f} steps/s"
            f" MEMHANDLE={candidate:.3f} steps/s"
            f" ratio_mem_over_raw={candidate/baseline:.3f}"
            f" raw_decode_ms={median(raw_runs,'decode_npu_ms_per_step'):.3f}"
            f" mem_decode_ms={median(mem_runs,'decode_npu_ms_per_step'):.3f}"
        )
        print("STAGE82_AB_NOTE: includes CPU mirroring 9216-byte KV slices;"
              " no profiler in A/B runs; repeat order RAW/MEMHANDLE for"
              " each trial; clock/temperature not controlled")
    if profile:
        print(f"STAGE82_PROFILE_RESULT: level={profile['level']} "
              f"calls={profile['graph_calls']} "
              f"events={profile['event_records']} "
              f"types={profile['profile_event_types']}")
    print(f"STAGE82_REPORT: {folder / 'summary.json'}")
    if preflight_verified:
        print("STAGE82_OK: output_checksums_identical=1 same_qnn_aot=1 "
              "prefill128=1 kv_capacity4096=1 "
              "matched_litert_256_256_1024=0")
    elif profile:
        print("STAGE82_PROFILE_OK: raw_qnn_profile=1"
              " memhandle_not_required=1"
              " matched_litert_256_256_1024=0")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("probe", "ab", "profile", "all"),
                   default="probe",
                   help="probe checks correctness only (recommended first); "
                        "ab adds throughput; profile collects API events; "
                        "all includes all stages")
    p.add_argument("--text", default="Что ты умеешь?",
                   help="Prompt fitting Prefill-128 (<128 actual tokens)")
    p.add_argument("--probe-steps", type=int, default=8,
                   help="Exact RAW/MEMHANDLE comparison steps, 1..32")
    p.add_argument("--steps", type=int, default=256,
                   help="Exact Decode steps per long run, 1..256")
    p.add_argument("--runs", type=int, default=3,
                   help="A/B run pairs, 1..10")
    p.add_argument("--profile-level", choices=("basic", "detailed"),
                   default="basic")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--runner", type=Path,
                   default=ROOT / "build/gemma4-qnn-run")
    args = p.parse_args()
    if not 1 <= args.probe_steps <= 32:
        p.error("--probe-steps must be 1..32")
    if not 1 <= args.steps <= 256:
        p.error("--steps must be 1..256")
    if not 1 <= args.runs <= 10:
        p.error("--runs must be 1..10")
    work = args.workdir.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not work.is_dir():
        raise BenchError("Missing work directory: " + str(work))
    check_models(work)
    runner_ready(runner)
    help_result = subprocess.run([str(runner), "--help"],
                                 capture_output=True, text=True, check=False)
    if (help_result.returncode != 0 or
            "--dynamic-kv-memhandle" not in help_result.stdout or
            "--dynamic-qnn-profile" not in help_result.stdout or
            "--dynamic-bench-checksums" not in help_result.stdout):
        raise BenchError("Stage 8.2 runner flags missing: rebuild the QNN binary")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(tempfile.mkdtemp(
        prefix="bench_stage8_2_" + stamp + "_", dir=work
    ))
    print(f"STAGE82_BEGIN: mode={args.mode} directory={folder}", flush=True)
    print("STAGE82_GOLDEN_RUNNING: unchanged RAW QNN model regression", flush=True)
    golden = run_once(work, runner, folder, "golden", GOLDEN_TEXT, 16, "stop")
    if (golden["decoded_ids"] != GOLDEN_IDS or
            golden["decoded_text"] != GOLDEN_ANSWER or
            golden["stop_reason"] != "eos"):
        raise BenchError("Original 2+2 QNN regression failed; no benchmarks")
    print("STAGE82_GOLDEN_OK: exact nine token IDs EOS106", flush=True)
    raw_probe=None
    mem_probe=None
    count=0
    if args.mode != "profile":
        print("STAGE82_PREFLIGHT: comparing ALL Decode and Head output checksums",
              flush=True)
        raw_probe=run_fixed(work,runner,folder,"probe_raw",args.text,
                            args.probe_steps,("--dynamic-bench-checksums",))
        mem_probe=run_fixed(work,runner,folder,"probe_memhandle",args.text,
                            args.probe_steps,("--dynamic-bench-checksums",
                                              "--dynamic-kv-memhandle"))
        raw_log=Path(raw_probe["log"]).read_text(errors="replace")
        mem_log=Path(mem_probe["log"]).read_text(errors="replace")
        assert_registration(raw_log,args.probe_steps,False)
        assert_registration(mem_log,args.probe_steps,True)
        count=compare_checksums(raw_log,mem_log,args.probe_steps)
        print(f"STAGE82_PREFLIGHT_OK: raw_memhandle_equal_steps={count} "
              "all_decode_outputs_equal=1 all_lm_head_logits_equal=1 "
              "all_30_KV_cache_bytes_equal_at_finish=1", flush=True)
    else:
        print("STAGE82_PROFILE_ONLY: RAW profiling independent of rpcmem",flush=True)
    raw_runs=[]
    mem_runs=[]
    if args.mode in ("ab","all"):
        for i in range(1,args.runs+1):
            print(f"STAGE82_AB: pair={i}/{args.runs} mode=RAW",flush=True)
            raw=run_fixed(work,runner,folder,f"ab_raw_{i:02d}",
                          args.text,args.steps)
            assert_registration(Path(raw["log"]).read_text(errors="replace"),
                                args.steps,False)
            raw_runs.append(raw)
            print(f"STAGE82_AB: pair={i}/{args.runs} mode=MEMHANDLE",flush=True)
            mem=run_fixed(work,runner,folder,f"ab_mem_{i:02d}",
                          args.text,args.steps,("--dynamic-kv-memhandle",))
            assert_registration(Path(mem["log"]).read_text(errors="replace"),
                                args.steps,True)
            mem_runs.append(mem)
    profile=None
    profile_run=None
    if args.mode in ("profile","all"):
        print(f"STAGE82_PROFILING: raw_mode=1 level={args.profile_level} "
              f"sampled_steps=1,2,16,64,128,{args.steps}",flush=True)
        profile_run=run_fixed(
            work,runner,folder,"profile_raw",args.text,args.steps,
            ("--dynamic-qnn-profile",args.profile_level)
        )
        profile=check_profile(
            Path(profile_run["log"]).read_text(errors="replace"),
            args.steps,args.profile_level
        )
    report={
        "schema":"gemma4-stage8-2-ab-rpcmem-v1",
        "platform":"Radxa Q8B",
        "method":"independent RAW/MEMHANDLE QNN processes, same Prefill-128 AOT and KV4096",
        "mode":args.mode,
        "profile_level":args.profile_level,
        "prompt":args.text,
        "capacity":4096,
        "prefill_graph_tokens":128,
        "long_steps":args.steps,
        "runs_per_binding":len(raw_runs),
        "not_matched_litert_256_256_1024":True,
        "both_io_modes_exact_same_decode_head_outputs_in_probe":count>0,
        "golden":golden,
        "probe":{"steps":count,"raw":raw_probe,"memhandle":mem_probe},
        "raw_runs":raw_runs,
        "memhandle_runs":mem_runs,
        "profile":profile,
        "profile_run":profile_run,
        "median":(
            {mode:{key:median(values,key)
                   for key in ("decode_steps_per_s",
                               "decode_npu_ms_per_step",
                               "lm_head_graph_npu_ms_per_step",
                               "kv_feedback_ms_per_step")}
             for mode,values in (("RAW",raw_runs),("MEMHANDLE",mem_runs))}
            if raw_runs and mem_runs else None
        )
    }
    with (folder/"summary.json").open("x",encoding="utf-8") as output:
        json.dump(report,output,ensure_ascii=False,indent=2)
        output.write("\n")
    present(raw_runs,mem_runs,profile,folder,count>0)


if __name__=="__main__":
    try:
        main()
    except (BenchError,OSError,ValueError,RuntimeError) as e:
        print("STAGE82_ERROR:",e,file=sys.stderr)
        raise SystemExit(1)
