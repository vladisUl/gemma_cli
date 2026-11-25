#!/usr/bin/env python3
"""Stage 8.1: exact N-step synthetic QNN Decode/KV benchmark on Q8B.

Every measured iteration feeds one input token, runs CPU non-KV + NPU
Decode + LM Head, and writes all 30 output KV slices into the next
position of the same RAM cache. The first input is the model-header LF
107, after which input token 236778 is repeatedly supplied to guarantee
a fixed duration even when the model predicts EOS or tied INT8 logits.

THIS IS NOT TEXT GENERATION AND NOT A MATCHED 256/256/1024 LiteRT TEST.
The existing fixed Prefill-128 AOT graph is retained; its actual
prompt length is less than 128 and Decode cache capacity remains 4096.
"""
import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from bench_stage8 import (
    ROOT, DEFAULT_WORK, GOLDEN_TEXT, GOLDEN_IDS, GOLDEN_ANSWER,
    BenchError, check_models, runner_ready, one, number, rate, run_once
)

FIXED_TOKEN = 236778
FIRST_TOKEN = 107
KV_SLICES = 30
KV_FEEDBACK_BYTES = 9216


def parse_fixed_log(log, requested_steps):
    lines = log.splitlines()
    mode = one(lines, "BENCH81_MODE:")
    result = one(lines, "BENCH81_OK:")
    prefill = one(lines, "BENCH8_PREFILL:")
    metadata = one(lines, "BENCH8_METADATA:")
    steps = [s for s in lines if s.startswith("BENCH81_STEP:")]
    heads = [s for s in lines if s.startswith("CHAIN_DECODE_HEAD_OK:")]
    if len(steps) != requested_steps or len(heads) != requested_steps:
        raise BenchError(
            f"Expected exactly {requested_steps} Step/Head lines; "
            f"got {len(steps)} Step and {len(heads)} Head"
        )
    for name in ("synthetic_fixed_input=1", "no_eos_stop=1",
                 "no_argmax_tie_stop=1", "kv_feedback_each_step=1",
                 "excludes_context_load_from_decode_rate=1"):
        if name not in mode:
            raise BenchError("Missing synthetic-mode contract " + name)
    if (int(number(mode, "requested_steps")) != requested_steps or
            int(number(mode, "decode_first_token")) != FIRST_TOKEN or
            int(number(mode, "decode_replay_token")) != FIXED_TOKEN or
            int(number(mode, "context_capacity")) != 4096):
        raise BenchError("Wrong fixed-input/capacity/step contract")
    actual_prefill = int(number(prefill, "real_tokens"))
    if (int(number(prefill, "graph_tokens")) != 128 or
            int(number(mode, "prefill_tokens")) != actual_prefill or
            "warmed_up=0" not in prefill):
        raise BenchError("Unexpected Prefill shape/warm-up")
    if not 1 <= actual_prefill < 128:
        raise BenchError("Invalid real Prefill tokens")
    if actual_prefill + requested_steps > 4096:
        raise BenchError("KV capacity exceeded")
    for key, expected in (
        ("executed_steps", requested_steps),
        ("kv_updates", requested_steps),
        ("kv_tensors_per_step", KV_SLICES),
        ("last_written_pos", actual_prefill + requested_steps - 1),
        ("fixed_token", FIXED_TOKEN),
        ("prefill_tokens", actual_prefill),
    ):
        if int(number(result, key)) != expected:
            raise BenchError(f"BENCH81_OK: {key} mismatch")
    if "synthetic_not_text_generation=1" not in result:
        raise BenchError("Synthetic workload not declared")

    measures = {
        "cpu_nonkv_ms": 0.0,
        "npu_decode_ms": 0.0,
        "head_call_ms": 0.0,
        "kv_feedback_ms": 0.0,
        "step_wall_ms": 0.0,
        "npu_head_execute_ms": 0.0,
    }
    first_step = None
    first_four = []
    last_four = []
    for i, (step, head) in enumerate(zip(steps, heads), 1):
        if int(number(step, "step")) != i:
            raise BenchError("Nonconsecutive fixed Decode step")
        if int(number(step, "pos")) != actual_prefill + i - 1:
            raise BenchError("Unexpected KV position / missed step")
        if int(number(step, "input_token")) != (
            FIRST_TOKEN if i == 1 else FIXED_TOKEN
        ):
            raise BenchError("Synthetic token stream changed")
        if (int(number(step, "kv_updated")) != 1 or
                int(number(step, "kv_tensors")) != KV_SLICES or
                int(number(step, "kv_slice_bytes")) != KV_FEEDBACK_BYTES):
            raise BenchError("Missing KV feedback in one Decode iteration")
        if "no_sample_logits_used=1" not in step:
            raise BenchError("Sample / saved input use not excluded")
        for key in measures:
            if key == "npu_head_execute_ms":
                value = number(head, "head_execute_ms")
            else:
                value = number(step, key)
            measures[key] += value
        step_wall = number(step, "step_wall_ms")
        components = sum(number(step, name) for name in (
            "cpu_nonkv_ms", "npu_decode_ms", "head_call_ms", "kv_feedback_ms"
        ))
        if abs(components - step_wall) > max(0.2, .02 * step_wall):
            raise BenchError("Step wall differs from component time sum")
        if number(head, "head_execute_ms") > number(step, "head_call_ms") + 0.2:
            raise BenchError("Head graphExecute exceeds full head call")
        if first_step is None:
            first_step = step_wall
        if i <= 4:
            first_four.append(step_wall)
        if i > requested_steps - 4:
            last_four.append(step_wall)
    wall_ms = measures["step_wall_ms"]
    npu_ms = measures["npu_decode_ms"] + measures["npu_head_execute_ms"]
    after_first = wall_ms - first_step
    if wall_ms <= 0 or npu_ms <= 0:
        raise BenchError("Zero duration cannot establish throughput")
    summary_ms = number(result, "measured_steps_total_ms")
    if abs(summary_ms - wall_ms) > max(1.0, 0.001 * wall_ms):
        raise BenchError("Sum of measured steps disagrees with C++ summary")
    summary_rate = number(result, "decode_steps_per_s")
    if abs(summary_rate - rate(requested_steps, wall_ms)) > max(0.05, .005 * summary_rate):
        raise BenchError("C++ vs Python throughput mismatch")
    return {
        "steps": requested_steps,
        "prefill_real_tokens": actual_prefill,
        "prefill_graph_tokens": 128,
        "kv_capacity": 4096,
        "kv_updates": requested_steps,
        "kv_tensors_per_step": KV_SLICES,
        "kv_slice_bytes_per_step": KV_FEEDBACK_BYTES,
        "kv_total_feedback_bytes": KV_FEEDBACK_BYTES * requested_steps,
        "kv_last_written_position": actual_prefill + requested_steps - 1,
        "fixed_token_id": FIXED_TOKEN,
        "first_decode_input_id": FIRST_TOKEN,
        "metadata_ms": number(metadata, "ms"),
        "cpu_prefill_ms": number(prefill, "cpu_inputs_ms"),
        "context_load_ms": number(prefill, "runtime_contexts_ms"),
        "npu_prefill_ms": number(prefill, "npu_execute_ms"),
        "kv_bridge_ms": number(prefill, "kv_bridge_ms"),
        "cpu_nonkv_ms": measures["cpu_nonkv_ms"],
        "decode_npu_ms": measures["npu_decode_ms"],
        "lm_head_graph_npu_ms": measures["npu_head_execute_ms"],
        "lm_head_call_ms": measures["head_call_ms"],
        "kv_feedback_ms": measures["kv_feedback_ms"],
        "decode_steps_wall_ms": wall_ms,
        "first_step_ms": first_step,
        "cpu_nonkv_ms_per_step": measures["cpu_nonkv_ms"] / requested_steps,
        "decode_npu_ms_per_step": measures["npu_decode_ms"] / requested_steps,
        "lm_head_graph_npu_ms_per_step": measures["npu_head_execute_ms"] / requested_steps,
        "kv_feedback_ms_per_step": measures["kv_feedback_ms"] / requested_steps,
        "decode_steps_per_s": rate(requested_steps, wall_ms),
        "npu_graphs_only_steps_per_s": rate(requested_steps, npu_ms),
        "steps_after_first_per_s": rate(
            requested_steps - 1, after_first) if requested_steps > 1 else None,
        "first_four_steps_ms": first_four,
        "last_four_steps_ms": last_four,
        "not_text_generation": True,
        "no_extra_warmup": True,
        "separate_context_per_run": True,
    }


def run_fixed(work, runner, target, label, prompt, steps):
    raw = target / (label + ".raw")
    logfile = target / (label + ".log")
    start = time.perf_counter()
    encode = subprocess.run(
        [sys.executable, str(ROOT / "scripts/gemma4_bpe.py"),
         "--tokenizer", str(work / "tokenizer.json"),
         "--golden-raw", str(work / "derived_prefill_128_token_ids_i32.raw"),
         "--text", prompt, "--output-new-raw", str(raw)],
        cwd=work, capture_output=True, text=True, check=False
    )
    after_bpe = time.perf_counter()
    if encode.returncode:
        raise BenchError("BPE error: " + (encode.stderr or encode.stdout)[-1000:])
    if raw.stat().st_size != 512:
        raise BenchError("BPE RAW file is not 512 bytes")
    sdk = str(work / "sdk247")
    env = os.environ.copy()
    env["LD_LIBRARY_PATH"] = sdk
    env["ADSP_LIBRARY_PATH"] = sdk
    command = [
        str(runner),
        "--system", str(work / "sdk247/libQnnSystem.so"),
        "--backend", str(work / "sdk247/libQnnHtp.so"),
        "--jobs", "persistent_jobs_decode_prefill_head.txt",
        "--dynamic-prefill", "--dynamic-benchmark",
        "--dynamic-bench-fixed-steps", str(steps),
        "--dynamic-prompt-tokens", str(raw),
        "--dynamic-section0", str(
            work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"),
        "--dynamic-section1", str(
            work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"),
        "--dynamic-section2", str(
            work / "cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"),
    ]
    with logfile.open("wb") as file:
        status = subprocess.run(command, cwd=work, env=env,
                                stdout=file, stderr=subprocess.STDOUT,
                                check=False)
    finish = time.perf_counter()
    if status.returncode:
        lines = logfile.read_text(errors="replace").splitlines()
        error = next((s for s in reversed(lines) if "QNN_RUN_ERROR:" in s), "")
        raise BenchError(
            f"QNN failed ({label}, exit {status.returncode}) {error} "
            f"diagnostics={logfile}"
        )
    metrics = parse_fixed_log(
        logfile.read_text(errors="replace"), steps)
    metrics.update({
        "label": label, "prompt": prompt,
        "bpe_wall_ms": 1000 * (after_bpe - start),
        "qnn_process_wall_ms": 1000 * (finish - after_bpe),
        "full_wall_ms": 1000 * (finish - start),
        "log": str(logfile)
    })
    return metrics


def calculate_medians(runs):
    exclude = {
        "steps", "prefill_real_tokens", "prefill_graph_tokens",
        "kv_capacity", "kv_updates", "kv_tensors_per_step",
        "kv_slice_bytes_per_step", "kv_total_feedback_bytes",
        "kv_last_written_position", "fixed_token_id",
        "first_decode_input_id", "not_text_generation",
        "no_extra_warmup", "separate_context_per_run",
        "first_four_steps_ms", "last_four_steps_ms",
        "label", "prompt", "log"
    }
    columns = [
        k for k, v in runs[0].items()
        if k not in exclude and isinstance(v, (int, float))
        and all(row.get(k) is not None for row in runs)
    ]
    return {k: statistics.median(row[k] for row in runs) for k in columns}


def present(runs, median, folder):
    print("\nSTAGE81_METHOD: synthetic fixed token 236778, not text generation;"
          " first LF107; KV feedback on EVERY step including final;"
          " fresh QNN process each run; no warmup")
    for run in runs:
        print(
            f"STAGE81_RUN: {run['label']} "
            f"steps={run['steps']} kv_updates={run['kv_updates']} "
            f"prefill_tokens={run['prefill_real_tokens']} "
            f"decode_steps_s={run['decode_steps_per_s']:.3f} "
            f"npu_graphs_steps_s={run['npu_graphs_only_steps_per_s']:.3f} "
            f"qnn_process_wall_s={run['qnn_process_wall_ms']/1000:.3f}"
        )
    print("\nМедианы (3 контекста загружаются заново в каждом процессе):")
    fields = [
        ("BPE", "bpe_wall_ms", "мс"),
        ("QNN metadata + I/O", "metadata_ms", "мс"),
        ("Context init + 3 AOT", "context_load_ms", "мс"),
        ("CPU Prefill", "cpu_prefill_ms", "мс"),
        ("NPU Prefill-128", "npu_prefill_ms", "мс"),
        ("KV bridge", "kv_bridge_ms", "мс"),
        ("CPU non-KV / шаг", "cpu_nonkv_ms_per_step", "мс"),
        ("NPU Decode / шаг", "decode_npu_ms_per_step", "мс"),
        ("NPU LM Head / шаг", "lm_head_graph_npu_ms_per_step", "мс"),
        ("KV feedback / шаг", "kv_feedback_ms_per_step", "мс"),
        ("Все Decode шаги", "decode_steps_wall_ms", "мс"),
        ("Первый Decode шаг", "first_step_ms", "мс"),
        ("Полная скорость Decode+Head+CPU+KV", "decode_steps_per_s", "шаг/с"),
        ("Только графы Decode+Head", "npu_graphs_only_steps_per_s", "шаг/с"),
        ("Шаги после первого", "steps_after_first_per_s", "шаг/с"),
        ("QNN процесс целиком", "qnn_process_wall_ms", "мс"),
    ]
    for title, key, unit in fields:
        if key in median:
            print(f"  {title}: {median[key]:.3f} {unit}")
    print(f"STAGE81_REPORT: {folder / 'summary.json'}")
    print("STAGE81_OK: fixed_length=1 all_kv_feedback_verified=1 "
          "no_reload_in_decode_rate=1 matched_litert_256_256_1024=0")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", default="Что ты умеешь?",
                        help="Fresh single-turn prompt: actual Prefill <128 tokens")
    parser.add_argument("--steps", type=int, default=256,
                        help="Exact N QNN Decode/Head + KV updates, 1..256")
    parser.add_argument("--runs", type=int, default=3,
                        help="Independent fresh-process trials, 1..10")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--runner", type=Path,
                        default=ROOT / "build/gemma4-qnn-run")
    args = parser.parse_args()
    if not 1 <= args.steps <= 256:
        parser.error("--steps must be 1..256")
    if not 1 <= args.runs <= 10:
        parser.error("--runs must be 1..10")
    work = args.workdir.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not work.is_dir():
        raise BenchError("Missing work directory " + str(work))
    check_models(work)
    runner_ready(runner)
    help_result = subprocess.run([str(runner), "--help"],
                                 capture_output=True, text=True, check=False)
    if help_result.returncode or \
            "--dynamic-bench-fixed-steps" not in help_result.stdout:
        raise BenchError("Runner lacks Stage 8.1; rebuild gemma4-qnn-run")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(tempfile.mkdtemp(
        prefix="bench_stage8_1_" + stamp + "_", dir=work
    ))
    print(f"STAGE81_BEGIN: {folder}", flush=True)
    print("STAGE81_GOLDEN_RUNNING: exact 9-token 2+2 inference first", flush=True)
    golden = run_once(work, runner, folder, "golden",
                      GOLDEN_TEXT, 16, "stop")
    if (golden["decoded_ids"] != GOLDEN_IDS or
            golden["decoded_text"] != GOLDEN_ANSWER or
            golden["stop_reason"] != "eos"):
        raise BenchError("GOLDEN_QNN_OUTPUT_CHANGED, abort performance trial")
    print("STAGE81_GOLDEN_OK: exact_9_ids=1 EOS106=1", flush=True)
    records = []
    for i in range(1, args.runs + 1):
        print(f"STAGE81_MEASURE: run={i}/{args.runs} steps={args.steps}",
              flush=True)
        record = run_fixed(
            work, runner, folder, f"run_{i:02d}", args.text, args.steps
        )
        records.append(record)
        print(
            f"STAGE81_MEASURE_OK: run={i} "
            f"steps={record['steps']} kv_updates={record['kv_updates']} "
            f"steps_s={record['decode_steps_per_s']:.3f}",
            flush=True
        )
    med = calculate_medians(records)
    report = {
        "schema": "gemma4-stage8-1-fixed-kv-v1",
        "platform": "Radxa Dragon Q8B",
        "prompt": args.text,
        "requested_steps": args.steps,
        "repetitions": args.runs,
        "fixed_replay_token": FIXED_TOKEN,
        "first_decode_token": FIRST_TOKEN,
        "cache_capacity": 4096,
        "prefill_graph_tokens": 128,
        "matched_litert_256_256_1024": False,
        "synthetic_not_text_generation": True,
        "all_kv_feedback_included": True,
        "separate_qnn_process_per_run": True,
        "no_extra_warmup": True,
        "normal_golden_before_measurement": True,
        "golden": golden,
        "runs": records,
        "median": med,
    }
    with (folder / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    present(records, med, folder)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print("STAGE81_ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
