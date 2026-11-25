#!/usr/bin/env python3
"""Stage 8.0: repeatable cold-process QNN Prefill/Decode benchmark for Q8B.

One independent end-to-end golden check, then N independent COLD processes.
No artificial NPU warmup: "steps_after_first" is NOT a warm-start benchmark.
Uses the exact verified Stage 7.2 dynamic path; no model/RAW modifications.
"""
import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

from gemma4_cli import ROOT, DEFAULT_WORK, check_models, runner_ready

GOLDEN_TEXT = "Привет! Сколько будет 2+2?"
GOLDEN_IDS = [236778, 900, 236743, 236778, 16632,
              5213, 236812, 84750, 106]
GOLDEN_ANSWER = "2 + 2 будет **4**."
STOP_IDS = {1, 106, 50}


class BenchError(ValueError):
    pass


def one(lines, prefix):
    found = [x for x in lines if x.startswith(prefix)]
    if len(found) != 1:
        raise BenchError(f"{prefix} expected once, found {len(found)}")
    return found[0]


def number(line, key):
    match = re.search(
        r"(?:^|\s)" + re.escape(key) + r"=(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
        r"(?=\s|$)", line
    )
    if not match:
        raise BenchError(f"missing numeric {key} in {line[:200]}")
    value = float(match.group(1))
    if not math.isfinite(value) or value < 0:
        raise BenchError(f"bad metric {key}={value}")
    return value


def rate(count, ms):
    return 1000.0 * count / ms if ms > 0 else 0.0


def parse_benchmark(log, response):
    """Strict matching of N head graph executes and N benchmark step lines."""
    lines = log.splitlines()
    prefill = one(lines, "BENCH8_PREFILL:")
    first = one(lines, "BENCH8_FIRST_TOKEN:")
    metadata = one(lines, "BENCH8_METADATA:")
    step_lines = [x for x in lines if x.startswith("BENCH8_STEP:")]
    head_lines = [x for x in lines if x.startswith("CHAIN_DECODE_HEAD_OK:")]
    token_lines = [x for x in lines if x.startswith("DYNAMIC_GENERATED_TOKEN:")]
    n = len(token_lines)
    if n < 1 or len(step_lines) != n or len(head_lines) != n:
        raise BenchError(
            f"step/head/token counts disagree: {len(step_lines)}/{len(head_lines)}/{n}"
        )
    if response["steps"] != n:
        raise BenchError("decoded token count differs from benchmark steps")
    if number(prefill, "graph_tokens") != 128 or "warmed_up=0" not in prefill:
        raise BenchError("unexpected Prefill shape or hidden warmup")
    real_tokens = int(number(prefill, "real_tokens"))
    if not 1 <= real_tokens < 128:
        raise BenchError("invalid real Prefill token count")
    qnn_decode = 0.0
    qnn_head = 0.0
    wall_steps = 0.0
    cpu_steps = 0.0
    feedback = 0.0
    steps_after_first_ms = 0.0
    first_step_ms = None
    for index, (step, head, token) in enumerate(
            zip(step_lines, head_lines, token_lines), 1):
        if int(number(step, "step")) != index or \
                int(number(token, "step")) != index:
            raise BenchError("nonconsecutive measured generation step")
        if number(step, "step_wall_ms") < number(step, "npu_decode_ms"):
            raise BenchError("step shorter than NPU Decode alone")
        step_ms = number(step, "step_wall_ms")
        if index == 1:
            first_step_ms = step_ms
        else:
            steps_after_first_ms += step_ms
        wall_steps += step_ms
        cpu_steps += number(step, "cpu_nonkv_ms")
        feedback += number(step, "kv_feedback_ms")
        qnn_decode += number(step, "npu_decode_ms")
        qnn_head += number(head, "head_execute_ms")
    text_ids = [i for i in response["ids"] if i not in STOP_IDS]
    text_tokens = len(text_ids)
    if text_tokens < 1:
        raise BenchError("no text tokens to measure")
    return {
        "real_prefill_tokens": real_tokens,
        "graph_prefill_tokens": 128,
        "metadata_ms": number(metadata, "ms"),
        "cpu_prefill_ms": number(prefill, "cpu_inputs_ms"),
        "qnn_runtime_context_load_ms": number(prefill, "runtime_contexts_ms"),
        "npu_prefill_ms": number(prefill, "npu_execute_ms"),
        "kv_bridge_ms": number(prefill, "kv_bridge_ms"),
        "qnn_first_token_ms": number(first, "from_job_start_ms"),
        "npu_prefill_effective_tok_s": rate(
            real_tokens, number(prefill, "npu_execute_ms")),
        "npu_prefill_fixed_block_tok_s": rate(
            128, number(prefill, "npu_execute_ms")),
        "cpu_plus_npu_prefill_effective_tok_s": rate(
            real_tokens,
            number(prefill, "cpu_inputs_ms") +
            number(prefill, "npu_execute_ms")),
        "decode_steps": n,
        "text_tokens": text_tokens,
        "decode_steps_wall_ms": wall_steps,
        "decode_first_step_ms": first_step_ms,
        "decode_cpu_nonkv_ms": cpu_steps,
        "decode_npu_graph_ms": qnn_decode,
        "lm_head_npu_graph_ms": qnn_head,
        "kv_feedback_ms": feedback,
        "decode_steps_per_s": rate(n, wall_steps),
        "generation_text_tok_s": rate(text_tokens, wall_steps),
        "decode_steps_after_first_per_s": rate(n - 1, steps_after_first_ms),
        "npu_decode_and_head_steps_per_s": rate(
            n, qnn_decode + qnn_head),
        "stop_reason": response["stop_reason"],
        "tie_resolutions": response.get("tie_resolutions", 0),
        "warmed_up": False,
    }


def run_once(work, runner, base, label, prompt, max_tokens, tie_policy):
    """Runs BPE, QNN and decoding in distinct processes with monotonic clocks."""
    raw_path = base / f"{label}.raw"
    log_path = base / f"{label}.log"
    clock = time.perf_counter

    whole_start = clock()
    bpe_args = [
        sys.executable, str(ROOT / "scripts/gemma4_bpe.py"),
        "--tokenizer", str(work / "tokenizer.json"),
        "--golden-raw", str(work / "derived_prefill_128_token_ids_i32.raw"),
        "--text", prompt, "--output-new-raw", str(raw_path),
    ]
    bpe = subprocess.run(bpe_args, cwd=work, capture_output=True, text=True,
                         check=False)
    bpe_end = clock()
    if bpe.returncode != 0:
        raise BenchError("BPE error: " + (bpe.stderr or bpe.stdout)[-1600:])
    if raw_path.stat().st_size != 512:
        raise BenchError("BPE prompt RAW has wrong size")
    env = os.environ.copy()
    sdk = str(work / "sdk247")
    env["LD_LIBRARY_PATH"] = sdk
    env["ADSP_LIBRARY_PATH"] = sdk
    command = [
        str(runner),
        "--system", str(work / "sdk247/libQnnSystem.so"),
        "--backend", str(work / "sdk247/libQnnHtp.so"),
        "--jobs", "persistent_jobs_decode_prefill_head.txt",
        "--dynamic-prefill",
        "--dynamic-benchmark",
        "--dynamic-prompt-tokens", str(raw_path),
        "--dynamic-section0", str(
            work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"),
        "--dynamic-section1", str(
            work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"),
        "--dynamic-section2", str(
            work / "cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"),
        "--dynamic-max-tokens", str(max_tokens),
    ]
    if tie_policy != "stop":
        command.extend(["--dynamic-tie-policy", tie_policy])
    with log_path.open("wb") as output:
        status = subprocess.run(command, cwd=work, env=env, stdout=output,
                                stderr=subprocess.STDOUT, check=False)
    qnn_end = clock()
    if status.returncode:
        lines = log_path.read_text(errors="replace").splitlines()
        err = next((s for s in reversed(lines) if "QNN_RUN_ERROR:" in s), "")
        raise BenchError(
            f"QNN failed for {label}, exit={status.returncode}: {err}; "
            f"log={log_path}"
        )
    decoder = subprocess.run(
        [sys.executable, str(ROOT / "scripts/gemma4_decode_log.py"),
         "--log", str(log_path), "--tokenizer", str(work / "tokenizer.json")],
        cwd=work, capture_output=True, text=True, check=False
    )
    finish = clock()
    if decoder.returncode:
        raise BenchError(
            f"decoder failed for {label}: {decoder.stderr[-1000:]}; "
            f"log={log_path}"
        )
    response = json.loads(decoder.stdout)
    numbers = parse_benchmark(log_path.read_text(errors="replace"), response)
    numbers.update({
        "label": label, "prompt": prompt,
        "bpe_wall_ms": 1000.0 * (bpe_end - whole_start),
        "qnn_process_wall_ms": 1000.0 * (qnn_end - bpe_end),
        "decoder_wall_ms": 1000.0 * (finish - qnn_end),
        "user_request_wall_ms": 1000.0 * (finish - whole_start),
        "decoded_text": response["text"],
        "decoded_ids": response["ids"],
        "log_file": str(log_path),
    })
    return numbers


def median_metrics(runs):
    keys = (
        "real_prefill_tokens", "bpe_wall_ms", "metadata_ms",
        "cpu_prefill_ms", "qnn_runtime_context_load_ms",
        "npu_prefill_ms", "kv_bridge_ms", "qnn_first_token_ms",
        "npu_prefill_effective_tok_s", "npu_prefill_fixed_block_tok_s",
        "cpu_plus_npu_prefill_effective_tok_s",
        "decode_steps", "text_tokens",
        "decode_steps_wall_ms", "decode_first_step_ms",
        "decode_cpu_nonkv_ms", "decode_npu_graph_ms",
        "lm_head_npu_graph_ms", "kv_feedback_ms",
        "decode_steps_per_s", "generation_text_tok_s",
        "decode_steps_after_first_per_s", "npu_decode_and_head_steps_per_s",
        "qnn_process_wall_ms", "decoder_wall_ms", "user_request_wall_ms"
    )
    return {key: statistics.median(row[key] for row in runs) for key in keys}


def print_summary(golden, runs, summary, folder):
    print("\nSTAGE8_GOLDEN_OK: exact_9_ids=1 original_QNN_samples_unmodified=1")
    print("STAGE8_METHOD: independent cold QNN processes, no warmup; "
          "post-first-step is NOT a fully warm-start comparison")
    for i, row in enumerate(runs, 1):
        print(f"STAGE8_RUN: {i} input={row['real_prefill_tokens']} "
              f"text_tokens={row['text_tokens']} stop={row['stop_reason']} "
              f"prefill_npu_ms={row['npu_prefill_ms']:.2f} "
              f"generation_tok_s={row['generation_text_tok_s']:.2f} "
              f"ttft_job_ms={row['qnn_first_token_ms']:.2f} "
              f"full_wall_ms={row['user_request_wall_ms']:.2f}")
    print("\nМедианы по отдельным запускам (мс, если не указано иначе):")
    fields = [
        ("BPE (CPU, загрузка JSON включена)", "bpe_wall_ms", "мс"),
        ("Метаданные QNN / создание буферов", "metadata_ms", "мс"),
        ("Подготовка CPU Prefill", "cpu_prefill_ms", "мс"),
        ("Инициализация backend + 3 QNN контекстов",
         "qnn_runtime_context_load_ms", "мс"),
        ("Prefill graphExecute (NPU)", "npu_prefill_ms", "мс"),
        ("Перенос Prefill KV в RAM", "kv_bridge_ms", "мс"),
        ("Prefill NPU эффективных токенов/с",
         "npu_prefill_effective_tok_s", "ток/с"),
        ("Prefill NPU условных 128 позиций/с",
         "npu_prefill_fixed_block_tok_s", "поз/с"),
        ("TTFT от входа в QNN jobs до ID1", "qnn_first_token_ms", "мс"),
        ("Первый Decode шаг целиком", "decode_first_step_ms", "мс"),
        ("Все шаги Decode, CPU+NPU+feedback", "decode_steps_wall_ms", "мс"),
        ("Decode graphExecute (NPU, сумма)", "decode_npu_graph_ms", "мс"),
        ("LM Head graphExecute (NPU, сумма)", "lm_head_npu_graph_ms", "мс"),
        ("Генерация текстовых токенов/с", "generation_text_tok_s", "ток/с"),
        ("Decode шагов/с, включая EOS", "decode_steps_per_s", "шаг/с"),
        ("Decode шагов/с после первого", "decode_steps_after_first_per_s", "шаг/с"),
        ("Один запрос, включая BPE+QNN+decoder", "user_request_wall_ms", "мс"),
    ]
    for title, key, unit in fields:
        print(f"  {title}: {summary[key]:.2f} {unit}")
    print(f"STAGE8_REPORT: {folder / 'summary.json'}")
    print("STAGE8_OK: baseline_cold_process=1 runtime_prefill_decode_instrumented=1")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--text", default="Что ты умеешь?",
                   help="Prompt for throughput measurements")
    p.add_argument("--runs", type=int, default=3,
                   help="Independent cold QNN process runs: 1..10")
    p.add_argument("--max-tokens", type=int, default=64,
                   help="Decode prediction limit, 1..256")
    p.add_argument("--tie-policy", choices=("stop", "lowest-id"),
                   default="stop", help="Experimental lowest-id ONLY if explicitly requested")
    p.add_argument("--workdir", type=Path, default=DEFAULT_WORK)
    p.add_argument("--runner", type=Path,
                   default=ROOT / "build/gemma4-qnn-run")
    args = p.parse_args()
    if not 1 <= args.runs <= 10:
        p.error("--runs must be 1..10")
    if not 1 <= args.max_tokens <= 256:
        p.error("--max-tokens must be 1..256")
    work = args.workdir.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not work.is_dir():
        raise BenchError("Missing work directory: " + str(work))
    check_models(work)
    runner_ready(runner)
    help_result = subprocess.run([str(runner), "--help"], capture_output=True,
                                 text=True, check=False)
    if "--dynamic-benchmark" not in help_result.stdout:
        raise BenchError("Runner is not Stage 8.0: cmake --build build -j2")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    folder = Path(tempfile.mkdtemp(prefix="bench_stage8_" + stamp + "_", dir=work))
    print("STAGE8_BEGIN: " + str(folder), flush=True)
    print("STAGE8_GOLDEN_RUNNING: first full sample-free control", flush=True)
    golden = run_once(work, runner, folder, "golden",
                      GOLDEN_TEXT, 16, "stop")
    if golden["decoded_ids"] != GOLDEN_IDS or \
            golden["decoded_text"] != GOLDEN_ANSWER or \
            golden["stop_reason"] != "eos":
        raise BenchError(
            "Stage8 gold mismatch: no throughput claims permitted; "
            f"trace={golden['log_file']}"
        )
    print("STAGE8_GOLDEN_OK: 9 tokens identical including EOS106", flush=True)
    runs = []
    for i in range(1, args.runs + 1):
        print(f"STAGE8_MEASURE: cold_run={i}/{args.runs}", flush=True)
        run = run_once(work, runner, folder, f"run_{i:02d}",
                       args.text, args.max_tokens, args.tie_policy)
        runs.append(run)
        print(f"STAGE8_MEASURE_OK: run={i} "
              f"decode_steps={run['decode_steps']} "
              f"text_tokens={run['text_tokens']} "
              f"stop_reason={run['stop_reason']}", flush=True)
    medians = median_metrics(runs)
    report = {
        "schema": "gemma4-stage8-cold-v1",
        "platform": "Radxa Dragon Q8B",
        "warmup_performed": False,
        "context_reloaded_between_measurements": True,
        "post_first_decode_not_equivalent_to_separate_warm_inference": True,
        "generation_sampler": "quantized_int8_argmax",
        "tie_policy": args.tie_policy,
        "prompt": args.text,
        "requested_max_tokens": args.max_tokens,
        "golden": golden,
        "runs": runs,
        "median": medians
    }
    with (folder / "summary.json").open("x", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print_summary(golden, runs, medians, folder)


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as error:
        print("STAGE8_ERROR:", error, file=sys.stderr)
        raise SystemExit(1)
