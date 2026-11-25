# Stage 8.0 — Q8B Prefill / Decode speed baseline

Status: source committed; measurements on the real board pending.

## Method

Stage 8 uses the same BPE → CPU Prefill → NPU Prefill → KV cache
→ CPU Decode inputs → NPU Decode → NPU LM Head path as the verified
Stage 7.2–7.5. It never modifies QNN binaries, the tokenizer JSON,
saved samples or reference RAW data. Before any throughput claim, it
runs the original 2+2 question through the sample-free path and
requires all 9 output IDs, the correct text and EOS106.

No extra warmup is performed. Each timed iteration starts a NEW QNN
process and loads all three QNN contexts afresh. File caches, DSP
frequencies, and hardware thermal state can persist across these
executions: this is a fresh-process comparison, NOT a cold boot.
Decode steps 2..N occur with contexts already loaded, but excluding
step 1 does NOT constitute a separate warm Prefill benchmark.

## Run on the Radxa Q8B

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/bench_stage8.py --runs 3 --max-tokens 64 \
  --text 'Что ты умеешь?'
```

Optional extended output experiment:

```bash
python3 scripts/bench_stage8.py --runs 3 --max-tokens 128 \
  --text 'Опиши коротко, как работает микроконтроллер.'
```

The default `--tie-policy stop` stops at an ambiguous INT8
maximum. Explicitly passing `--tie-policy lowest-id` produces
an EXPERIMENTAL continuation and does not recover true FP32 ranking.
The script accepts 1..10 independent runs (default 3).

## Outputs

A unique directory is created under `~/gemma4_qnn_test/`:

```text
bench_stage8_YYYYMMDD_HHMMSS_xxxxxxxx/
  golden.raw
  golden.log
  run_01.raw
  run_01.log
  run_02.raw
  run_02.log
  run_03.raw
  run_03.log
  summary.json
```

The terminal and summary.json provide each run, its stop reason,
number of emitted text tokens and median across runs.
Median rates are medians of each run's own ratio, NOT rates
recomputed from medians of timing and token counts.

| Metric | Definition |
|---|---|
| BPE ms | New stdlib Python tokenizer process + JSON loading |
| Metadata ms | Read QNN AOT binaries and construct metadata and I/O buffers |
| CPU Prefill ms | Packed embeddings and TFLite Section1 inputs |
| Context load ms | Initialize QNN backend and load three contexts |
| NPU Prefill ms | Wall-clock time around blocking Prefill graphExecute |
| KV bridge ms | Copy 30 cache blocks from Prefill to Decode RAM KV |
| Prefill effective tok/s | Non-padding Prefill tokens / NPU Prefill duration |
| Prefill fixed-block pos/s | 128 padded graph positions / NPU Prefill duration |
| TTFT ms | From QNN jobs preparation to first token prediction; includes context load but excludes BPE and process startup |
| Decode step ms | CPU non-KV preparation, NPU Decode, NPU LM Head, sampling, feedback |
| NPU Decode ms | Host wall duration of Decode graphExecute only |
| NPU LM Head ms | Existing head_execute_ms around LM Head graphExecute |
| Generation tok/s | Non-EOS output tokens / sum of measured Decode steps (including EOS evaluation) |
| Decode steps/s | All executed Decode+Head iterations / total step duration |
| Steps 2..N/s | Iterations after first / their wall durations |
| Total user wall ms | BPE + QNN process + separate output decoder |

A host steady_clock around synchronous graphExecute measures blocking
API execution time, including possible driver overhead. It is NOT
the accelerator's raw cycle counter. Short responses and early EOS
make the tok/s ratio noisier; read counts and stop reasons.

QNN performance markers are opt-in via `--dynamic-benchmark`,
keeping the normal generation path unchanged. The labels are
`BENCH8_METADATA`, `BENCH8_PREFILL`, `BENCH8_FIRST_TOKEN`,
`BENCH8_STEP` and the existing `CHAIN_DECODE_HEAD_OK`.

## Measured baseline: Dragon Q8B, 2026-11-25

The hardware run reported `STAGE8_OK` and a successful full golden
regression (`2 + 2 будет **4**.`, nine exact IDs, EOS106).
All 14 CTest cases passed after the Stage 8 unit-test fixture fix.
Exact saved local report path:

`/home/radxa/gemma4_qnn_test/bench_stage8_20261125_184428_kjg2ps6_/summary.json`

Conditions: prompt `Что ты умеешь?`, **13 actual Prefill tokens**
(excludes final LF replayed by Decode), graph fixed at 128 positions;
three independent fresh-process runs, no added warm-up;
64 predicted text tokens, `stop_reason=limit` for every run;
safe default INT8 tie policy. The output did not reach EOS, so it is
a speed baseline, not evidence of answer completion.

| Run | NPU Prefill (ms) | Text tok/s | QNN-jobs TTFT (ms) | Full request wall (ms) |
|---|---:|---:|---:|---:|
| 1 | 60.72 | 11.02 | 1362.68 | 8557.59 |
| 2 | 59.64 | 11.83 | 1339.19 | 8113.35 |
| 3 | 56.31 | 11.84 | 1348.78 | 8104.60 |
| **Median** | **59.64** | **11.83** | **1348.78** | **8113.35** |

Median component details:

| Metric | Value |
|---|---:|
| BPE Python startup, tokenizer JSON + encoding | 950.05 ms |
| QNN metadata / allocation | 459.13 ms |
| CPU Prefill input construction | 36.54 ms |
| QNN backend and three context loads | 703.28 ms |
| Blocking Prefill graphExecute | 59.64 ms |
| Prefill-to-Decode KV bridge | 1.81 ms |
| Effective Prefill (13 real tokens) | 217.99 tok/s |
| Nominal 128 graph-position throughput | 2146.37 pos/s |
| CPU + NPU effective Prefill | 135.16 tok/s |
| TTFT from QNN jobs start, includes context loading | 1348.78 ms |
| First Decode iteration | 85.80 ms |
| 64 Decode iterations including CPU, Head and KV | 5408.05 ms |
| 64 Decode graphExecute calls | 4147.85 ms |
| 64 LM Head graphExecute calls | 713.60 ms |
| User-visible text generation | 11.83 tok/s |
| NPU Decode + Head graphExecute only | 13.17 steps/s |
| Decode iterations after first | 11.84 steps/s |

Per iteration from median summed wall times: Decode NPU ≈64.81 ms,
LM Head ≈11.15 ms, CPU/other ≈8.54 ms; approximately 77% of
iteration wall time is in NPU Decode, 13% in Head. Note these are
ratios of marginal medians, not a breakdown of one exact run.

### LiteRT-LM figures supplied for comparison

The user measured the same E2B model through standard LiteRT-LM.
The reported figures (independent test; methodology may differ) were:

| Backend | Init (s) | TTFT (s) | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|---:|
| GPU | 4.308 | 0.811 | 340.807 | 16.698 |
| CPU | 10.593 | 0.635 | 429.293 | 25.704 |
| CPU YNNPACK | 8.270 | 0.480 | 577.284 | 27.689 |
| Our QNN HTP | included in individual timings | 1.349 from jobs start | 217.99 effective over 13 tokens | 11.83 |

Raw decode throughput QNN vs LiteRT GPU is about 0.71×;
vs CPU about 0.46×; vs CPU YNNPACK about 0.43×.
These comparisons should **not** be interpreted as
matched-hardware, matched-prompt, matched-sampling benchmarks without
confirming the measurement method, model quantization and runtime
settings.

In particular, the QNN TTFT 1.349 s includes 0.459 s metadata
and 0.703 s context loading. Their difference (~0.186 s) is only
an inferred after-init estimate, not an independent warm TTFT sample.
QNN Prefill throughput uses just 13 genuine tokens despite running
all 128 graph positions: 2146 nominal padded positions/s is NOT
2146 real input tokens/s.

Next benchmark questions: NPU/DSP clocks and thermal behavior;
constant-token-length workload; context-load removal for interactive
use; NPU Decode graph latency (~64.8ms per token) versus Head (~11.2ms)
and CPU input prep (~8.5ms). Do not optimize BPE and claim that will
remove the dominant Decode graphExecute latency.

