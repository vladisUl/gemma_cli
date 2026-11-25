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

No numerical speed claim is made until measured on the Q8B.
