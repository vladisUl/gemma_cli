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



## Stage 8.1 — 256 exact Decode steps with KV feedback (synthetic load test)

Stage 8.1 is implemented in the **existing** sample-free dynamic path
behind an opt-in flag: `--dynamic-bench-fixed-steps N` (1..256),
which additionally requires `--dynamic-prefill --dynamic-benchmark`.
Normal user-facing inference and Stage 8.0 stay unchanged.

**Method:** use the normal BPE prompt and Prefill-128 once; use its KV
for the first Decode token (the chat-header LF, token 107), and then
feed a deterministic token ID **236778** (the known BPE token for "2")
at every later Decode iteration, irrespective of predicted token IDs.
The head graph runs at **every** step, but EOS and ties are ignored.
Every iteration updates **all 30 KV slices** in the SAME live Decode
KV buffers, INCLUDING the last step. Only the new slot is modified
on the host; QNN continues to receive the complete declared KV RAW
inputs. `BENCH81_STEP` verifies all 30 slices, exactly 9216 feedback
bytes per step, input ID and next KV position.

The benchmark is an **artificial, fixed-length workload**, **NOT**
semantic text generation, a faithful FP32 sampler, or a standalone
warm-start inference. It is intended to isolate the cost of
`CPU non-KV → QNN Decode → QNN LM Head → KV update` while the three
AOT contexts stay loaded for all 256 measured steps.

No extra warm-up iterations are inserted into the measured process.
Before measuring, a golden run in a separate process must reproduce
the exact nine tokens and `2 + 2 будет **4**.` including EOS106.
That golden run can still change OS caches and NPU clock state;
the following measured processes start with their own fresh QNN
contexts and are not cold boots.

### Run on the Q8B

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/bench_stage8_1.py --runs 3 --steps 256 \
  --text 'Что ты умеешь?'
```

The output contains three `STAGE81_RUN` lines, a table of medians,
`STAGE81_OK`, and a unique `summary.json` and per-run logs under
`~/gemma4_qnn_test/bench_stage8_1_.../`. Do not present synthetic
`decode_steps_per_s` as natural-language text tokens/s; it is the
rate of completed Decode+Head+KV update iterations.

Per-step timings use host `steady_clock`:
- `cpu_nonkv_ms`: packed embeddings + TFLite auxiliary outputs +
  metadata validation and binding of buffers
- `npu_decode_ms`: blocking QNN Decode `graphExecute`
- `head_call_ms`: full Head call, including NPU execution and
  diagnostic argmax/log output
- `head_execute_ms`: strictly the LM Head graphExecute, from the
  existing `CHAIN_DECODE_HEAD_OK` marker
- `kv_feedback_ms`: copy/update **all 30 output cache slices**
  into the new token slot
- `step_wall_ms`: sum of CPU, Decode, Head and feedback before
  benchmark logging (no AOT load included)

Both full-loop step/s and NPU-only graphExecute step/s are printed.
NPU graph times are synchronous host API durations, not HTP cycle
counters. `steps_after_first_per_s` is separately provided.

### Comparison against LiteRT-LM

The user's LiteRT-LM 0.19.1 E2B YNNPACK benchmark is
`prefill_tokens=256`, `decode_tokens=256`,
`max_num_tokens=1024`, 8 CPU threads, MTP disabled; measured
27.689 decoded tok/s. In Stage 8.0 our natural-greedy sample-free
QNN ran 64 text tokens from a **13-token** Prefill context at
11.83 tok/s, with cache capacity 4096.

Stage 8.1 now controls the number of Decode steps (256) and ensures
complete KV feedback, but still uses fixed AOT Prefill-128 and cache
capacity 4096. The data stream is synthetic fixed token 236778;
LiteRT-LM measures normal decode from 256 Prefill tokens. Therefore
Stage 8.1 is NOT yet a matched 256/256/1024 comparison.
Future Stage 8.2 would require a correct, validated 256-token
Prefill and a matching QNN Decode AOT with 1024 KV capacity.

### Stage 8.1 measured results — 2026-11-25, Dragon Q8B

Hardware run passed **16/16 CTest**, the original nine-token
2+2 golden regression (EOS106), and all three Stage 8.1 runs.
Each individual run executed **exactly 256 synthetic Decode+Head
iterations and 256 updates to all 30 KV slices**. Each started
from a fresh QNN process; none reloaded contexts between its
256 Decode steps. No extra warmup was inserted.

Original board-side report (not uploaded to this repository):

`/home/radxa/gemma4_qnn_test/bench_stage8_1_20261125_191731_0m4rz4hn/summary.json`

| Run | Fixed Decode steps | KV updates | Step/s (CPU+QNN+KV) | NPU graph-only step/s | QNN process wall (s) |
|---|---:|---:|---:|---:|---:|
| 1 | 256 | 256 | 11.756 | 13.201 | 23.152 |
| 2 | 256 | 256 | 10.409 | 11.544 | 25.995 |
| 3 | 256 | 256 | 10.297 | 11.363 | 26.202 |
| **Median** | **256** | **256** | **10.409** | **11.544** | **25.995** |

Reported median phase measurements:

| Phase | Milliseconds |
|---|---:|
| BPE (Python + tokenizer JSON) | 1027.484 |
| QNN metadata / buffers | 455.837 |
| Initialize QNN backend + three contexts | 719.180 |
| CPU Prefill | 36.584 |
| NPU Prefill-128 | 61.357 |
| Prefill → Decode RAM KV transfer | 1.838 |
| CPU non-KV preparation per Decode step | 8.012 |
| **NPU Decode graphExecute per step** | **72.903** |
| **NPU LM Head graphExecute per step** | **13.650** |
| CPU KV slot-feedback per step | 0.128 |
| All 256 Decode steps (median) | 24593.645 |
| First Decode step | 96.533 |
| Steps after first | 10.414 steps/s |

Median across the three independent runs showed a decrease from
11.756 to 10.297 step/s (~12.4%). Causes are **not established**:
measure DSP/HTP clocks and temperatures, and inspect step latency by
position before attributing the effect to throttling or growing KV.

Although the CPU-side writing of 30 KV slices costs only ~0.128 ms,
that does **not** establish that QNN/HTP avoids full-cache transfers or
synchronization inside its Decode graphExecute (measured ~72.9 ms).
KV cache is reused and never recomputed from prompt at each step,
but the full 4096-position buffers are bound to Decode as QNN RAW
tensor inputs. The rate is dominated by blocking Decode and Head
calls, not by Python BPE or host KV slot updates.

Compared numerically with LiteRT-LM 0.19.1 E2B YNNPACK at
27.689 decoded tok/s, the Stage 8.1 synthetic loop's 10.409
Decode+Head+KV steps/s is ~2.66x lower. **Do not call this a matched
apples-to-apples benchmark:** QNN has 13 real Prefill tokens, a
Prefill-128 graph, a 4096-token cache capacity, and fixed repeated
token ID 236778. LiteRT-LM uses 256 Prefill tokens, 256 normal Decode
tokens, 1024 KV capacity, eight CPU threads, MTP disabled. The next
performance investigation should first check per-step latency across
positions and HTP clock/thermal behavior; true 256/256/1024 requires
different validated QNN contexts.


## Stage 8.2 — QNN HTP profile + RAW/MEMHANDLE KV A/B

Status: **source committed; QAIRT 2.47 compilation and Q8B execution
NOT yet confirmed**. This is an opt-in hardware compatibility
experiment. No original AOT context or model is rebuilt or modified.

### Problem

QNN Decode takes 30 INT8 KV input tensors, each with context
capacity 4096; together they total **37,748,736 bytes (36 MiB)**.
Only 30 new slices totaling **9216 bytes (9 KiB)** are updated
on the CPU after each Decode step. Stage 8.1 measured ~72.903 ms
per blocking Decode graphExecute at the median of three runs. That
time includes possible RPC/driver input transfer, not only HTP cycles.

Current QNN I/O binds KV as QNN_TENSORMEMTYPE_RAW from std::vector.
Qualcomm documents optional HTP RPC shared-buffer registration:
a QnnMem_register handle for an rpcmem_alloc allocation may allow the
backend to avoid the per-step input copy. The exact benefit and ABI
support on Q8B with QAIRT 2.47 are to be determined experimentally.

### Controlled implementation

Three opt-in flags **only** accepted with `--dynamic-prefill`,
`--dynamic-benchmark` and `--dynamic-bench-fixed-steps`:

* `--dynamic-kv-memhandle`: allocate 30 RPCmem regions via
  `libcdsprpc.so`, copy the 36 MiB KV once after ordinary Prefill,
  register each fd using `QnnMem_register` in the **Decode QNN
  context**, then bind the 30 KV Decode inputs with
  `QNN_TENSORMEMTYPE_MEMHANDLE`. The remaining eight Decode inputs,
  all Prefill inputs and LM Head continue in RAW. The original RAM
  KV is also updated, with only the 9216 changed bytes mirrored
  into registered RPCmem on each Decode step. KV bytes are verified
  against the RAM copy at initialization and after the final step.
  `QnnMem_deRegister` and `rpcmem_free` run before the contexts
  are freed. If rpcmem is absent or registration fails, the test
  **fails instead of silently switching back to RAW**. The optional
  `GEMMA4_RPCMEM_LIBRARY` environment variable can point to the
  actual libcdsprpc.so location.

* `--dynamic-bench-checksums`: log 64-bit FNV-1a checksums of
  every Decode output tensor and all LM Head INT8 logits after each
  step. The short correctness preflight compares every checksum
  from RAW and MEMHANDLE runs on the same fixed-token stream;
  mismatches abort the benchmark.

* `--dynamic-qnn-profile basic|detailed`: profile sampled
  Decode and Head graphExecute calls (steps 1, 2, 16, 64, 128,
  and the final step where present) via QNN profileCreate,
  profileGetEvents, profileGetEventData and profileGetSubEvents.
  Logs contain event type and unit **numeric IDs** as supplied by
  the backend; never guess that an event represents raw HTP
  compute. Profile collection is isolated from A/B speed runs.

The ordinary Stage 7 CLI, original Stage 8.0 and the default
Stage 8.1 RAW run are unchanged. Only the synthetic fixed-step
benchmark permits MEMHANDLE; natural-language correctness still
requires the unchanged exact 2+2 golden RAW regression, and
registered memory requires matching ALL Decode+Head outputs in
a separate synthetic multi-step preflight.

### Start with compatibility/correctness only

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/bench_stage8_2.py --mode probe --probe-steps 8
```

Expect `STAGE82_GOLDEN_OK`, followed by `STAGE82_PREFLIGHT_OK`
and `STAGE82_OK`. If the board lacks the RPCmem library, export
`GEMMA4_RPCMEM_LIBRARY=/actual/path/libcdsprpc.so` before
retrying. A QNN MEM_REGISTER error means the ION fd or ABI may
not be accepted by this backend; do not claim zero-copy until
registration and byte-for-byte verification succeed.

### Then run A/B and separate sampled profiling

```bash
python3 scripts/bench_stage8_2.py --mode ab \
  --steps 256 --runs 3 --text 'Что ты умеешь?'
python3 scripts/bench_stage8_2.py --mode profile \
  --steps 256 --profile-level basic --text 'Что ты умеешь?'
```

Or after the compatibility check, `--mode all` performs both
in one session. The script always checks 2+2; MEMHANDLE probe and A/B modes additionally
require exact eight-step RAW/MEMHANDLE output equivalence before long trials.
The RAW-only `--mode profile` does not require a working RPCmem library
or a MEMHANDLE registration, so profiling can be investigated separately. A/B runs
use independent fresh-process QNN contexts with NO QNN profiler,
but all same synthetic 256 Decode+Head steps, full KV feedback
and identical model binaries. Profile runs sample individual
calls and deliberately should NOT be compared for throughput.

Each run creates `~/gemma4_qnn_test/bench_stage8_2_.../` with
golden log, two 8-step preflight logs, per-run RAW/MEMHANDLE
QNN logs, optional QNN profile logs and `summary.json`.
The summary explicitly states this remains **Prefill-128 with
13-ish actual tokens, fixed Decode KV4096** and not the matched
LiteRT-LM 256/256/1024 workload. Actual Prefill token count
is read from the runtime. The fixed repeated token 236778 is
synthetic; its step/s is not user-visible natural-language tok/s.

### Interpretation

The A/B ratio assesses the **entire decode loop** including
updated KV slot mirrors, with all other knobs held constant.
A successful MEMHANDLE path confirms registration and output
equivalence, but alone does not prove zero-copy DMA behavior.
QNN profiling may expose HOST RPC and HTP runtime events; their
availability and semantics must be checked against the specific
QAIRT 2.47 headers and backend outputs.

Links:
- Qualcomm QNN HTP Shared Buffer Tutorial
- Qualcomm QNN HTP Qmem Graph documentation
- QNN Profile API, QnnMem registration API

No clock or thermal conclusions are drawn without measurements.
