# Stage 5 — CPU LiteRT bring-up on Radxa Dragon Q8B

## Verified baseline

The Q8B completed a strict two-step replay:
`Decode(step3) → KV feedback in RAM → Decode(step4) → LM Head(step4)`.
All 30 full K/V-cache tensors, all 31 Decode outputs and all
262144 INT8 LM Head logits bytes matched the corresponding saved step-4 references.
Decode, Prefill and LM Head QNN contexts were kept loaded simultaneously,
in the verified **Decode → Prefill → LM Head** order.

The **eight non-KV Decode inputs** (embeddings, per-layer embeddings, two masks and
four RoPE arrays) still come from saved files. We now need CPU implementations
on Q8B, without WSL, Genie or host cooperation during inference.

## Model signatures from the original graph report

| Model | Size reported | Signature | Input | Output |
| --- | --- | --- | --- | --- |
| `Section0_TFLiteModel_tf_lite_embedder.tflite` | 106959512 B | `decode_embedder` | INT32 token_ids [1,1] | FLOAT32 embeddings [1,1,1536] |
| `Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite` | 1394630752 B | `decode_per_layer_embedder` | INT32 token_ids [1,1] | FLOAT32 embeddings [1,1,35,256] |
| `Section1_TFLiteModel_tf_lite_aux.tflite` | 85480 B | `decode_mask` | INT32 token_ids [1,1], INT32 time_step scalar | two INT8 masks [1,1,1,4097] |
| Same Section1 | 85480 B | `decode_rope` | INT32 input_pos [1] | four INT8 tensors: global cos/sin (512 each), local cos/sin (256 each) |
| Same Section1 | 85480 B | `decode_cache_update` | position, K/V caches, slices | 30 updated caches; already verified in our direct C++ code |

The prefill variants include `prefill_embedder_128`,
`prefill_per_layer_embedder_128`, `prefill_mask_128` and
`prefill_rope_128`. Tokenizer/model assets for Section11 are not yet located
on Q8B. These are **reported source signatures**, not a claim that the model
files or CPU runtime are installed on the board.

## First diagnostic — locate files and compatible CPU runtime

The script [`scripts/probe_cpu_stack.sh`](../scripts/probe_cpu_stack.sh)
performs a *read-only* file/runtime inventory without loading the 1.4 GB
Section2 model, installing anything or modifying QNN binaries.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
bash scripts/probe_cpu_stack.sh 2>&1 | tee ~/gemma4_cpu_probe.log
```

It searches `$HOME` to a depth of seven directories. If the model
files are elsewhere, set `GEMMA4_CPU_SCAN_ROOT` to their parent directory.
Send the emitted `CPU_MODEL_*`, `CPU_TOKENIZER_*`,
`CPU_RUNTIME_*` and `CPU_PYTHON_PACKAGE` lines.

The priority is to run and verify **Section1 decode_mask/decode_rope** and
**Section0 decode_embedder** locally against existing QNN Decode input buffers.
The large Section2 per-layer embedder comes next. Quantized auxiliary output
bytes must be compared exactly; FLOAT32 embeddings should be checked with
appropriate precision, then QNN output equality.

For Linux the official LiteRT/TFLite C library is
`libtensorflowlite_c.so`, provided by a compatible ARM64 build.
See https://developers.google.com/edge/litert/build/cmake and
https://developers.google.com/edge/litert/conversion/tensorflow/signatures .
Do not assume an Android binary is usable on native Linux.

**Next milestone:** CPU outputs replace the eight saved non-KV Decode
inputs, while the known-good QNN executor and KV-feedback remain unchanged.

## Q8B: first CPU inventory result (2026-11-25)

The board is AArch64 (kernel 7.0.11-7-qcom). Sections 0, 1, 2 were not found
under $HOME (depth 7). System runtime was found:
`/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1`.
Gemma tokenizer JSON exists at `/home/radxa/gemma4_qnn_test/tokenizer.json`
(32,155,733 bytes). The Python TFLite and SentencePiece packages were not
found, but Python packages are not needed for the native C++ generation path.

Availability of **TFLite C++ headers**, a linker `.so` symlink, the required
exported symbols, compatibility of operations, and the source `.tflite` files
still needs independent confirmation. The original probe matched only Section
filenames; the new script also looks for other `.tflite` names and inspects
the tokenizer JSON metadata.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
bash scripts/probe_cpu_native.sh 2>&1 | tee ~/gemma4_cpu_native.log
```

If Section0/1/2 only exist on the WSL development machine, they must be
transferred **once** to the Q8B (preferably under `~/gemma4_qnn_test/cpu_models`).
Do not copy over, modify or re-export the three verified QNN context binaries.

## Q8B native CPU check (2026-11-25)

The board has `libtensorflow-lite.so.2.14.1` (ARM64, Ubuntu/Debian
`libtensorflow-lite2.14.1:arm64`). Exports include
`TfLiteInterpreterCreate` and `TfLiteInterpreterCreateWithSelectedOps`.
No C or C++ TFLite headers were found under the scanned paths, and there were no
`.tflite` source models anywhere under `/home/radxa`.

**Tokenizer correction:** the 262144-vocabulary `tokenizer.json` reports
`model.type=BPE`, not Unigram. It has Replace normalizer, Split
pre-tokenizer, Sequence decoder and TemplateProcessing post-processor.
A C++ tokenizer must implement these actual JSON-defined transforms;
do not assume a classic SentencePiece Unigram model or use a generic
`spm` tokenizer without equivalence tests.

### Native C ABI probe

Added `gemma4-cpu-abi` (C++17, `dlopen/dlsym`, no TensorFlow headers).
It checks 27 required TFLite C API symbols, including *named signature
runner* methods, and prints library version and missing functions.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
./build/gemma4-cpu-abi 2>&1 | tee ~/gemma4_cpu_abi.log
```

If all symbols are present, expect `CPU_C_ABI_READY: required=27 missing=0
signature_runner_available=1`. Missing symbols produce `CPU_C_ABI_INCOMPLETE`
and exit 1. **This only checks dynamic symbol availability.** Actual
Section0/1/2 model loading, operators and tensor values require the model
files and separate tests.

The C signature-runner API exists in TensorFlow Lite 2.14 source; we must
still check that this particular distro shared library exports its symbols.
Reference: https://sources.debian.org/src/tensorflow/2.14.1%2Bdfsg-3.1/tensorflow/lite/c/c_api_signature_runner_test.cc

**Remaining dependency:** source TFLite files Section0 (106959512 bytes),
Section1 (85480 bytes), Section2 (1394630752 bytes) must be found on the
development machine and copied to Q8B once. Their byte lengths are from
the earlier model report; validate copies and avoid touching the three QNN
context binaries.

## Stage 5.1 — CPU TFLite signature runner (code ready, model needed)

Q8B passed the complete native ABI check:

```text
CPU_C_ABI_VERSION: 2.14.1
CPU_C_ABI_READY: required=27 missing=0 signature_runner_available=1
```

Added `gemma4-cpu-run`: a C++17 standalone TFLite signature runner that
loads the installed CPU library via `dlopen/dlsym` (no developer headers).
It uses the actual TFLite named signature API, checks tensor dtype/shapes and
byte sizes and supports scalar INT32 inputs or exact-sized RAW input files.
Output can optionally be byte-compared to an explicitly chosen reference.

Build and verify CLI on Q8B (model files need not be present yet):

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
./build/gemma4-cpu-run --help
```

Once `Section1_TFLiteModel_tf_lite_aux.tflite` is present on Q8B at
`~/gemma4_qnn_test/cpu_models/`:

```bash
cd ~/gemma4_qnn_test
../gemma4-cli-stage1/build/gemma4-cpu-run \
  --model cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite --list

../gemma4-cli-stage1/build/gemma4-cpu-run \
  --model cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite \
  --signature decode_rope --i32 input_pos=3 \
  --output-dir cpu_rope_pos3
```

For `decode_mask`, both inputs are scalar-sized INT32, but they must be
derived from the **correct actual token and step** to reproduce a saved
reference. Example only (not yet verified against a saved QNN input):

```bash
../gemma4-cli-stage1/build/gemma4-cpu-run \
  --model cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite \
  --signature decode_mask --i32 input_tokens=2 --i32 time_step=3 \
  --output-dir cpu_mask_example
```

If the required input dtype is not INT32, or dimensions cannot be allocated,
the program fails; it never silently converts or supplies zeros. Output paths
are never overwritten. To check saved data, use `--expect OUTPUT_NAME=FILE.raw`
for each named TFLite signature output.

We must still **locate/copy Section1 from the development machine**.
TFLite 2.14.1 successfully exporting its C API does not guarantee that
the particular source model's op versions are supported; the first
real model load is a separate test.


## Stage 5.2 — подтверждён первый CPU вызов Section1 и подготовлена сверка с QNN

На Q8B через системную TFLite 2.14.1 успешно выполнена **настоящая** сигнатура `decode_rope` из оригинальной `Section1_TFLiteModel_tf_lite_aux.tflite`:

```text
CPU_TFLITE_VERSION: 2.14.1
CPU_SIGNATURE_COUNT: 6
CPU_INPUT_META: input_pos type=2 shape=[1] bytes=4
CPU_INVOKE_BEGIN: decode_rope
CPU_OUTPUT: pos_emb_cos type=9 shape=[1,1,512] bytes=512
CPU_OUTPUT: pos_emb_local_cos type=9 shape=[1,1,256] bytes=256
CPU_OUTPUT: pos_emb_local_sin type=9 shape=[1,1,256] bytes=256
CPU_OUTPUT: pos_emb_sin type=9 shape=[1,1,512] bytes=512
CPU_SIGNATURE_OK: decode_rope inputs=1 outputs=4
```

Все четыре файла сохранены под `cpu_rope_pos3/`, пока **не сверены** с QNN Decode. Из сохранённого отчёта ввода `qnn_partition_0` установлено точное соответствие: `pos_emb_cos = input[3]`, `pos_emb_sin = input[5]`, `pos_emb_local_sin = input[6]`, `pos_emb_local_cos = input[18]`. Соответствующие размеры: 512/512/256/256 байт. Отдельно для будущей проверки масок: `mask_local = input[22]`, `mask_global = input[27]`, по 4097 байт.

Добавлен читающий-only `scripts/verify_cpu_aux.py`; он разбирает реальный qnn-net-run input-list (38 путей в порядке QNN metadata), проверяет имена при `name:=file`, наличие файлов и размеры, после чего сравнивает **все байты**. Используем именно сохранённую выборку шага 3, соответствующую `input_pos=3`:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main

cd ~/gemma4_qnn_test
python3 ../gemma4-cli-stage1/scripts/verify_cpu_aux.py \
  --kind rope --cpu-dir cpu_rope_pos3 \
  --decode-list auto_decode_step3_list.txt
```

Ожидаемый результат при полном совпадении: `CPU_AUX_ALL_MATCH: kind=rope tensors=4 bytes=1536`. В случае отличий — `CPU_AUX_MISMATCH` с названием, индексом QNN-входа и первым различающимся байтом; **не** считать уже подтверждённое исполнение CPU доказательством побайтовой точности до этой проверки.

После RoPE можно выполнить `decode_mask` с правильным токеном и `time_step`, сохранить маски в отдельный каталог и запустить тот же скрипт с `--kind mask`. Значения реального токена пока не извлечены из сохранённых шагов; не подставлять случайный ID при проверке соответствия.
