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

## Stage 5.3 — CPU decode_mask для сохранённого шага 3

Пользователь подтвердил побайтовое совпадение всех четырёх результатов Section1
`decode_rope` с четырьмя входами QNN Decode шага 3. Stage 5.2 проверен на Q8B.

Следующий CPU AUX тест: две маски `decode_mask` длиной по 4097 INT8 байт.
Точные индексы входов QNN: `mask_local = 22`, `mask_global = 27`.
Новый скрипт выполняет Section1 `decode_mask`, пишет выходы в уникальный каталог,
затем проверяет оба массива по сохранённому `auto_decode_step3_list.txt`.
Сохранённые .raw и QNN бинарники не изменяются.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
bash scripts/test_cpu_mask.sh 2>&1 | tee ~/gemma4_cpu_mask.log
```

По умолчанию используются `input_tokens=2`, `time_step=3`.
ID 2 является только **пробным непустым токеном**, а не установленным
настоящим token ID шага 3. Совпадение докажет точность масок для данного
случая, но не подтвердит идентичность самого token ID.
При несовпадении нужно установить фактический токен и значение time_step;
их можно задать через `GEMMA4_MASK_TOKEN` и `GEMMA4_MASK_STEP`.

Ожидаемый успех (проверка ещё не проводилась):

```text
CPU_AUX_REFERENCE_MATCH: mask_local qnn_input_index=22 bytes=4097
CPU_AUX_REFERENCE_MATCH: mask_global qnn_input_index=27 bytes=4097
CPU_AUX_ALL_MATCH: kind=mask tensors=2 bytes=8194
CPU_MASK_STEP3_VERIFIED: outputs byte-identical to saved QNN inputs
```

## Stage 5.4 — CPU Section1 → QNN Decode без промежуточных .raw (проверено на Q8B)

**Stage 5.3 подтверждён пользователем:** оба выхода `decode_mask`
побайтово совпали с сохранёнными входами QNN Decode шага 3.
Вместе с Stage 5.2 это шесть INT8 AUX-тензоров общей длиной
`512+512+256+256+4097+4097 = 9730` байт.

Добавлен модуль `src/cpu_aux.cpp` и новая опция `gemma4-qnn-run --jobs`.
При `--cpu-aux-model ... --cpu-aux-pos N --cpu-aux-token ID` исполняются
две именованные TFLite-сигнатуры `decode_rope` и `decode_mask` через
установленную системную `libtensorflow-lite.so.2.14.1` (C ABI / `dlopen`).
Выходы C API **не записываются в файлы**, а прямо подставляются в шесть
QNN Decode input buffers, найденных по именам в метаданных QNN.
Для безопасности перед заменой CPU-выходы обязательно проверяются
побайтово против шести исходных сохранённых входов текущего Decode.
Если есть несовпадение, исполнение NPU не начинается.

Для первого теста оставляем другие 32 входа Decode (30 KV + два
FLOAT32 embedding) из сохранённого списка и снова сверяем все 31
выход Decode с оригинальным `qnn-net-run` эталоном.
Остальные QNN-графы в jobs по-прежнему должны совпадать с эталонами.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure

cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
set -o pipefail

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --jobs persistent_jobs_decode_prefill_head.txt \
  --cpu-aux-model "$PWD/cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite" \
  --cpu-aux-pos 3 --cpu-aux-token 2 \
  2>&1 | tee cpu_aux_qnn_decode.log
```

Получено на Q8B (побайтовая сверка прошла):

```text
CPU_AUX_SIGNATURE_OK: decode_rope outputs=4
CPU_AUX_SIGNATURE_OK: decode_mask outputs=2
CPU_AUX_QNN_REPLACED: tensors=6 bytes=9730 from=Section1_TFLite_2.14 verified_against_saved_decode=1
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
```

Флаги `--cpu-aux-pos` и `--cpu-aux-token` обязательны, по умолчанию не
подставляются. Для данного эталона пользователь уже получил побайтовое
совпадение при `pos=3`, пробном непустом `token=2`. Это **не**
доказательство идентичности реального ID токена шага 3,
но достаточно для строгого сравнения шести входов и 31 выхода QNN.

**Граница Stage 5.4:** CPU AUX замещает только входы **первого** Decode
из jobs. В режиме `--kv-run-next` второй Decode пока использует
восемь не-KV входов из следующего сохранённого списка; прямое
питание от CPU AUX для следующего шага — отдельное развитие.
Модель Section0 и Section2 не участвуют в этом тесте.

### Stage 5.4 — подтверждённый результат пользователя

На Q8B успешно прошёл полный `gemma4-qnn-run --jobs --cpu-aux-model`:

```text
CPU_AUX_VERSION: 2.14.1 position=3 token_id=2
CPU_AUX_SIGNATURE_OK: decode_rope outputs=4
CPU_AUX_SIGNATURE_OK: decode_mask outputs=2
CPU_AUX_QNN_REPLACED: tensors=6 bytes=9730 from=Section1_TFLite_2.14 verified_against_saved_decode=1
PERSISTENT_CONTEXTS_READY: 3 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_0 outputs=31 execute_ms=77.3643
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 execute_ms=60.9959
PERSISTENT_GRAPH_OK: qnn_partition_1 outputs=1 execute_ms=11.0824
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
```

Это подтверждает, что шесть входных AUX-тензоров **фактически заменены**
вычислениями CPU Section1 в оперативной памяти и поданы в QNN Decode;
все выходы графов остались побайтово равны эталонам.

## Stage 5.5 — CPU AUX для обоих последовательных Decode (новый код)

Существующий Stage 4.4 проверил: шаг 3 → 30 обновлений KV-cache в RAM →
шаг 4 → LM Head(4), с побайтовым совпадением всех 31 выходов Decode(4)
и 262144 байт LM Head(4). Stage 5.4 проверил подачу CPU AUX на первом шаге.

Stage 5.5 объединяет оба достижения:

- Первый Decode получает 6 масок/RoPE из TFLite Section1 (pos=3, token=2).
- KV-feedback обновляет все 30 полных кэшей в RAM для второго шага.
- На втором шаге из сохранённого списка сначала читаются 8 не-KV входов;
  **шесть AUX-тензоров из них затем заменяются CPU TFLite результатами**
  (pos=4, token=2) и сверяются побайтово с сохранённой выборкой шага 4.
- Два FLOAT32 embeddings второго шага всё ещё берутся из файлов.
- После замены второе исполнение Decode и LM Head должно совпасть с
  оригинальными QNN-эталонами шага 4.

Для удобства добавлен тестовый скрипт:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_cpu_aux_two_step.sh
```

Вывод сохраняется в `~/gemma4_qnn_test/cpu_aux_two_step.log`.
Тест ещё не запущен на Q8B. Ожидаемые итоги:

```text
CPU_AUX_NEXT_QNN_REPLACED: tensors=6 bytes=9730 cpu_pos=4 cpu_token=2
KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_AUX_TWO_STEP_EXIT_OK: all requested comparisons completed
```

Если `CPU_AUX_REFERENCE_MISMATCH` возникнет **до второго NPU Decode**,
это означает, что выбранные CPU pos/token не воспроизводят сохранённый
тестовый шаг 4. Не обходить сверку. Пробный token=2 гарантирует только
непустое значение, не идентифицирует оригинальный токен.

## Stage 5.5 — Q8B подтвердил оба CPU AUX шага

Запуск `scripts/test_cpu_aux_two_step.sh` завершился успешно:

```text
CPU_AUX_QNN_REPLACED: tensors=6 bytes=9730 from=Section1_TFLite_2.14 verified_against_saved_decode=1 scope=first
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
KV_AUTO_POS_OK: pos=3 changed_cache_tensors=30 total_cache_tensors=30
KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical
CPU_AUX_NEXT_QNN_REPLACED: tensors=6 bytes=9730 cpu_pos=4 cpu_token=2
DECODE_NEXT_INPUTS_READY: kv_from_RAM=30 non_kv_from_saved_next=8 cpu_aux_replaced=6
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_AUX_TWO_STEP_EXIT_OK: all requested comparisons completed
```

CPU TFLite Section1 создал по 6 AUX-входов на каждом из двух Decode
(19 460 байт суммарно). Исходные сохранённые входы использовались только
для проверки/всего остального. Для второго Decode 30 KV получены через
CPU-feedback в RAM. Из восьми не-KV входов шесть были заменены на CPU AUX
перед вызовом QNN, а **два FLOAT32 embeddings пока оставлены из сохранённых .raw**.
Три QNN-контекста персистентны, output эталоны не изменяются.

## Stage 5.6 — подготовка CPU Section0 и Section2 embeddings

Осталось два входа Decode: `decode_embeddings_0` FLOAT32 1536 значений
(6144 байта, QNN input[33]) и `decode_per_layer_embeddings_0` FLOAT32
[1,1,35,256] (35840 байт, QNN input[7]).

Оригинальные модели, согласно прежнему графическому отчёту:

| Модель | Размер, байты | CPU decode-сигнатура | Ожидаемый выход |
|---|---:|---|---|
| `Section0_TFLiteModel_tf_lite_embedder.tflite` | 106959512 | `decode_embedder` | 6144 байта |
| `Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite` | 1394630752 | `decode_per_layer_embedder` | 35840 байт |

Файл Section1 был найден на машине `comp` под `/root/gemma4/unpacked/`.
Нужно проверить наличие **Section0 и Section2** в той же папке и один раз
перенести на Q8B. Во время инференса WSL / SSH не используются.

```bash
# На comp (root):
ls -lh /root/gemma4/unpacked/Section0_TFLiteModel_tf_lite_embedder.tflite \
       /root/gemma4/unpacked/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite

scp /root/gemma4/unpacked/Section0_TFLiteModel_tf_lite_embedder.tflite \
    /root/gemma4/unpacked/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite \
    radxa@192.168.0.161:/home/radxa/gemma4_qnn_test/cpu_models/
```

Затем Q8B, **сначала только Section0**:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
bash scripts/test_cpu_embedder.sh 2>&1 | tee ~/gemma4_cpu_embed0.log
```

После успешного Section0 запустить большой Section2 отдельно:

```bash
GEMMA4_EMBEDDER_SECTION=2 bash scripts/test_cpu_embedder.sh \
  2>&1 | tee ~/gemma4_cpu_embed2.log
```

Тест использует `token_ids=2` **только для smoke-test CPU-модели**.
Этот ID не считается фактическим токеном сохранённого шага 3.
Побайтовая сверка с QNN Decode будет корректной, только когда известен
настоящий ID токена оригинального сохранённого шага.

Пока ожидаем только `CPU_EMBED_TEST_OK: section=0 output_bytes=6144` и
`CPU_EMBED_TEST_OK: section=2 output_bytes=35840`; ни один из этих тестов
ещё не выполнялся на Q8B.
