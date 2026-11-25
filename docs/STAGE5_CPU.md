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

## Stage 5.6 — ошибка несовместимости TFLite и разбор модели

Первый запуск **Section0 на Radxa Q8B завершился на этапе создания CPU-интерпретатора**:

```text
CPU_EMBED_TEST_START: section=0 token_probe=2
ERROR: Didn't find op for builtin opcode 'EMBEDDING_LOOKUP' version '4'.
ERROR: Registration failed.
CPU_TFLITE_VERSION: 2.14.1
CPU_RUN_ERROR: CPU_INTERPRETER_CREATE_FAILED
```

Это не ошибка QNN и не результат вычисления embeddings. Код C++ собрался
успешно; Section1 (mask/RoPE) и Stage 5.5 **остаются подтверждёнными**.
Старый runtime 2.14.1 не поддерживает запрошенную версию builtin opcode.
В истории релизов TensorFlow Lite поддержка INT4 и поканального квантования
для EmbeddingLookup описана в релизе 2.18.0:
https://github.com/tensorflow/tensorflow/releases/tag/v2.18.0 .
Это сильная причина тестировать более свежий runtime, но окончательную
совместимость с **обеими** Section0 и Section2 докажет только реальный запуск.

**Запрещённый shortcut:** не редактировать TFLite FlatBuffer
для принудительной замены версии op4 на op3 — INT4/поканальная
квантизация может менять семантику, выходы будут недостоверными.
Нельзя подменять системную libtensorflow-lite.so.2.14.1;
её использует проверенный CPU Section1. Нужную новую библиотеку
надо разместить отдельно (например, под ~/gemma4_qnn_test/cpu_runtime/).

### Безопасный инвентаризационный тест без TFLite/Python-пакетов

В репозитории добавлен `scripts/inspect_tflite_ops.py` на Python stdlib.
Он использует **read-only mmap** и не копирует даже 1.4 GB Section2
целиком в оперативную память. Выводит opcode/версии, сигнатуры,
форматы/размеры таблиц embeddings и тип квантизации.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/inspect_tflite_ops.py \
  ~/gemma4_qnn_test/cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite \
  ~/gemma4_qnn_test/cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite \
  2>&1 | tee ~/gemma4_embedder_opcodes.log
```

Если второй файл отсутствует, скрипт завершится ошибкой после отчёта
о первом. Тогда запустить только для Section0. Выводы
`TFLITE_OPCODE`, `TFLITE_EMBED_INPUT`, `TFLITE_SIGNATURE`
покажут, можно ли ограничиться новой версией LiteRT
или нужна также специализированная поддержка нестандартных типов.

### Подключение нового ARM64 runtime отдельно от системного

Скрипт `test_cpu_embedder.sh` теперь принимает переменную
`GEMMA4_TFLITE_LIB=/absolute/path/to/new/libtensorflowlite_c.so`.
Она передаётся в существующий `gemma4-cpu-run --lib` (dlopen/dlsym),
системная TFLite 2.14.1 не заменяется. Новый файл должен быть именно
**Linux aarch64 C API shared library** с символами
`TfLiteInterpreterCreate`, `TfLiteInterpreterGetSignatureRunner` и
`TfLiteSignatureRunnerInvoke`. Обычная Python `.so` не гарантирует C API.
При появлении такой библиотеки:

```bash
GEMMA4_TFLITE_LIB=/path/to/compatible/aarch64/libtensorflowlite_c.so \
  bash scripts/test_cpu_embedder.sh
```

Документация сборки TensorFlow Lite ARM64 C API:
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/guide/build_arm.md .

## Stage 5.7 — экспериментальный C++ INT2/INT4 embedding lookup (без TFLite runtime)

Инспекция исходных моделей подтвердила:

- **Section0:** одна матрица [262144,1536], тип **INT2 = TensorType 19**
  (`TYPE_19` в старой версии инспектора), 100663296 байт упакованных
  весов и 262144 шкал. Нужны `EMBEDDING_LOOKUP v4` и финальный `MUL`.
- **Section2:** 35 таблиц [262144,256] типа **INT4 = TensorType 17**,
  каждая содержит 33554432 байта весов и 262144 шкал.
  Используются 35 × `EMBEDDING_LOOKUP v4` и `CONCATENATION`.
- Исходные `decode_embedder`/`decode_per_layer_embedder` имеют выходы
  FLOAT32 6144 и 35840 байт соответственно.

Источник алгоритма распаковки —
`tensorflow/lite/kernels/embedding_lookup.cc` версии v2.21.0.
Алгоритм использует little-endian упакованные signed INT2/INT4
и умножение выбранного кода на шкалу **конкретной строки токена**.

Для самостоятельного CLI добавлен экспериментальный C++17
`gemma4-packed-embed`: отображает оригинальный TFLite FlatBuffer в память
через `mmap(MAP_PRIVATE, PROT_READ)`, читает одну строку указанного токена
из каждой таблицы и рассчитывает FLOAT32. Обрабатывает MUL Section0
по скалярной константе непосредственно из оригинального FlatBuffer
и CONCATENATION Section2 по порядку входов TFLite-графа.
Программа проверяет типы, число операторов, shapes, размеры весов,
количество row-scales и zero-point. Запрашивает токен только
в диапазоне 0..262143. Не выполняет общую TFLite-модель;
поддерживает исключительно обнаруженные Decode-топологии.

Результат пока **не проверен на реальных файлах Q8B**. Первоначальный
smoke-test проверяет только корректность загрузки/разбора/размеров.
Для подтверждения побайтовой эквивалентности предусмотрен
`--reference SAVED_FLOAT32.raw` (только когда установлен реальный ID
токена из соответствующего сохранённого шага QNN).

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_packed_embedder.sh 2>&1 | tee ~/gemma4_packed_embed0.log
```

Если Section0 пройдёт, Section2 запускается отдельно:

```bash
GEMMA4_PACKED_SECTION=2 bash scripts/test_packed_embedder.sh \
  2>&1 | tee ~/gemma4_packed_embed2.log
```

В конце ожидается `PACKED_TEST_OK: section=0 output_bytes=6144`
или `PACKED_TEST_OK: section=2 output_bytes=35840`.
Это **не** является доказательством правильности модели до байтовой
сверки с настоящими embeddings. Только после строгой сверки
эти функции будут добавлены в единый CPU→NPU процесс.

Инспектор `inspect_tflite_ops.py` также исправлен:
`TYPE_19` теперь распознаётся как INT2, четыре опкода получили
официальные имена (`GREATER_EQUAL`, `LESS`, `LOGICAL_AND`,
`SELECT_V2`), `BUILTIN_2` заменён на `CONCATENATION`, и
`SignatureDef.subgraph_index` читается из правильного поля 4
(поле 3 — устаревший `deprecated_tag`).

### Stage 5.7.1 — исправлена передача результата через RESHAPE (2026-11-25)

На Q8B новый `gemma4-packed-embed` собрался, все четыре CTest прошли.
Первый запуск Section0 с пробным `token=2` подтвердил вычисление последнего
`MUL` с константой `39.1918`, но завершился ошибкой
`PACKED_ERROR: no computed output tensor for original graph`.

Причина в первоначальной реализации: она сознательно игнорировала
`RESHAPE` в том числе после `MUL`, поэтому итоговый FLOAT32 буфер не
попал в индекс выходного тензора FlatBuffer. Добавлена передача значений
через `RESHAPE` только для уже вычисленных FLOAT32 embeddings с
проверкой размерности; первый INT32 token-side RESHAPE по-прежнему не
запускается. Это исправление **не прошло повторный тест на Q8B**.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
bash scripts/test_packed_embedder.sh 2>&1 | tee ~/gemma4_packed_embed0_retry.log
```

Ожидаемые дополнительные строки:

```text
PACKED_EMBED_MUL: scalar=39.1918
PACKED_EMBED_RESHAPE: elements=1536 ...
PACKED_EMBED_OK: ... bytes=6144
PACKED_TEST_OK: section=0 output_bytes=6144 (not yet reference-verified)
```

Сверка с исходным FLOAT32 reference от конкретного токена нужна отдельно.

## Stage 5.8 — token ID из сохранённых QNN embeddings и строгая сверка Section0/2

Пользователь подтвердил успешные smoke-tests **обоих** экспериментальных
`gemma4-packed-embed` на Q8B после исправления завершающего RESHAPE.
Это пока подтверждение корректной загрузки оригинальных TFLite, распаковки
INT2/INT4, применения MUL и CONCAT и размеров выходных FLOAT32.
**Точность полученных FLOAT32 до этого шага ещё не доказана.**

Подготовлен `scripts/verify_packed_embeddings.py`:

1. Разбирает оригинальные `auto_decode_step3_list.txt` / `auto_decode_step4_list.txt`
   и извлекает входы QNN **[33] decode_embeddings_0 (6144 байта)** и
   **[7] decode_per_layer_embeddings_0 (35840 байт)**.
2. По знакам 32 FLOAT32 значений сохранённого Section0 ищет совпадающие
   упакованные INT2-строки таблицы оригинальной модели. Для каждого из 262144
   токенов проверяются только первые 8 байт строки (знаковые шаблоны INT2),
   поэтому не требуется 262144 полных CPU-инференса.
3. Каждый кандидат полностью пересчитывается `gemma4-packed-embed`; token ID
   считается подтверждённым, **только если все 6144 байта Section0 совпадают**.
4. С найденным token ID рассчитывает Section2 и сравнивает **35840 байт**
   с сохранённым входом QNN этого же Decode. Любое различие — ошибка.
5. Делает всё отдельно для шагов 3 и 4; оригиналы и QNN эталоны не изменяет.

Нулевые/положительные/отрицательные значения исходной INT2-квантизации
сохраняют знак при положительном per-token scale и при положительном
Section0 MUL (`39.1918` на Q8B). Поэтому поиск лишь сокращает кандидатов;
полный результат обязательно подтверждается строгой побайтовой сверкой.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/verify_packed_embeddings.py --steps 3 \
  2>&1 | tee ~/gemma4_packed_verify_step3.log
```

После успешного шага 3 можно проверить шаг 4:

```bash
python3 scripts/verify_packed_embeddings.py --steps 4 \
  2>&1 | tee ~/gemma4_packed_verify_step4.log
```

Программа выводит `PACKED_TOKEN_CANDIDATES`,
`PACKED_TOKEN_IDENTIFIED`, `PACKED_EMBED_REFERENCE_MATCH`,
`PACKED_STEP_VERIFIED` — только при реальных совпадениях.
Любое несовпадение показывает первый отличающийся байт, число
несовпавших FLOAT32 и максимальную абсолютную разницу.
Режим пока **не тестировался на Q8B**.

## Stage 5.8 — оба embeddings на шаге 3 и 4 побайтово совпали

Пользователь прислал подтверждённые результаты Q8B:

```text
PACKED_TOKEN_CANDIDATES: step=3 count=1 ids=[568]
PACKED_TOKEN_IDENTIFIED: step=3 token=568 proof=Section0_6144_bytes_exact
PACKED_EMBED_REFERENCE_MATCH: step=3 section=0 token=568 bytes=6144
PACKED_EMBED_REFERENCE_MATCH: step=3 section=2 token=568 bytes=35840
PACKED_STEP_VERIFIED: step=3 token=568 sections=0,2 total_bytes=41984
PACKED_TOKEN_CANDIDATES: step=4 count=1 ids=[568]
PACKED_TOKEN_IDENTIFIED: step=4 token=568 proof=Section0_6144_bytes_exact
PACKED_EMBED_REFERENCE_MATCH: step=4 section=0 token=568 bytes=6144
PACKED_EMBED_REFERENCE_MATCH: step=4 section=2 token=568 bytes=35840
PACKED_STEP_VERIFIED: step=4 token=568 sections=0,2 total_bytes=41984
```

Таким образом, оригинальные INT2/INT4 модели Section0 и Section2
воспроизведены собственным C++17 исполнителем `mmap` побайтово точно
на обоих сохранённых Decode шагах: 2 × 41984 байта.

## Stage 5.9 — 8 не-KV входов Decode вычисляются внутри gemma4-qnn-run

**Новый код пока не испытан на Q8B.** Тот же `packed_embedder.cpp`,
который прошёл строгую проверку Stage 5.8, собирается также в статическую
библиотеку `gemma4-packed-embed-core` без main. Функция
`gemma4::compute_packed_embedding` возвращает float32-байты напрямую
в RAM. `gemma4-qnn-run` находит два входа Decode по метаданным QNN:

- `decode_embeddings_0`: Section0, float32 [1,1,1536], 6144 байта;
- `decode_per_layer_embeddings_0`: Section2, float32 [1,1,35,256], 35840 байт.

По двум новым опциям моделей и ID токена runner вычисляет embeddings
из оригинальных TFLite (не используя runtime TFLite для Section0/2),
побайтово сверяет **сохранённые QNN-входы**, и только после совпадения
заменяет ими QNN input_buffers. Section1 аналогично заменяет шесть AUX
входов. На втором Decode 30 кэшей поступают из KV-feedback в RAM;
два embeddings и шесть AUX заново вычисляются по следующим pos/token.
Перезаписывать оригинальные .raw запрещено.

Новые CLI-опции для jobs:

```text
--cpu-embed-section0 Section0_TFLiteModel_tf_lite_embedder.tflite
--cpu-embed-section2 Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite
--cpu-embed-token 568
--cpu-embed-next-token 568
```

Section1 mask-test продолжает использовать `--cpu-aux-token 2`
(любое согласованное непустое значение); это **не тот же токен, который
вычислен из сохранённого embeddings**. Для embeddings token=568
подтверждён строгим сравнением.

### Готовый тест (на Q8B)

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_cpu_full_nonkv_two_step.sh
```

Лог: `~/gemma4_qnn_test/cpu_full_nonkv_two_step.log`.
Ключевые **ожидаемые**, пока не подтверждённые строки:

```text
CPU_EMBED_QNN_REPLACED: scope=first token=568 tensors=2 bytes=41984 verified_against_saved_decode=1
CPU_AUX_QNN_REPLACED: tensors=6 bytes=9730 ... scope=first
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical
CPU_EMBED_QNN_REPLACED: scope=next token=568 tensors=2 bytes=41984 verified_against_saved_decode=1
CPU_AUX_NEXT_QNN_REPLACED: tensors=6 bytes=9730 cpu_pos=4 cpu_token=2
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_FULL_NONKV_TWO_STEP_OK: CPU Section0/1/2 all non-KV inputs + in-RAM KV feedback verified
```

**Граница:** у первого Decode начальные 30 полных K/V пока загружаются
из сохранённых файлов эталона, Prefill входы пока из сохранённого
списка, а tokenizer, выбор следующего ID и настоящий prompt → prefill →
decode loop ещё не связаны. Поэтому этот этап не является
полностью самостоятельной текстовой генерацией.

## Stage 5.9 — Q8B подтвердил все восемь входов Decode через CPU + RAM

Пользователь прислал успешный `scripts/test_cpu_full_nonkv_two_step.sh`:

```text
CPU_AUX_QNN_REPLACED: tensors=6 bytes=9730 ... scope=first
CPU_EMBED_QNN_REPLACED: scope=first token=568 tensors=2 bytes=41984 verified_against_saved_decode=1
PERSISTENT_CONTEXTS_READY: 3 graphs; one backend/device; all contexts alive
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
KV_AUTO_POS_OK: pos=3 changed_cache_tensors=30 total_cache_tensors=30
KV_FEEDBACK_OK: pos=3 cache_tensors=30 inserted_bytes=9216 layout=K_strided,V_contiguous
KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical
CPU_AUX_NEXT_QNN_REPLACED: tensors=6 bytes=9730 cpu_pos=4 cpu_token=2
CPU_EMBED_QNN_REPLACED: scope=next token=568 tensors=2 bytes=41984 verified_against_saved_decode=1
DECODE_NEXT_INPUTS_READY: kv_from_RAM=30 non_kv_from_saved_next=8 cpu_aux_replaced=6 cpu_embeddings_replaced=2
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_FULL_NONKV_TWO_STEP_OK: CPU Section0/1/2 all non-KV inputs + in-RAM KV feedback verified
```

**Подтверждение:** на обоих Decode шагах все 8 non-KV input tensors
теперь прямо заполняются из Section1 TFLite (маски/RoPE, 9730 байт)
и из собственного INT2/INT4 C++ обработчика Section0/Section2
(41984 байта), всего 51714 байт/шаг. Отдельные prefill/head графы
сохранили reference match, Decode(4) 31/31 и LM Head(4)
262144/262144 байта. Не требуется читать non-KV `.raw` для
**вычисления**; они остаются в тесте как обязательные эталоны.
Начальные KV первого Decode и Prefill QNN inputs остаются сохранёнными.

## Stage 6.0 — предстоящая связка Prefill-128 → Decode

Оригинальный `qnn_io_report.txt` для `qnn_partition_2` содержит
36 входов: **28 K/V cache** и **8 non-KV**. По порядку QNN:

| QNN index | CPU output | Bytes |
|---:|---|---:|
| 3 | `prefill_128_pos_emb_cos_0` | 65536 |
| 5 | `prefill_128_pos_emb_sin_0` | 65536 |
| 6 | `prefill_128_pos_emb_local_sin_0` | 32768 |
| 7 | `prefill_128_per_layer_embeddings_0` | 4587520 |
| 18 | `prefill_128_pos_emb_local_cos_0` | 32768 |
| 21 | `prefill_128_mask_local_0` | 540672 |
| 26 | `prefill_128_mask_global_0` | 540672 |
| 32 | `prefill_128_embeddings_0` | 786432 |

Здесь 30 выходов Prefill существуют независимо от того, что
входных K/V-кэшей только 28. Алгоритм связи Prefill outputs →
Decode initial caches должен опираться на реальные размеры,
имена/слои и раскладку — **не делать предположение 30:30**.

Добавлен `scripts/probe_prefill_metadata.py`, который не требует
TFLite runtime и QNN backend: читает Original Section0/1/2
read-only mmap, выводит точные `prefill_*` сигнатуры, их
именованные входы/выходы, формы и типы; из
`prefill_real_input_list.txt` показывает исходные QNN non-KV
файлы и сверяет их размеры с QNN metadata.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/probe_prefill_metadata.py \
  2>&1 | tee ~/gemma4_prefill_interface.log
```

Ключевые строки: `PREFILL_CPU_SIGNATURE`, `PREFILL_CPU_IO`,
`PREFILL_QNN_NONKV` и `PREFILL_INTERFACE_PROBE_OK`.
После реального Q8B отчёта можно подготовить **правильные** CPU
входы prefill_rope_128 / prefill_mask_128, вычислить
Section0/Section2 embeddings для 128 token IDs и строго проверить
результаты с сохранённым QNN input list. Текущий Stage 6.0
**не доказал** автоматическую связку Prefill→Decode.

## Stage 6.0 на Q8B — метаданные Prefill подтверждены

Пользователь получил:

```text
PREFILL_QNN_LIST_OK: inputs=36 non_kv=8 kv=28 source=prefill_real_input_list.txt
PREFILL_INTERFACE_PROBE_OK: metadata inspection completed
```

Имена и типы CPU-сигнатур в полном логе пока не передавались.
Скрипт проверил, что исходные Section0/1/2 модели присутствуют и
что восемь сохранённых non-KV Prefill входов имеют размеры из QNN-метаданных.

## Stage 6.1 — извлечение 128 token IDs из сохранённого Prefill

Для оригинального `prefill_real_input_list.txt` создан
`scripts/recover_prefill_tokens.py`. Сначала берёт
`prefill_128_embeddings_0` QNN input index 32, FLOAT32
`[1,128,1536]` (786432 байта). Каждый из 128 рядов —
одна строка embeddings от Section0.

Кандидаты ID по каждому ряду подбираются по знакам первых 32 FLOAT32:
из оригинальной INT2-таблицы [262144,1536] формируется
компактная 64-битная подпись знаков (нулевой, положительный,
отрицательный). По всей таблице делается **один проход**,
по 8 упакованных байт на строку, сразу для всех искомых
подписей — без 128 отдельных полнословарных переборов.

Затем **каждый** кандидат проходит через уже побайтово проверенный
`gemma4-packed-embed --section 0`; сравнение должно дать
точное совпадение всех **6144 байт** с сохранённой строкой.
Если у одного ряда нет **ровно одного** подходящего ID, скрипт
останавливается. Даже при повторяющихся токенах проверяются
все 128 позиций; промежуточные выходы кешируются в RAM.

Только после 128 успешных сверок будет создан
`~/gemma4_qnn_test/derived_prefill_128_token_ids_i32.raw`
размером 512 байт (128 × little-endian INT32). Существующий
несовпадающий файл **никогда не перезаписывается**.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/recover_prefill_tokens.py \
  2>&1 | tee ~/gemma4_prefill_tokens_stage6_1.log
```

Искомые результаты:

```text
PREFILL_TOKEN_SEARCH_READY: rows=128 ...
PREFILL_TOKEN_REFERENCE_MATCH: position=0 token=... bytes=6144
...
PREFILL_TOKEN_REFERENCE_MATCH: position=127 token=... bytes=6144
PREFILL_TOKENS_SAVED: .../derived_prefill_128_token_ids_i32.raw bytes=512
PREFILL_TOKENS_VERIFIED: count=128 unique=... reference_bytes=786432 ...
```

Код Stage 6.1 ещё не проверялся на Q8B. Следующий этап
после подтверждения ID — разобрать полноценные Prefill-подписи
CPU Section1, вычислить 128 эмбеддингов Section0/2 одним проходом
и строго сравнить все восемь входов Prefill. Первые 28 KV
входов Prefill останутся отдельным вопросом, как и
перенос его 30 выходов к начальному состоянию Decode.

## Stage 6.1 — 128 токенов восстановлены на Q8B

Пользователь прислал успешный отчёт `scripts/recover_prefill_tokens.py`:

```text
PREFILL_TOKENS_SAVED: /home/radxa/gemma4_qnn_test/derived_prefill_128_token_ids_i32.raw bytes=512
PREFILL_TOKEN_IDS: 2,105,2364,107,116130,236888,217598,16632,236743,236778,236862,236778,236881,106,107,105,4368,0,0,...,0
PREFILL_TOKENS_VERIFIED: count=128 unique=15 reference_bytes=786432 first=2 last=0
```

Первые 17 токенов ненулевые, остальные 111 позиций исходного
образца заполнены нулями. Каждый ID подтверждён 6144 байтами
исходного Section0 FLOAT32, а не просто близостью значений.
Для Stage 6.2 используем именно созданный файл с оригинальными IDs
(128 x int32, little-endian, 512 байт).

## Stage 6.2 — 128 × Section0/Section2 embeddings, CPU только

Экспериментальный код **ещё не испытан на Q8B**. Добавлена библиотечная
функция `gemma4::compute_packed_prefill_embeddings` в уже проверенный
`src/packed_embedder.cpp`; она делает ровно один `mmap` большой модели
за весь Prefill-128, переиспользуя проверенные INT2/INT4 таблицы
и сохраняя FLOAT32 биты результата для каждого токена.
Повторные ID кэшируются (в сохранённом примере уникальных — 15).
Выходные буферы собираются в token-major порядке:

- Section0 FLOAT32 `[1,128,1536]`: 786432 байта;
- Section2 FLOAT32 `[1,128,35,256]`: 4587520 байт.

Новый CLI `gemma4-packed-prefill` принимает
`--section {0,2} --model MODEL.tflite --tokens IDS_512.raw`
`--reference QNN_FLOAT32.raw --output NEW.raw`.
Обязательно выполняет побайтовое сравнение с сохранённым
QNN Prefill input **до** записи нового output. При несовпадении
сообщает первый байт/позицию. Не перезаписывает файлы.

Простой оркестратор `scripts/test_packed_prefill.py` получает
правильные QNN reference paths из `prefill_real_input_list.txt`:

- input[32] `prefill_128_embeddings_0` (Section0),
- input[7] `prefill_128_per_layer_embeddings_0` (Section2).

Оригинальные `Section0/Section2` `.tflite` читаются только через
`mmap(PROT_READ, MAP_PRIVATE)`; ни QNN runtime, ни Section1/LibTFLite
пока не запускаются. Программа не модифицирует эталоны.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/test_packed_prefill.py \
  2>&1 | tee ~/gemma4_prefill_embeddings_stage6_2.log
```

Ожидаемые **пока не проверенные** строки:

```text
PACKED_PREFILL_RAM_READY: section=0 positions=128 unique_tokens=15 bytes=786432 mmap_count=1
PACKED_PREFILL_REFERENCE_MATCH: section=0 bytes=786432 positions=128
PREFILL_EMBED_INPUT_MATCH: section=0 qnn_index=32 bytes=786432 positions=128
PACKED_PREFILL_RAM_READY: section=2 positions=128 unique_tokens=15 bytes=4587520 mmap_count=1
PACKED_PREFILL_REFERENCE_MATCH: section=2 bytes=4587520 positions=128
PREFILL_EMBED_INPUT_MATCH: section=2 qnn_index=7 bytes=4587520 positions=128
PREFILL_EMBED_ALL_MATCH: sections=0,2 tensors=2 bytes=5373952 positions=128 byte_identical=1
```

После этого отдельно подключаем исходные Section1 CPU-сигнатуры
`prefill_rope_128` и `prefill_mask_128`, сравнивая 6 INT8
Prefill non-KV входов, и лишь затем можно переводить
prefill graph QNN на полностью локальные входы.
Перенос KV Prefill → Decode и tokenizer/generation loop остаётся
отдельной задачей.

## Stage 6.2 — Q8B подтвердил оба Prefill embeddings

Пользователь прислал успешный конец теста
`scripts/test_packed_prefill.py`: 

```text
PACKED_PREFILL_RAM_READY: section=2 positions=128 unique_tokens=15 bytes=4587520 mmap_count=1
PACKED_PREFILL_REFERENCE_MATCH: section=2 bytes=4587520 positions=128
PREFILL_EMBED_INPUT_MATCH: section=2 qnn_index=7 bytes=4587520 positions=128
PREFILL_EMBED_ALL_MATCH: sections=0,2 tensors=2 bytes=5373952 positions=128 byte_identical=1
```

Сводный `PREFILL_EMBED_ALL_MATCH` подтверждает проверку и Section0
(786432 байта), и Section2 (4587520 байт). Вычисления выполняются
с оригинальными INT2/INT4 моделями без TFLite runtime, одним mmap
на модель и кэшированием 15 уникальных ID.

## Stage 6.3 — Section1 CPU Prefill RoPE + маски

**Новый код ещё не проверен на Q8B.** Добавлен
`scripts/test_cpu_prefill_aux.py`: читает из Section1 `.tflite`
реальные `prefill_rope_128` / `prefill_mask_128` signature maps
(имена, dtype и формы входов/выходов),
формирует INT32 `input_tokens` из подтверждённого
`derived_prefill_128_token_ids_i32.raw`, генерирует позиции
и `time_step` для первого Prefill (по умолчанию 0), затем
вызывает проверенный `gemma4-cpu-run` через системную TFLite 2.14.1.
Каждый CPU-выход сверяется **байт-в-байт** с файлом из
`prefill_real_input_list.txt`, без NPU и без изменений исходных файлов.

| QNN input индекс | Имя | Байты |
|---:|---|---:|
| 3 | `prefill_128_pos_emb_cos_0` | 65536 |
| 5 | `prefill_128_pos_emb_sin_0` | 65536 |
| 6 | `prefill_128_pos_emb_local_sin_0` | 32768 |
| 18 | `prefill_128_pos_emb_local_cos_0` | 32768 |
| 21 | `prefill_128_mask_local_0` | 540672 |
| 26 | `prefill_128_mask_global_0` | 540672 |
| **Итого** | **6 INT8 тензоров** | **1277952** |

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/test_cpu_prefill_aux.py \
  2>&1 | tee ~/gemma4_prefill_aux_stage6_3.log
```

Ожидаемые (ещё не подтверждённые) итоговые строки:

```text
PREFILL_AUX_SIGNATURE_OK: prefill_rope_128 outputs=4
PREFILL_AUX_SIGNATURE_OK: prefill_mask_128 outputs=2
PREFILL_AUX_ALL_MATCH: tensors=6 bytes=1277952 pos_start=0 time_step=0 verified_against_saved_qnn=1
```

Если именованная сигнатура содержит неизвестные входы или CPU-выход
отличается от эталона, программа **останавливается без успеха**.
Для диагностики будут напечатаны `PREFILL_AUX_SIGNATURE_IO` и
`CPU_RUN_ERROR` с исходными именами/типами.
Параметры `--pos-start N` и `--time-step N` доступны, но
изменять их наугад не следует: это лишь гипотеза о
координатах исходного Prefill, пока сравнение её не подтвердило.

После успеха Stage 6.3 все **восемь** не-KV Prefill входов
будут доказаны как CPU-вычислимые (6 INT8 Section1 +
2 FLOAT32 Section0/2). Далее их связываем в один QNN runtime
и отдельно решаем init 28 входных KV и переход 30 выходных
Prefill slices к Decode KV cache.

## Stage 6.3 — первый запуск на Q8B: расхождение mask_global у padding

Новый CPU Section1 `prefill_mask_128` был вызван с исходной
строго восстановленной последовательностью 128 ID и `time_step=0`.
Runtime 2.14.1 создаёт обе маски. Сравнение `mask_global`
с оригинальным QNN input[26] остановилось:

```text
CPU_OUTPUT: mask_global type=9 shape=[1,1,128,4224] bytes=540672
CPU_RUN_ERROR: CPU_REFERENCE_MISMATCH: mask_global first_byte=75904
```

Оффсет 75904 = `17*4224+4096`: **строка 17, колонка 4096**.
Это именно первая строка после 17 ненулевых Prefill токенов;
остальные 111 ID восстановленного sample равны 0.
Совпадение границы — сильный признак различий при обработке padding
или исходного `input_tokens` маски, но **не доказанная причина**.
RoPE был первым в порядке вызовов `test_cpu_prefill_aux.py`; раз
вызов mask уже начался, все четыре RoPE сравнения прошли.
Нельзя удалять `--expect` и объявлять успех, равно как копировать
сохранённый `mask_global.raw` вместо CPU маски.

Добавлен изолированный скрипт
`scripts/diagnose_prefill_mask.py`: запускает только Section1
`prefill_mask_128` через `gemma4-cpu-run`, перебирая небольшой
контролируемый набор проб:

- исходные 128 token IDs; и те же ID с заменой только suffix-паддинга
  (`0` после 17-й позиции) на ненулевые probe-значения `2` или `1`;
- значения `time_step` 0, 1, 17, 128.

Для каждой пары параметров сравниваются **обе** маски с оригинальными
QNN input[21] и input[26]. Отчёт:

- точное число различающихся байтов;
- первый отличающийся byte, row/column, actual/expected;
- количество различий до padding, в колонках 0..4095 и 4096..4223;
- пять ближайших к эталону проб и наличие/отсутствие полного совпадения.

Замена суффикса на 2/1 — исключительно диагностический эксперимент:
она **не исправляет настоящие ID Prefill и не меняет входы генерации**.
Изменений в рабочий CPU→NPU код пока нет.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
python3 scripts/diagnose_prefill_mask.py \
  2>&1 | tee ~/gemma4_prefill_mask_diagnostic.log
```

Искомые строки:
`PREFILL_MASK_PADDING_BOUNDARY`, `PREFILL_MASK_PROBE`,
`PREFILL_MASK_PROBE_RANK`, `PREFILL_MASK_DIAGNOSTIC_DONE`.
Если `PREFILL_MASK_PROBE_EXACT` появится, необходимо понять
почему исходный QNN sample использовал именно эту семантику маски.
Если нет — изучить по отличиям actual/expected первую строку
padding и проверить подготовку `time_step`/mask для sample.

## Stage 6.3.1 — установлена точная геометрия расхождения padding mask

На Q8B скрипт `diagnose_prefill_mask.py` подтвердил для обоих
`mask_local` и `mask_global` при исходном `time_step=0`: 

```text
PREFILL_MASK_PADDING_BOUNDARY: first_zero=17 zero_suffix=111
PREFILL_MASK_PROBE: candidate=original time_step=0 output=mask_local diff_bytes=8103 first_byte=75904 first_row=17 first_column=4096 actual=127 expected=128 prefix_rows_diff=0 cache_columns_diff=0 current_window_diff=8103 bad_rows=111
PREFILL_MASK_PROBE: candidate=original time_step=0 output=mask_global diff_bytes=8103 first_byte=75904 first_row=17 first_column=4096 actual=127 expected=128 prefix_rows_diff=0 cache_columns_diff=0 current_window_diff=8103 bad_rows=111
```

Ровно **8103 = sum(r+1, r=17..127) = 128*129/2 - 17*18/2**.
Это полный нижний причинный треугольник для 111 padded query rows.
Первые 17 rows и все 4096 исторических cache columns совпадают.
Пробные `time_step` 1,17,128 добавляют расхождения в cache columns,
а замена padded `input_tokens` на 1/2 не меняет ошибки: Section1
Prefill выдаёт causal-mask, а QNN эталон исключает padding query rows.
Это выведено из совпадения структуры ошибок; точное направление
маскирования подтверждается значениями CPU 127, QNN 128
(для signed INT8 это -128).

**Исправление**, только в `scripts/test_cpu_prefill_aux.py`:

- после выполнения TFLite `prefill_mask_128` определяет границу
  padding из строго восстановленных token IDs (нулевой suffix);
- для каждой строки r начиная с padding boundary устанавливает
  INT8 = -128 (байт `0x80`) только в текущем окне, для
  causal columns `0..r` (смещение `r*4224 + 4096 + c`);
- до преобразования требует **ровно 8103** исходных различий
  на обеих масках, и на каждом ожидаемом байте пару (127,128);
- после преобразования строго сравнивает все 540672 байта
  `mask_local` и все 540672 байт `mask_global` с QNN.
  Любое несоответствие вне треугольника блокирует успех.

Эталонные `.raw` остаются read-only. Изменение выполняется только
временной копии CPU-маски. Для Prefill здесь речь не об изменении
оригинального TFLite, а о явном логическом query-padding mask
после выполнения его исходного causal op.

Добавлена стандартная синтетическая регрессия
`scripts/test_prefill_padding_unit.py` (CTest при наличии Python3).
Проверяются 8103 замен, неприкосновенность reference и
отклонение неправильных байтов вне разрешённого треугольника.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
ctest --test-dir build --output-on-failure
python3 scripts/test_cpu_prefill_aux.py \
  2>&1 | tee ~/gemma4_prefill_aux_stage6_3_retry.log
```

**Пока повторный запуск на Q8B не подтверждён.** Искомые строки:

```text
PREFILL_MASK_PADDING_UNPATCHED: name=mask_local mismatching_bytes=8103 expected_padding_triangle=8103
PREFILL_MASK_PADDING_APPLIED: name=mask_local rows=111 rewritten_bytes=8103 from_i8=127 to_i8=-128 reference_match=1
PREFILL_MASK_PADDING_APPLIED: name=mask_global rows=111 rewritten_bytes=8103 from_i8=127 to_i8=-128 reference_match=1
PREFILL_AUX_ALL_MATCH: tensors=6 bytes=1277952 pos_start=0 time_step=0 verified_against_saved_qnn=1
```

После успешной проверки можно будет объединять в Prefill QNN
Section0+Section2 FLOAT32 embeddings и Section1 RoPE +
явную padding-masked Section1 masks внутри одного процесса.

## Stage 6.3.1 — подтверждён на Q8B: все шесть Prefill RoPE/mask выходов

Пользователь прислал полный успешный тест Section1 Prefill 128:

```text
CPU_SIGNATURE_OK: prefill_rope_128 inputs=1 outputs=4
PREFILL_AUX_SIGNATURE_OK: prefill_rope_128 outputs=4
CPU_SIGNATURE_OK: prefill_mask_128 inputs=2 outputs=2
PREFILL_MASK_PADDING_UNPATCHED: name=mask_local mismatching_bytes=8103 expected_padding_triangle=8103
PREFILL_MASK_PADDING_APPLIED: name=mask_local rows=111 rewritten_bytes=8103 from_i8=127 to_i8=-128 reference_match=1
PREFILL_MASK_PADDING_UNPATCHED: name=mask_global mismatching_bytes=8103 expected_padding_triangle=8103
PREFILL_MASK_PADDING_APPLIED: name=mask_global rows=111 rewritten_bytes=8103 from_i8=127 to_i8=-128 reference_match=1
PREFILL_AUX_SIGNATURE_OK: prefill_mask_128 outputs=2
PREFILL_AUX_ALL_MATCH: tensors=6 bytes=1277952 pos_start=0 time_step=0 verified_against_saved_qnn=1
```

Таким образом, все восемь не-KV входов Prefill доказаны как CPU-вычислимые:

- Section0/2 FLOAT32: 5373952 байта, INT2/INT4 на C++ без TFLite runtime;
- Section1 INT8 RoPE/mask: 1277952 байта, TFLite 2.14.1 и
  строго установленная треугольная query-padding коррекция;
- Всего 6651904 байта/Prefill, все по независимым оригинальным QNN RAW
  эталонам. Начальные 28 K/V Prefill пока из сохранённого input list.

Реальные TFLite input signatures подтверждены на Q8B:
`prefill_rope_128(input_pos INT32[128])`,
`prefill_mask_128(input_tokens INT32[1,128],time_step INT32 scalar)`.
Для проверенного образца `input_pos=[0..127]`, `time_step=0`,
17 значащих ID и 111 нулевых padded IDs.

## Stage 6.4 — первые общие CPU→QNN Prefill и Decode в одном процессе

**Новый код ожидает теста на Q8B.** Функция
`gemma4::run_prefill_aux` в `src/cpu_aux.cpp` теперь выполняет
обе `prefill_*` сигнатуры в одной in-process TFLite 2.14.1
интерпретации. Позиции/маски формируются по 128 INT32
токенам, без посторонних RAW, и две маски корректируются строго
по нулевому суффиксу: только causal-allow байты 127 у padded
query rows заменяются на INT8 -128. Все остальные байты
не трогаются.

В `gemma4-qnn-run --jobs` добавлены четыре совместно обязательных
аргумента:

```text
--cpu-prefill-tokens derived_prefill_128_token_ids_i32.raw
--cpu-prefill-section0 cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite
--cpu-prefill-section2 cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite
--cpu-prefill-aux-model cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite
```

Перед NPU-выполнением `qnn_partition_2` из CPU вычисляются
две FLOAT32 embeddings через packed INT2/INT4 (единственный mmap
каждой модели и кеширование повторных ID) и шесть Section1 INT8
RoPE/mask tensors. **Каждый** QNN input index, metadata dtype,
имя и размер проверяются; затем все байты CPU результата сравниваются
со старым input_buffer из сохранённого Prefill QNN sample.
Только после byte-match соответствующий QNN input buffer заменяется
вычисленным в памяти буфером. K/V не изменяются. При отличии
хотя бы одного байта NPU ещё не инициализируется.

Запуск реплея **полной цепочки проверенных контекстов** в порядке
Decode → Prefill → LM Head; все 62 выхода по исходным эталонам;
оба Decode шага также получают свои 8 CPU входов, а 30 K/V
между шагами обновляются в RAM:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_cpu_full_prefill_decode.sh
```

Лог: `~/gemma4_qnn_test/cpu_full_prefill_decode.log`.
Искомые **пока не подтверждённые** маркеры:

```text
CPU_PREFILL_PADDING_APPLIED: prefill_128_mask_local_0 first_pad=17 rewritten_bytes=8103 signed_int8=-128
CPU_PREFILL_PADDING_APPLIED: prefill_128_mask_global_0 first_pad=17 rewritten_bytes=8103 signed_int8=-128
CPU_PREFILL_ALL_NONKV_QNN_REPLACED: tensors=8 embeddings_bytes=5373952 aux_bytes=1277952 total_bytes=6651904 kv_inputs_from_saved=28 verified_against_saved_qnn=1
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
KV_NEXT_CACHE_MATCH: 30 cache tensors byte-identical
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_FULL_PREFILL_DECODE_OK: Prefill-128 eight CPU inputs, 3 QNN graphs 62 outputs, two Decode steps and LM Head byte-identical
```

**Важная граница:** три QNN контекста исполняют самостоятельные
референсные samples, а не одно и то же продолжение prompt.
Initial Prefill-128 28 K/V inputs — из saved baseline, initial
Decode 30 K/V — отдельно из saved step3. Prefill 30 output
KV slices **ещё не передаются** в Decode cache. Даже при успехе
Stage 6.4 это не автономный `prompt→prefill→decode` loop.

Stage 6.5: определить правильную укладку 30 Prefill output slices
в Decode K/V (слоты, слои, порядок K strided / V contiguous),
инициализацию cache и позиции первого Decode; потом tokenizer
и генерацию последовательных ID без saved QNN input.

## Stage 6.4 на Q8B — полный CPU Prefill + Decode reference replay подтверждён

Пользователь получил итоговую строку `CPU_FULL_PREFILL_DECODE_OK` и
подтвердил отсутствие ошибок. В конце лога:

```text
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
CHAIN_DECODE_HEAD_OK: ... logits_bytes=262144 ... int8_argmax_id=568
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_FULL_PREFILL_DECODE_OK: Prefill-128 eight CPU inputs, 3 QNN graphs 62 outputs, two Decode steps and LM Head byte-identical
```

Stage 6.4 прошёл: два embeddings Prefill (Section0 INT2, Section2 INT4)
и шесть AUX tensors Prefill (Section1 RoPE/masks) вычисляются в RAM
внутри `gemma4-qnn-run`, сравниваются побайтово с исходными QNN
input RAW и передаются в Prefill NPU. Decode два шага с CPU non-KV
и RAM feedback 30 K/V продолжает давать побайтово точные результаты.
Все 3 QNN контекста живут одновременно. Важное ограничение:
Prefill и Decode исполняются на отдельных сохранённых K/V samples;
`argmax_id=568` не означает, что новая Prefill последовательность
связана с этим Decode результатом.

## Stage 6.5 — безопасная диагностика K/V Prefill → Decode в RAM

**Код в GitHub, на Q8B ещё не проверен.** Добавлен
`gemma4::update_kv_block` (`src/kv_cache.cpp`), который переносит
первые `valid_rows` из выходного Prefill блока из
`[1,1,channels,128]` для K или `[1,1,128,channels]` для V
в cache `[1,1,channels,4096]` для K или
`[1,1,4096,channels]` для V. K переносится по каналам (strided),
V — подряд (contiguous). Позиции после `valid_rows` не трогаются.
Расширены тесты `tests/kv_cache_test.cpp`: K/V layouts,
соответствие однопозиционному update и обработка неверных размеров.

Новый флаг `gemma4-qnn-run --prefill-kv-probe` работает
**только как отдельный диагностический код** поверх полного
проверенного реплея. Требует `--jobs` и все 4
`--cpu-prefill-*` аргумента. После выполнения 3 QNN графов
и проверки их 62 исходных output tensors:

1. По подтверждённому Prefill token file вычисляет `valid_tokens=17`
   и `padding_rows_ignored=111`, не угадывая позицию.
2. Сверяет метаданные всех 30 Decode cache inputs с
   30 Prefill output tensors `StatefulPartitionedCall_1_0..29`
   по именам, типу, shape, размерам и INT8 scale/offset.
3. Создаёт **30 отдельных нулевых RAM cache буферов**
   размером `channels*4096` и переносит только 17 непустых
   строк из сохранённых, уже byte-validated Prefill outputs.
4. Проверяет каждый перенесённый байт через независимое чтение,
   а также что слоты 17..4095 остались нулями.
5. Выводит `KV_PREFILL_SOURCE_CACHE_INFO` — сколько из
   **28 исходных Prefill K/V input** байтов были ненулевыми.
   Это отдельное условие для настоящей инициализации cache.

При любой неоднозначности или несовпадении scale/offset —
**ошибка без изменения Decode input buffers**. Созданный cache
пока находится только в scratch RAM, не сохраняется и не
используется для NPU Decode. Это не proof correctness
автономного Prefill→Decode; нужно отдельно проверить
семантику исходных 28 cache inputs и следующий token ID.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_kv_bridge.sh
```

Лог: `~/gemma4_qnn_test/cpu_full_prefill_decode.log`.
Ожидаемые, пока **не подтверждённые** метки:

```text
KV_PREFILL_SOURCE_CACHE_INFO: kv_inputs=28 bytes=... nonzero_bytes=... scratch_decode_cache_zero_initialized=1
KV_PREFILL_PROBE_LINK_OK: output=StatefulPartitionedCall_1_0 ... rows_copied=17 padding_rows_skipped=111 ...
KV_PREFILL_BRIDGE_PROBE_OK: tensors=30 layers=15 valid_tokens=17 padding_rows_ignored=111 next_decode_position=17 ... quantization_match=1 no_active_decode_mutation=1 NPU_decode_link_not_executed=1
KV_PREFILL_STAGE6_5_OK: 30 source slices mapped, 17 valid token rows copied into scratch RAM, 111 padding rows excluded
```

Если появится `KV_PREFILL_PROBE_QUANT_ENCODING_UNSUPPORTED`
или `KV_PREFILL_PROBE_QUANT_MISMATCH`, не переносить байты
напрямую в настоящий cache: нужны сведения о масштабе
квантования из QNN metadata или явная конверсия.

## Stage 6.5 — успешно проверен на Q8B

Пользователь подтвердил успешный `test_prefill_kv_bridge.sh`: 

```text
DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical
DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical
CPU_FULL_PREFILL_DECODE_OK: Prefill-128 eight CPU inputs, 3 QNN graphs 62 outputs, two Decode steps and LM Head byte-identical
KV_PREFILL_STAGE6_5_OK: 30 source slices mapped, 17 valid token rows copied into scratch RAM, 111 padding rows excluded
```

Таким образом, Stage 6.5 подтвердил 30 метаданных K/V
Prefill → Decode и перенос 17 не-padding строк в отдельные
RAM-буферы без изменения рабочих Decode входов. В представленном
хвосте лога **нет значения** `KV_PREFILL_SOURCE_CACHE_INFO:`
`nonzero_bytes=...`, поэтому нельзя утверждать, что исходные
28 Prefill cache действительно нулевые.

## Stage 6.6 — первый настоящий Prefill KV → Decode(pos17) → LM Head

**Экспериментальный код ещё не проверен на Q8B.**
Добавлены опции `gemma4-qnn-run`:

```text
--prefill-kv-probe --prefill-kv-run --prefill-kv-token 568
```

В дополнение к четырём `--cpu-prefill-*` моделям/token file,
обычным `--cpu-aux-model` и `--cpu-embed-section0/2`, и
`--chain-decode-head`. Самостоятельная тестовая оболочка
`scripts/test_prefill_decode_link.sh` задаёт параметры автоматически.

Принцип работы:

1. Старый baseline сначала выполняет три QNN графа (в безопасном
   порядке Decode → Prefill → LM Head), проверяет все 62 output tensors.
2. Stage 6.5 проверяет 30 Prefill K/V tensor links и квантование,
   копирует 17 row в 30 отдельные нулевые scratch K/V буферы.
   В режиме запуска проверяет, что **все 28 начальных входов Prefill
   cache байтово нулевые** и что QNN signed INT8 cache имеет
   offset=0; иначе немедленно отказывается строить фиктивный
   fresh-prompt KV.
3. Прежний двухшаговый Decode c KV-feedback и LM Head
   по-прежнему сверяется с сохранёнными эталонами.
4. Сразу после этого в отдельном тестовом запуске
   **реально исполняется `qnn_partition_0`** с 30 scratch K/V
   из Prefill и **новыми 8 CPU non-KV** из Section0/1/2
   для `position=17, token=568`. Никаких сохранённых Decode
   `input_buffers` этот вызов не читает и не заменяет;
   единственное изменение — новые Decode `output_buffers`.
5. `qnn_partition_1` (LM Head) исполняется на hidden output этого
   связанного Decode. Печатает реальный INT8 `argmax`, но
   **НЕ** сравнивает его с независимым Decode/LM Head sample
   и не выдаёт его за ответ на исходный prompt.

Token ID **568 — только явно выбранный probe token**, ранее
полученный на независимом Decode sample. Наш `qnn_partition_2`
Prefill контекст выдаёт 30 K/V output slices без LM Head logits;
поэтому настоящий первый token ID после Prefill **пока не
извлечён** из его финального скрытого состояния.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_decode_link.sh
```

Отдельный лог:
`~/gemma4_qnn_test/cpu_prefill_link_stage6_6.log`.
Искомые, пока не подтверждённые строки:

```text
PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=17 input_token=568 saved_decode_inputs_used=0 fresh_prefill_initial_kv_zero=1
PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=17 input_token=568 ... independent_decode_reference_comparison=0
PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1 no_saved_decode_reference_claim=1
PREFILL_LINK_STAGE6_6_OK: Prefill 30 KV blocks (17 rows) -> real Decode(pos17, token568) -> real LM Head, no independent reference claim
```

В случае
`PREFILL_LINK_REQUIRES_ZERO_INITIAL_28_KV`
сначала разобраться с Prefill initial cache — не отключать
это условие ради успешного сообщения. Если
`PREFILL_LINK_NONZERO_QUANT_ZERO_POINT`, подобрать
корректное квантованное значение нуля и подтвердить его.

Stage 6.6 устанавливает возможность связанного выполнения NPU
но ещё **не обеспечивает семантически корректную генерацию текста**:
для неё потребуются реальный first-next token, tokenizer,
выбор/сэмплинг ID и дальнейшие Decode-шаги без saved refs.

## Stage 6.6 — настоящий связанный Prefill → Decode → LM Head подтверждён на Q8B

Пользователь прислал успешный конец лога `test_prefill_decode_link.sh`:

```text
PREFILL_LINK_INPUTS_READY: kv_from_prefill_ram=30 nonkv_from_cpu=8 position=17 input_token=568 saved_decode_inputs_used=0 fresh_prefill_initial_kv_zero=1
PREFILL_LINK_DECODE_OK: graph=qnn_partition_0 outputs=31 position=17 input_token=568 execute_ms=69.2985 independent_decode_reference_comparison=0
CHAIN_DECODE_HEAD_OK: ... logits_bytes=262144 head_execute_ms=12.2503 int8_argmax_id=236778 max_logit_q=111 argmax_ties=1
PREFILL_LINK_HEAD_OK: logits_bytes=262144 linked_prefill_kv=1 no_saved_decode_reference_claim=1
CPU_FULL_PREFILL_DECODE_OK: Prefill-128 eight CPU inputs, 3 QNN graphs 62 outputs, two Decode steps and LM Head byte-identical
PREFILL_LINK_STAGE6_6_OK: Prefill 30 KV blocks (17 rows) -> real Decode(pos17, token568) -> real LM Head, no independent reference claim
PREFILL_LINK_WRAPPER_OK: Stage 6.6 completed with linked NPU execution
```

**Доказано:** 30 K/V кэшей из настоящих Prefill output buffers переданы
в настоящий Decode QNN с позиционированием 17 и вновь вычисленными
8 non-KV; его hidden output использован непосредственно в LM Head.
У Stage 6.6 **нет независимого QNN эталона** для связанного вывода,
поэтому correctness подтверждена как связность и успешное исполнение,
а не идентичность отдельно сохранённому другому Decode sample.
Изначальные 28 Prefill K/V и их quantized zero прошли guard.
Токен 568 был явным probe, **не получен из самого Prefill**.
После него QNN LM Head вернул уникальный INT8 argmax ID **236778**.

## Stage 6.7 — два и более последовательных Prefill-linked Decode

**Код добавлен, Q8B ещё не проверял.** Опциональный параметр
`--prefill-kv-steps N` (`1..4`, по умолчанию 1) доступен только
совместно с `--prefill-kv-run`. Stage 6.6 остаётся по умолчанию
одним linked Decode. Предел 4 — защита от неограниченного вывода
на стадии эксперимента.

Алгоритм при `N=2`:

1. Исполняется старый строго проверенный трёхконтекстный QNN
   baseline и подтверждённый Stage 6.6 linked Decode
   **pos17, probe token568**, затем linked LM Head.
2. Из его реальных 262144 INT8 logits выбирается
   единственный max `236778` (если несколько совпадающих
   максимумов — отказ, а не произвольное продолжение).
3. Все 30 настоящих выходных Decode K/V slices
   записываются через `update_kv_slot` в scratch cache на
   **позицию 17** (K strided / V contiguous, 9216 байт);
   каждый записанный байт проверяется.
4. Для **позиции 18** и токена `236778` заново вычисляются
   Section0 INT2, Section2 INT4 embeddings и шесть
   Section1 Decode RoPE/mask inputs — всего восемь tensors.
5. Перед исполнением QNN заново привязываются все 38
   RAM pointers к QNN tensor descriptors. Запускаются
   второй QNN Decode и затем LM Head на его hidden output.
6. Для `N=3..4` цикл повторяется, используя
   argmax logits предыдущего linked шага и его KV output.
   Не считывается **ни один** saved Decode input для этих
   связных шагов; 30 KV сохраняют Prefill+feedback state.

**Тест на Q8B:**

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_decode_multistep.sh
```

Вывод: `~/gemma4_qnn_test/cpu_prefill_multistep_stage6_7.log`.
Ожидаемые (**ещё не подтверждены Q8B**) строки:

```text
PREFILL_LINK_GREEDY_NEXT: executed_step=1 token_id=236778 max_logit_q=111 argmax_ties=1 quantized_int8_argmax=1
PREFILL_LINK_KV_FEEDBACK_OK: executed_step=1 position=17 cache_tensors=30 inserted_bytes=9216 layout=K_strided,V_contiguous
PREFILL_LINK_NEXT_INPUTS_READY: step=2 position=18 token=236778 kv_from_feedback_ram=30 nonkv_from_cpu=8 saved_decode_inputs_used=0
PREFILL_LINK_NEXT_DECODE_OK: step=2 position=18 input_token=236778 outputs=31 ... reference_comparison=0
PREFILL_LINK_NEXT_HEAD_OK: step=2 logits_bytes=262144 linked_KV_feedback=1 independent_reference_comparison=0
PREFILL_LINK_MULTI_STEP_OK: steps=2 first_position=17 last_position=18 first_token=568 all_KV_from_prefill_or_feedback_RAM=1 all_nonKV_from_CPU=1 independent_reference_claim=0
PREFILL_LINK_MULTISTEP_STAGE6_7_OK: 2 Prefill-linked Decode steps with KV feedback and INT8 argmax, no saved Decode inputs
PREFILL_MULTISTEP_WRAPPER_OK: Stage 6.7 linked two Decode passes completed
```

Добавлена unit-проверка `tests/kv_cache_test.cpp` для последовательной
записи двух K/V feedback slots без повреждения исходных Prefill
позиций и будущих нулевых слотов.

**Ограничение неизменно:** первый token `568` не является
next token, вычисленным из Prefill — это пробный ID.
Новые argmax после него подтверждают возможность связанного
вычисления, **но не корректную выдачу текста по исходному prompt**.
Tokenizer, финальный Prefill hidden-state/first token и sampler
ещё не связаны. INT8 greedy с неоднозначным максимумом
отклоняется специально; это не production sampling.

## Stage 6.7 — подтверждены два последовательных связанных Decode на Q8B

Пользователь прислал успешный `test_prefill_decode_multistep.sh`:

```text
CHAIN_DECODE_HEAD_OK: ... logits_bytes=262144 head_execute_ms=11.7449 int8_argmax_id=236862 max_logit_q=118 argmax_ties=1
PREFILL_LINK_NEXT_HEAD_OK: step=2 logits_bytes=262144 linked_KV_feedback=1 independent_reference_comparison=0
PREFILL_LINK_MULTI_STEP_OK: steps=2 first_position=17 last_position=18 first_token=568 all_KV_from_prefill_or_feedback_RAM=1 all_nonKV_from_CPU=1 independent_reference_claim=0
PREFILL_LINK_MULTISTEP_STAGE6_7_OK: 2 Prefill-linked Decode steps with KV feedback and INT8 argmax, no saved Decode inputs
PREFILL_MULTISTEP_WRAPPER_OK: Stage 6.7 linked two Decode passes completed
```

Получился технический цикл: pos17 token568 (заданный вручную)
→ LM Head argmax236778 → 30 KV slices через RAM feedback на
позицию 17 → pos18 token236778 → LM Head argmax236862.
Побайтовые baseline регрессии всех 3 QNN графов также прошли.
Эти logits *не принадлежат* семантически корректному продолжению
изначального prompt, так как первый token 568 не был предсказан.

## Stage 6.8 — первый generated token без пробного токена 568

**Новое дополнение, ещё не проверено на Q8B.**
`qnn_partition_2` Prefill выдаёт 30 K/V blocks, но не выдаёт
финальный hidden state для LM Head. Для получения логитов
после реального prompt используем проверенный autoregressive трюк:
повторно считаем **последний настоящий токен** через Decode,
передав ему из Prefill только K/V префикса до этого токена.

У текущего byte-verified prompt 17 настоящих ID,
последний `4368` на позиции `16` (0-based).
Новый флаг `--prefill-seed-last-prompt` (вместо
`--prefill-kv-token 568`) изменяет linked эксперимент:

1. Prefill запускается как прежде на 128 позициях,
   проверяя восемь CPU inputs и 30 выходных KV blocks.
2. При формировании отдельных KV-кэшей Decode копируются
   **только первые 16 K/V rows**, а строка для токена 4368
   удерживается для независимой перепроверки.
3. NPU Decode получает pos16 + token4368, правильные
   30 KV префиксных буферов и восемь новых Section0/1/2
   CPU inputs. Сохранённые Decode inputs не используются.
4. 30 выходных KV slices Decode сравниваются с
   соответствующими последними K/V rows оригинального Prefill.
   Выводятся количество отличающихся байтов для каждого
   слоя и итог `PREFILL_LAST_PROMPT_KV_COMPARE`. **Побайтовое
   равенство между разными QNN kernels не предположено:**
   mismatch — диагностический факт, а не автоматическое
   утверждение неверной семантики. Важно оценить результаты.
5. LM Head получает hidden state этого же Decode;
   по уникальному INT8 argmax выводится
   `PREFILL_FIRST_TOKEN_FROM_PROMPT: token_id=...`.
   Это первый token-кандидат, обусловленный исходным prompt,
   а не независимым sample. Он выбран квантованным greedy
   способом, без production sampling.
6. Тест `N=2` помещает 30 KV slices Decode(pos16) в слот16;
   на pos17 запускает Decode уже с **действительно полученным
   первым токеном**; затем исполняет LM Head второй раз.
   Далее цикл может повторяться до ранее установленного лимита4.

Режим доступен только с `--prefill-kv-run`,
`--prefill-kv-probe`, всеми CPU моделями и
`--prefill-kv-steps 1..4`. Одновременный
`--prefill-seed-last-prompt` и `--prefill-kv-token` запрещён.
Остальные скрипты Stage6.5/6.6/6.7 сохраняют поведение.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_first_token.sh
```

Отдельный лог:
`~/gemma4_qnn_test/cpu_prefill_first_token_stage6_8.log`.
При успехе ожидаются пока **не подтверждённые Q8B** строки:

```text
PREFILL_LAST_PROMPT_SEED: prompt_tokens=17 replay_position=16 replay_token=4368 kv_rows_from_prefill=16 first_generated_token_from_LM_Head=1
PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216 mismatching_bytes=... exact_tensors=... bit_exact=... position=16 numerical_equivalence_not_assumed=1
PREFILL_FIRST_TOKEN_FROM_PROMPT: token_id=... prompt_tokens=17 replayed_last_prompt_position=16 quantized_greedy=1 independent_reference_claim=0
PREFILL_LINK_KV_FEEDBACK_OK: executed_step=1 position=16 cache_tensors=30 inserted_bytes=9216 layout=K_strided,V_contiguous
PREFILL_LINK_NEXT_INPUTS_READY: step=2 position=17 token=<first_generated_token> kv_from_feedback_ram=30 nonkv_from_cpu=8 saved_decode_inputs_used=0
PREFILL_FIRST_TOKEN_STAGE6_8_OK: last_prompt_id=4368 replay_pos=16 first_generated_id=... next_decode_pos=17 saved_decode_inputs_used=0
```

**Ограничения:** до окончательной автономной CLI генерации
ещё нужны tokenizer (text → реальные токены и обратно),
новый Prefill для произвольного prompt (не сохранённого 17-token
образца), правильная позиция/маски для произвольной длины,
EOS handling и полноценный sampler. Текущий linked first-token
тест вычисляет только реальный кандидат для **этого** prompt
и сравнивает K/V численно. Не выдавать новый ID за готовый
читаемый текст без декодирования SentencePiece/tokenizer.

## Stage 6.8 — на Q8B получен первый prompt-conditioned token ID

Пользователь прислал окончание успешного `test_prefill_first_token.sh`:

```text
PREFILL_LINK_NEXT_INPUTS_READY: step=2 position=17 token=236921 kv_from_feedback_ram=30 nonkv_from_cpu=8 saved_decode_inputs_used=0
PREFILL_LINK_NEXT_DECODE_OK: step=2 position=17 input_token=236921 outputs=31 decode_ms=64.8482 reference_comparison=0
CHAIN_DECODE_HEAD_OK: ... logits_bytes=262144 head_execute_ms=11.5586 int8_argmax_id=152294 max_logit_q=79 argmax_ties=2
PREFILL_LINK_NEXT_HEAD_OK: step=2 logits_bytes=262144 linked_KV_feedback=1 independent_reference_comparison=0
PREFILL_LINK_MULTI_STEP_OK: steps=2 first_position=16 last_position=17 first_token=4368 last_prompt_replayed=1 all_KV_from_prefill_or_feedback_RAM=1 all_nonKV_from_CPU=1 independent_reference_claim=0
PREFILL_FIRST_TOKEN_STAGE6_8_OK: last_prompt_id=4368 replay_pos=16 first_generated_id=236921 next_decode_pos=17 saved_decode_inputs_used=0
```

**Факт:** replay last prompt token4368 в pos16 с KV Prefill
позиций0..15 сгенерировал по unique INT8 argmax
**первый токен236921**; следующий Decode в pos17
успешно исполнился именно с236921 и RAM KV-feedback.
Его LM Head выдал `max_logit_q=79, argmax_ties=2`,
первый ID в QNN массиве среди лидеров — **152294**;
второй ID в предоставленном хвосте лога не указан.
Нельзя утверждать, что 152294 — единственно правильный
следующий ID: INT8 квантование стёрло порядок двух максимумов.

**Недостающая часть лога:** пользователь не прислал
`PREFILL_LAST_PROMPT_KV_COMPARE: ... mismatching_bytes=...`;
скрипт Stage6.8 проверяет наличие строки, но не требует
`bit_exact=1`. Для оценки cross-graph equivalence
нужно изучить точную строку и побайтовые различия.

## Stage 6.9 — диагностика неоднозначных INT8 логитов

**Новый код в GitHub, ожидает проверки на Q8B.**
Теперь связанный LM Head на каждом шаге выводит
`CHAIN_LINKED_INT8_MAX_TOKENS: max_logit_q=... ties=... token_ids=...`,
включая все ID, которые разделяют максимальный quantized logit.

В `gemma4-qnn-run --jobs` добавлен
`--prefill-tie-policy stop|lowest-id`:

- `stop` (по умолчанию): при `argmax_ties>1` прекращаем
  автопродолжение, оставляя оба ID диагностике;
- `lowest-id`: явный экспериментальный tie-break,
  который выбирает наименьший ID и пишет
  `PREFILL_LINK_TIE_RESOLUTION: ...
  experimental_arbitrary_tiebreak=1 no_FP32_ordering_claim=1`.
  Это **не FP32 greedy**, и не следует использовать его
  для заявления о качестве ответа модели.

Новый `scripts/test_prefill_int8_tie.sh` повторяет
успешный Stage6.8 с `--prefill-kv-steps 3`,
`--prefill-seed-last-prompt` и
`--prefill-tie-policy lowest-id`. По логу проверяет
обе tied candidates у pos17, их порядок, выбранный
минимальный ID, 30 K/V feedback slots из pos17 и
реальный Decode+LM Head pos18, не читая saved Decode
inputs. Изолированный лог:
`~/gemma4_qnn_test/cpu_prefill_tie_stage6_9.log`.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_int8_tie.sh
```

Искомые, пока не проверенные строки:

```text
PREFILL_LINK_GREEDY_CANDIDATES: executed_step=2 max_logit_q=79 ties=2 candidate_token_ids=152294,...
PREFILL_LINK_TIE_RESOLUTION: executed_step=2 policy=lowest-id chosen_token_id=152294 ... experimental_arbitrary_tiebreak=1 no_FP32_ordering_claim=1
PREFILL_LINK_NEXT_DECODE_OK: step=3 position=18 ...
PREFILL_LINK_NEXT_HEAD_OK: step=3 logits_bytes=262144 ...
PREFILL_LINK_MULTI_STEP_OK: steps=3 first_position=16 last_position=18 first_token=4368 last_prompt_replayed=1 int8_tie_policy=lowest-id-experimental ...
PREFILL_TIE_STAGE6_9_OK: first_generated_id=236921 tied_candidates=... experimental_selected=... third_decode_position=18 ...
```

Строгая семантическая корректность generation пока не установлена:
нужно оценить `PREFILL_LAST_PROMPT_KV_COMPARE` и подключить
правильный tokenizer модели. В репозитории сейчас
нет tokenizer/vocab файлов или реализации text decoding;
только работу с integer token ID и QNN raw tensors проверяли.

## Stage 6.9 — подтверждён на Q8B, включая идентификацию всех INT8 tied IDs

Пользователь прислал успешный результат Stage 6.9:

```text
CHAIN_LINKED_INT8_MAX_TOKENS: max_logit_q=124 ties=1 token_ids=236888 quantized_logits_only=1
PREFILL_LINK_NEXT_HEAD_OK: step=3 logits_bytes=262144 linked_KV_feedback=1 independent_reference_comparison=0
PREFILL_LINK_MULTI_STEP_OK: steps=3 first_position=16 last_position=18 first_token=4368 last_prompt_replayed=1 int8_tie_policy=lowest-id-experimental all_KV_from_prefill_or_feedback_RAM=1 all_nonKV_from_CPU=1 independent_reference_claim=0
PREFILL_TIE_STAGE6_9_OK: first_generated_id=236921 tied_candidates=152294,236787 experimental_selected=152294 third_decode_position=18 real_KV_feedback=1 not_FP32_greedy=1
```

Значит, модель от последнего реального prompt ID=4368 (pos16)
выдала ID=236921, следующий LM Head на pos17 имеет
**двух равных лидеров INT8: 152294 и 236787** (logit79).
В эксперименте выбран min ID=152294 и на pos18 следующий
LM Head дал **236888** (logit124, ties1). Это проверенный
технический цикл, но **не** основание считать 152294
FP32-максимумом: два ID имеют одинаковый INT8 logit.

Самое важное: пользователь также выполнил
`grep PREFILL_LAST_PROMPT_KV_COMPARE` на полном Stage 6.8 логе:

```text
PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216 mismatching_bytes=6975 exact_tensors=0 bit_exact=0 position=16 numerical_equivalence_not_assumed=1
```

Из 9216 сравнённых K/V байт после **одного и того же**
последнего токена 4368 отличаются 6975 (75.68%);
ни один из 30 срезов не совпал побайтово.
Это **существенное расхождение**, которое нельзя
по умолчанию списывать на разницу округлений NPU.
Возможные причины, пока не установленные: неправильная
временная ось или layer mapping Prefill output, разные
раскладки/квантование промежуточных операций, различие
содержимого attention и KV между Prefill и Decode.
Эти результаты не опровергают корректную передачу 30 RAM KV
или успешное исполнение NPU, но **не доказывают**
численную/семантическую эквивалентность Prefill vs
пошагового Decode для этого prompt.

## Stage 6.10 — диагностический анализ K/V расхождений по слоям/времени

**Код в GitHub, ещё не проверен на Q8B.** Новый opt-in
`gemma4-qnn-run --prefill-kv-diagnose` разрешён только с
`--prefill-kv-run --prefill-seed-last-prompt`.
В прежнем Stage 6.8 сохранённом linked-тесте, после
получения pos16 Decode output K/V, для всех 30 пар
Prefill slice vs Decode output проверяются:

- знаковый INT8 |delta|=1, |delta|=2, |delta|>2,
  сумма и максимум абсолютной разницы, доля точных байт
  по полю `mismatched_bytes`;
- поиск лучшей строки (временной позиции) 0..127
  в **том же** Prefill tensor по числу равных байт;
- поиск лучшего другого Prefill output с **той же K/V
  формой, числом каналов и точными scale/offset QNN cache**;
- агрегаты: сколько пар имеют лучшую строку pos16,
  сколько имеют лучший источник с ожидаемым именем,
  сколько имеют точное совпадение при альтернативной
  строке или источнике.

Не выполняются перемаркировка слоёв, сдвиг или копирование
«лучшей» строки в настоящий K/V: диагностика только читает
готовые 30 Prefill outputs и 30 Decode slices.
Сохраняются проверенные ранние Stage 5.9/6.4/6.6/6.8.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_kv_diagnostics.sh
```

Лог:
`~/gemma4_qnn_test/cpu_prefill_kv_diagnostic_stage6_10.log`.
Ожидаемые, **ещё не подтверждённые на Q8B** маркеры:

```text
PREFILL_KV_DIAG_TENSOR: mapped_output=StatefulPartitionedCall_1_0 ...
... (30 lines: signed differences, best row and matching layer)
PREFILL_KV_DIAG_SUMMARY: tensors=30 compared_bytes=9216 mismatched_bytes=... best_same_tensor_row_is_expected=... best_compatible_layer_is_expected=... exact_some_same_tensor_row=... exact_some_compatible_layer=... no_relabel_or_patch_applied=1 diagnostic_only=1
PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216 mismatching_bytes=...
PREFILL_KV_STAGE6_10_OK: all 30 KV tensors inspected; original reference data unchanged
```

Не заявлять о полноценной корректной генерации текста до
разбора причины 75.68% cross-graph KV divergence,
подтверждения модели/токенизатора и стратегии работы с
INT8 ties. Успех `PREFILL_KV_STAGE6_10_OK` означает лишь
получение всех диагностик, а не устранение расхождения.

## Stage 6.10 — Q8B подтвердил корректные строки/слои, но KV численно расходятся

Пользователь прислал полный 30-тензорный отчёт
`PREFILL_KV_DIAG_TENSOR` и агрегат:

```text
PREFILL_KV_DIAG_SUMMARY: tensors=30 compared_bytes=9216 mismatched_bytes=6975 best_same_tensor_row_is_expected=30 best_compatible_layer_is_expected=30 exact_some_same_tensor_row=0 exact_some_compatible_layer=0 no_relabel_or_patch_applied=1 diagnostic_only=1
PREFILL_LAST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216 mismatching_bytes=6975 exact_tensors=0 bit_exact=0 position=16 numerical_equivalence_not_assumed=1
```

**Доказанное уточнение к Stage 6.9:** никакого правильного
альтернативного row либо layer mapping обнаружить не удалось.
Для **всех 30** лучшими остаются собственный тензор и
временная строка16. Из 9216 байт несовпадают 6975 (75.68%),
но величины INT8 отличий распределены неравномерно:

| Группа | Байт | Не совпало | Доля |
|---|---:|---:|---:|
| K (15 tensors) | 4608 | 3221 | 69.9% |
| V (15 tensors) | 4608 | 3754 | 81.5% |
| Кэш-каналы 256 (24 tensors) | 6144 | 4209 | 68.5% |
| Кэш-каналы 512 (6 tensors) | 3072 | 2766 | 90.0% |

Различие signed INT8 по абсолютной величине:
ровно1 — **2911** байт, ровно2 — **1407** байт,
больше2 — **2657** байт; максимум27. Из >2 отклонений
**1890** относятся к 512-канальным тензорам (3072 байта),
**767** — к 256-канальным (6144 байта).
Суммарная абсолютная signed INT8 дельта по всем 9216
байтам **20423**.

Все 256-канальные K преимущественно отличаются на1..2;
512-канальные K/V показывают усиленные различия
(например `k_14` max24, `v_14` max27).
Это **согласуется** с численной чувствительностью/накоплением
ошибок в некоторых слоях, но пока **не доказывает** причину.
Режим Stage 6.10 не изменил ни одного QNN входа или KV.

## Stage 6.11 — изолированный численный тест без истории: Decode pos0 vs Prefill row0

**Новый код в GitHub, Q8B ещё не проверял.**
Добавлен флаг `--prefill-seed-first-prompt`,
который исключает все Prefill KV из входного Decode
и вместо последнего ID воспроизводит **первый реальный
ID=2** исходного prompt на **позиции0** с **30 нулевыми
K/V кэшами**. Исходные 28 Prefill cache уже проверены
нулевыми в Stage6.6. Важно: запрет на ненулевые
cache offset/quant по-прежнему действует.

В отличие от стандартного Stage6.8, этот эксперимент
сравнивает 30 выходных Decode KV slices с соответствующими
**нулевыми строками** 30 Prefill blocks. После сравнения
печатаются `PREFILL_FIRST_PROMPT_KV_COMPARE` и подробные
`PREFILL_KV_DIAG_TENSOR` по всем слоям, времени и
INT8-разнице. Результаты НЕ подменяют нормальный K/V.
Ввиду потенциальных разных QNN kernels полное байтовое
совпадение не предполагается — проверяем распределение и
величину расхождений.

Изолированная оболочка `scripts/test_prefill_first_row_kv.sh`
запускает стандартный byte-exact baseline Decode/Prefill/Head,
после чего отдельно один `Decode(pos0,token2)` и LM Head;
требует проверки всех 30 KV slices, 38 CPU/KV входов и
отсутствия сохранённых Decode input bytes.
Лог Stage6.11:
`~/gemma4_qnn_test/cpu_prefill_first_row_stage6_11.log`.

Дополнительный stdlib-only скрипт
`scripts/compare_prefill_kv_diagnostics.py` автоматически
сравнивает количественные отчёты pos0 и сохранённый
`cpu_prefill_kv_diagnostic_stage6_10.log` pos16.
Группирует K/V по 256/512 каналам и считает процент
несовпадающих байт и среднюю абсолютную signed INT8 дельту.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_first_row_kv.sh
```

Контрольные строки, ещё **не подтверждённые Q8B**:

```text
PREFILL_FIRST_PROMPT_SEED: prompt_tokens=17 replay_position=0 replay_token=2 kv_rows_from_prefill=0 all_KV_raw_zero=1 no_previous_attention_context=1
PREFILL_FIRST_PROMPT_KV_COMPARE: tensors=30 compared_bytes=9216 mismatching_bytes=... exact_tensors=... bit_exact=... position=0
PREFILL_KV_DIAG_SUMMARY: tensors=30 compared_bytes=9216 ...
PREFILL_KV_FIRST_ROW_STAGE6_11_OK: 30 zero-history Decode(pos0,token2) KV slices compared against Prefill row0 without rewriting caches or reference files
PREFILL_KV_POSITION_TOTAL: row=0 tensors=30 bytes=9216 different_bytes=...
PREFILL_KV_POSITION_TOTAL: row=16 tensors=30 bytes=9216 different_bytes=6975 ...
PREFILL_KV_POSITION_COMPARISON_OK: mapped_tensors=30 row0_mismatches=... row16_mismatches=6975 delta_mismatches=...
```

**Интерпретация:** если KV расхождение велико уже при pos0,
то накопление ошибки при передаче Prefill cache не является
единственной причиной. Если в pos0 расхождений почти нет,
а на pos16 их 6975, изучать attention masks/арифметику
в истории и рост ошибочного состояния.
Ни один из выводов нельзя объявлять семантической
эквивалентностью без дополнительного численного контроля.


## Stage 6.12 — проверка причин расхождения Prefill и Decode на первом токене

**Режим подготовлен; ожидает испытания на реальном Q8B.**
A=Prefill-128(BOS=2 + 127 паддингов ID0),
B=уже проверенный Prefill-128(17 реальных токенов),
C=Decode(pos0, BOS=2, пустой KV).
Сравниваем 30 срезов K/V только для row0 (9216 байт).
Новый Prefill-A использует тот же QNN context, но отдельные tensor descriptors,
8 заново вычисленных CPU-входов, 28 неизменных нулевых K/V и 30 независимых
выходных буферов. Сохранённые результаты B/C проверяются на неизменность.
Существующие Stage 6.8–6.11 не изменены.

Запуск:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_bos_only.sh
```

Лог: `~/gemma4_qnn_test/cpu_prefill_bos_only_stage6_12.log`.

Маркеры: `PREFILL_BOS_ONLY_INPUTS_READY`,
`PREFILL_BOS_ONLY_NPU_OK`, 30 строк
`PREFILL_BOS_ONLY_KV_TENSOR`, 4 строки
`PREFILL_BOS_ONLY_KV_GROUP`, `PREFILL_BOS_ONLY_KV_TOTAL`
и `PREFILL_BOS_ONLY_STAGE6_12_OK`.

AB — BOS-only Prefill против Prefill(17), AC — BOS-only Prefill
против Decode(pos0), BC — прежнее сравнение Prefill(17) с Decode.
MAE вычисляется в деквантованных единицах с QNN scale,
проверяется совместимость quant metadata выходных тензоров.
AB=0 указывает на независимость Prefill row0 от будущих токенов/pad
в проверенном режиме. Если AB>0, нужно проверить causal mask,
padding и CPU AUX входы. Маркер OK не доказывает семантическое совпадение.


## Stage 6.12 — результат Q8B

На реальной Q8B: `AB_mismatches=0/9216` для всех 30 K/V, `AC=BC=5449/9216` на позиции 0.
BOS-only Prefill и Prefill(17 tokens) имеют побайтово одинаковые K/V первой строки.
Это исключает влияние **будущих токенов и padding** на row0 в данном тесте;
не доказывает равенство Prefill и Decode.

## Stage 6.13 — сверка 8 не-KV входов первого токена (код, ожидает Q8B)

Три источника: A = CPU-входы Prefill(BOS-only), B = уже проверенные
CPU-входы Prefill(17 tokens), C = реальные 8 Decode входов
перед QNN graphExecute(pos0, token2, нулевой KV) из Stage6.11.

`PREFILL_FIRST_INPUT_TENSOR` логируется 8 раз:
- `embeddings`: 6144 байта
- `per_layer_embeddings`: 35840 байт
- global RoPE sin/cos: 512 + 512
- local RoPE sin/cos: 256 + 256
- local/global attention masks: по 4097 байт

Итого ровно 51714 байт общих подпространств входных тензоров.
Для масок Prefill `[128,4224]`, Decode `[4097]`: сравнивается row0,
колонки 0..4096, а дополнительные Prefill колонки 4097..4223
публикуются отдельно (`mask_tail_masked/unmasked/other`).
Полные буферы несовместимых размеров НЕ сравниваются.
Для квантованных INT8 тензоров обязательно совпадение scale/offset.
Печатаются `AB_mismatches`, `AC_mismatches`, `BC_mismatches`,
`first_AC_offset`; итог `PREFILL_FIRST_INPUT_TOTAL`.

Запуск:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
bash scripts/test_prefill_first_inputs.sh
```

Лог: `~/gemma4_qnn_test/cpu_prefill_first_inputs_stage6_13.log`.

Интерпретация: если `AC_mismatches=0` для всех 8, первый токен
Prefill и Decode получает одинаковые смысловые CPU-входы (при
сравнимой области attention masks), и расхождение K/V нужно искать
в самих QNN графах / вычислениях и в их операционной структуре.
Если `AC>0`, нужно локализовать конкретный вход, проверить
маски/RoPE/эмбеддинги, прежде чем обвинять NPU kernels.
`_OK` означает только завершение диагностического режима.
