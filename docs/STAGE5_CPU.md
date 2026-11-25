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
