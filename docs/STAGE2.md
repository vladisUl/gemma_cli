# Stage 2 — прямой QNN Graph Execute

Этот этап **не является генерацией текста**. Он проверяет работу одного графа Gemma 4 без Genie, ONNX Runtime и перезапуска `qnn-net-run`.

Что реализовано:

- `gemma4-qnn-run --inspect` — читает QNN context binary через QNN System API и печатает имена, форму, тип, размер и scale/offset входов и выходов;
- `gemma4-qnn-run --input ... --output ...` — загружает context и граф один раз, вызывает `QnnGraph_execute()` на Hexagon HTP;
- `--reference` — побайтово сравнивает результат с эталоном, включая точную длину;
- `--repeat N` — исполняет тот же граф несколько раз, сообщая среднее время QNN execute.

**Ограничение этапа 2:** `--run` поддерживает только **1 вход / 1 выход**. Начинаем с `qnn_partition_1_1.bin` (LM Head). У Decode/Prefill десятки входов/выходов, для них будет отдельный следующий этап.

## Подготовка на Radxa Q8B

Заголовки QAIRT **2.47** должны лежать в `~/qairt247-headers/include/QNN/`. Работающие .so и .bin уже находятся в `~/gemma4_qnn_test/sdk247` и `~/gemma4_qnn_test`.

После обновления `main`:

```sh
cd ~/gemma4-cli-stage1
git pull --ff-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQAIRT_ROOT="$HOME/qairt247-headers"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

На этом этапе не нужно заменять или пересоздавать QNN-бинарники.

## Тест 1: метаданные LM Head

```sh
cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/sdk247"

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --binary "$PWD/qnn_partition_1_1.bin" \
  --inspect | tee lm_head_metadata.log
```

Ищем `QNN_META_OK`, `GRAPH qnn_partition_1`, **1 input / 1 output**. Размеры и типы смотрим из реальных метаданных, а не подставляем в код.

Этот режим не загружает `libQnnHtp.so` и не вызывает инференс. Если не сработал, приложите лог `lm_head_metadata.log`; пока не переходите к следующему тесту.

## Проверенный результат на Radxa Q8B (2026-11-25)

Сборка Stage 2 прошла с GCC 15.2.0 и заголовками QAIRT 2.47; `ctest` — **1/1 passed**. На реальной плате получен результат режима `--inspect` для `qnn_partition_1_1.bin`:

```text
BINARY /home/radxa/gemma4_qnn_test/qnn_partition_1_1.bin bytes=405118976
QNN_META_OK: graphs=1
GRAPH qnn_partition_1 inputs=1 outputs=1
  input[0] LanguageModel_decode_graph_LanguageModel_transformer_stack_transformer_transformer_final_norm_composite_quantized sfix8 shape=[1,1,1536] bytes=1536 scale=0.4921325445 offset=10
  output[0] StatefulPartitionedCall_30 sfix8 shape=[1,1,262144] bytes=262144 scale=0.2333765179 offset=1
```

Это подтверждает **только чтение метаданных QNN System**. `QnnGraph_execute` ещё не проверен на этом этапе. Не следует путать смещение QNN (offset=+1) с TFLite zero point (-1).

Перед вызовом LM Head найдите сохранённый вход **от того же тестового запуска** (он должен быть 1536 байт):

```sh
find ~/gemma4_qnn_test -type f -name '*.raw' -size 1536c -print
```

Если вход хранится только в WSL, ищите там среди артефактов теста; копируйте нужный файл на Radxa через `scp`. По одному лишь размеру нельзя определить, какой файл является настоящим входом LM Head. Эталонные 262144 байта logits также должны соответствовать **этому же входу**.

## Результат проверки `QnnGraph_execute` на Q8B (2026-11-25)

Реальный тест с сохранённой парой `real_result_decode_17` и `real_result_head_17` **успешно выполнен** на устройстве:

```text
QNN_EXECUTE_OK: qnn_partition_1 load_ms=81.007 avg_execute_ms=12.716 repeat=1 ... bytes=262144
REFERENCE_MATCH: 262144 bytes identical
```

Вход: 1536 байт sfix8 из `real_result_decode_17/Result_0/LanguageModel_decode_graph_LanguageModel_transformer_stack_transformer_transformer_final_norm_composite_quantized_native.raw`.

Эталон: `real_result_head_17/Result_0/StatefulPartitionedCall_30_native.raw` (262144 байта sfix8).

Это подтверждает прямой QNN inference LM Head и точное совпадение с `qnn-net-run`. Для следующего этапа — 38 входов Decode — см. [Stage 3](STAGE3.md).

## Тест 2: LM Head на NPU

Используйте **именно сохранённый вход LM Head** из уже удачного запуска `qnn-net-run`. Это сырой квантованный тензор, а не текст и не float32 embeddings. Никаких нулевых заглушек.

```sh
cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/sdk247"

INPUT_RAW=/absolute/path/to/previous/lm_head_hidden_int8.raw

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --binary "$PWD/qnn_partition_1_1.bin" \
  --graph qnn_partition_1 \
  --input "$INPUT_RAW" \
  --output "$PWD/lm_head_direct_qnn.raw"
```

Успех: `QNN_EXECUTE_OK: qnn_partition_1 ...`.

Сравните с сохранённым QNN выходом **того же входа**:

```sh
cmp -s /absolute/path/to/previous/lm_head_logits.raw \
  ~/gemma4_qnn_test/lm_head_direct_qnn.raw \
  && echo REFERENCE_MATCH || echo REFERENCE_MISMATCH
```

Или используйте `--reference /absolute/path/to/previous/lm_head_logits.raw`. Команда сама завершится с ошибкой при отличии хотя бы одного байта. Файл результатов сохраняется даже при несовпадении.

### Важные ограничения

- Не используйте `--input` и `--output` с одним и тем же путём.
- Размер входа проверяется по QNN System метаданным. Неподходящий `.raw` не будет отправлен на NPU.
- Input/output передаются через `QNN_TENSORMEMTYPE_RAW`; оптимизация общей памяти RPCMem будет позднее.
- Время `avg_execute_ms` **не включает загрузку контекста** и не равно полной скорости генерации.
- Бинарники `qnn_partition_0_0.bin`, `qnn_partition_1_1.bin`, `qnn_partition_2_2.bin` не изменяются.

## Следующий шаг

После подтверждения одинаковых LM Head logits добавим произвольное число входных и выходных тензоров (Decode 38/31, Prefill 36/30), загрузку всех графов по одному разу и цикл управления KV-cache. Только после этого переносим на Radxa CPU LiteRT / SentencePiece и реализуем `gemma4-cli` с интерактивным чатом.
