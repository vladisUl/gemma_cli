# Stage 4 — Persistent QNN contexts (smoke test)

## Предпосылка: Stage 3 завершён

На Radxa Q8B все три графа по отдельности исполнены через прямой QNN C++ API и побайтово сравнивались с `qnn-net-run`:

- LM Head `qnn_partition_1`: 262144/262144 байта совпали;
- Decode `qnn_partition_0`: 31/31 выходов совпали;
- Prefill-128 `qnn_partition_2`: 30/30 выходов совпали (`PREFILL_VERIFIED`).

Последняя диагностическая строка `rpcmem_deinit:dummy call` не означает ошибку при нулевом статусе и наличии `PREFILL_VERIFIED`.

## Что добавлено

`gemma4-qnn-run --jobs FILE` поднимает **один QNN backend/device**, читает метаданные QNN System, валидирует все входы и эталонные выходы, затем **загружает все перечисленные QNN context binaries в один процесс до первого вызова `graphExecute`**, исполняет каждый граф и проверяет все выходы побайтово.

Это тест **совместного размещения** QNN-контекстов и их корректного выполнения, но **ещё не** токенизация, KV-cache feedback или генерация текста. Графы получают независимые, ранее сохранённые входные данные.

Формат файла (по одной строке на граф, 4 поля без пробелов в путях):

```text
# BINARY  GRAPH  INPUT_LIST  REFERENCE_RESULT_0
qnn_partition_2_2.bin qnn_partition_2 prefill_real_input_list.txt result_prefill_real_prompt/Result_0
qnn_partition_0_0.bin qnn_partition_0 auto_decode_step3_list.txt auto_result_decode_3/Result_0
qnn_partition_1_1.bin qnn_partition_1 lm_head_saved_input_list.txt real_result_head_17/Result_0
```

**Это шаблон:** строки Decode и LM Head подставлять только после проверки наличия файлов, использованных в успешных предыдущих тестах; `lm_head_saved_input_list.txt` нужно создать, указав **реальный 1536-байтовый вход** того же LM Head шага. Нельзя подменять входы случайными `.raw` или смешивать эталоны разных шагов.

## Запуск на Q8B

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQAIRT_ROOT="$HOME/qairt247-headers"
cmake --build build -j2
ctest --test-dir build --output-on-failure

cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
```

Для начальной проверки нового режима используйте **один уже известный рабочий Prefill**:

```bash
printf '%s\n' 'qnn_partition_2_2.bin qnn_partition_2 prefill_real_input_list.txt result_prefill_real_prompt/Result_0' > persistent_jobs_prefill.txt

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --jobs persistent_jobs_prefill.txt 2>&1 | tee persistent_prefill.log
```

Ожидается:

```text
QNN_CONTEXT_READY: qnn_partition_2 ...
PERSISTENT_CONTEXTS_READY: 1 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 ...
ALL_JOBS_REFERENCES_MATCH: 1 graphs 30 output tensors
```

После этого создайте `persistent_jobs.txt` с тремя реальными наборами входов и эталонов. Нужные входные списки и каталоги можно найти:

```bash
cd ~/gemma4_qnn_test
find . -maxdepth 3 -type f \( -name '*decode*list*.txt' -o -name '*input*list*.txt' \) -print | sort
find . -maxdepth 3 -type d -name 'Result_0' -print | sort
```

**Критический тест Stage 4:** в логе должны появиться **три** строки `QNN_CONTEXT_READY`, затем `PERSISTENT_CONTEXTS_READY: 3 graphs`, затем три `PERSISTENT_GRAPH_OK` и финальная `ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors` (30 Prefill + 31 Decode + 1 LM Head).


## Проверка всех трёх контекстов (следующий тест)

**Одиночный persistent Prefill прошёл на Q8B (2026-11-25):**

```text
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 execute_ms=54.8585
ALL_JOBS_REFERENCES_MATCH: 1 graphs 30 output tensors
Exit code: 0
```

Следующий тест через [`scripts/test_persistent_three.sh`](../scripts/test_persistent_three.sh) собирает три строки `--jobs` из сохранённых данных, проверяет существование и размеры эталонов **до NPU init**, а затем загружает все три QNN-контекста в один процесс.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
# C++ уже собран в предыдущем тесте; повторная сборка не нужна.

cd ~/gemma4_qnn_test
bash ../gemma4-cli-stage1/scripts/test_persistent_three.sh
```

По умолчанию используются именно ранее документированные пары:

- Prefill: `prefill_real_input_list.txt` + `result_prefill_real_prompt/Result_0`.
- Decode: `auto_decode_step3_list.txt` + `auto_result_decode_3/Result_0`.
- LM Head: скрытое состояние из `real_result_decode_17/Result_0` (1536 байт) + `real_result_head_17/Result_0` (262144 байта logits).

Если Decode 31/31 у вас выполнялся с **другим** списком и эталоном, укажите **оба** соответствующих пути, например:

```bash
DECODE_LIST=мой_проверенный_decode_input_list.txt \
DECODE_REF=каталог_того_же_decode/Result_0 \
bash ../gemma4-cli-stage1/scripts/test_persistent_three.sh
```

Аналогично можно переопределить `HEAD_INPUT` и `HEAD_REF`; не смешивайте разные шаги генерации. Скрипт не подменяет отсутствующие данные фиктивными; при неполном комплекте выведет `PERSISTENT_THREE_ERROR` до загрузки NPU.

При успехе:

```text
PERSISTENT_CONTEXTS_READY: 3 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 ...
PERSISTENT_GRAPH_OK: qnn_partition_0 outputs=31 ...
PERSISTENT_GRAPH_OK: qnn_partition_1 outputs=1 ...
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
PERSISTENT_THREE_VERIFIED: 3 graphs, 62 output tensors byte-identical
```

Если fail уже при загрузке контекстов — покажите первые `QNN_CONTEXT_READY` и `QNN_RUN_ERROR`, а также `free -h`; возможно ограничение памяти HTP. Полный лог сохраняется в `persistent_three.log`.

## Ограничения и диагностика

- Три контекста могут не поместиться в ресурсы CPU/DSP при одновременной загрузке. Если загрузка одного из них даст QNN error, зафиксировать первую ошибку и состояние памяти `free -h`.
- `--jobs` не пишет новые `.raw`, а сравнивает выходы непосредственно из памяти. Существующие режимы `--input-list`, `--reference-dir`, `--inspect` не меняются.
- Пути из manifest разрешаются относительно **текущей рабочей директории**, обычно `~/gemma4_qnn_test`.
- Для каждого контекста сохраняются исходный binary buffer и объект QNN System Metadata; по окончании работы все QNN context освобождаются, затем QNN device/backend.
- Один вызов каждого графа не является тестом KV-cache feedback. Следующий этап после успешного совместного исполнения — persistent входные/выходные буферы, циклическая подача K/V-срезов, CPU LiteRT и tokenizer.

## Stage 4: ошибка одновременной загрузки Prefill → Decode (Q8B)

Проверка трёх контекстов (2026-11-25) остановилась при загрузке второго контекста:

```text
QNN_CONTEXT_READY: qnn_partition_2 load_ms=167.312
fastrpc memory map ... length: 1887436800 failed
Could not allocate persistent weights buffer!
QNN_RUN_ERROR: contextCreateFromBinary(qnn_partition_0): QNN error 1002
```

При этом доступная RAM Linux — около **13 ГиБ**. Это ошибка отображения весов в DSP через FastRPC/SMMU; она **не доказывает**, что не хватает обычной RAM, и не указывает на ошибку вычислений Prefill или Decode. После отказа DSP также сообщил SSR, поэтому при следующих тестах возможны проблемы с состоянием DSP-сессии. Точный предел памяти/адресного пространства HTP пока неизвестен.

Для диагностики порядка и общего объёма отображения добавлен [`scripts/test_persistent_order.sh`](../scripts/test_persistent_order.sh). Скрипт берёт уже валидированные пары входов и эталонов из `persistent_jobs_all3.txt`, созданного предыдущим тестом. Он принимает названия графов в порядке загрузки и вызывает существующий `gemma4-qnn-run --jobs`; модель не перекомпилируется, исходные тестовые файлы не меняются.

**Успешный эксперимент (2026-11-25):** Decode загружен первым, LM Head вторым, оба контекста одновременно активны, все 32 выхода идентичны эталонам:

```text
QNN_CONTEXT_READY: qnn_partition_0 load_ms=344.451
QNN_CONTEXT_READY: qnn_partition_1 load_ms=74.3263
PERSISTENT_CONTEXTS_READY: 2 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_0 outputs=31 execute_ms=73.8492
PERSISTENT_GRAPH_OK: qnn_partition_1 outputs=1 execute_ms=11.2431
ALL_JOBS_REFERENCES_MATCH: 2 graphs 32 output tensors
PERSISTENT_ORDER_VERIFIED: decode_head graphs=2 outputs=32
```

Результат доказывает, что Decode и LM Head могут сосуществовать на Q8B. Но **не доказывает**, что одновременно поместятся также и веса Prefill. Не интерпретировать один замер времени как устойчивый benchmark.

**Успешный эксперимент (2026-11-25):** Decode → Prefill, оба контекста в одном процессе; все 61 выход идентичен эталонам:

```text
QNN_CONTEXT_READY: qnn_partition_0 load_ms=355.276
QNN_CONTEXT_READY: qnn_partition_2 load_ms=160.368
PERSISTENT_CONTEXTS_READY: 2 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_0 outputs=31 execute_ms=75.384
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 execute_ms=54.8381
ALL_JOBS_REFERENCES_MATCH: 2 graphs 61 output tensors
PERSISTENT_ORDER_VERIFIED: decode_prefill graphs=2 outputs=61
```

В отличие от ранее неудачного порядка `Prefill → Decode`, обратный порядок **прошёл**. Это согласуется с зависимостью выделения/отображения DSP-памяти от последовательности загрузки, но не устанавливает точную причину или предел памяти.

**Следующий эксперимент — три контекста:** Decode → Prefill → LM Head в одном процессе с неизменными проверенными входами.

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cd ~/gemma4_qnn_test
bash ../gemma4-cli-stage1/scripts/test_persistent_order.sh decode prefill head
```

Успех: `PERSISTENT_ORDER_VERIFIED: decode_prefill_head graphs=3 outputs=62`. При отказе FastRPC/QNN сохранить `persistent_decode_prefill_head.log` и прислать первые строки ошибки, а не повторять сразу после возможного DSP SSR. Этот тест проверяет **одновременное размещение и воспроизводимость вычислений**, а не связанный цикл генерации.

Обратите внимание: существующие `--jobs` / graphExecute работали с реальными входами; новый скрипт только собирает варианты manifest. Результат одновременной загрузки ещё не подтверждён.

## Stage 4.1 — прямое связывание Decode → LM Head без файлов

После успешного теста трёх одновременно загруженных графов на Q8B (2026-11-25):

```text
QNN_CONTEXT_READY: qnn_partition_0 load_ms=334.76
QNN_CONTEXT_READY: qnn_partition_2 load_ms=147.26
QNN_CONTEXT_READY: qnn_partition_1 load_ms=75.2312
PERSISTENT_CONTEXTS_READY: 3 graphs; one backend/device; all contexts alive
PERSISTENT_GRAPH_OK: qnn_partition_0 outputs=31 execute_ms=68.4304
PERSISTENT_GRAPH_OK: qnn_partition_2 outputs=30 execute_ms=54.3325
PERSISTENT_GRAPH_OK: qnn_partition_1 outputs=1 execute_ms=10.9321
ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors
PERSISTENT_ORDER_VERIFIED: decode_prefill_head graphs=3 outputs=62
```

Три контекста могут сосуществовать на Hexagon v68 **в указанном порядке**. Следующий шаг — связать вычисления без `.raw` на границе графов.

Добавлен необязательный флаг `--chain-decode-head` к режиму `--jobs`. Он:

1. Загружает все контексты в порядке manifest, валидирует все сохранённые входы и побайтово сверяет все 62 независимых результата.
2. Находит 1536-байтный выход Decode, совпадающий с входом LM Head по `shape`, `dtype` и квантованию (`scale/offset`), **не по номеру выхода**.
3. Повторно вызывает `QnnGraph_execute` для LM Head, передавая `Decode.output_buffers[i].data()` в качестве входа. Не пишет и не читает промежуточный hidden state с диска.
4. Выводит `CHAIN_DECODE_HEAD_OK` с числом логитов и диагностическим INT8 argmax (при ничьей фиксирует число одинаковых максимумов).

**Важно:** проверенные отдельные Decode и LM Head относятся к разным сохранённым шагам генерации. Следовательно, новая выходная последовательность логитов НЕ должна автоматически сравниваться с ранее сохранённым `real_result_head_17`: это был бы неверный эталон. Успех `CHAIN_DECODE_HEAD_OK` пока означает корректное исполнение связки и совместимость метаданных, а не точность полного текстового ответа.

На Q8B:

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
  --jobs persistent_jobs_all3.txt \
  --chain-decode-head 2>&1 | tee chain_decode_head.log
```

Ожидаются уже проверенные строки `ALL_JOBS_REFERENCES_MATCH: 3 graphs 62 output tensors`, а затем `CHAIN_DECODE_HEAD_OK: ... hidden_bytes=1536 logits_bytes=262144`. Этот режим **не генерирует текст** и пока не обновляет KV-cache между шагами; для этого потребуется CPU LiteRT и токенизатор на Radxa.

