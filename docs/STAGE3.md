# Stage 3 — Decode и Prefill-128 (прямой QNN, сверка с qnn-net-run)

**Уже доказано на Q8B:** LM Head (`qnn_partition_1`) исполняется напрямую через `QnnGraph_execute` и выдаёт **262144/262144 идентичных байта** с `qnn-net-run`. Для проверенного шага 17: загрузка контекста 81.007 мс, один вызов execute 12.716 мс. Это результаты **одного** шага, не скорость генерации.

**Decode на Q8B проверен (2026-11-25):** пользовательский тест завершился строками `REFERENCE_MATCH: 1536 bytes identical` и `ALL_REFERENCES_MATCH: 31 output tensors`. Это означает побайтовое совпадение всех 31 выходов Decode с результатами `qnn-net-run` для того же набора входов. Prefill-128 — следующая проверка. Добавлен режим `gemma4-qnn-run`, принимающий 38 и больше входных тензоров, выполняющий граф один раз и сохраняющий **каждый** выход с именем `<tensor_name>_native.raw`. Сверку делает с прежними папками `Result_0` от `qnn-net-run`.

## 1. Обновить и собрать на Q8B

```sh
cd ~/gemma4-cli-stage1
git pull --ff-only
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQAIRT_ROOT="$HOME/qairt247-headers"
cmake --build build -j2
ctest --test-dir build --output-on-failure
```

Предыдущий режим LM Head `--input /path --output /path --reference /path` остался без изменений.

## 2. Посмотреть метаданные Decode (без NPU)

```sh
cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/sdk247"

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --binary "$PWD/qnn_partition_0_0.bin" \
  --inspect > direct_decode_metadata.log

grep -E '^QNN_META_OK|^GRAPH' direct_decode_metadata.log
```

Ожидаем `GRAPH qnn_partition_0 inputs=38 outputs=31`. Эта проверка выполняется без NPU. Полный лог сохраняется в `direct_decode_metadata.log`.

## 3. Найти готовый input_list от прошлого теста

В старых тестах Q8B `qnn-net-run` получал входы через **одну строку с 38 путями** к `.raw` (в порядке `graphInputs`). Например, сценарий автоматического теста создавал `auto_decode_step3_list.txt` и `auto_decode_step4_list.txt`, результат шёл в `auto_result_decode_3/Result_0` и т. д.

```sh
cd ~/gemma4_qnn_test
find . -maxdepth 2 -type f \( -name '*decode*list*.txt' -o -name '*input*list*.txt' \) -print | sort
```

**Не переписывайте входные данные ради теста.** Выберите тот список, которым действительно запускался `qnn-net-run`; все входные файлы должны по-прежнему существовать.

## 4. Проверить размеры всех 38 входов (без NPU)

Пример с тестом позиции 3, если сохранился соответствующий файл:

```sh
cd ~/gemma4_qnn_test
LIST="$PWD/auto_decode_step3_list.txt"

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --binary "$PWD/qnn_partition_0_0.bin" \
  --graph qnn_partition_0 \
  --input-list "$LIST" \
  --outputs-dir "$PWD/direct_result_decode_3/Result_0" \
  --check-inputs
```

При полном успехе: `INPUTS_VALIDATED: 38 tensors; no NPU execution`. Проверяются точный размер и тип каждого тензора по метаданным QNN. Ни один вход не подменяется нулями. **Путь к каждому input `.raw` будет напечатан.**

Если файл `auto_decode_step3_list.txt` отсутствует, используйте путь к найденному списку и **соответствующей ему** папке результатов.

## 5. Выполнить Decode на NPU и сравнить все 31 выхода

```sh
cd ~/gemma4_qnn_test
LIST="$PWD/auto_decode_step3_list.txt"

../gemma4-cli-stage1/build/gemma4-qnn-run \
  --system "$PWD/sdk247/libQnnSystem.so" \
  --backend "$PWD/sdk247/libQnnHtp.so" \
  --binary "$PWD/qnn_partition_0_0.bin" \
  --graph qnn_partition_0 \
  --input-list "$LIST" \
  --outputs-dir "$PWD/direct_result_decode_3/Result_0" \
  --reference-dir "$PWD/auto_result_decode_3/Result_0"
```

Искомый результат: `QNN_MULTI_EXECUTE_OK: qnn_partition_0 inputs=38 outputs=31`, 31 строк `REFERENCE_MATCH: N bytes identical` и итог `ALL_REFERENCES_MATCH: 31 output tensors`. Последняя строка возможна только после успешной побайтовой сверки всех выходов.

## Поддерживаемые формы входа

- `--input-list FILE`: одна непустая строка с путями к `.raw`, разделёнными пробелами, как в наших готовых `qnn-net-run` сценариях. Поддержан также формат `tensor_name:=path` **в исходном порядке метаданных**. Дополнительные строки (несколько samples) намеренно отклоняются. Относительные пути разрешаются от **текущего рабочего каталога**.
- `--inputs-dir DIR`: вместо списка ищет `<tensor_name>_native.raw`, `<tensor_name>.raw` или `input_<index>.raw`, для каждого input должно быть ровно одно совпадение. Если есть неоднозначность, используйте `--input-list`.
- `--outputs-dir DIR`: сохраняет все выходы с именами `<tensor_name>_native.raw`. Не используйте исходную папку эталонов, иначе они будут перезаписаны.
- `--reference-dir DIR`: эталонная директория **`Result_0`**, где лежат `.raw` от `qnn-net-run`. Сверка побайтовая и строгая.
- `--repeat N`: повторяет вызов графа с теми же входами без повторной загрузки контекста. Не тестирует обновление KV-cache — это будет отдельный шаг.

## Что не реализовано

Это ещё **не** `gemma4-cli` с автогенерацией текста. Нет длительно живущих трёх контекстов, динамического KV-cache, CPU LiteRT embedder/AUX, SentencePiece и интерактивного чата. После подтверждения `Decode 38 → 31` следующим шагом будет `Prefill 36 → 30`, затем единый persistent-executor и локальный CPU pipeline.


## 6. Проверить Prefill-128: 36 входов → 30 выходов

**Новый скрипт:** [`scripts/test_prefill.sh`](../scripts/test_prefill.sh). Для него не требуется переделывать C++: текущий `gemma4-qnn-run` уже поддерживает произвольное число тензоров.

Выполнить на Q8B после `git pull --ff-only` и сборки:

```bash
cd ~/gemma4-cli-stage1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DQAIRT_ROOT="$HOME/qairt247-headers"
cmake --build build -j2
ctest --test-dir build --output-on-failure

cd ~/gemma4_qnn_test
find . -maxdepth 3 -type f \( -name '*prefill*list*.txt' -o -name '*prefill*input*.txt' \) -print | sort
find . -maxdepth 4 -type d -iname '*prefill*' -print | sort
```

Нужно выбрать сохранённый **Prefill input-list** (или папку 36 входных файлов) и соответствующий ему **эталонный `Result_0`**, полученный при вызове `qnn-net-run` **с точно теми же входами**. Автоматически подбирать несовместимые результаты по имени каталога нельзя. Все относительные пути в списке входов трактуются относительно `~/gemma4_qnn_test`.

```bash
# Внимание: заменить на существующие реальные пути из find.
LIST='ПУТЬ_К_PREFILL_INPUT_LIST.txt'
REF='ПУТЬ_К_ЭТАЛОНУ/Result_0'

bash ../gemma4-cli-stage1/scripts/test_prefill.sh "$LIST" "$REF"
```

Скрипт выполняет два этапа:

1. `--check-inputs`: сверяет все 36 входов по QNN System metadata, без NPU.
2. `QnnGraph_execute` графа `qnn_partition_2` с сохранением всех 30 выходов и обязательным побайтовым сравнением с эталоном.

Искомый успешный финал:

```text
INPUTS_VALIDATED: 36 tensors; no NPU execution
QNN_MULTI_EXECUTE_OK: qnn_partition_2 inputs=36 outputs=30 ...
ALL_REFERENCES_MATCH: 30 output tensors
PREFILL_VERIFIED: 30/30 outputs byte-identical
```

Файлы результатов сохраняются в `~/gemma4_qnn_test/direct_result_prefill_128/Result_0/`, лог — в `~/gemma4_qnn_test/prefill_reference_test.log`. При несовпадении хотя бы одного байта скрипт завершается ошибкой, не выдавая `PREFILL_VERIFIED`.

Если сохранённого входного списка или эталона нет, сначала найдите их; **не создавайте нулевые входы и не сравнивайте с другим шагом**. Скрипт не меняет три QNN-бинарника.


### Транспортная ошибка QNN 14001 при тесте Prefill (2026-11-25)

При первом запуске `scripts/test_prefill.sh` проверены **36/36 входов** (строка `INPUTS_VALIDATED`), однако до выполнения `QnnGraph_execute` тест не дошёл: `deviceCreate: QNN error 14001`, `QNN_TRANSPORT_CONFIG crc32 failed`, `Failed to load skel, error: 1002`. **Результаты Prefill этим тестом ещё не подтверждены**; это ошибка транспорта DSP, а не несовпадение выходных тензоров.

Скрипт теперь запускается в изолированном окружении библиотек `sdk247`:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cd ~/gemma4_qnn_test
bash ../gemma4-cli-stage1/scripts/test_prefill.sh \\
  ./prefill_real_input_list.txt ./result_prefill_real_prompt/Result_0
```

На этот раз и stdout, и stderr записываются в `prefill_reference_test.log`. Если ошибка `14001` повторится, проверить отдельно инициализацию QNN HTP (без загрузки контекста):

```bash
cd ~/gemma4_qnn_test
LD_LIBRARY_PATH="$PWD/sdk247" ADSP_LIBRARY_PATH="$PWD/sdk247" \\
  ../gemma4-cli-stage1/build/gemma4-qnn-probe "$PWD/sdk247/libQnnHtp.so"
```

Если и `gemma4-qnn-probe` завершится ошибкой транспорта, сначала нужно разбираться с DSP/Skel/QAIRT и состоянием устройства, не меняя модель и её входы. Если probe успешен, но Prefill по-прежнему падает, потребуются полные логи QNN и анализ загрузки контекста.
