# Gemma 4 E2B — автономный CLI для Radxa Q8B

## Цель

Разработать `gemma4-cli` в стиле llama.cpp, непосредственно на Linux ARM64 Radxa Dragon Q8B (SC8280XP, HTP v68) **без Genie, ONNX Runtime, SSH и WSL во время инференса**. Сохраняем три AOT QNN context binary, созданные и проверенные с QAIRT 2.47:

- `qnn_partition_2_2.bin` — prefill_128 (`qnn_partition_2`)
- `qnn_partition_0_0.bin` — decode (`qnn_partition_0`)
- `qnn_partition_1_1.bin` — lm_head (`qnn_partition_1`)

Исходный полнофункциональный инференс уже проверен на запросе «Привет! Сколько будет 2+2?» (ответ: `2 + 2 будет **4**.`). В исходном прототипе CPU LiteRT выполнялся в WSL, а QNN graph execution — на Q8B.

## Что делает текущая поставка

**Stage 1 — `gemma4-qnn-probe`:** проверяет прямой вызов `QnnInterface_getProviders`, `backendCreate`, `deviceCreate`, `contextCreateFromBinary`, `graphRetrieve`, `contextFree` без Genie. Это диагностический первый этап, **не полноценный CLI генерации**. Бинарник компилируется из исходников на Q8B с заголовками QAIRT 2.47; наши `.bin` не изменяются.

### Сборка на Q8B

Нужны `g++`, `cmake`, заголовки *той же версии* QAIRT 2.47 в читаемой директории, например `~/qairt247-headers/include/QNN`. Заголовки можно скопировать из имеющегося у пользователя WSL SDK (копирование, а не изменение read-only SDK):

```bash
# В WSL, если заголовки отсутствуют на Q8B:
ssh radxa@192.168.0.161 'mkdir -p ~/qairt247-headers/include'
scp -r /mnt/e/radxa/qairt/2.47.0.260601/include/QNN \
  radxa@192.168.0.161:~/qairt247-headers/include/
```

Скопировать исходники этой папки на Q8B, затем в терминале Q8B:

```bash
cd ~/gemma4-cli-stage1
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DQAIRT_ROOT="$HOME/qairt247-headers"
cmake --build build -j2

cd ~/gemma4_qnn_test
export LD_LIBRARY_PATH="$PWD/sdk247:${LD_LIBRARY_PATH:-}"
export ADSP_LIBRARY_PATH="$PWD/sdk247"
../gemma4-cli-stage1/build/gemma4-qnn-probe \
  "$PWD/sdk247/libQnnHtp.so" \
  "$PWD/qnn_partition_1_1.bin" qnn_partition_1
```

Ожидаем `QNN_CONTEXT_OK: qnn_partition_1`. При ошибке `QNN_PROBE_ERROR` надо исправить API/окружение прежде чем строить верхние слои.

**Внимание:** исходники проверены на структуру и корректность вызовов по документации QNN, но сборка и исполнение на реальной Q8B не проводились в этом окружении. Версия SDK header ABI может требовать точечной адаптации.


## Stage 2 — прямое исполнение QNN (в разработке)

Добавлен `gemma4-qnn-run`: чтение I/O-метаданных QNN context binary и прямой `QnnGraph_execute()` для графов с одним входом и одним выходом (первый тест — LM Head). Команды сборки, запуска и побайтовой сверки с эталоном описаны в [docs/STAGE2.md](docs/STAGE2.md).

**Проверено на Radxa Q8B (2026-11-25):** сборка C++ и CTest (1/1), чтение QNN System метаданных LM Head, **`QnnGraph_execute` успешно запущен напрямую на NPU**, а все **262144 байта** логитов совпали с `qnn-net-run` (пара `real_result_decode_17` / `real_result_head_17`). Загрузка LM Head — 81.007 мс, время одного исполнения — 12.716 мс. Детали — в [docs/STAGE2.md](docs/STAGE2.md).

**Stage 3 (Decode и Prefill подтверждены на Q8B):** `gemma4-qnn-run` расширен до произвольного числа входных/выходных QNN-тензоров. Теперь можно напрямую подавать **ту же** `qnn-net-run` строку с путями к 38 входным файлам и побайтово сравнивать все 31 выхода Decode с эталоном. Инструкции — в [docs/STAGE3.md](docs/STAGE3.md). Тест Decode 38/31 успешно прошёл побайтовую сверку всех 31 выходов (`ALL_REFERENCES_MATCH: 31 output tensors`). Prefill-128 36/30 также прошёл строгую проверку: `PREFILL_VERIFIED: 30/30 outputs byte-identical`. Скрипт [scripts/test_prefill.sh](scripts/test_prefill.sh) сохранён. Рабочие AOT `.bin` не модифицируются.

**Stage 4 — persistent QNN contexts (3/3 подтверждены на Q8B):** в `gemma4-qnn-run` добавлен режим `--jobs`, позволяющий загрузить до трёх QNN-контекстов в одном процессе и побайтово проверить их выходы без повторного `contextCreateFromBinary` между вызовами. Одиночный persistent Prefill уже проверен (`PERSISTENT_GRAPH_OK`, `ALL_JOBS_REFERENCES_MATCH: 1 graphs 30 output tensors`, `Exit code: 0`, 54.8585 мс graphExecute). В попытке одновременной загрузки Prefill → Decode получен `QNN error 1002`: FastRPC не смог отобразить 1 887 436 800 байт весов Decode при ~13 ГиБ доступной RAM Linux. Добавлен [scripts/test_persistent_order.sh](scripts/test_persistent_order.sh) для проверки сочетаний и порядка загрузки с исходными эталонами. Порядок `decode head` успешно проверен: 32/32 выхода совпали. Порядок `decode prefill` также успешно проверен: оба контекста одновременно загружены, 61/61 выход совпал (`PERSISTENT_ORDER_VERIFIED: decode_prefill graphs=2 outputs=61`). Неудача `prefill decode` указывает на зависимость успешного размещения от порядка загрузки. Проверка `decode prefill head` также прошла: **3 одновременно загруженных контекста и 62/62 побайтово идентичных выхода** (`PERSISTENT_ORDER_VERIFIED: decode_prefill_head graphs=3 outputs=62`). **Stage 4.1 пройден на Q8B:** `--chain-decode-head` напрямую передал 1536-байтный hidden state из Decode в LM Head без файла, 262144 INT8 логитов получены за 9,6176 мс (диагностический argmax ID 568). [scripts/test_persistent_three.sh](scripts/test_persistent_three.sh) сохранён. Инструкция: [docs/STAGE4.md](docs/STAGE4.md). Сохранённые входы и эталоны обязательны; это ещё не динамический KV-cache/генерация.

**Stage 4.2 (проверен на Q8B):** опция `--kv-feedback-pos N` обновляет в RAM все 30 INT8 K/V-кэшей Decode по соответствующим срезам. K — страйдовая запись, V — последовательная; добавлен CTest `kv_cache_slot_layout`. Подтверждены `KV_MAPPING_OK: 30`, `KV_FEEDBACK_OK: pos=3 cache_tensors=30 inserted_bytes=9216` и 62/62 эталонных выхода. Опция `--kv-next-list` сверяет полученное состояние с сохранёнными кэшами следующего шага. См. [docs/STAGE4.md](docs/STAGE4.md).

**Stage 4.3 (новый режим, тест на Q8B впереди):** флаг `--kv-auto-pos --kv-next-list FILE` вычисляет позицию KV-записи из различий между двумя сохранёнными соседними состояниями Decode. Отвергает изменения в нескольких слотах, затем требует совпадения всех 30 обновлённых кэшей с эталоном. Не требует заранее угадывать `N`. Наличие настоящей пары соседних input-list ещё нужно подтвердить.

## Дорожная карта

1. **Stage 1 — backend/context smoke test** (исходный код). Выявить несовместимости C API с QAIRT 2.47, проверить время загрузки контекста.
2. **Stage 2 — QNN graph executor.** Прочитать метаданные I/O с помощью QNN System API, создать корректные Qnn_Tensor_t, выполнять LM Head через `graphExecute` и побайтово сравнить все 262144 INT8 логита с `qnn-net-run`.
3. **Stage 3 — прямой decode и prefill.** Подключить другие два контекста, сохранить корректное сопоставление 30 K/V-срезов (не по файловым суффиксам), проверить KV-cache и выходы против эталонного инференса. Контексты должны загружаться по одному разу.
4. **Stage 4 — локальные CPU-модели.** Перенести `Section0` embedder, `Section2` per-layer embedder, `Section1` AUX и `Section11` SentencePiece на Linux ARM64 Q8B (TFLite/LiteRT C++ и tokenizer). Сохранить двухфазный prefill+decode и padding mask=-1, embeddings=0.
5. **Stage 5 — `gemma4-cli` MVP.** Запуск `gemma4-cli -m ~/models/gemma4 -p '...' -n 64`, генерация токенов со streaming stdout, остановка 1/50/106, персистентный KV cache в RAM, измерение TTFT и токенов/с, `-i` interactive.
6. **Stage 6 — удобства llama.cpp.** `--temp`, `--top-k`, `--top-p`, `--seed`, системный prompt/шаблон чата, `--ctx-size`, `--repeat-penalty`, `/reset`, Ctrl-C, многоходовый чат, логи, бенчмарки, тесты регрессии. Функции мультимодальности, grammar/LoRA и API-сервера — отдельная фаза, не обещаем их в MVP.

Не используем `genie-t2t-run` в production path; он уже показал несовместимость интерфейса наших LiteRT AOT-графов с требованиями Genie.
# gemma_cli
