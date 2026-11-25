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

**Stage 4.3 (проверен на Q8B):** флаг `--kv-auto-pos --kv-next-list FILE` вычисляет позицию KV-записи из различий между двумя сохранёнными соседними состояниями Decode. Отвергает изменения в нескольких слотах, затем требует совпадения всех 30 обновлённых кэшей с эталоном. Не требует заранее угадывать `N`. Проверено на `auto_decode_step3_list.txt` → `auto_decode_step4_list.txt`: определена позиция `3`, все 30 полных KV-cache побайтово совпали после обновления 9216 байт (`KV_NEXT_CACHE_MATCH`).

**Stage 4.4 полностью проверен на Q8B (два Decode + два LM Head):** флаг `--kv-run-next` позволяет выполнить **второй** Decode в том же процессе и том же QNN context: 30 KV-cache обновляются напрямую в RAM, а следующие 8 не-KV входов читаются из сохранённого шага 4. Первое двухшаговое исполнение прошло: 30/30 K/V-кэшей совпали с шагом 4, второй Decode выполнен за 78.323 мс, LM Head повторно получил 262144 логита. Найдены эталоны `auto_result_decode_4/Result_0` и `auto_result_head_4/Result_0`; теперь `--kv-next-reference-dir` проверит все 31 выхода второго Decode, а новый `--kv-next-head-reference-dir` — все 262144 байта второго LM Head. **Обе строгие проверки прошли:** `DECODE_NEXT_REFERENCE_MATCH: 31 output tensors byte-identical` и `DECODE_NEXT_HEAD_REFERENCE_MATCH: 262144 logits bytes identical`. Сохранённый шаг 3 → KV-feedback → повторный Decode шага 4 → LM Head шага 4 воспроизводится на Q8B побайтово. См. [docs/STAGE4.md](docs/STAGE4.md).

**Stage 5 CPU (первый Section1 `decode_rope` успешно выполнен на Q8B):** определить на Q8B наличие исходных Section0/Section1/Section2 TFLite-моделей, токенизатора и совместимой CPU LiteRT/TFLite-библиотеки; утилита [scripts/probe_cpu_stack.sh](scripts/probe_cpu_stack.sh) не меняет модель и не требует установки пакетов. Результаты: найдена системная `libtensorflow-lite.so.2.14.1` (C API, заголовки отсутствуют), `tokenizer.json` с **BPE**, vocab 262144; исходные Section0/1/2 `.tflite` на Q8B пока не обнаружены. Новый `gemma4-cpu-abi` проверяет наличие 27 символов C API для запуска именованных сигнатур через `dlopen/dlsym` без заголовков. Вычислены четыре INT8 RoPE-выхода (512+512+256+256 байт) через CPU TFLite 2.14.1, без WSL. Для побайтовой сверки с исходными четырьмя входами QNN Decode создан [scripts/verify_cpu_aux.py](scripts/verify_cpu_aux.py). Все четыре RoPE-тензора, по подтверждению пользователя, побайтово совпали с сохранёнными QNN-входами; Stage 5.2 закрыт. `gemma4-cpu-abi` подтвердил на Q8B `CPU_C_ABI_READY: required=27 missing=0`, включая именованные TFLite-сигнатуры. Добавлен новый `gemma4-cpu-run` для CPU-инференса сигнатур через C API без TensorFlow-заголовков (`--list`, `--signature`, `--i32`, `--input`, `--expect`); модель Section1 ещё нужно перенести один раз на плату. План — [docs/STAGE5_CPU.md](docs/STAGE5_CPU.md).

**Stage 5.3 (следующая проверка):** новый [scripts/test_cpu_mask.sh](scripts/test_cpu_mask.sh) запускает TFLite `decode_mask` и сравнивает две маски по 4097 байт с соответствующими сохранёнными QNN-входами шага 3. По умолчанию использует тестовый непустой токен 2, не выдавая его за фактический token ID сохранённой последовательности. 

**Stage 5.3 (проверен на Q8B):** обе INT8-маски (`mask_local`, `mask_global`, по 4097 байт) совпали с сохранёнными входами QNN шага 3. Вместе с RoPE подтверждены все шесть CPU AUX-тензоров, **9730 байт**.

**Stage 5.4 (код добавлен, тест на Q8B впереди):** `--cpu-aux-model ... --cpu-aux-pos 3 --cpu-aux-token 2` выполняет `Section1` в том же процессе, напрямую заменяет шесть входов QNN Decode на CPU-выходы в RAM (только после строгой сверки с сохранёнными входами), затем повторяет проверку всех 62 выходных QNN-тензоров. См. [docs/STAGE5_CPU.md](docs/STAGE5_CPU.md).

**Stage 5.4 (проверен на Q8B):** 6 AUX тензоров (9730 байт) рассчитаны в том же C++ процессе через CPU TFLite Section1, сверены с входами QNN Decode и поданы напрямую в RAM; все 62 выхода трёх QNN-графов совпали побайтово с эталонами.

**Stage 5.5 (код в GitHub, проверка Q8B впереди):** при `--kv-run-next` новые `--cpu-aux-next-pos 4 --cpu-aux-next-token 2` позволяют заменить **ещё шесть AUX-входов второго Decode** вычислениями CPU Section1; 30 K/V-кэшей второго шага берутся из обратной связи в RAM. Готовый тест — [scripts/test_cpu_aux_two_step.sh](scripts/test_cpu_aux_two_step.sh).

**Stage 5.5 (проверен на Q8B):** два последовательных Decode используют 6 CPU AUX-тензоров на каждом шаге, KV-feedback обновляет все 30 кэшей в RAM. Повторный Decode дал 31/31 точных выходов, повторный LM Head — все 262144 байта логитов в точности. Следующий блокер: **обычные и per-layer embeddings**. Подготовлен [scripts/test_cpu_embedder.sh](scripts/test_cpu_embedder.sh) для проверки исходных Section0/Section2 по отдельности, после однократного переноса моделей на Q8B.

**Stage 5.6 (выявлена блокировка):** оригинальная CPU Section0 требует `EMBEDDING_LOOKUP v4`, не зарегистрированный в системной TFLite 2.14.1. Сборка C++ успешна; Stage 5.5 не затронут. Добавлен [scripts/inspect_tflite_ops.py](scripts/inspect_tflite_ops.py) для просмотра версий операторов и INT4/квантизации без сторонних пакетов. CPU embedder-тест принимает `GEMMA4_TFLITE_LIB` для отдельной совместимой ARM64 библиотеки. Не заменять системный runtime.

**Stage 5.7 (экспериментальный код, не проверен на Q8B):** обе исходные модели используют `EMBEDDING_LOOKUP v4`: Section0 — **INT2**, Section2 — 35 таблиц **INT4**, всё с 262144 row-scales. Добавлен [src/packed_embedder.cpp](src/packed_embedder.cpp), прямой `mmap`/unpack/dequantize/MUL/CONCATENATION без новых TFLite библиотек, и [scripts/test_packed_embedder.sh](scripts/test_packed_embedder.sh). Сначала тест совместимости на реальных моделях, затем обязательная побайтовая сверка с QNN embeddings.

**Stage 5.7.1 (исправление после теста Q8B):** `gemma4-packed-embed` собирается, но Section0 после `MUL` завершался ошибкой из-за непереданного завершающего `RESHAPE`. Исправлено в [src/packed_embedder.cpp](src/packed_embedder.cpp), ожидается повторное испытание и последующая побайтовая проверка.

**Stage 5.7 (smoke-тесты Section0 и Section2 подтверждены на Q8B):** обе нативные C++ распаковки INT2/INT4 создают FLOAT32 нужной длины. Следующая строгая проверка — **Stage 5.8**: [scripts/verify_packed_embeddings.py](scripts/verify_packed_embeddings.py) автоматически ищет реальный token ID сохранённого QNN Decode по знакам INT2 и проверяет все 6144 байта Section0, затем все 35840 байт Section2 отдельно для шагов 3 и 4. Проверка ещё не запущена на Q8B.

**Stage 5.8 (проверен на Q8B):** идентифицирован токен **568** для обоих Decode шагов 3/4, нативный Section0 INT2 совпал с QNN входом по всем 6144 байтам и Section2 INT4 по всем 35840 байтам **для каждого шага**. **Stage 5.9 (код в GitHub, ожидает проверку на Q8B):** прямое CPU Section0/1/2 → QNN в RAM для всех восьми не-KV входов обоих Decode; готов [scripts/test_cpu_full_nonkv_two_step.sh](scripts/test_cpu_full_nonkv_two_step.sh). Начальные 30 K/V и тестовые Prefill входы пока из сохранённых файлов.

**Stage 5.9 (Q8B подтверждён, 25.11.2026):** оба последовательных Decode получают все **8 non-KV CPU входов** напрямую в RAM (Section1 9730 байт + Section0/2 41984 байта), 30 K/V между шагами обновляются через CPU feedback; второй Decode 31/31 выходов и LM Head 262144/262144 байта совпали с QNN reference. **Stage 6.0** готовит Prefill-128: по QNN metadata у него 36 входов — 28 K/V и 8 non-KV. Добавлен [scripts/probe_prefill_metadata.py](scripts/probe_prefill_metadata.py) для проверки исходных CPU prefill-сигнатур и размеров сохранённых QNN входов перед соединением Prefill→Decode. Это пока диагностика, не автономная генерация.

**Stage 6.0 подтверждён на Q8B:** 36 входов Prefill-128 = 28 K/V + 8 non-KV, метаданные всех трёх исходных CPU-моделей доступны. **Stage 6.1 (новый код, ещё не проверен):** [scripts/recover_prefill_tokens.py](scripts/recover_prefill_tokens.py) однократно сканирует INT2 таблицу и проверяет 128 сохранённых Prefill embeddings по 6144 байт каждый, чтобы определить реальные 128 token IDs без догадок и без изменения входных эталонов.

## Дорожная карта

1. **Stage 1 — backend/context smoke test** (исходный код). Выявить несовместимости C API с QAIRT 2.47, проверить время загрузки контекста.
2. **Stage 2 — QNN graph executor.** Прочитать метаданные I/O с помощью QNN System API, создать корректные Qnn_Tensor_t, выполнять LM Head через `graphExecute` и побайтово сравнить все 262144 INT8 логита с `qnn-net-run`.
3. **Stage 3 — прямой decode и prefill.** Подключить другие два контекста, сохранить корректное сопоставление 30 K/V-срезов (не по файловым суффиксам), проверить KV-cache и выходы против эталонного инференса. Контексты должны загружаться по одному разу.
4. **Stage 4 — локальные CPU-модели.** Перенести `Section0` embedder, `Section2` per-layer embedder, `Section1` AUX и `Section11` SentencePiece на Linux ARM64 Q8B (TFLite/LiteRT C++ и tokenizer). Сохранить двухфазный prefill+decode и padding mask=-1, embeddings=0.
5. **Stage 5 — `gemma4-cli` MVP.** Запуск `gemma4-cli -m ~/models/gemma4 -p '...' -n 64`, генерация токенов со streaming stdout, остановка 1/50/106, персистентный KV cache в RAM, измерение TTFT и токенов/с, `-i` interactive.
6. **Stage 6 — удобства llama.cpp.** `--temp`, `--top-k`, `--top-p`, `--seed`, системный prompt/шаблон чата, `--ctx-size`, `--repeat-penalty`, `/reset`, Ctrl-C, многоходовый чат, логи, бенчмарки, тесты регрессии. Функции мультимодальности, grammar/LoRA и API-сервера — отдельная фаза, не обещаем их в MVP.

Не используем `genie-t2t-run` в production path; он уже показал несовместимость интерфейса наших LiteRT AOT-графов с требованиями Genie.
# gemma_cli
