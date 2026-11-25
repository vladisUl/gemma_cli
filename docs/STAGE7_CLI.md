# Stage 7.3 — Gemma 4 terminal interface (Q8B)

This is a thin **user-facing CLI** built on the successfully tested Stage 7.2
dynamic inference path; not a new QNN model nor an API service.

## Confirmed hardware baseline: Stage 7.2

On Radxa Dragon Q8B, QAIRT 2.47.0.260601:

- A new BPE prompt was accepted (no saved Decode/Prefill QNN reference inputs).
- Three persistent graphs were loaded in a single generation process:
  Prefill-128, Decode and LM Head.
- First NPU answer: `2 + 2 будет **4**.`, exact 9 ID golden control.
- Second NPU answer from new text `Сколько будет 3 + 5?`:
  `3 + 5 = **8**`, 9 unique INT8 argmax predictions and EOS106.
- All embeddings, RoPE/masks from local CPU; KV in local Q8B RAM.
- This does **not** validate arbitrary open-ended accuracy, multi-turn
  chat, or a real FP32 sampler. New text may encounter tied INT8 maxima.

## Stage 7.3: minimal terminal frontend

Programs:

- `scripts/gemma4_cli.py`: external launcher / interactive frontend
- `scripts/gemma4_decode_log.py`: log-to-text parser, separate subprocess
- `tests/test_stage7_3_cli_unit.py`: no-QNN parser regression tests

Example one-shot question:

```bash
cd ~/gemma4-cli-stage1
git pull --ff-only origin main
cmake --build build -j2
ctest --test-dir build --output-on-failure
python3 scripts/gemma4_cli.py --text 'Сколько будет 3 + 5?'
```

Interactive prompt loop:

```bash
python3 scripts/gemma4_cli.py
# Вы> Сколько будет 3 + 5?
# Gemma 4> 3 + 5 = **8**
# Вы> /exit
```

`--max-tokens` defaults to 32, allowed range 1–32, consistent with the
existing `--dynamic-max-tokens` QNN flag. The result is explicitly marked
incomplete when QNN stops at the limit or an ambiguous INT8 max. An
ambiguous argmax ID is **never** inserted into the user-visible text.

Architecture per QUESTION:
1. Separate stdlib-only BPE Python subprocess loads `tokenizer.json`,
   independently validates the original 9-ID golden input and creates
   a new 128×int32 prompt file under a private temporary directory.
2. BPE subprocess exits; `gemma4-qnn-run --dynamic-prefill` loads the
   actual three AOT contexts, executes one response and writes its output
   into a temporary log (not the historical `Result_0` files).
3. QNN subprocess exits and releases resources; a separate decoder process
   loads `tokenizer.json` to map IDs to text according to
   Replace(▁→space),ByteFallback,Fuse.
4. The frontend prints only the response; temporary prompt/log are deleted
   on success. On QNN or decoder errors, a uniquely named diagnostic log
   is saved to `~/gemma4_qnn_test/gemma4_cli_error_*.log`.

**Limitations:** each question reloads all QNN contexts and uses a completely
fresh prompt and KV cache. The frontend is interactive but it is NOT a
multi-turn, persistent-context chat server. Do not interpret test success
as validation across all BPE inputs or as a general benchmark.

The existing read-only diagnostic scripts remain available unchanged:
`scripts/test_dynamic_text_stage7_2.sh`,
`scripts/test_correct_header_generation_eos.sh`, etc. No QNN
context binary, tokenizer JSON, saved input RAW, or benchmark reference
is modified by this CLI.

Useful commands:

```bash
python3 scripts/gemma4_cli.py --help
python3 scripts/gemma4_cli.py --text 'Как тебя зовут?' --max-tokens 32
python3 scripts/gemma4_cli.py --text 'Сколько будет 3 + 5?'
python3 scripts/gemma4_cli.py
```

**Status:** Stage 7.3 code committed, not yet compiled/tested on Q8B.
