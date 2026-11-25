#!/usr/bin/env python3
"""Simple terminal frontend for Gemma 4 E2B-it on Radxa Dragon Q8B.

Examples:
  python3 scripts/gemma4_cli.py --text "Сколько будет 3 + 5?"
  python3 scripts/gemma4_cli.py

The QNN contexts currently reload once per QUESTION. This is not yet a
single persistent multi-turn conversation: each question starts with a fresh
KV cache and a single-user-turn chat template. No network or extra packages.

The standalone BPE subprocess exits BEFORE QNN loads; the standalone decoder
runs AFTER QNN exits. That avoids simultaneous high Python/QNN memory use.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORK = Path.home() / "gemma4_qnn_test"


class CliError(RuntimeError):
    pass


def check_models(work):
    files = [
        work / "sdk247/libQnnSystem.so",
        work / "sdk247/libQnnHtp.so",
        work / "persistent_jobs_decode_prefill_head.txt",
        work / "tokenizer.json",
        work / "derived_prefill_128_token_ids_i32.raw",
        work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite",
        work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite",
        work / "cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite",
    ]
    for path in files:
        if not path.is_file():
            raise CliError("Required input missing: " + str(path))


def runner_ready(runner):
    if not runner.is_file() or not os.access(runner, os.X_OK):
        raise CliError("QNN binary missing: " + str(runner))
    help_cmd = subprocess.run(
        [str(runner), "--help"], capture_output=True, text=True,
        check=False
    )
    if help_cmd.returncode != 0 or "--dynamic-prefill" not in help_cmd.stdout:
        raise CliError("Outdated QNN binary: rebuild using cmake --build build -j2")


def save_error_log(work, log):
    destination = work / ("gemma4_cli_error_" + uuid.uuid4().hex[:12] + ".log")
    with destination.open("xb") as target:
        with log.open("rb") as original:
            shutil.copyfileobj(original, target)
    return destination


def generate(text, work, runner, max_tokens):
    """Run exactly one fresh request, return decoded response JSON."""
    if not text.strip():
        raise CliError("Empty user question")
    with tempfile.TemporaryDirectory(prefix="gemma4_cli_", dir=work) as dirname:
        scratch = Path(dirname)
        token_path = scratch / "prompt.raw"
        log_path = scratch / "qnn.log"
        tokenizer_path = work / "tokenizer.json"

        # Subprocess BPE avoids holding huge merge tables during NPU execution.
        encode = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gemma4_bpe.py"),
             "--tokenizer", str(tokenizer_path),
             "--golden-raw",
             str(work / "derived_prefill_128_token_ids_i32.raw"),
             "--text", text, "--output-new-raw", str(token_path)],
            capture_output=True, text=True, check=False
        )
        if encode.returncode:
            message = (encode.stderr or encode.stdout).strip()
            raise CliError("BPE encoder rejected request: " + message[-1200:])
        if token_path.stat().st_size != 512:
            raise CliError("BPE prompt RAW not 512 bytes")

        sdk = str(work / "sdk247")
        env = os.environ.copy()
        env["LD_LIBRARY_PATH"] = sdk
        env["ADSP_LIBRARY_PATH"] = sdk
        command = [
            str(runner),
            "--system", str(work / "sdk247/libQnnSystem.so"),
            "--backend", str(work / "sdk247/libQnnHtp.so"),
            "--jobs", "persistent_jobs_decode_prefill_head.txt",
            "--dynamic-prefill",
            "--dynamic-prompt-tokens", str(token_path),
            "--dynamic-section0", str(
                work / "cpu_models/Section0_TFLiteModel_tf_lite_embedder.tflite"),
            "--dynamic-section1", str(
                work / "cpu_models/Section1_TFLiteModel_tf_lite_aux.tflite"),
            "--dynamic-section2", str(
                work / "cpu_models/Section2_TFLiteModel_tf_lite_per_layer_embedder.tflite"),
            "--dynamic-max-tokens", str(max_tokens),
        ]
        with log_path.open("wb") as log_file:
            status = subprocess.run(
                command, cwd=work, env=env, stdout=log_file,
                stderr=subprocess.STDOUT, check=False
            )
        if status.returncode:
            diagnostics = log_path.read_text(errors="replace").splitlines()
            error = next((line for line in reversed(diagnostics)
                          if "QNN_RUN_ERROR:" in line), "")
            saved = save_error_log(work, log_path)
            raise CliError(
                "QNN execution failed (exit " + str(status.returncode) +
                "). " + error + " Log: " + str(saved)
            )

        # Decoder loads BPE JSON only AFTER QNN has released its contexts.
        decoded = subprocess.run(
            [sys.executable, str(ROOT / "scripts/gemma4_decode_log.py"),
             "--log", str(log_path), "--tokenizer", str(tokenizer_path)],
            cwd=work, capture_output=True, text=True, check=False
        )
        if decoded.returncode:
            saved = save_error_log(work, log_path)
            raise CliError("QNN output parsing failed: " +
                           (decoded.stderr or decoded.stdout).strip()[-900:] +
                           ". Log: " + str(saved))
        try:
            result = json.loads(decoded.stdout)
        except json.JSONDecodeError as exc:
            raise CliError("Malformed generation JSON: " + str(exc)) from exc
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--text", help="One-shot question; omit for terminal loop")
    parser.add_argument("--max-tokens", type=int, default=32,
                        help="Max QNN Decode predictions per question, 1..32")
    parser.add_argument("--workdir", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--runner", type=Path,
                        default=ROOT / "build/gemma4-qnn-run")
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 32:
        parser.error("--max-tokens must be between 1 and 32")
    work = args.workdir.expanduser().resolve()
    runner = args.runner.expanduser().resolve()
    if not work.is_dir():
        raise CliError("Work directory not found: " + str(work))
    check_models(work)
    runner_ready(runner)

    def ask(question):
        result = generate(question, work, runner, args.max_tokens)
        print("Gemma 4> " + result["text"], flush=True)
        if result["stop_reason"] == "limit":
            print("[Лимит токенов: ответ может быть неполным.]", flush=True)
        elif result["stop_reason"] == "tie":
            print("[Остановка: неоднозначный INT8 максимум; "
                  "токен не выбирался произвольно.]", flush=True)
        return result

    if args.text is not None:
        ask(args.text)
        return
    print("Gemma 4 E2B-it / Radxa Q8B (experimental, one fresh turn per query)")
    print("Введите вопрос. Для завершения: /exit")
    while True:
        try:
            line = input("Вы> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.strip().lower() in {"/exit", "/quit", "выход"}:
            break
        if not line.strip():
            continue
        try:
            ask(line)
        except CliError as exc:
            print("Ошибка: " + str(exc), file=sys.stderr, flush=True)


if __name__ == "__main__":
    try:
        main()
    except (CliError, OSError, subprocess.SubprocessError) as exc:
        print("GEMMA4_CLI_ERROR:", exc, file=sys.stderr)
        raise SystemExit(1)
