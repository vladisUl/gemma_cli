#!/usr/bin/env bash
# Read-only diagnostics for Q8B CPU TFLite 2.14.1 / Gemma tokenizer assets.
set -euo pipefail
printf 'CPU_NATIVE_BEGIN: arch=%s\n' "$(uname -m)"
for cmd in g++ cmake pkg-config ldconfig nm readelf ldd dpkg-query; do
  if command -v "$cmd" >/dev/null 2>&1; then
    printf 'CPU_TOOL_AVAILABLE: %s=%s\n' "$cmd" "$(command -v "$cmd")"
  else
    printf 'CPU_TOOL_MISSING: %s\n' "$cmd"
  fi
done

echo 'CPU_TFLITE_HEADERS:'
have_cpp=0
have_c=0
for root in /usr/include /usr/local/include /opt "$HOME"; do
  [[ -d "$root" ]] || continue
  while IFS= read -r f; do
    case "$f" in
      */tensorflow/lite/interpreter.h)
        printf 'CPU_TFLITE_CPP_HEADER: %s\n' "$f"; have_cpp=1 ;;
      */tensorflow/lite/c/c_api.h)
        printf 'CPU_TFLITE_C_HEADER: %s\n' "$f"; have_c=1 ;;
    esac
  done < <(find "$root" -maxdepth 8 -type f \( \
      -path '*/tensorflow/lite/interpreter.h' -o \
      -path '*/tensorflow/lite/c/c_api.h' \) -print 2>/dev/null | sort)
done
if (( have_cpp == 0 )); then echo 'CPU_TFLITE_CPP_HEADERS_NOT_FOUND'; fi
if (( have_c == 0 )); then echo 'CPU_TFLITE_C_HEADERS_NOT_FOUND'; fi

echo 'CPU_TFLITE_RUNTIME_DETAILS:'
lib="${GEMMA4_TFLITE_LIB:-/usr/lib/aarch64-linux-gnu/libtensorflow-lite.so.2.14.1}"
if [[ -f "$lib" ]]; then
  printf 'CPU_TFLITE_RUNTIME_FILE: %s\n' "$lib"
  if command -v file >/dev/null 2>&1; then file -L "$lib"; fi
  if command -v readelf >/dev/null 2>&1; then
    readelf -d "$lib" 2>/dev/null | awk '/SONAME/ {print "CPU_TFLITE_SONAME:",$0}'
  fi
  if command -v ldd >/dev/null 2>&1; then
    ldd "$lib" 2>&1 | awk '/not found/ {print "CPU_TFLITE_DEPENDENCY_MISSING:",$0}'
  fi
  if command -v nm >/dev/null 2>&1; then
    nm -D -C "$lib" 2>/dev/null |
      awk '/tflite::InterpreterBuilder|tflite::Interpreter::GetSignatureRunner|tflite::SignatureRunner|TfLiteInterpreterCreate/ {
        if (n++ < 15) print "CPU_TFLITE_EXPORTED_SYMBOL:", $0
      }' || true
  fi
else
  printf 'CPU_TFLITE_RUNTIME_FILE_NOT_FOUND: %s\n' "$lib"
fi
for p in /usr/lib/aarch64-linux-gnu/libtensorflow-lite.so /usr/local/lib/libtensorflow-lite.so; do
  if [[ -e "$p" ]]; then
    printf 'CPU_TFLITE_LINKER_SYMLINK: %s -> %s\n' "$p" "$(readlink -f "$p")"
  fi
done
if command -v pkg-config >/dev/null 2>&1; then
  for pkg in tensorflow-lite tensorflowlite tensorflowlite_c; do
    if pkg-config --exists "$pkg"; then
      printf 'CPU_TFLITE_PKGCONFIG: %s\n' "$pkg"
      pkg-config --cflags --libs "$pkg" || true
    fi
  done
fi
if command -v dpkg-query >/dev/null 2>&1; then
  dpkg-query -W -f='CPU_TFLITE_DPKG: ${binary:Package} ${Version}\n' \
    'libtensorflow-lite*' 2>/dev/null || true
fi

echo 'CPU_TFLITE_MODEL_CANDIDATES:'
root="${GEMMA4_CPU_SCAN_ROOT:-$HOME}"
if [[ -d "$root" ]]; then
  found=0
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    printf 'CPU_ANY_TFLITE_MODEL: %s bytes=%s\n' "$p" "$(stat -c %s "$p")"
    found=1
  done < <(find "$root" -maxdepth 8 -type f \( \
           -iname '*.tflite' -o -iname '*.litert' -o \
           -iname '*tf_lite_aux*' -o -iname '*per_layer_embedder*' \) \
           -print 2>/dev/null | sort)
  if (( found == 0 )); then
    printf 'CPU_ANY_TFLITE_MODEL_NOT_FOUND: root=%s\n' "$root"
  fi
fi

echo 'CPU_GEMMA_TOKENIZER_METADATA:'
tokenizer="${GEMMA4_TOKENIZER_JSON:-$HOME/gemma4_qnn_test/tokenizer.json}"
if [[ -f "$tokenizer" ]] && command -v python3 >/dev/null 2>&1; then
  python3 - "$tokenizer" <<'PY'
import json
import sys
try:
    with open(sys.argv[1], encoding="utf-8") as stream:
        data = json.load(stream)
    model = data.get("model") or {}
    vocab = model.get("vocab")
    print("CPU_TOKENIZER_JSON:", sys.argv[1])
    print("CPU_TOKENIZER_MODEL_TYPE:", model.get("type", "(unspecified)"))
    print("CPU_TOKENIZER_VOCAB_SIZE:",
          len(vocab) if isinstance(vocab, (list, dict)) else "(unknown)")
    for key in ("normalizer", "pre_tokenizer", "decoder", "post_processor"):
        val = data.get(key)
        print("CPU_TOKENIZER_COMPONENT:", key,
              val.get("type", "(unspecified)") if isinstance(val, dict) else "(none)")
except (OSError, json.JSONDecodeError, TypeError) as exc:
    print("CPU_TOKENIZER_JSON_ERROR:", type(exc).__name__, str(exc))
PY
else
  echo 'CPU_TOKENIZER_JSON_CANNOT_INSPECT'
fi
echo 'CPU_NATIVE_COMPLETE: read-only; no model execution'
