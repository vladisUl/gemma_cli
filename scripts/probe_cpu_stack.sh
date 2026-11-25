#!/usr/bin/env bash
# Read-only inventory for local Gemma 4 CPU LiteRT/TFLite components.
set -euo pipefail

ROOT="${GEMMA4_CPU_SCAN_ROOT:-$HOME}"
DEPTH="${GEMMA4_CPU_SCAN_DEPTH:-7}"

if (( $# )); then
  echo 'Usage: GEMMA4_CPU_SCAN_ROOT=/path GEMMA4_CPU_SCAN_DEPTH=7 bash scripts/probe_cpu_stack.sh' >&2
  exit 2
fi
if [[ ! -d "$ROOT" ]]; then
  printf 'CPU_PROBE_ERROR: scan root does not exist: %s\n' "$ROOT" >&2
  exit 1
fi
if [[ ! "$DEPTH" =~ ^[0-9]+$ ]] || (( DEPTH < 1 || DEPTH > 12 )); then
  echo 'CPU_PROBE_ERROR: GEMMA4_CPU_SCAN_DEPTH must be 1..12' >&2
  exit 1
fi

printf 'CPU_PROBE_BEGIN: root=%s depth=%s\n' "$ROOT" "$DEPTH"
printf 'CPU_ARCH: %s\n' "$(uname -m)"
printf 'CPU_KERNEL: %s\n' "$(uname -r)"

echo 'CPU_MODELS:'
for idx in 0 1 2; do
  found=0
  while IFS= read -r path; do
    [[ -n "$path" ]] || continue
    printf 'CPU_MODEL_SECTION%s: %s bytes=%s\n' \
      "$idx" "$path" "$(stat -c %s "$path")"
    found=1
  done < <(find "$ROOT" -maxdepth "$DEPTH" -type f \
                -iname "Section${idx}*.tflite" -print 2>/dev/null | sort)
  if (( found == 0 )); then
    printf 'CPU_MODEL_SECTION%s_NOT_FOUND: under selected scan root\n' "$idx"
  fi
done

echo 'CPU_TOKENIZER_FILES:'
found=0
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  printf 'CPU_TOKENIZER_CANDIDATE: %s bytes=%s\n' \
    "$path" "$(stat -c %s "$path")"
  found=1
done < <(find "$ROOT" -maxdepth "$DEPTH" -type f \( \
            -iname 'Section11*' -o -iname '*sentencepiece*.model' -o \
            -iname '*tokenizer*.model' -o -iname '*.spm' -o \
            -iname 'tokenizer.json' \) -print 2>/dev/null | sort)
if (( found == 0 )); then
  echo 'CPU_TOKENIZER_NOT_FOUND: under selected scan root'
fi

echo 'CPU_RUNTIME_LDCONFIG:'
if command -v ldconfig >/dev/null 2>&1; then
  ldconfig -p 2>/dev/null | grep -E 'libtensorflowlite_c|libtensorflow-lite|liblitert' ||
    echo 'CPU_RUNTIME_NOT_REGISTERED: no TFLite/LiteRT C library in ldconfig'
else
  echo 'CPU_RUNTIME_LDCONFIG_UNAVAILABLE'
fi
echo 'CPU_RUNTIME_FILES:'
found=0
while IFS= read -r path; do
  [[ -n "$path" ]] || continue
  printf 'CPU_RUNTIME_CANDIDATE: %s\n' "$path"
  if command -v file >/dev/null 2>&1; then
    file -L "$path" || true
  fi
  found=1
done < <(find "$ROOT" -maxdepth "$DEPTH" -type f \( \
           -name 'libtensorflowlite_c.so*' -o -name 'libtensorflow-lite.so*' \
           -o -name 'liblitert*.so*' \) -print 2>/dev/null | sort)
if (( found == 0 )); then
  echo 'CPU_RUNTIME_NOT_FOUND_IN_SCAN'
fi

echo 'CPU_PYTHON_PACKAGES:'
if command -v python3 >/dev/null 2>&1; then
  python3 - <<'PY'
from importlib.util import find_spec
for name in ("ai_edge_litert", "tflite_runtime", "tensorflow", "sentencepiece"):
    try:
        yes = find_spec(name) is not None
        print("CPU_PYTHON_PACKAGE:", name, "available" if yes else "missing")
    except Exception as exc:
        print("CPU_PYTHON_PACKAGE_ERROR:", name, type(exc).__name__)
PY
else
  echo 'CPU_PYTHON_UNAVAILABLE'
fi
echo 'CPU_PROBE_COMPLETE: no inference or file modification'
