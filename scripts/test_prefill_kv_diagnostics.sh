#!/usr/bin/env bash
# Stage 6.10: isolate the 6975 / 9216 last prompt token KV byte mismatch.
# Per-layer raw signed INT8 delta histogram, row search, shape+scale+offset
# compatible layer search. All inference/caches/initial prompt unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"
GEMMA4_PREFILL_KV_DIAGNOSE=1 bash "$ROOT/scripts/test_prefill_first_token.sh"
LOG="$WORK/cpu_prefill_kv_diagnostic_stage6_10.log"
[[ -f "$LOG" ]] || { echo "PREFILL_KV_DIAG_ERROR: log not found" >&2; exit 1; }
count="$(grep -c '^PREFILL_KV_DIAG_TENSOR:' "$LOG" || true)"
[[ "$count" == 30 ]] || {
    echo "PREFILL_KV_DIAG_ERROR: expected 30 per-tensor reports, got $count" >&2
    exit 1
}
grep -Fq 'PREFILL_KV_DIAG_SUMMARY: tensors=30 compared_bytes=9216' "$LOG" || {
    echo "PREFILL_KV_DIAG_ERROR: summary missing" >&2; exit 1
}
grep -Fq 'no_relabel_or_patch_applied=1 diagnostic_only=1' "$LOG" || {
    echo "PREFILL_KV_DIAG_ERROR: read-only mode not confirmed" >&2; exit 1
}
echo "PREFILL_KV_STAGE6_10_OK: all 30 KV tensors inspected; original reference data unchanged"
grep -E '^(PREFILL_KV_DIAG_SUMMARY:|PREFILL_LAST_PROMPT_KV_COMPARE:)' "$LOG"
