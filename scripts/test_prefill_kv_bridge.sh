#!/usr/bin/env bash
# Stage 6.5: read-only in-RAM Prefill-128 KV -> Decode cache mapping probe.
# Uses original validated QNN samples but does NOT link actual Decode.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${GEMMA4_TEST_ROOT:-$HOME/gemma4_qnn_test}"

GEMMA4_PREFILL_KV_PROBE=1 bash "$ROOT/scripts/test_cpu_full_prefill_decode.sh"

LOG="$WORK/cpu_full_prefill_decode.log"
if [[ ! -f "$LOG" ]]; then
    echo "KV_PREFILL_PROBE_ERROR: missing log $LOG" >&2
    exit 1
fi
LINKS="$(grep -c '^KV_PREFILL_PROBE_LINK_OK:' "$LOG" || true)"
if [[ "$LINKS" != 30 ]]; then
    echo "KV_PREFILL_PROBE_ERROR: expected 30 mappings; got $LINKS" >&2
    exit 1
fi
grep -Fq 'KV_PREFILL_BRIDGE_PROBE_OK: tensors=30 layers=15 valid_tokens=17 padding_rows_ignored=111 copied_rows=17 next_decode_position=17 last_prompt_replay=0' "$LOG" || {
    echo "KV_PREFILL_PROBE_ERROR: missing strict block bridge marker" >&2
    exit 1
}
grep -Fq 'quantization_match=1 no_active_decode_mutation=1 NPU_decode_link_not_executed=1' "$LOG" || {
    echo "KV_PREFILL_PROBE_ERROR: missing quantization / no-mutation proof" >&2
    exit 1
}
echo "KV_PREFILL_STAGE6_5_OK: 30 source slices mapped, 17 valid token rows copied into scratch RAM, 111 padding rows excluded"
