#!/usr/bin/env bash
# Stage 6.7: Prefill -> Decode(pos17, probe token568) -> Head ->
# KV-feedback -> Decode(pos18, token from previous Head argmax) -> Head.
# Experimental only: first input 568 was not predicted by Prefill.
# Preserves all original reference files, and uses a separate log.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA4_PREFILL_KV_PROBE=1 GEMMA4_PREFILL_KV_RUN=1 GEMMA4_PREFILL_KV_STEPS=2 \
    bash "$ROOT/scripts/test_cpu_full_prefill_decode.sh"
echo "PREFILL_MULTISTEP_WRAPPER_OK: Stage 6.7 linked two Decode passes completed"
