#!/usr/bin/env bash
# Stage 6.6: first real QNN Prefill KV -> Decode -> LM Head execution.
# Does NOT use the independent saved Decode KV as the linked input.
# Token 568 is an explicitly chosen test probe, not the Prefill next token.
# Will refuse linked mode if the original 28 Prefill initial caches are nonzero.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA4_PREFILL_KV_PROBE=1 GEMMA4_PREFILL_KV_RUN=1 \
  bash "$ROOT/scripts/test_cpu_full_prefill_decode.sh"

# Outer marker only printed if every byte-check and linked execution passed.
echo "PREFILL_LINK_WRAPPER_OK: Stage 6.6 completed with linked NPU execution"
