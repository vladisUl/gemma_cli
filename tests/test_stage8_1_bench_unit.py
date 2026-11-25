#!/usr/bin/env python3
"""Unit contracts for Stage 8.1 fixed decode, no QNN required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_stage8_1 import (  # noqa: E402
    BenchError, FIXED_TOKEN, KV_FEEDBACK_BYTES,
    calculate_medians, parse_fixed_log
)


def step_line(n, pos, token, cpu, npu, head, feedback, total):
    return (
        f"BENCH81_STEP: step={n} pos={pos} input_token={token}"
        f" cpu_nonkv_ms={cpu} npu_decode_ms={npu}"
        f" head_call_ms={head} kv_feedback_ms={feedback}"
        f" step_wall_ms={total} kv_updated=1 kv_tensors=30"
        f" kv_slice_bytes={KV_FEEDBACK_BYTES} no_sample_logits_used=1"
    )


def fixture():
    return "\n".join([
        "BENCH8_METADATA: ms=459 sample_inputs_read=0 saved_references_read=0",
        "BENCH8_PREFILL: real_tokens=13 graph_tokens=128 cpu_inputs_ms=36"
        " runtime_contexts_ms=703 npu_execute_ms=60 kv_bridge_ms=2 warmed_up=0",
        f"BENCH81_MODE: requested_steps=3 prefill_tokens=13"
        f" decode_first_token=107 decode_replay_token={FIXED_TOKEN}"
        " context_capacity=4096 synthetic_fixed_input=1 no_eos_stop=1"
        " no_argmax_tie_stop=1 kv_feedback_each_step=1"
        " excludes_context_load_from_decode_rate=1",
        "CHAIN_DECODE_HEAD_OK: head_execute_ms=11.0 argmax_ties=1",
        step_line(1, 13, 107, 4, 60, 12, 1, 77),
        "CHAIN_DECODE_HEAD_OK: head_execute_ms=10.0 argmax_ties=2",
        step_line(2, 14, FIXED_TOKEN, 3, 63, 11, 1, 78),
        "CHAIN_DECODE_HEAD_OK: head_execute_ms=11.0 argmax_ties=1",
        step_line(3, 15, FIXED_TOKEN, 3, 62, 12, 1, 78),
        "BENCH81_OK: executed_steps=3 kv_updates=3"
        " kv_tensors_per_step=30 last_written_pos=15"
        f" fixed_token={FIXED_TOKEN} prefill_tokens=13"
        " cpu_nonkv_total_ms=10 npu_decode_total_ms=185"
        " head_call_total_ms=35 kv_feedback_total_ms=3"
        " measured_steps_total_ms=233 first_step_ms=77"
        " decode_steps_per_s=12.8755 workload_wall_ms=234"
        " synthetic_not_text_generation=1",
    ])


class Stage81Unit(unittest.TestCase):
    def test_exact_replay_with_all_kv_writes(self):
        result = parse_fixed_log(fixture(), 3)
        self.assertEqual(result["steps"], 3)
        self.assertEqual(result["kv_updates"], 3)
        self.assertEqual(result["kv_total_feedback_bytes"], 3 * 9216)
        self.assertEqual(result["kv_last_written_position"], 15)
        self.assertEqual(result["decode_npu_ms"], 185)
        self.assertEqual(result["lm_head_graph_npu_ms"], 32)
        self.assertEqual(result["kv_feedback_ms"], 3)
        self.assertEqual(result["decode_steps_wall_ms"], 233)
        self.assertAlmostEqual(result["decode_steps_per_s"], 3000 / 233)
        self.assertTrue(result["not_text_generation"])

    def test_missing_qnn_head_rejected(self):
        corrupted = fixture().replace(
            "CHAIN_DECODE_HEAD_OK: head_execute_ms=10.0", "HEAD_MISSING: 10.0"
        )
        with self.assertRaisesRegex(BenchError, "Expected exactly"):
            parse_fixed_log(corrupted, 3)

    def test_missing_kv_feedback_rejected(self):
        corrupted = fixture().replace(
            "step=2 pos=14 input_token=236778",
            "step=2 pos=14 input_token=236778"
        ).replace("step_wall_ms=78 kv_updated=1",
                  "step_wall_ms=78 kv_updated=0", 1)
        with self.assertRaisesRegex(BenchError, "Missing KV feedback"):
            parse_fixed_log(corrupted, 3)

    def test_kv_position_skipped_rejected(self):
        corrupted = fixture().replace(
            "step=2 pos=14 input_token=236778",
            "step=2 pos=16 input_token=236778"
        )
        with self.assertRaisesRegex(BenchError, "Unexpected KV position"):
            parse_fixed_log(corrupted, 3)

    def test_ignored_eos_policy_must_be_declared(self):
        corrupted = fixture().replace("no_eos_stop=1", "no_eos_stop=0")
        with self.assertRaisesRegex(BenchError, "Missing synthetic-mode"):
            parse_fixed_log(corrupted, 3)

    def test_last_kv_slot_must_be_committed(self):
        corrupted = fixture().replace("last_written_pos=15",
                                       "last_written_pos=14")
        with self.assertRaisesRegex(BenchError, "last_written_pos"):
            parse_fixed_log(corrupted, 3)

    def test_wrong_replay_token_rejected(self):
        corrupted = fixture().replace(
            "step=3 pos=15 input_token=236778",
            "step=3 pos=15 input_token=107"
        )
        with self.assertRaisesRegex(BenchError, "Synthetic token stream"):
            parse_fixed_log(corrupted, 3)

    def test_median_of_individual_rates(self):
        b = parse_fixed_log(fixture(), 3)
        runs = [
            dict(b, decode_steps_per_s=x) for x in (8.0, 12.0, 60.0)
        ]
        result = calculate_medians(runs)
        self.assertEqual(result["decode_steps_per_s"], 12.0)


if __name__ == "__main__":
    unittest.main()
