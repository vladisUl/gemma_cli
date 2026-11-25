#!/usr/bin/env python3
"""Unit tests for Stage 8.0 timing parser, no proprietary QNN libs needed."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from bench_stage8 import BenchError, parse_benchmark, median_metrics  # noqa: E402


def fake_log():
    return "\n".join([
        "BENCH8_METADATA: ms=9 sample_inputs_read=0 saved_references_read=0",
        "BENCH8_PREFILL: real_tokens=17 graph_tokens=128"
        " cpu_inputs_ms=11 runtime_contexts_ms=101"
        " npu_execute_ms=20 kv_bridge_ms=5 warmed_up=0",
        "BENCH8_FIRST_TOKEN: from_job_start_ms=170"
        " includes_metadata_cpu_and_load=1",
        "CHAIN_DECODE_HEAD_OK: head_execute_ms=8.0",
        "DYNAMIC_GENERATED_TOKEN: step=1 response_pos=18"
        " id=236778 max_q=119 ties=1",
        "BENCH8_STEP: step=1 cpu_nonkv_ms=4 npu_decode_ms=15"
        " head_call_ms=8.5 sampling_and_logging_ms=1"
        " kv_feedback_ms=1 step_wall_ms=30 feedback_written=1"
        " stop=continue",
        "CHAIN_DECODE_HEAD_OK: head_execute_ms=7.0",
        "DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19"
        " id=106 max_q=120 ties=1",
        "BENCH8_STEP: step=2 cpu_nonkv_ms=4 npu_decode_ms=14"
        " head_call_ms=7.5 sampling_and_logging_ms=1"
        " kv_feedback_ms=0 step_wall_ms=27 feedback_written=0"
        " stop=eos",
    ])


RESPONSE = {"steps": 2, "ids": [236778, 106], "stop_reason": "eos"}


class Stage8Unit(unittest.TestCase):
    def test_counts_and_rates(self):
        r = parse_benchmark(fake_log(), RESPONSE)
        self.assertEqual(r["real_prefill_tokens"], 17)
        self.assertAlmostEqual(r["metadata_ms"], 9)
        self.assertAlmostEqual(r["npu_prefill_ms"], 20)
        self.assertAlmostEqual(r["decode_steps_wall_ms"], 57)
        self.assertAlmostEqual(r["decode_npu_graph_ms"], 29)
        self.assertAlmostEqual(r["lm_head_npu_graph_ms"], 15)
        self.assertAlmostEqual(r["generation_text_tok_s"], 1000 / 57)
        self.assertAlmostEqual(r["npu_prefill_effective_tok_s"], 850)
        self.assertAlmostEqual(r["decode_steps_after_first_per_s"], 1000 / 27)
        self.assertFalse(r["warmed_up"])

    def test_reject_missing_step_marker(self):
        log = fake_log().replace("BENCH8_STEP: step=2", "BROKEN: step=2")
        with self.assertRaisesRegex(BenchError, "counts disagree"):
            parse_benchmark(log, RESPONSE)

    def test_reject_missing_head_graph_execute(self):
        log = fake_log().replace("CHAIN_DECODE_HEAD_OK: head_execute_ms=7.0",
                                 "BROKEN_HEAD: head_execute_ms=7.0")
        with self.assertRaisesRegex(BenchError, "counts disagree"):
            parse_benchmark(log, RESPONSE)

    def test_reject_false_warmup(self):
        with self.assertRaisesRegex(BenchError, "hidden warmup"):
            parse_benchmark(fake_log().replace("warmed_up=0", "warmed_up=1"),
                            RESPONSE)

    def test_reject_negative_prefill_time(self):
        with self.assertRaisesRegex(BenchError, "bad metric"):
            parse_benchmark(fake_log().replace("npu_execute_ms=20",
                                               "npu_execute_ms=-20"),
                            RESPONSE)

    def test_reject_incorrect_summary_step_count(self):
        with self.assertRaisesRegex(BenchError, "differs"):
            parse_benchmark(fake_log(),
                            {"steps": 1, "ids": [236778], "stop_reason": "limit"})

    def test_median_not_mean(self):
        # parse_benchmark returns QNN-internal measurements; run_once adds
        # process-level durations before median_metrics is called. Mimic
        # the complete production records, including these four fields.
        baseline = parse_benchmark(fake_log(), RESPONSE)
        baseline.update({
            "bpe_wall_ms": 120.0,
            "qnn_process_wall_ms": 800.0,
            "decoder_wall_ms": 90.0,
            "user_request_wall_ms": 1010.0,
        })
        runs = [dict(baseline, npu_prefill_ms=i) for i in (1.0, 2.0, 99.0)]
        summary = median_metrics(runs)
        self.assertEqual(summary["npu_prefill_ms"], 2.0)
        self.assertEqual(summary["bpe_wall_ms"], 120.0)
        self.assertEqual(summary["user_request_wall_ms"], 1010.0)


if __name__ == "__main__":
    unittest.main()
