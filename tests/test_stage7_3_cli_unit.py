#!/usr/bin/env python3
"""Unit tests for terminal Gemma 4 log decoder; QNN hardware not required."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gemma4_decode_log import (  # noqa: E402
    GenerationParseError, decode_ids, parse_dynamic,
)

PREFIX = "\n".join([
    "DYNAMIC_PROMPT_READY: chat_tokens=18 prefill_tokens=17",
    "DYNAMIC_PREFILL_NPU_OK: inputs=36 outputs=30 valid_rows=17",
    "DYNAMIC_KV_BRIDGE_OK: from_prefill=30 prefix_rows=17",
])
TOKEN1 = ("DYNAMIC_GENERATED_TOKEN: step=1 response_pos=18 id=236778"
          " max_q=119 ties=1 input_pos=17 input_token=107")
TOKEN2 = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=106"
          " max_q=118 ties=1 input_pos=18 input_token=236778")
VOCAB = {236778: "2", 900: "▁+", 106: "<turn|>",
         30: "<0xC3>", 31: "<0xA9>"}


class Stage73Unit(unittest.TestCase):
    def test_decode_byte_fallback(self):
        self.assertEqual(decode_ids([30, 31, 900], VOCAB), "é +")

    def test_eos_excluded_from_response(self):
        log = "\n".join([
            PREFIX, TOKEN1, TOKEN2,
            "DYNAMIC_STOP_EOS: step=2 eos_id=106 eos_not_submitted_to_decode=1",
            "DYNAMIC_GENERATION_OK: npu_decode_steps=2 text_tokens=1"
            " stop_reason=eos saved_QNN_inputs_used=0",
        ])
        r = parse_dynamic(log, VOCAB)
        self.assertEqual(r["text"], "2")
        self.assertEqual(r["ids"], [236778, 106])
        self.assertEqual(r["stop_reason"], "eos")
        self.assertTrue(r["complete"])

    def test_limit_marked_incomplete(self):
        log = "\n".join([
            PREFIX, TOKEN1,
            "DYNAMIC_GENERATION_OK: npu_decode_steps=1 text_tokens=1"
            " stop_reason=limit",
        ])
        r = parse_dynamic(log, VOCAB)
        self.assertEqual(r["text"], "2")
        self.assertEqual(r["stop_reason"], "limit")
        self.assertFalse(r["complete"])

    def test_tie_candidate_never_rendered(self):
        tied = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=900"
                " max_q=118 ties=5 input_pos=18 input_token=236778")
        log = "\n".join([
            PREFIX, TOKEN1, tied,
            "DYNAMIC_STOP_TIE: step=2 max_q=118 ties=5"
            " no_arbitrary_tiebreak=1",
        ])
        r = parse_dynamic(log, VOCAB)
        self.assertEqual(r["ids"], [236778])
        self.assertEqual(r["text"], "2")
        self.assertEqual(r["stop_reason"], "tie")
        self.assertFalse(r["complete"])

    def test_invalid_eos_marker_rejected(self):
        log = "\n".join([
            PREFIX, TOKEN1, TOKEN2,
            "DYNAMIC_GENERATION_OK: npu_decode_steps=2 text_tokens=1"
            " stop_reason=eos",
        ])
        with self.assertRaisesRegex(GenerationParseError, "EOS stop"):
            parse_dynamic(log, VOCAB)

    def test_nonconsecutive_steps_rejected(self):
        wrong = ("DYNAMIC_GENERATED_TOKEN: step=3 response_pos=19 id=106"
                 " max_q=118 ties=1")
        log = "\n".join([
            PREFIX, TOKEN1, wrong,
            "DYNAMIC_STOP_EOS: step=3 eos_id=106",
            "DYNAMIC_GENERATION_OK: npu_decode_steps=2 text_tokens=1"
            " stop_reason=eos",
        ])
        with self.assertRaisesRegex(GenerationParseError, "Nonconsecutive"):
            parse_dynamic(log, VOCAB)


    def test_tie_candidates_shown_but_no_arbitrary_output(self):
        tied = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=900"
                " max_q=118 ties=2 input_pos=18 input_token=236778")
        details = ("DYNAMIC_TIE_CANDIDATES: step=2 max_q=118 ties=2"
                   " candidate_ids=900,1234 shown=2 truncated=0"
                   " quantized_logits_only=1 no_fp32_ranking_claim=1")
        log = "\n".join([
            PREFIX, TOKEN1, details, tied,
            "DYNAMIC_STOP_TIE: step=2 max_q=118 ties=2"
            " no_arbitrary_tiebreak=1",
        ])
        r = parse_dynamic(log, VOCAB)
        self.assertEqual(r["ids"], [236778])
        self.assertEqual(r["text"], "2")
        self.assertEqual(r["stop_reason"], "tie")
        self.assertEqual(r["tie_details"][-1]["candidate_ids"], [900, 1234])
        self.assertEqual(r["tie_details"][-1]["candidate_pieces"][0], "▁+")
        self.assertFalse(r["experimental_tie_break"])

    def test_explicit_lowest_id_tie_resolves_with_warning_metadata(self):
        tied = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=900"
                " max_q=118 ties=2 input_pos=18 input_token=236778")
        details = ("DYNAMIC_TIE_CANDIDATES: step=2 max_q=118 ties=2"
                   " candidate_ids=900,1234 shown=2 truncated=0"
                   " quantized_logits_only=1 no_fp32_ranking_claim=1")
        resolved = ("DYNAMIC_TIE_RESOLVED: step=2 ties=2 chosen_id=900"
                    " policy=lowest-id-experimental no_fp32_ranking_claim=1")
        third = ("DYNAMIC_GENERATED_TOKEN: step=3 response_pos=20 id=106"
                 " max_q=120 ties=1 input_pos=19 input_token=900")
        log = "\n".join([
            PREFIX, TOKEN1, details, tied, resolved, third,
            "DYNAMIC_STOP_EOS: step=3 eos_id=106 eos_not_submitted_to_decode=1",
            "DYNAMIC_GENERATION_OK: npu_decode_steps=3 text_tokens=2"
            " tie_resolutions=1 stop_reason=eos",
        ])
        r = parse_dynamic(log, VOCAB)
        self.assertEqual(r["ids"], [236778, 900, 106])
        self.assertEqual(r["text"], "2 +")
        self.assertTrue(r["complete"])
        self.assertTrue(r["experimental_tie_break"])
        self.assertEqual(r["tie_resolutions"], 1)

    def test_forged_tie_resolution_is_rejected(self):
        tied = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=900"
                " max_q=118 ties=2")
        wrong = ("DYNAMIC_TIE_RESOLVED: step=2 ties=2 chosen_id=1234"
                 " policy=lowest-id-experimental no_fp32_ranking_claim=1")
        log = "\n".join([
            PREFIX, TOKEN1, tied, wrong,
            "DYNAMIC_GENERATION_OK: npu_decode_steps=2 text_tokens=2"
            " tie_resolutions=1 stop_reason=limit",
        ])
        with self.assertRaisesRegex(GenerationParseError, "Untrusted tie"):
            parse_dynamic(log, VOCAB)

    def test_experiment_requires_tie_count_in_final_summary(self):
        tied = ("DYNAMIC_GENERATED_TOKEN: step=2 response_pos=19 id=900"
                " max_q=118 ties=2")
        resolved = ("DYNAMIC_TIE_RESOLVED: step=2 ties=2 chosen_id=900"
                    " policy=lowest-id-experimental no_fp32_ranking_claim=1")
        log = "\n".join([
            PREFIX, TOKEN1, tied, resolved,
            "DYNAMIC_GENERATION_OK: npu_decode_steps=2 text_tokens=2"
            " stop_reason=limit",
        ])
        with self.assertRaisesRegex(GenerationParseError, "Unreported"):
            parse_dynamic(log, VOCAB)


if __name__ == "__main__":
    unittest.main()
