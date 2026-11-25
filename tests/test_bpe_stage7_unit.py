#!/usr/bin/env python3
"""Unit tests for Stage 7.1; use tiny in-memory vocab, no 32 MB Q8B file."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from gemma4_bpe import GemmaBPE, ContractError  # noqa: E402


def fixture():
    tok = object.__new__(GemmaBPE)
    tok.forward = {
        "a": 10, "b": 11, "c": 12, "▁": 13,
        "ab": 14, "bc": 15, "abc": 16,
        "▁b": 17, "a▁b": 18,
        "<0xC3>": 30, "<0xA9>": 31,
        "<unk>": 3,
    }
    tok.reverse = {v: k for k, v in tok.forward.items()}
    tok.ranks = {
        (11, 12): (0, 15),
        (10, 15): (1, 16),
        (10, 11): (2, 14),
        (13, 11): (3, 17),
        (10, 17): (4, 18),
    }
    tok.special = ["<bos>"]
    return tok


class BpeUnit(unittest.TestCase):
    def test_rank_order_not_greedy_vocab(self):
        tok = fixture()
        self.assertEqual(tok.encode_plain("abc"), [16])

    def test_merge_space_after_normalizer(self):
        tok = fixture()
        self.assertEqual(tok.encode_plain("a b"), [18])
        self.assertEqual(tok.decode_ids([18]), "a b")

    def test_byte_fallback_unicode_roundtrip(self):
        tok = fixture()
        self.assertEqual(tok.encode_plain("é"), [30, 31])
        self.assertEqual(tok.decode_ids([30, 31]), "é")

    def test_unknown_fuses_consecutive_missing_codepoints(self):
        tok = fixture()
        self.assertEqual(tok.encode_plain("☃★"), [3])

    def test_disallow_added_token_literals_without_added_vocabulary(self):
        tok = fixture()
        with self.assertRaisesRegex(ContractError, "AddedVocabulary"):
            tok.encode_plain("hello <bos> world")

    def test_empty_input(self):
        tok = fixture()
        self.assertEqual(tok.encode_plain(""), [])


if __name__ == "__main__":
    unittest.main()
