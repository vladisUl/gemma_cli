#!/usr/bin/env python3
"""Stage 8.2 parser checks — run without HTP hardware."""
import sys
import unittest
from pathlib import Path

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from bench_stage8_2 import (  # noqa: E402
    BenchError, compare_checksums, get_digests,
    assert_registration, check_profile
)


def digest_fixture(steps=3, flip=False):
    result=[]
    for i in range(1,steps+1):
        head=20+i+(1 if flip and i==3 else 0)
        result.append(
            f"STAGE82_CHECKSUM: step={i} decode_fnv={10+i}"
            f" head_fnv={head} comparable_raw_memhandle=1"
        )
    return "\n".join(result)


class Stage82Unit(unittest.TestCase):
    def test_checksum_exact_match(self):
        data=digest_fixture()
        self.assertEqual(compare_checksums(data,data,3),3)
        self.assertEqual(len(get_digests(data,3)),3)

    def test_checksum_mismatch_aborts(self):
        with self.assertRaisesRegex(BenchError,"outputs differ"):
            compare_checksums(digest_fixture(),digest_fixture(flip=True),3)

    def test_checksum_missing_step_aborts(self):
        with self.assertRaisesRegex(BenchError,"Missing or reordered"):
            compare_checksums(digest_fixture(),digest_fixture(2),3)

    def test_checksum_duplicate_step_aborts(self):
        corrupted=digest_fixture()+"\n"+digest_fixture(1)
        with self.assertRaisesRegex(BenchError,"Duplicate"):
            get_digests(corrupted,3)

    def test_checksum_malformed_aborts(self):
        with self.assertRaisesRegex(BenchError,"Malformed"):
            get_digests("STAGE82_CHECKSUM: invalid",1)

    def test_mem_registration_requires_full_validation(self):
        good="\n".join([
            "STAGE82_MEM_REGISTERED: tensors=30 bytes=37748736",
            "STAGE82_KV_MIRROR_OK: stage=initial full_cache_equal=1 tensors=30",
            "STAGE82_KV_MIRROR_OK: stage=final full_cache_equal=1 tensors=30",
            "BENCH81_OK: executed_steps=8 kv_binding=memhandle",
        ])
        assert_registration(good,8,True)
        with self.assertRaisesRegex(BenchError,"MEMHANDLE run missing"):
            assert_registration(good.replace(
                "stage=final full_cache_equal=1","stage=final full_cache_equal=0"
            ),8,True)

    def test_partial_memhandle_probe_contract(self):
        mixed="\n".join([
            "STAGE82_MEM_REGISTERED: tensors=1 bytes=1258291 "
            "requested=1 raw_remaining=29",
            "STAGE82_KV_MIRROR_OK: stage=initial full_cache_equal=1 tensors=1",
            "STAGE82_KV_MIRROR_OK: stage=final full_cache_equal=1 tensors=1",
            "BENCH81_OK: executed_steps=8 kv_binding=memhandle",
        ])
        assert_registration(mixed,8,True,expected_tensors=1)
        with self.assertRaisesRegex(BenchError,"MEMHANDLE run missing"):
            assert_registration(mixed,8,True,expected_tensors=30)

    def test_raw_binding_requires_no_registered_kv(self):
        raw="BENCH81_OK: executed_steps=8 kv_binding=raw"
        assert_registration(raw,8,False)
        with self.assertRaisesRegex(BenchError,"RAW run"):
            assert_registration(raw+" STAGE82_MEM_REGISTERED:",8,False)

    def test_profile_requires_sampled_decode_and_head(self):
        # Steps=3 => sample 1,2,3, each Decode and Head exactly once.
        calls=[
            f"STAGE82_PROFILE_CALL: graph={graph} step={i} "
            "graph_call_ms=20 top_events=1 level=basic"
            for i in (1,2,3) for graph in ("decode","head")
        ]
        profile="\n".join(calls+[
            "STAGE82_PROFILE_EVENT: graph=decode step=1 depth=0 "
            "type=3001 unit=1 value=40 identifier=RPC",
        ])
        parsed=check_profile(profile,3,"basic")
        self.assertEqual(parsed["graph_calls"],6)
        self.assertEqual(parsed["profile_event_types"],[3001])
        with self.assertRaisesRegex(BenchError,"Expected"):
            check_profile("\n".join(calls[:-1]+[
                "STAGE82_PROFILE_EVENT: graph=decode step=1 depth=0 "
                "type=3001 unit=1 value=40 identifier=RPC",
            ]),3,"basic")


if __name__=="__main__":
    unittest.main()
