#!/usr/bin/env python3
"""Regression tests for the exact Stage 6.3 Prefill causal-padding mask fix."""
import pathlib
import tempfile

from test_cpu_prefill_aux import apply_query_padding, padding_start

TOKENS=128
ROW=4096+TOKENS

def synthetic():
    # Build a causal INT8 mask, then mask all padded query rows in reference.
    actual=bytearray([128])*(TOKENS*ROW)
    for r in range(TOKENS):
        for c in range(r+1):
            actual[r*ROW+4096+c]=127
    expected=bytearray(actual)
    for r in range(17,TOKENS):
        for c in range(r+1):
            expected[r*ROW+4096+c]=128
    return bytes(actual),bytes(expected)

def attempt(actual,reference,should_pass,expected_error=None):
    with tempfile.TemporaryDirectory(prefix="prefill_padding_unit_") as directory:
        path=pathlib.Path(directory)
        cpu=path/"cpu.raw"
        saved=path/"qnn_reference.raw"
        cpu.write_bytes(actual)
        saved.write_bytes(reference)
        try:
            apply_query_padding(cpu,saved,TOKENS*ROW,17,"mask_global")
            success=True
        except ValueError as error:
            success=False
            if expected_error and expected_error not in str(error):
                raise AssertionError(f"Unexpected error: {error}") from error
        if success != should_pass:
            raise AssertionError(
                f"Expected success={should_pass}, got success={success}")
        if saved.read_bytes()!=reference:
            raise AssertionError("Reference was modified")
        if should_pass and cpu.read_bytes()!=reference:
            raise AssertionError("CPU padding correction not byte-identical")

def main():
    assert padding_start([2]*17+[0]*111)==17
    try:
        padding_start([2]*17+[0,2]+[0]*109)
    except ValueError:
        pass
    else:
        raise AssertionError("Noncontiguous zero padding was accepted")
    original,reference=synthetic()
    assert sum(a!=b for a,b in zip(original,reference))==8103
    attempt(original,reference,True)
    # An extra wrong byte in the cache columns must fail.
    damaged=bytearray(original)
    damaged[0]^=1
    attempt(bytes(damaged),reference,False,"PREFILL_PADDING_UNEXPECTED_DIFF_COUNT")
    # A wrong byte outside the permitted triangle must be rejected even
    # when the total mismatch count is kept unchanged.
    altered_reference=bytearray(reference)
    altered_reference[17*ROW+4096]=127
    altered_reference[0]^=1
    attempt(original,bytes(altered_reference),False,"PREFILL_PADDING_UNEXPECTED_VALUE")
    print("PREFILL_PADDING_UNIT_OK: 8103 expected bytes, strict negative cases, reference immutable")

if __name__=="__main__":
    main()
