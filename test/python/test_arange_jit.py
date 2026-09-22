import ctypes
import math
import os
import sys

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "numeric"))

import numpy as np
import torch

from mlir_numeric.ir import Context, Module
from mlir_numeric.passmanager import PassManager
from mlir_numeric.execution_engine import ExecutionEngine
from mlir_numeric.runtime import get_ranked_memref_descriptor
from mlir_numeric.dialects import numeric_nanobind as numeric_d


LOWER_TO_LLVM_PIPELINE = (
    "builtin.module("
    "convert-numeric-to-arith,"
    "one-shot-bufferize{bufferize-function-boundaries=true},"
    "convert-linalg-to-loops,"
    "convert-scf-to-cf,"
    "convert-arith-to-llvm,"
    "convert-cf-to-llvm,"
    "finalize-memref-to-llvm,"
    "convert-func-to-llvm,"
    "reconcile-unrealized-casts"
    ")"
)


def run_arange_jit(start_v, end_v, step_v):
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @arange_test(%start: i64, %end: i64, %step: i64, %out: memref<?xi64>)
            attributes {llvm.emit_c_interface} {
          %0 = numeric.arange %start, %end, %step : (i64, i64, i64) -> tensor<?xi64>
          %buf = bufferization.to_buffer %0 : tensor<?xi64> to memref<?xi64>
          memref.copy %buf, %out : memref<?xi64> to memref<?xi64>
          return
        }
        """)
        PassManager.parse(LOWER_TO_LLVM_PIPELINE).run(module.operation)
        engine = ExecutionEngine(module)

        n = math.ceil((end_v - start_v) / step_v)
        start = ctypes.c_int64(start_v)
        end = ctypes.c_int64(end_v)
        step = ctypes.c_int64(step_v)
        out_arr = np.zeros(n, dtype=np.int64)

        engine.invoke(
            "arange_test",
            ctypes.pointer(start),
            ctypes.pointer(end),
            ctypes.pointer(step),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(out_arr))),
        )
        return out_arr


def test_arange_matches_torch_basic():
    result = run_arange_jit(2, 20, 3)
    expected = torch.arange(2, 20, 3, dtype=torch.int64).numpy()
    assert np.array_equal(result, expected)


def test_arange_matches_torch_from_zero():
    result = run_arange_jit(0, 10, 1)
    expected = torch.arange(0, 10, 1, dtype=torch.int64).numpy()
    assert np.array_equal(result, expected)


def test_arange_matches_torch_larger_step():
    result = run_arange_jit(5, 100, 7)
    expected = torch.arange(5, 100, 7, dtype=torch.int64).numpy()
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    test_arange_matches_torch_basic()
    print("test_arange_matches_torch_basic: PASS")
    test_arange_matches_torch_from_zero()
    print("test_arange_matches_torch_from_zero: PASS")
    test_arange_matches_torch_larger_step()
    print("test_arange_matches_torch_larger_step: PASS")