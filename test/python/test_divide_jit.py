import ctypes
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
    "func.func(tosa-to-linalg),"
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


def run_divide_jit(n_arr, d_arr, mlir_type, np_dtype):
    n = n_arr.shape[0]
    with Context():
        numeric_d.register_dialects()
        module = Module.parse(f"""
        func.func @divide_test(%n: memref<{n}x{mlir_type}>, %d: memref<{n}x{mlir_type}>,
                                %out: memref<{n}x{mlir_type}>)
            attributes {{llvm.emit_c_interface}} {{
          %n_t = bufferization.to_tensor %n restrict : memref<{n}x{mlir_type}> to tensor<{n}x{mlir_type}>
          %d_t = bufferization.to_tensor %d restrict : memref<{n}x{mlir_type}> to tensor<{n}x{mlir_type}>
          %0 = numeric.divide %n_t, %d_t : tensor<{n}x{mlir_type}>
          %buf = bufferization.to_buffer %0 : tensor<{n}x{mlir_type}> to memref<{n}x{mlir_type}>
          memref.copy %buf, %out : memref<{n}x{mlir_type}> to memref<{n}x{mlir_type}>
          return
        }}
        """)
        PassManager.parse(LOWER_TO_LLVM_PIPELINE).run(module.operation)
        engine = ExecutionEngine(module)

        out_arr = np.zeros(n, dtype=np_dtype)
        engine.invoke(
            "divide_test",
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(n_arr))),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(d_arr))),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(out_arr))),
        )
        return out_arr


def test_divide_int_matches_torch():
    n_arr = np.array([10, 20, 33, 7], dtype=np.int64)
    d_arr = np.array([3, 4, 5, 2], dtype=np.int64)
    result = run_divide_jit(n_arr, d_arr, "i64", np.int64)
    expected = torch.div(
        torch.from_numpy(n_arr), torch.from_numpy(d_arr), rounding_mode="trunc"
    ).numpy()
    print("MLIR numeric.divide (int) :", result)
    print("torch.div (trunc)         :", expected)
    assert np.array_equal(result, expected)


def test_divide_float_matches_torch():
    n_arr = np.array([10.0, 21.0, 33.0, 7.5], dtype=np.float32)
    d_arr = np.array([4.0, 3.0, 6.0, 2.5], dtype=np.float32)
    result = run_divide_jit(n_arr, d_arr, "f32", np.float32)
    expected = torch.div(torch.from_numpy(n_arr), torch.from_numpy(d_arr)).numpy()
    print("MLIR numeric.divide (float):", result)
    print("torch.div                 :", expected)
    assert np.allclose(result, expected, rtol=1e-5)


if __name__ == "__main__":
    test_divide_int_matches_torch()
    print("test_divide_int_matches_torch: PASS")
    test_divide_float_matches_torch()
    print("test_divide_float_matches_torch: PASS")