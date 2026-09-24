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


def run_addcmul_jit(input_arr, t1_arr, t2_arr, value):
    n = input_arr.shape[0]
    with Context():
        numeric_d.register_dialects()
        module = Module.parse(f"""
        func.func @addcmul_test(%input: memref<{n}xi64>, %t1: memref<{n}xi64>,
                                 %t2: memref<{n}xi64>, %value: i64, %out: memref<{n}xi64>)
            attributes {{llvm.emit_c_interface}} {{
          %in_t = bufferization.to_tensor %input restrict : memref<{n}xi64> to tensor<{n}xi64>
          %t1_t = bufferization.to_tensor %t1 restrict : memref<{n}xi64> to tensor<{n}xi64>
          %t2_t = bufferization.to_tensor %t2 restrict : memref<{n}xi64> to tensor<{n}xi64>
          %0 = numeric.addcmul %in_t, %t1_t, %t2_t, %value
              : (tensor<{n}xi64>, tensor<{n}xi64>, tensor<{n}xi64>, i64) -> tensor<{n}xi64>
          %buf = bufferization.to_buffer %0 : tensor<{n}xi64> to memref<{n}xi64>
          memref.copy %buf, %out : memref<{n}xi64> to memref<{n}xi64>
          return
        }}
        """)
        PassManager.parse(LOWER_TO_LLVM_PIPELINE).run(module.operation)
        engine = ExecutionEngine(module)

        value_c = ctypes.c_int64(value)
        out_arr = np.zeros(n, dtype=np.int64)

        engine.invoke(
            "addcmul_test",
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(input_arr))),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(t1_arr))),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(t2_arr))),
            ctypes.pointer(value_c),
            ctypes.pointer(ctypes.pointer(get_ranked_memref_descriptor(out_arr))),
        )
        return out_arr


def test_addcmul_matches_torch():
    input_arr = np.array([1, 2, 3, 4], dtype=np.int64)
    t1_arr = np.array([2, 2, 2, 2], dtype=np.int64)
    t2_arr = np.array([3, 4, 5, 6], dtype=np.int64)
    value = 2

    result = run_addcmul_jit(input_arr, t1_arr, t2_arr, value)
    expected = torch.addcmul(
        torch.from_numpy(input_arr),
        torch.from_numpy(t1_arr),
        torch.from_numpy(t2_arr),
        value=value,
    ).numpy()

    print("MLIR numeric.addcmul :", result)
    print("torch.addcmul        :", expected)
    assert np.array_equal(result, expected)


if __name__ == "__main__":
    test_addcmul_matches_torch()
    print("test_addcmul_matches_torch: PASS")