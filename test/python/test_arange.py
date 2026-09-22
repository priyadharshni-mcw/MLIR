import os
import sys

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "numeric"))

from mlir_numeric.ir import Context, Module
from mlir_numeric.passmanager import PassManager
from mlir_numeric.dialects import numeric_nanobind as numeric_d


def test_arange_parses_and_verifies():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @arange_int(%start: i64, %end: i64, %step: i64) -> tensor<?xi64> {
          %0 = numeric.arange %start, %end, %step : (i64, i64, i64) -> tensor<?xi64>
          return %0 : tensor<?xi64>
        }
        """)
        module.operation.verify()
        assert "numeric.arange" in str(module)


def test_arange_lowers_to_linalg_generic():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @arange_int(%start: i64, %end: i64, %step: i64) -> tensor<?xi64> {
          %0 = numeric.arange %start, %end, %step : (i64, i64, i64) -> tensor<?xi64>
          return %0 : tensor<?xi64>
        }
        """)
        PassManager.parse("builtin.module(convert-numeric-to-arith)").run(module.operation)
        text = str(module)
        assert "numeric.arange" not in text
        assert "linalg.generic" in text
        assert "tensor.empty" in text


if __name__ == "__main__":
    test_arange_parses_and_verifies()
    print("test_arange_parses_and_verifies: PASS")
    test_arange_lowers_to_linalg_generic()
    print("test_arange_lowers_to_linalg_generic: PASS")