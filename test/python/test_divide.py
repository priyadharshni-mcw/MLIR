import os
import sys

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "numeric"))

from mlir_numeric.ir import Context, Module
from mlir_numeric.passmanager import PassManager
from mlir_numeric.dialects import numeric_nanobind as numeric_d


def test_divide_parses_and_verifies():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @divide_int(%n: tensor<4xi64>, %d: tensor<4xi64>) -> tensor<4xi64> {
          %0 = numeric.divide %n, %d : tensor<4xi64>
          return %0 : tensor<4xi64>
        }
        """)
        module.operation.verify()
        assert "numeric.divide" in str(module)


def test_divide_int_lowers_to_intdiv():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @divide_int(%n: tensor<4xi64>, %d: tensor<4xi64>) -> tensor<4xi64> {
          %0 = numeric.divide %n, %d : tensor<4xi64>
          return %0 : tensor<4xi64>
        }
        """)
        PassManager.parse("builtin.module(convert-numeric-to-arith)").run(module.operation)
        text = str(module)
        assert "numeric.divide" not in text
        assert "tosa.intdiv" in text


def test_divide_float_lowers_to_reciprocal_mul():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @divide_float(%n: tensor<4xf32>, %d: tensor<4xf32>) -> tensor<4xf32> {
          %0 = numeric.divide %n, %d : tensor<4xf32>
          return %0 : tensor<4xf32>
        }
        """)
        PassManager.parse("builtin.module(convert-numeric-to-arith)").run(module.operation)
        text = str(module)
        assert "numeric.divide" not in text
        assert "tosa.reciprocal" in text
        assert "tosa.mul" in text






if __name__ == "__main__":
    test_divide_parses_and_verifies()
    print("test_divide_parses_and_verifies: PASS")
    test_divide_int_lowers_to_intdiv()
    print("test_divide_int_lowers_to_intdiv: PASS")
    test_divide_float_lowers_to_reciprocal_mul()
    print("test_divide_float_lowers_to_reciprocal_mul: PASS")
   
    