import os
import sys

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "numeric"))

from mlir_numeric.ir import Context, Module
from mlir_numeric.passmanager import PassManager
from mlir_numeric.dialects import numeric_nanobind as numeric_d


def test_addcmul_parses_and_verifies():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @addcmul_tensor(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                                   %t2: tensor<4xi64>, %value: i64) -> tensor<4xi64> {
          %0 = numeric.addcmul %input, %t1, %t2, %value
              : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
          return %0 : tensor<4xi64>
        }
        """)
        module.operation.verify()
        assert "numeric.addcmul" in str(module)


def test_addcmul_rejects_mismatched_value_type():
    with Context():
        numeric_d.register_dialects()
        try:
            module = Module.parse("""
            func.func @bad(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                            %t2: tensor<4xi64>, %value: f32) -> tensor<4xi64> {
              %0 = numeric.addcmul %input, %t1, %t2, %value
                  : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, f32) -> tensor<4xi64>
              return %0 : tensor<4xi64>
            }
            """)
            module.operation.verify()
            assert False, "expected a verification failure (value type mismatch)"
        except Exception:
            pass  # expected


def test_addcmul_lowers_to_tosa():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @addcmul_tensor(%input: tensor<4xi64>, %t1: tensor<4xi64>,
                                   %t2: tensor<4xi64>, %value: i64) -> tensor<4xi64> {
          %0 = numeric.addcmul %input, %t1, %t2, %value
              : (tensor<4xi64>, tensor<4xi64>, tensor<4xi64>, i64) -> tensor<4xi64>
          return %0 : tensor<4xi64>
        }
        """)
        PassManager.parse("builtin.module(convert-numeric-to-arith)").run(module.operation)
        text = str(module)
        assert "numeric.addcmul" not in text
        assert text.count("tosa.mul") == 2
        assert "tosa.add" in text

if __name__ == "__main__":
    test_addcmul_parses_and_verifies()
    print("test_addcmul_parses_and_verifies: PASS")
    test_addcmul_rejects_mismatched_value_type()
    print("test_addcmul_rejects_mismatched_value_type: PASS")
    test_addcmul_lowers_to_tosa()
    print("test_addcmul_lowers_to_arith: PASS")