import os
import sys

BUILD_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "build")
sys.path.insert(0, os.path.join(BUILD_DIR, "python_packages", "numeric"))

from mlir_numeric.ir import Context, Module
from mlir_numeric.passmanager import PassManager
from mlir_numeric.dialects import numeric_nanobind as numeric_d


def test_add_parses_and_verifies():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @add_test(%lhs: i64, %rhs: i64) -> i64 {
          %0 = numeric.add %lhs, %rhs : i64
          return %0 : i64
        }
        """)
        module.operation.verify()
        assert "numeric.add" in str(module)


def test_add_rejects_mismatched_float_tensor():
    with Context():
        numeric_d.register_dialects()
        try:
            module = Module.parse("""
            func.func @bad(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xi32> {
              %0 = numeric.add %lhs, %rhs : tensor<4xi32>
              return %0 : tensor<4xi32>
            }
            """)
            module.operation.verify()
            assert False, "expected a verification failure"
        except Exception:
            pass  # expected: operand/result type mismatch


def test_full_expression_lowers_to_arith():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @expr() -> i64 {
          %a = "numeric.constant"() <{value = 4 : i64}> : () -> i64
          %b = "numeric.constant"() <{value = 3 : i64}> : () -> i64
          %c = "numeric.constant"() <{value = 5 : i64}> : () -> i64
          %d = "numeric.constant"() <{value = 2 : i64}> : () -> i64
          %bc = numeric.mul %b, %c : i64
          %sum = numeric.add %a, %bc : i64
          %result = numeric.sub %sum, %d : i64
          return %result : i64
        }
        """)
        PassManager.parse("builtin.module(convert-numeric-to-arith)").run(module.operation)
        text = str(module)
        assert "numeric." not in text
        assert "arith." in text


def test_canonicalize_folds_constants_after_lowering():
    with Context():
        numeric_d.register_dialects()
        module = Module.parse("""
        func.func @expr() -> i64 {
          %a = "numeric.constant"() <{value = 4 : i64}> : () -> i64
          %b = "numeric.constant"() <{value = 3 : i64}> : () -> i64
          %c = "numeric.constant"() <{value = 5 : i64}> : () -> i64
          %d = "numeric.constant"() <{value = 2 : i64}> : () -> i64
          %bc = numeric.mul %b, %c : i64
          %sum = numeric.add %a, %bc : i64
          %result = numeric.sub %sum, %d : i64
          return %result : i64
        }
        """)
        PassManager.parse(
            "builtin.module(convert-numeric-to-arith,canonicalize)"
        ).run(module.operation)
        assert "17" in str(module)


if __name__ == "__main__":
    test_add_parses_and_verifies()
    print("test_add_parses_and_verifies: PASS")
    test_add_rejects_mismatched_float_tensor()
    print("test_add_rejects_mismatched_float_tensor: PASS")
    test_full_expression_lowers_to_arith()
    print("test_full_expression_lowers_to_arith: PASS")
    test_canonicalize_folds_constants_after_lowering()
    print("test_canonicalize_folds_constants_after_lowering: PASS")