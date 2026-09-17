// RUN: numeric-opt %s -convert-numeric-to-arith -convert-arith-to-llvm -convert-func-to-llvm -reconcile-unrealized-casts | FileCheck %s

// CHECK-LABEL: llvm.func @expr_scalar_int
// CHECK-NOT: numeric.
// CHECK-NOT: arith.
// CHECK-NOT: func.func
func.func @expr_scalar_int() -> i64 {
  %a = "numeric.constant"() <{value = 4 : i64}> : () -> i64
  %b = "numeric.constant"() <{value = 3 : i64}> : () -> i64
  %c = "numeric.constant"() <{value = 5 : i64}> : () -> i64
  %d = "numeric.constant"() <{value = 2 : i64}> : () -> i64
  %bc = numeric.mul %b, %c : i64
  %a_plus_bc = numeric.add %a, %bc : i64
  %result = numeric.sub %a_plus_bc, %d : i64
  return %result : i64
}