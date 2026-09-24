// RUN: numeric-opt %s -convert-numeric-to-arith | FileCheck %s

// CHECK-LABEL: func.func @divide_int
func.func @divide_int(%n: tensor<4xi64>, %d: tensor<4xi64>) -> tensor<4xi64> {
  // CHECK: tosa.intdiv %arg0, %arg1
  // CHECK-NOT: numeric.
  %0 = numeric.divide %n, %d : tensor<4xi64>
  return %0 : tensor<4xi64>
}

// CHECK-LABEL: func.func @divide_float
func.func @divide_float(%n: tensor<4xf32>, %d: tensor<4xf32>) -> tensor<4xf32> {
  // CHECK: %[[RECIP:.*]] = tosa.reciprocal %arg1
  // CHECK: %[[SHIFT:.*]] = "tosa.const"()
  // CHECK: tosa.mul %arg0, %[[RECIP]], %[[SHIFT]]
  // CHECK-NOT: numeric.
  %0 = numeric.divide %n, %d : tensor<4xf32>
  return %0 : tensor<4xf32>
}

// CHECK-LABEL: func.func @add_scalar
func.func @add_scalar(%a: i64, %b: i64) -> i64 {
  // CHECK: arith.addi
  // CHECK-NOT: numeric.
  %0 = numeric.add %a, %b : i64
  return %0 : i64
}

// CHECK-LABEL: func.func @add_tensor
func.func @add_tensor(%a: tensor<4xi64>, %b: tensor<4xi64>) -> tensor<4xi64> {
  //