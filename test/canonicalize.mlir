// RUN: numeric-opt %s -canonicalize | FileCheck %s

// CHECK-LABEL: func.func @fold_add
func.func @fold_add() -> i64 {
  %a = "numeric.constant"() <{value = 3 : i64}> : () -> i64
  %b = "numeric.constant"() <{value = 4 : i64}> : () -> i64
  // CHECK-NOT: numeric.add
  // CHECK: "numeric.constant"() <{value = 7 : i64}>
  %s = numeric.add %a, %b : i64
  return %s : i64
}

// CHECK-LABEL: func.func @fold_add_tensor
func.func @fold_add_tensor() -> tensor<4xi32> {
  %a = "numeric.constant"() <{value = dense<[1, 2, 3, 4]> : tensor<4xi32>}> : () -> tensor<4xi32>
  %b = "numeric.constant"() <{value = dense<[10, 20, 30, 40]> : tensor<4xi32>}> : () -> tensor<4xi32>
  // CHECK-NOT: numeric.add
  // CHECK: dense<[11, 22, 33, 44]>
  %s = numeric.add %a, %b : tensor<4xi32>
  return %s : tensor<4xi32>
}

// CHECK-LABEL: func.func @add_zero
func.func @add_zero(%x: i64) -> i64 {
  %z = "numeric.constant"() <{value = 0 : i64}> : () -> i64
  // CHECK-NOT: numeric.add
  // CHECK: return %arg0
  %s = numeric.add %x, %z : i64
  return %s : i64
}

// CHECK-LABEL: func.func @mul_one
func.func @mul_one(%x: i64) -> i64 {
  %one = "numeric.constant"() <{value = 1 : i64}> : () -> i64
  // CHECK-NOT: numeric.mul
  // CHECK: return %arg0
  %p = numeric.mul %one, %x : i64
  return %p : i64
}

// CHECK-LABEL: func.func @mul_zero
func.func @mul_zero(%x: i64) -> i64 {
  %z = "numeric.constant"() <{value = 0 : i64}> : () -> i64
  // CHECK-NOT: numeric.mul
  // CHECK: "numeric.constant"() <{value = 0 : i64}>
  %p = numeric.mul %x, %z : i64
  return %p : i64
}

// CHECK-LABEL: func.func @sub_self
func.func @sub_self(%x: i64) -> i64 {
  // CHECK-NOT: numeric.sub
  // CHECK: "numeric.constant"() <{value = 0 : i64}>
  %d = numeric.sub %x, %x : i64
  return %d : i64
}

// Negative test: float x - x must NOT fold (NaN and inf break the identity).
// CHECK-LABEL: func.func @float_sub_self_is_kept
func.func @float_sub_self_is_kept(%x: f32) -> f32 {
  // CHECK: numeric.sub
  %d = numeric.sub %x, %x : f32
  return %d : f32
}

// CHECK-LABEL: func.func @divide_by_one
func.func @divide_by_one(%x: tensor<4xi64>) -> tensor<4xi64> {
  %one = "numeric.constant"() <{value = dense<1> : tensor<4xi64>}> : () -> tensor<4xi64>
  // CHECK-NOT: numeric.divide
  // CHECK: return %arg0
  %q = numeric.divide %x, %one : tensor<4xi64>
  return %q : tensor<4xi64>
}