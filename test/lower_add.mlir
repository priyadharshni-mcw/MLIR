// RUN: numeric-opt %s -convert-numeric-to-arith | FileCheck %s

// CHECK-LABEL: func.func @lower_add_scalar_float
func.func @lower_add_scalar_float(%lhs: f32, %rhs: f32) -> f32 {
  // CHECK: arith.addf
  // CHECK-NOT: numeric.add
  %0 = numeric.add %lhs, %rhs : f32
  return %0 : f32
}

// CHECK-LABEL: func.func @lower_add_scalar_int
func.func @lower_add_scalar_int(%lhs: i64, %rhs: i64) -> i64 {
  // CHECK: arith.addi
  // CHECK-NOT: numeric.add
  %0 = numeric.add %lhs, %rhs : i64
  return %0 : i64
}

// CHECK-LABEL: func.func @lower_add_tensor
func.func @lower_add_tensor(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
  // CHECK: arith.addi
  // CHECK-NOT: numeric.add
  %0 = numeric.add %lhs, %rhs : tensor<4xi32>
  return %0 : tensor<4xi32>
}