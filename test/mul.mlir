// RUN: numeric-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @mul_scalar_int
  func.func @mul_scalar_int(%lhs: i64, %rhs: i64) -> i64 {
    // CHECK: numeric.mul %{{.*}}, %{{.*}} : i64
    %0 = numeric.mul %lhs, %rhs : i64
    return %0 : i64
  }

  // CHECK-LABEL: func.func @mul_scalar_float
  func.func @mul_scalar_float(%lhs: f32, %rhs: f32) -> f32 {
    // CHECK: numeric.mul %{{.*}}, %{{.*}} : f32
    %0 = numeric.mul %lhs, %rhs : f32
    return %0 : f32
  }

  // CHECK-LABEL: func.func @mul_tensor
  func.func @mul_tensor(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
    // CHECK: numeric.mul %{{.*}}, %{{.*}} : tensor<4xi32>
    %0 = numeric.mul %lhs, %rhs : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}