// RUN: numeric-opt %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @add_scalar
  func.func @add_scalar(%lhs: i64, %rhs: i64) -> i64 {
    // CHECK: numeric.add %{{.*}}, %{{.*}} : i64
    %0 = numeric.add %lhs, %rhs : i64
    return %0 : i64
  }

  // CHECK-LABEL: func.func @add_tensor
  func.func @add_tensor(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
    // CHECK: numeric.add %{{.*}}, %{{.*}} : tensor<4xi32>
    %0 = numeric.add %lhs, %rhs : tensor<4xi32>
    return %0 : tensor<4xi32>
  }

  // CHECK-LABEL: func.func @add_tensor_float
  func.func @add_tensor_float(%lhs: tensor<4xf32>, %rhs: tensor<4xf32>) -> tensor<4xf32> {
    // CHECK: numeric.add %{{.*}}, %{{.*}} : tensor<4xf32>
    %0 = numeric.add %lhs, %rhs : tensor<4xf32>
    return %0 : tensor<4xf32>
  }
}