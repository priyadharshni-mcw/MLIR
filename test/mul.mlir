module {
  func.func @mul_scalar_int(%lhs: i64, %rhs: i64) -> i64 {
    %0 = numeric.mul %lhs, %rhs : i64
    return %0 : i64
  }

  func.func @mul_scalar_float(%lhs: f32, %rhs: f32) -> f32 {
    %0 = numeric.mul %lhs, %rhs : f32
    return %0 : f32
  }

  func.func @mul_tensor(%lhs: tensor<4xi32>, %rhs: tensor<4xi32>) -> tensor<4xi32> {
    %0 = numeric.mul %lhs, %rhs : tensor<4xi32>
    return %0 : tensor<4xi32>
  }
}